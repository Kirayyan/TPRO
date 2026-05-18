#!/usr/bin/env python3
"""Score one cd test set using saved cd_model.pkl (grid TPRO, no road map)."""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import average_precision_score

from tpro_baseline.grid_tpro import trajs_to_grid_seqs


def load_labels(path: Path, num_test: int) -> np.ndarray:
  arr = np.load(path, allow_pickle=True)
  arr = np.asarray(arr).reshape(-1)
  if arr.size == num_test and np.max(arr) <= 1:
    return arr.astype(np.int32)
  labels = np.zeros(num_test, dtype=np.int32)
  for idx in arr:
    labels[int(idx)] = 1
  return labels


def load_grid_seqs(trajs: Sequence, cache: Optional[Path]) -> List[List[int]]:
  if cache and cache.exists():
    with cache.open("rb") as f:
      return pickle.load(f)
  seqs = trajs_to_grid_seqs(trajs)
  if cache:
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
      pickle.dump(seqs, f)
  return seqs


def main() -> None:
  p = argparse.ArgumentParser(description="cd grid TPRO scoring")
  p.add_argument("--test", required=True)
  p.add_argument("--labels", required=True)
  p.add_argument("--model_cache", default=".tpro_cache/cd_model.pkl")
  p.add_argument("--test_cache", default="")
  p.add_argument("--scores_out", default="")
  args = p.parse_args()

  model_path = ROOT / args.model_cache
  test_path = Path(args.test)
  labels_path = Path(args.labels)
  test_cache = Path(args.test_cache) if args.test_cache else None

  print("load model", model_path)
  with model_path.open("rb") as f:
    blob = pickle.load(f)
  if not blob.get("grid_only", True):
    print("error: model is not grid/cd mode")
    sys.exit(1)
  model = blob["model"]

  test_trajs = np.load(test_path, allow_pickle=True)
  print("grid encode test ...")
  t0 = time.time()
  test_seqs = load_grid_seqs(test_trajs, test_cache)
  print("done (%.1fs), n=%d" % (time.time() - t0, len(test_seqs)))

  labels = load_labels(labels_path, len(test_trajs))
  scores = np.array(
      model.anomaly_scores([s if len(s) >= 2 else [] for s in test_seqs]),
      dtype=np.float64,
  )
  pr_auc = float(average_precision_score(labels, scores))
  print("test size:", len(labels), "positives:", int(labels.sum()))
  print("PR-AUC:", f"{pr_auc:.6f}")

  if args.scores_out:
    out = Path(args.scores_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, scores)
    print("saved", out)


if __name__ == "__main__":
  main()
