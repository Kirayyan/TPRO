#!/usr/bin/env python3
"""Score one porto test set using saved porto_model.pkl (no train / no fit)."""

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

from tpro_baseline.map_graph import RoadGraph, mst_traj_to_edge_route
from tpro_baseline.tpro import TPRO

MAP_AREA = (
    41.140519 - (41.175893 - 41.140519) / 10,
    41.175893 + (41.175893 - 41.140519) / 10,
    -8.651993 - (-8.579304 + 8.651993) / 10,
    -8.579304 + (-8.579304 + 8.651993) / 10,
)


def load_labels(path: Path, num_test: int) -> np.ndarray:
  arr = np.load(path, allow_pickle=True)
  arr = np.asarray(arr).reshape(-1)
  if arr.size == num_test and np.max(arr) <= 1:
    return arr.astype(np.int32)
  labels = np.zeros(num_test, dtype=np.int32)
  for idx in arr:
    labels[int(idx)] = 1
  return labels


def match_routes(
    graph: RoadGraph,
    trajs: Sequence,
    cache: Optional[Path],
) -> List[List[int]]:
  if cache and cache.exists():
    with cache.open("rb") as f:
      return pickle.load(f)
  routes = []
  t0 = time.time()
  for i, traj in enumerate(trajs):
    routes.append(mst_traj_to_edge_route(graph, traj))
    if (i + 1) % 5000 == 0:
      print(f"[test] {i + 1}/{len(trajs)} ({time.time() - t0:.0f}s)")
  print(f"[test] map-match ok {sum(1 for r in routes if r)}/{len(trajs)}")
  if cache:
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
      pickle.dump(routes, f)
  return routes


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--test", required=True)
  p.add_argument("--labels", required=True)
  p.add_argument("--model_cache", default=".tpro_cache/porto_model.pkl")
  p.add_argument("--test_cache", default="")
  p.add_argument("--map_dir", default="map")
  p.add_argument("--scores_out", default="")
  args = p.parse_args()

  model_path = ROOT / args.model_cache
  test_path = Path(args.test)
  labels_path = Path(args.labels)
  test_cache = Path(args.test_cache) if args.test_cache else None

  print("load model", model_path)
  with model_path.open("rb") as f:
    model = pickle.load(f)["model"]

  graph = RoadGraph(ROOT / args.map_dir, area=MAP_AREA)
  test_trajs = np.load(test_path, allow_pickle=True)
  test_routes = match_routes(graph, test_trajs, test_cache)
  labels = load_labels(labels_path, len(test_trajs))

  scores = np.array(
      model.anomaly_scores([r if r else [] for r in test_routes]),
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
