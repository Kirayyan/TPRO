#!/usr/bin/env python3
"""Score one porto test set using saved porto_model.pkl (no train / no fit)."""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from sklearn.metrics import average_precision_score

from tpro_baseline.map_graph import RoadGraph
from tpro_baseline.run_test import match_routes, load_labels, MAP_AREA
from tpro_baseline.tpro import TPRO


def main() -> None:
  p = argparse.ArgumentParser()
  p.add_argument("--test", required=True, help="outliers_data_*.npy")
  p.add_argument("--labels", required=True, help="outliers_idx_*.npy")
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
    blob = pickle.load(f)
  model = blob["model"]

  graph = RoadGraph(ROOT / args.map_dir, area=MAP_AREA)
  test_trajs = np.load(test_path, allow_pickle=True)
  print("map-match test ...")
  test_routes = match_routes(graph, test_trajs, test_cache, "test")
  labels = load_labels(labels_path, len(test_trajs))
  test_for_score = [r if r else [] for r in test_routes]

  print("scoring ...")
  scores = np.array(model.anomaly_scores(test_for_score), dtype=np.float64)
  pr_auc = float(average_precision_score(labels, scores))
  print("test size:", len(labels), "positives:", int(labels.sum()))
  print("PR-AUC:", f"{pr_auc:.6f}")

  if args.scores_out:
    out = Path(args.scores_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, scores)


if __name__ == "__main__":
  main()
