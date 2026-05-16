"""
Evaluate TPRO on MST-OATD Porto data with the same labels and PR-AUC metric.

Expected MST-OATD layout (after preprocess + generate_outliers):
  <mst_oatd_root>/data/porto/train_data_init.npy
  <mst_oatd_root>/data/porto/outliers_data_init_{d}_{fraction}_{rho}.npy
  <mst_oatd_root>/data/porto/outliers_idx_init_{d}_{fraction}_{rho}.npy
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np
from sklearn.metrics import average_precision_score

from .map_graph import RoadGraph, load_edge_routes_csv, mst_traj_to_edge_route
from .tpro import TPRO

# TPRO.cpp TPROTest defaults
DEFAULT_LON_BLOCKS = 10
DEFAULT_LAT_BLOCKS = 20
DEFAULT_TOP_K = 5

# Porto map-matching region used by the original TPRO release (area +/- 10% padding)
_MIN_LAT, _MAX_LAT = 41.140519, 41.175893
_MIN_LON, _MAX_LON = -8.651993, -8.579304
_D_LAT = _MAX_LAT - _MIN_LAT
_D_LON = _MAX_LON - _MIN_LON
PORTO_AREA = (
    _MIN_LAT - _D_LAT / 10,
    _MAX_LAT + _D_LAT / 10,
    _MIN_LON - _D_LON / 10,
    _MAX_LON + _D_LON / 10,
)


def pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
  """Same metric as MST-OATD utils.auc_score (average_precision_score)."""
  return float(average_precision_score(labels, scores))


def trajectories_to_routes(
    graph: RoadGraph,
    trajs: Sequence,
    cache_path: Optional[Path],
    split_name: str,
) -> List[List[int]]:
  if cache_path and cache_path.exists():
    with cache_path.open("rb") as f:
      cached = pickle.load(f)
    print(f"[cache] loaded {len(cached)} routes from {cache_path}")
    return cached

  routes: List[List[int]] = []
  skipped = 0
  t0 = time.time()
  for i, traj in enumerate(trajs):
    route = mst_traj_to_edge_route(graph, traj)
    if route:
      routes.append(route)
    else:
      routes.append([])
      skipped += 1
    if (i + 1) % 5000 == 0:
      print(f"[{split_name}] map-matched {i + 1}/{len(trajs)} ({time.time() - t0:.1f}s)")

  valid = sum(1 for r in routes if r)
  print(f"[{split_name}] valid routes: {valid}/{len(trajs)}, empty: {skipped}")
  if cache_path:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("wb") as f:
      pickle.dump(routes, f)
    print(f"[cache] saved -> {cache_path}")
  return routes


def build_labels(num_test: int, outlier_indices: np.ndarray) -> np.ndarray:
  labels = np.zeros(num_test, dtype=np.int32)
  for idx in outlier_indices:
    labels[int(idx)] = 1
  return labels


def run_mst_oatd_eval(args: argparse.Namespace) -> float:
  mst_root = Path(args.mst_oatd_dir)
  data_dir = mst_root / "data" / args.dataset
  tag = f"{args.distance}_{args.fraction}_{args.obeserved_ratio}"

  train_path = data_dir / "train_data_init.npy"
  test_path = data_dir / f"outliers_data_init_{tag}.npy"
  idx_path = data_dir / f"outliers_idx_init_{tag}.npy"

  for p in (train_path, test_path, idx_path):
    if not p.exists():
      raise FileNotFoundError(
          f"Missing {p}.\n"
          f"If anomalies are already generated, use:\n"
          f"  python -m tpro_baseline.run_test --train <train.npy> --test <outliers.npy> --labels <idx.npy>"
      )

  train_trajs = np.load(train_path, allow_pickle=True)
  test_trajs = np.load(test_path, allow_pickle=True)
  outlier_idx = np.load(idx_path, allow_pickle=True)

  map_dir = Path(args.map_dir)
  min_lat, max_lat, min_lon, max_lon = PORTO_AREA
  graph = RoadGraph(map_dir, area=(min_lat, max_lat, min_lon, max_lon))

  cache_dir = Path(args.cache_dir) if args.cache_dir else None
  train_cache = cache_dir / f"train_routes_{args.dataset}.pkl" if cache_dir else None
  test_cache = cache_dir / f"test_routes_{args.dataset}_{tag}.pkl" if cache_dir else None

  print("Map-matching train trajectories...")
  train_routes = trajectories_to_routes(graph, train_trajs, train_cache, "train")
  print("Map-matching test trajectories...")
  test_routes = trajectories_to_routes(graph, test_trajs, test_cache, "test")

  train_routes = [r for r in train_routes if r]
  test_routes_valid = [r if r else [0] for r in test_routes]

  tpro = TPRO(
      graph,
      min_lat=41.140519,
      max_lat=41.175893,
      min_lon=-8.651993,
      max_lon=-8.579304,
      lon_blocks=args.lon_blocks,
      lat_blocks=args.lat_blocks,
      top_k=args.top_k,
  )
  print(f"Training TPRO on {len(train_routes)} routes...")
  t0 = time.time()
  tpro.fit(train_routes)
  print(f"Training done in {time.time() - t0:.1f}s")

  print("Scoring test trajectories...")
  scores = np.array(tpro.anomaly_scores(test_routes_valid), dtype=np.float64)
  labels = build_labels(len(test_trajs), outlier_idx)

  auc = pr_auc(labels, scores)
  print(f"PR-AUC: {auc:.6f}")
  print(f"Anomaly ratio in test set: {labels.mean():.4f}")

  if args.scores_out:
    out = Path(args.scores_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, scores)
    np.save(out.with_name(out.stem + "_labels.npy"), labels)
    print(f"Saved scores -> {out}")

  return auc


def run_edges_eval(args: argparse.Namespace) -> float:
  map_dir = Path(args.map_dir)
  min_lat, max_lat, min_lon, max_lon = PORTO_AREA
  graph = RoadGraph(map_dir, area=(min_lat, max_lat, min_lon, max_lon))

  train_routes = load_edge_routes_csv(args.train_edges)
  test_routes = load_edge_routes_csv(args.test_edges)
  labels = np.load(args.test_labels)

  tpro = TPRO(
      graph,
      min_lat=41.140519,
      max_lat=41.175893,
      min_lon=-8.651993,
      max_lon=-8.579304,
      lon_blocks=args.lon_blocks,
      lat_blocks=args.lat_blocks,
      top_k=args.top_k,
  )
  tpro.fit(train_routes)
  scores = np.array(tpro.anomaly_scores(test_routes), dtype=np.float64)
  auc = pr_auc(labels, scores)
  print(f"PR-AUC: {auc:.6f}")
  return auc


def main(argv: Optional[Sequence[str]] = None) -> int:
  parser = argparse.ArgumentParser(description="TPRO baseline evaluation (MST-OATD compatible)")
  parser.add_argument("--mode", choices=("mst_oatd", "edges"), default="mst_oatd")
  parser.add_argument("--mst_oatd_dir", type=str, default="../MST-OATD", help="MST-OATD repo root")
  parser.add_argument("--dataset", type=str, default="porto", choices=("porto", "cd"))
  parser.add_argument("--map_dir", type=str, default="TPRO/map", help="folder with nodeOSM.txt / edgeOSM.txt")
  parser.add_argument("--distance", type=int, default=2)
  parser.add_argument("--fraction", type=float, default=0.2)
  parser.add_argument("--obeserved_ratio", type=float, default=1.0)
  parser.add_argument("--lon_blocks", type=int, default=DEFAULT_LON_BLOCKS)
  parser.add_argument("--lat_blocks", type=int, default=DEFAULT_LAT_BLOCKS)
  parser.add_argument("--top_k", type=int, default=DEFAULT_TOP_K)
  parser.add_argument("--cache_dir", type=str, default=".tpro_cache", help="cache map-matched routes")
  parser.add_argument("--scores_out", type=str, default="", help="optional .npy path for scores")
  parser.add_argument("--train_edges", type=str, default="TPRO/cleaned_mm_edges.txt")
  parser.add_argument("--test_edges", type=str, default="")
  parser.add_argument("--test_labels", type=str, default="")
  args = parser.parse_args(argv)

  if args.mode == "mst_oatd":
    run_mst_oatd_eval(args)
  else:
    if not args.test_edges or not args.test_labels:
      parser.error("--test_edges and --test_labels are required for edges mode")
    run_edges_eval(args)
  return 0


if __name__ == "__main__":
  sys.exit(main())
