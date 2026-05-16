"""
用 TPRO 对已生成的异常轨迹做检测评测（PR-AUC，与 MST-OATD 一致）。

只需三类文件：
  1. 训练轨迹（正常，用于学习热门路径）
  2. 测试轨迹（你已注入异常的数据，.npy object 数组）
  3. 标签：outliers_idx（异常轨迹下标）或 0/1 标签向量

示例：
  python -m tpro_baseline.run_test \\
    --train /path/to/train_data_init.npy \\
    --test  /path/to/outliers_data_init_2_0.2_1.0.npy \\
    --labels /path/to/outliers_idx_init_2_0.2_1.0.npy
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Union

import numpy as np
from sklearn.metrics import average_precision_score

from .map_graph import RoadGraph, load_edge_routes_csv, mst_traj_to_edge_route
from .tpro import TPRO

TPRO_MIN_LAT, TPRO_MAX_LAT = 41.140519, 41.175893
TPRO_MIN_LON, TPRO_MAX_LON = -8.651993, -8.579304
_D_LAT = TPRO_MAX_LAT - TPRO_MIN_LAT
_D_LON = TPRO_MAX_LON - TPRO_MIN_LON
MAP_AREA = (
    TPRO_MIN_LAT - _D_LAT / 10,
    TPRO_MAX_LAT + _D_LAT / 10,
    TPRO_MIN_LON - _D_LON / 10,
    TPRO_MAX_LON + _D_LON / 10,
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
    name: str,
) -> List[List[int]]:
  if cache and cache.exists():
    with cache.open("rb") as f:
      routes = pickle.load(f)
    print(f"[{name}] cache {len(routes)} routes <- {cache}")
    return routes

  routes: List[List[int]] = []
  t0 = time.time()
  for i, traj in enumerate(trajs):
    routes.append(mst_traj_to_edge_route(graph, traj))
    if (i + 1) % 5000 == 0:
      print(f"[{name}] {i + 1}/{len(trajs)} ({time.time() - t0:.0f}s)")
  ok = sum(1 for r in routes if r)
  print(f"[{name}] map-match ok {ok}/{len(trajs)}")
  if cache:
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
      pickle.dump(routes, f)
  return routes


def run(
    train_path: Union[str, Path],
    test_path: Optional[Union[str, Path]],
    labels_path: Union[str, Path],
    map_dir: Union[str, Path] = "TPRO/map",
    train_routes_cache: Optional[Union[str, Path]] = None,
    test_routes_cache: Optional[Union[str, Path]] = None,
    train_edges: Optional[Union[str, Path]] = None,
    test_edges: Optional[Union[str, Path]] = None,
    lon_blocks: int = 10,
    lat_blocks: int = 20,
    top_k: int = 5,
    scores_out: Optional[Union[str, Path]] = None,
) -> float:
  labels_path = Path(labels_path)
  map_dir = Path(map_dir)
  graph = RoadGraph(map_dir, area=MAP_AREA)

  if train_edges:
    train_routes = [r for r in load_edge_routes_csv(train_edges) if r]
    print(f"[train] edges {train_edges}: {len(train_routes)}")
  else:
    train_trajs = np.load(Path(train_path), allow_pickle=True)
    train_routes = match_routes(
        graph,
        train_trajs,
        Path(train_routes_cache) if train_routes_cache else None,
        "train",
    )
    train_routes = [r for r in train_routes if r]

  if test_edges:
    test_routes = load_edge_routes_csv(test_edges)
    labels = load_labels(labels_path, len(test_routes))
    print(f"[test] edges {test_edges}: {len(test_routes)}")
  else:
    if not test_path:
      raise ValueError("需要 --test 或 --test_edges")
    test_trajs = np.load(Path(test_path), allow_pickle=True)
    test_routes = match_routes(
        graph,
        test_trajs,
        Path(test_routes_cache) if test_routes_cache else None,
        "test",
    )
    labels = load_labels(labels_path, len(test_trajs))

  test_for_score = [r if r else [] for r in test_routes]

  tpro = TPRO(
      graph,
      min_lat=TPRO_MIN_LAT,
      max_lat=TPRO_MAX_LAT,
      min_lon=TPRO_MIN_LON,
      max_lon=TPRO_MAX_LON,
      lon_blocks=lon_blocks,
      lat_blocks=lat_blocks,
      top_k=top_k,
  )
  print(f"TPRO fit on {len(train_routes)} routes ...")
  t0 = time.time()
  tpro.fit(train_routes)
  print(f"fit done ({time.time() - t0:.1f}s)")

  scores = np.array(tpro.anomaly_scores(test_for_score), dtype=np.float64)
  pr_auc = float(average_precision_score(labels, scores))

  print(f"test size: {len(labels)}, positives: {int(labels.sum())}")
  print(f"PR-AUC: {pr_auc:.6f}")

  if scores_out:
    out = Path(scores_out)
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, scores)
    label_out = out.with_name(out.stem + "_labels.npy")
    np.save(label_out, labels)
    print(f"saved {out} , {label_out}")

  return pr_auc


def main(argv: Optional[Sequence[str]] = None) -> int:
  p = argparse.ArgumentParser(description="TPRO 检测评测（异常已生成，直接测试）")
  p.add_argument("--train", required=True, help="训练 .npy 或配合 --train_edges")
  p.add_argument("--test", default="", help="测试 .npy（outliers_data_*.npy）")
  p.add_argument("--labels", required=True, help="outliers_idx 或 0/1 标签 .npy")
  p.add_argument("--map_dir", default="TPRO/map")
  p.add_argument("--train_cache", default="")
  p.add_argument("--test_cache", default="")
  p.add_argument("--train_edges", default="")
  p.add_argument("--test_edges", default="")
  p.add_argument("--lon_blocks", type=int, default=10)
  p.add_argument("--lat_blocks", type=int, default=20)
  p.add_argument("--top_k", type=int, default=5)
  p.add_argument("--scores_out", default="")
  args = p.parse_args(argv)

  run(
      train_path=args.train,
      test_path=args.test or None,
      labels_path=args.labels,
      map_dir=args.map_dir,
      train_routes_cache=args.train_cache or None,
      test_routes_cache=args.test_cache or None,
      train_edges=args.train_edges or None,
      test_edges=args.test_edges or None,
      lon_blocks=args.lon_blocks,
      lat_blocks=args.lat_blocks,
      top_k=args.top_k,
      scores_out=args.scores_out or None,
  )
  return 0


if __name__ == "__main__":
  sys.exit(main())
