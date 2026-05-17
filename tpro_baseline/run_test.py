# TPRO detection + PR-AUC on pre-generated outlier npy (porto/cd, optional grid-only).

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path
from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
from sklearn.metrics import average_precision_score

from .dataset_config import DatasetConfig, get_config
from .grid_tpro import GridTPRO, trajs_to_grid_seqs
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
    print(f"[{name}] route cache {len(routes)} <- {cache}")
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
    print(f"[{name}] saved route cache -> {cache}")
  return routes


def load_grid_seqs(trajs: Sequence, cache: Optional[Path], name: str) -> List[List[int]]:
  if cache and cache.exists():
    with cache.open("rb") as f:
      seqs = pickle.load(f)
    print(f"[{name}] grid cache {len(seqs)} <- {cache}")
    return seqs
  seqs = trajs_to_grid_seqs(trajs)
  if cache:
    cache.parent.mkdir(parents=True, exist_ok=True)
    with cache.open("wb") as f:
      pickle.dump(seqs, f)
  print(f"[{name}] grid sequences: {len(seqs)}")
  return seqs


def save_model_cache(path: Path, model: Union[TPRO, GridTPRO], grid_only: bool) -> None:
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("wb") as f:
    pickle.dump({"grid_only": grid_only, "model": model}, f)
  print(f"[model] saved -> {path}")


def load_model_cache(path: Path) -> Tuple[bool, Union[TPRO, GridTPRO]]:
  with path.open("rb") as f:
    blob = pickle.load(f)
  print(f"[model] loaded <- {path}")
  return blob["grid_only"], blob["model"]


def build_and_fit_model(
    train_path: Path,
    dataset: str,
    use_grid: bool,
    map_dir: Path,
    train_routes_cache: Optional[Path],
    train_edges: Optional[Path],
    lon_blocks: int,
    lat_blocks: int,
    top_k: int,
) -> Union[TPRO, GridTPRO]:
  cfg = get_config(dataset)
  if use_grid:
    train_trajs = np.load(train_path, allow_pickle=True)
    train_seqs = [
        s
        for s in load_grid_seqs(train_trajs, train_routes_cache, "train")
        if len(s) >= 2
    ]
    cfg = DatasetConfig(
        cfg.name, cfg.lat_grid_num, cfg.lng_grid_num, lon_blocks, lat_blocks, top_k, True
    )
    model = GridTPRO(cfg)
    print(f"GridTPRO fit on {len(train_seqs)} trajectories ...")
    t0 = time.time()
    model.fit(train_seqs)
    print(f"fit done ({time.time() - t0:.1f}s)")
    return model

  graph = RoadGraph(map_dir, area=MAP_AREA)
  if train_edges:
    train_routes = [r for r in load_edge_routes_csv(train_edges) if r]
  else:
    train_trajs = np.load(train_path, allow_pickle=True)
    train_routes = match_routes(graph, train_trajs, train_routes_cache, "train")
    train_routes = [r for r in train_routes if r]

  model = TPRO(
      graph, TPRO_MIN_LAT, TPRO_MAX_LAT, TPRO_MIN_LON, TPRO_MAX_LON, lon_blocks, lat_blocks, top_k
  )
  print(f"TPRO fit on {len(train_routes)} routes ...")
  t0 = time.time()
  model.fit(train_routes)
  print(f"fit done ({time.time() - t0:.1f}s)")
  return model


def score_test_set(
    model: Union[TPRO, GridTPRO],
    use_grid: bool,
    test_path: Optional[Path],
    labels_path: Path,
    map_dir: Path,
    test_routes_cache: Optional[Path],
    test_edges: Optional[Path],
) -> Tuple[np.ndarray, np.ndarray]:
  if use_grid:
    if test_edges:
      raise ValueError("--test_edges not supported in grid mode")
    if not test_path:
      raise ValueError("need --test")
    test_trajs = np.load(test_path, allow_pickle=True)
    test_seqs = load_grid_seqs(test_trajs, test_routes_cache, "test")
    labels = load_labels(labels_path, len(test_trajs))
    test_for_score = [s if len(s) >= 2 else [] for s in test_seqs]
    scores = np.array(model.anomaly_scores(test_for_score), dtype=np.float64)
    return labels, scores

  graph = model.graph if isinstance(model, TPRO) else RoadGraph(map_dir, area=MAP_AREA)
  if test_edges:
    test_routes = load_edge_routes_csv(test_edges)
    labels = load_labels(labels_path, len(test_routes))
  else:
    if not test_path:
      raise ValueError("need --test")
    test_trajs = np.load(test_path, allow_pickle=True)
    test_routes = match_routes(graph, test_trajs, test_routes_cache, "test")
    labels = load_labels(labels_path, len(test_trajs))
  test_for_score = [r if r else [] for r in test_routes]
  scores = np.array(model.anomaly_scores(test_for_score), dtype=np.float64)
  return labels, scores


def run(
    train_path: Union[str, Path],
    test_path: Optional[Union[str, Path]],
    labels_path: Union[str, Path],
    dataset: str = "porto",
    grid_only: bool = False,
    map_dir: Union[str, Path] = "map",
    train_routes_cache: Optional[Union[str, Path]] = None,
    test_routes_cache: Optional[Union[str, Path]] = None,
    model_cache: Optional[Union[str, Path]] = None,
    train_edges: Optional[Union[str, Path]] = None,
    test_edges: Optional[Union[str, Path]] = None,
    lon_blocks: int = 10,
    lat_blocks: int = 20,
    top_k: int = 5,
    scores_out: Optional[Union[str, Path]] = None,
    model: Optional[Union[TPRO, GridTPRO]] = None,
) -> Tuple[float, Union[TPRO, GridTPRO]]:
  cfg = get_config(dataset)
  use_grid = grid_only or cfg.grid_only
  labels_path = Path(labels_path)
  map_dir = Path(map_dir)
  train_path = Path(train_path)
  model_cache_path = Path(model_cache) if model_cache else None
  train_cache_path = Path(train_routes_cache) if train_routes_cache else None
  test_cache_path = Path(test_routes_cache) if test_routes_cache else None

  mode = "grid TPRO (no road map)" if use_grid else "road-map TPRO"
  print(f"mode: {mode}, dataset={cfg.name}")

  if model is not None:
    print("[model] reuse in-memory model (skip train/fit)")
  elif model_cache_path and model_cache_path.exists():
    cached_grid, model = load_model_cache(model_cache_path)
    if cached_grid != use_grid:
      raise ValueError(f"model cache grid_only={cached_grid} != current mode grid_only={use_grid}")
    print("[model] skip train/fit (loaded from cache)")
  else:
    model = build_and_fit_model(
        train_path,
        dataset,
        use_grid,
        map_dir,
        train_cache_path,
        Path(train_edges) if train_edges else None,
        lon_blocks,
        lat_blocks,
        top_k,
    )
    if model_cache_path:
      save_model_cache(model_cache_path, model, use_grid)

  labels, scores = score_test_set(
      model,
      use_grid,
      Path(test_path) if test_path else None,
      labels_path,
      map_dir,
      test_cache_path,
      Path(test_edges) if test_edges else None,
  )

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

  return pr_auc, model


def main(argv: Optional[Sequence[str]] = None) -> int:
  p = argparse.ArgumentParser(description="TPRO eval on pre-generated outliers")
  p.add_argument("--train", required=True, help="required on first run; ignored if --model_cache exists")
  p.add_argument("--test", default="")
  p.add_argument("--labels", required=True)
  p.add_argument("--dataset", default="porto", choices=("porto", "cd"))
  p.add_argument("--grid_only", action="store_true")
  p.add_argument("--map_dir", default="map")
  p.add_argument(
      "--train_cache",
      default="",
      help="cached map-matched train routes / grid seqs (skip ~hours of train map-match)",
  )
  p.add_argument("--test_cache", default="", help="cached test routes / grid seqs")
  p.add_argument(
      "--model_cache",
      default="",
      help="cached fitted TPRO model (skip train load + fit on next runs)",
  )
  p.add_argument("--train_edges", default="")
  p.add_argument("--test_edges", default="")
  p.add_argument("--lon_blocks", type=int, default=10)
  p.add_argument("--lat_blocks", type=int, default=20)
  p.add_argument("--top_k", type=int, default=5)
  p.add_argument("--scores_out", default="")
  args = p.parse_args(argv)

  grid_only = args.grid_only or args.dataset == "cd"
  run(
      train_path=args.train,
      test_path=args.test or None,
      labels_path=args.labels,
      dataset=args.dataset,
      grid_only=grid_only,
      map_dir=args.map_dir,
      train_routes_cache=args.train_cache or None,
      test_routes_cache=args.test_cache or None,
      model_cache=args.model_cache or None,
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
