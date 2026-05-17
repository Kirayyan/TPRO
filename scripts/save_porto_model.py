#!/usr/bin/env python3
"""Build .tpro_cache/porto_model.pkl from existing porto_train.pkl (route cache)."""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tpro_baseline.map_graph import RoadGraph  # noqa: E402
from tpro_baseline.run_test import MAP_AREA, save_model_cache  # noqa: E402
from tpro_baseline.tpro import TPRO  # noqa: E402

TPRO_MIN_LAT, TPRO_MAX_LAT = 41.140519, 41.175893
TPRO_MIN_LON, TPRO_MAX_LON = -8.651993, -8.579304


def main() -> None:
  train_cache = ROOT / ".tpro_cache" / "porto_train.pkl"
  model_cache = ROOT / ".tpro_cache" / "porto_model.pkl"
  map_dir = ROOT / "map"

  if not train_cache.exists():
    print(f"error: missing {train_cache}")
    print("run porto train map-match first with --train_cache .tpro_cache/porto_train.pkl")
    sys.exit(1)

  print(f"load routes from {train_cache}")
  with train_cache.open("rb") as f:
    routes = pickle.load(f)
  train_routes = [r for r in routes if r]
  print(f"valid routes: {len(train_routes)} / {len(routes)}")

  print("load road map ...")
  graph = RoadGraph(map_dir, area=MAP_AREA)
  model = TPRO(
      graph,
      TPRO_MIN_LAT,
      TPRO_MAX_LAT,
      TPRO_MIN_LON,
      TPRO_MAX_LON,
      lon_blocks=10,
      lat_blocks=20,
      top_k=5,
  )
  print("TPRO fit ...")
  t0 = time.time()
  model.fit(train_routes)
  print(f"fit done ({time.time() - t0:.1f}s)")

  save_model_cache(model_cache, model, grid_only=False)
  print(f"done: {model_cache} ({model_cache.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
  main()
