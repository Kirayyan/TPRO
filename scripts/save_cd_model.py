#!/usr/bin/env python3
"""Build .tpro_cache/cd_model.pkl from train_data_init.npy (grid TPRO, no road map)."""

from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np

from tpro_baseline.dataset_config import DatasetConfig
from tpro_baseline.grid_tpro import GridTPRO, trajs_to_grid_seqs


def main() -> None:
  train_npy = ROOT / "data_placeholder"
  if len(sys.argv) > 1:
    train_npy = Path(sys.argv[1])
  else:
    default = Path("/root/autodl-tmp/opensource_release_20260511/src/project_v2/data/cd/train_data_init.npy")
    train_npy = default if default.exists() else ROOT / "train_data_init.npy"

  model_cache = ROOT / ".tpro_cache" / "cd_model.pkl"
  train_cache = ROOT / ".tpro_cache" / "cd_train_grid.pkl"

  if not train_npy.exists():
    print("usage: python3 scripts/save_cd_model.py /path/to/train_data_init.npy")
    sys.exit(1)

  print("load", train_npy)
  trajs = np.load(train_npy, allow_pickle=True)
  if train_cache.exists():
    with train_cache.open("rb") as f:
      seqs = pickle.load(f)
    print("grid cache", len(seqs))
  else:
    seqs = trajs_to_grid_seqs(trajs)
    train_cache.parent.mkdir(parents=True, exist_ok=True)
    with train_cache.open("wb") as f:
      pickle.dump(seqs, f)
    print("saved grid cache ->", train_cache)

  train_seqs = [s for s in seqs if len(s) >= 2]
  print("valid trajectories:", len(train_seqs))

  cfg = DatasetConfig("cd", 167, 154, 10, 20, 5, True)
  model = GridTPRO(cfg)
  print("GridTPRO fit ...")
  t0 = time.time()
  model.fit(train_seqs)
  print("fit done (%.1fs)" % (time.time() - t0))

  model_cache.parent.mkdir(parents=True, exist_ok=True)
  with model_cache.open("wb") as f:
    pickle.dump({"grid_only": True, "model": model}, f)
  print("saved ->", model_cache, "(%.1f MB)" % (model_cache.stat().st_size / 1e6))


if __name__ == "__main__":
  main()
