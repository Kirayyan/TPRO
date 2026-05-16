# Batch TPRO evaluation for outliers_data_*.npy with auto-paired outliers_idx_*.npy

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .run_test import run


def labels_path_for_test(test_path: Path) -> Path:
  name = test_path.name
  if name.startswith("outliers_data_"):
    idx_name = "outliers_idx_" + name[len("outliers_data_") :]
    return test_path.with_name(idx_name)
  raise ValueError(f"cannot infer label file from {test_path}")


def main() -> int:
  p = argparse.ArgumentParser(description="Batch TPRO eval on porto outlier npy files")
  p.add_argument("--data_dir", required=True)
  p.add_argument("--train", default="train_data_init.npy")
  p.add_argument("--pattern", default="outliers_data_*.npy")
  p.add_argument("--map_dir", default="TPRO/map")
  p.add_argument("--train_cache", default=".tpro_cache/train.pkl")
  p.add_argument("--test_cache_dir", default=".tpro_cache/tests")
  p.add_argument("--lon_blocks", type=int, default=10)
  p.add_argument("--lat_blocks", type=int, default=20)
  p.add_argument("--top_k", type=int, default=5)
  p.add_argument("--scores_dir", default="results/tpro_scores")
  args = p.parse_args()

  data_dir = Path(args.data_dir)
  train_path = data_dir / args.train
  if not train_path.exists():
    for alt in ("train_data_1.npy", "train_data_init.npy"):
      cand = data_dir / alt
      if cand.exists():
        train_path = cand
        print(f"[train] using {train_path}")
        break
    else:
      print(f"error: train file not found under {data_dir}", file=sys.stderr)
      return 1

  tests = sorted(data_dir.glob(args.pattern))
  if not tests:
    print(f"error: no files match {args.pattern} in {data_dir}", file=sys.stderr)
    return 1

  cache_train = Path(args.train_cache)
  cache_test_dir = Path(args.test_cache_dir)
  scores_dir = Path(args.scores_dir)
  scores_dir.mkdir(parents=True, exist_ok=True)

  print(f"data_dir: {data_dir}")
  print(f"train:    {train_path}")
  print(f"tests:    {len(tests)}\n")

  results = []
  for test_path in tests:
    try:
      label_path = labels_path_for_test(test_path)
    except ValueError as e:
      print(f"skip {test_path.name}: {e}")
      continue
    if not label_path.exists():
      print(f"skip {test_path.name}: missing {label_path.name}")
      continue

    stem = test_path.stem
    print("=" * 60)
    print(f"test:   {test_path.name}")
    print(f"labels: {label_path.name}")

    test_cache = cache_test_dir / f"{stem}.pkl"
    score_out = scores_dir / f"{stem}_scores.npy"

    try:
      auc = run(
          train_path=train_path,
          test_path=test_path,
          labels_path=label_path,
          map_dir=args.map_dir,
          train_routes_cache=cache_train,
          test_routes_cache=test_cache,
          lon_blocks=args.lon_blocks,
          lat_blocks=args.lat_blocks,
          top_k=args.top_k,
          scores_out=score_out,
      )
      results.append((test_path.name, auc))
    except Exception as exc:
      print(f"failed {test_path.name}: {exc}")
      results.append((test_path.name, None))

  print("\n" + "=" * 60)
  print("PR-AUC summary:")
  for name, auc in results:
    if auc is None:
      print(f"  {name}: FAILED")
    else:
      print(f"  {name}: {auc:.6f}")

  summary = scores_dir / "summary.txt"
  with summary.open("w") as f:
    for name, auc in results:
      f.write(f"{name}\t{auc if auc is not None else 'FAILED'}\n")
  print(f"\nwrote {summary}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
