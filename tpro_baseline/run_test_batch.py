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


def resolve_train_path(data_dir: Path, train_arg: str, test_files: list[Path]) -> Path:
  """Find train npy: explicit name, then train_data_{id} from outliers, then any train*.npy."""
  explicit = data_dir / train_arg
  if explicit.exists():
    return explicit

  candidates = sorted(data_dir.glob("train*.npy"))
  if not candidates:
    candidates = sorted(data_dir.glob("**/train*.npy"))

  # e.g. outliers_data_1_stay_... -> train_data_1.npy
  if test_files:
    first = test_files[0].name
    parts = first.split("_")
    if len(parts) >= 3 and parts[0] == "outliers" and parts[1] == "data":
      split_id = parts[2]
      for name in (f"train_data_{split_id}.npy", f"train_{split_id}.npy"):
        cand = data_dir / name
        if cand.exists():
          print(f"[train] matched split {split_id} -> {cand.name}")
          return cand

  for name in ("train_data_init.npy", "train_data_1.npy", "train.npy"):
    cand = data_dir / name
    if cand.exists():
      print(f"[train] using {cand.name}")
      return cand

  if len(candidates) == 1:
    print(f"[train] using only train*.npy: {candidates[0].name}")
    return candidates[0]

  if candidates:
    print(f"[train] using largest train*.npy: {candidates[-1].name}")
    return candidates[-1]

  return explicit


def main() -> int:
  p = argparse.ArgumentParser(description="Batch TPRO eval on porto outlier npy files")
  p.add_argument("--data_dir", required=True)
  p.add_argument("--train", default="train_data_init.npy")
  p.add_argument("--pattern", default="outliers_data_*.npy")
  p.add_argument("--dataset", default="porto", choices=("porto", "cd"))
  p.add_argument("--grid_only", action="store_true", help="no road map; required for cd")
  p.add_argument("--map_dir", default="map")
  p.add_argument("--train_cache", default=".tpro_cache/train.pkl")
  p.add_argument("--test_cache_dir", default=".tpro_cache/tests")
  p.add_argument("--lon_blocks", type=int, default=10)
  p.add_argument("--lat_blocks", type=int, default=20)
  p.add_argument("--top_k", type=int, default=5)
  p.add_argument("--scores_dir", default="results/tpro_scores")
  args = p.parse_args()

  data_dir = Path(args.data_dir)
  if not data_dir.is_dir():
    print(f"error: data_dir not found: {data_dir}", file=sys.stderr)
    return 1

  tests = sorted(data_dir.glob(args.pattern))
  if not tests:
    print(f"error: no files match {args.pattern} in {data_dir}", file=sys.stderr)
    print("hint: ls", data_dir, file=sys.stderr)
    return 1

  train_path = resolve_train_path(data_dir, args.train, tests)
  if not train_path.exists():
    print(f"error: train file not found under {data_dir}", file=sys.stderr)
    print("available .npy files:", file=sys.stderr)
    for p in sorted(data_dir.glob("*.npy"))[:30]:
      print(f"  {p.name}", file=sys.stderr)
    if len(list(data_dir.glob("*.npy"))) > 30:
      print("  ...", file=sys.stderr)
    print("fix: pass --train <actual_train_file.npy>", file=sys.stderr)
    return 1

  cache_train = Path(args.train_cache)
  cache_test_dir = Path(args.test_cache_dir)
  scores_dir = Path(args.scores_dir)
  scores_dir.mkdir(parents=True, exist_ok=True)

  grid_only = args.grid_only or args.dataset == "cd"
  print(f"data_dir: {data_dir}")
  print(f"train:    {train_path}")
  print(f"dataset:  {args.dataset}, grid_only={grid_only}")
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
          dataset=args.dataset,
          grid_only=grid_only,
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
