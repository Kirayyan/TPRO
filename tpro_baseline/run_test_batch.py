"""
批量评测 porto 目录下所有 outliers_data_*.npy（自动配对 outliers_idx_*.npy）。

示例：
  python3 -m tpro_baseline.run_test_batch \\
    --data_dir /autodl-tmp/opensource_release_20260511/src/project_v2/data/porto \\
    --train train_data_init.npy \\
    --map_dir TPRO/map
"""

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
  raise ValueError(f"无法从文件名推断标签: {test_path}")


def main() -> int:
  p = argparse.ArgumentParser(description="批量 TPRO 评测 porto 异常数据")
  p.add_argument(
      "--data_dir",
      required=True,
      help="数据目录，如 /autodl-tmp/.../data/porto",
  )
  p.add_argument(
      "--train",
      default="train_data_init.npy",
      help="训练文件名（相对 data_dir），默认 train_data_init.npy",
  )
  p.add_argument(
      "--pattern",
      default="outliers_data_*.npy",
      help="测试集 glob，默认 outliers_data_*.npy",
  )
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
        print(f"[train] 使用 {train_path}")
        break
    else:
      print(f"错误: 未找到训练文件 {data_dir / args.train}", file=sys.stderr)
      return 1

  tests = sorted(data_dir.glob(args.pattern))
  if not tests:
    print(f"错误: {data_dir} 下没有匹配 {args.pattern} 的文件", file=sys.stderr)
    return 1

  cache_train = Path(args.train_cache)
  cache_test_dir = Path(args.test_cache_dir)
  scores_dir = Path(args.scores_dir)
  scores_dir.mkdir(parents=True, exist_ok=True)

  print(f"数据目录: {data_dir}")
  print(f"训练集:   {train_path}")
  print(f"测试集数: {len(tests)}\n")

  results = []
  for test_path in tests:
    try:
      label_path = labels_path_for_test(test_path)
    except ValueError as e:
      print(f"跳过 {test_path.name}: {e}")
      continue
    if not label_path.exists():
      print(f"跳过 {test_path.name}: 缺少标签 {label_path.name}")
      continue

    stem = test_path.stem  # outliers_data_1_stay_0p3_10_1
    print("=" * 60)
    print(f"评测: {test_path.name}")
    print(f"标签: {label_path.name}")

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
      print(f"失败 {test_path.name}: {exc}")
      results.append((test_path.name, None))

  print("\n" + "=" * 60)
  print("汇总 PR-AUC:")
  for name, auc in results:
    if auc is None:
      print(f"  {name}: FAILED")
    else:
      print(f"  {name}: {auc:.6f}")

  summary = scores_dir / "summary.txt"
  with summary.open("w") as f:
    for name, auc in results:
      f.write(f"{name}\t{auc if auc is not None else 'FAILED'}\n")
  print(f"\n已写入 {summary}")
  return 0


if __name__ == "__main__":
  sys.exit(main())
