#!/usr/bin/env bash
# 在 AutoDL 服务器上运行 TPRO 批量评测（异常已生成）
set -euo pipefail

DATA_DIR="/autodl-tmp/opensource_release_20260511/src/project_v2/data/porto"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

cd "$REPO_DIR"
pip install -q -r requirements.txt

# 单个异常类型（示例：stay）
# python3 -m tpro_baseline.run_test \
#   --train  "${DATA_DIR}/train_data_init.npy" \
#   --test   "${DATA_DIR}/outliers_data_1_stay_0p3_10_1.npy" \
#   --labels "${DATA_DIR}/outliers_idx_1_stay_0p3_10_1.npy" \
#   --map_dir TPRO/map \
#   --train_cache .tpro_cache/train.pkl \
#   --test_cache  .tpro_cache/outliers_data_1_stay_0p3_10_1.pkl \
#   --scores_out results/tpro_scores/outliers_data_1_stay_0p3_10_1_scores.npy

# 批量：目录下所有 outliers_data_*.npy
python3 -m tpro_baseline.run_test_batch \
  --data_dir "$DATA_DIR" \
  --train train_data_init.npy \
  --map_dir TPRO/map \
  --train_cache .tpro_cache/train.pkl \
  --test_cache_dir .tpro_cache/tests \
  --scores_dir results/tpro_scores
