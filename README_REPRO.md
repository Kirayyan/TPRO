# TPRO 基线：对已生成异常做检测（Python）

异常数据已准备好时，只需运行 TPRO 打分并计算 **PR-AUC**（与 MST-OATD `average_precision_score` 相同）。

## 安装

```bash
pip install -r requirements.txt
```

## AutoDL / project_v2 数据路径（你的场景）

数据目录：`/autodl-tmp/opensource_release_20260511/src/project_v2/data/porto`

异常文件形如 `outliers_data_1_stay_0p3_10_1.npy`，标签为同名 `outliers_idx_1_stay_0p3_10_1.npy`。

**单个文件评测：**

```bash
cd /path/to/TPRO
pip install -r requirements.txt

python3 -m tpro_baseline.run_test \
  --train  /autodl-tmp/opensource_release_20260511/src/project_v2/data/porto/train_data_init.npy \
  --test   /autodl-tmp/opensource_release_20260511/src/project_v2/data/porto/outliers_data_1_stay_0p3_10_1.npy \
  --labels /autodl-tmp/opensource_release_20260511/src/project_v2/data/porto/outliers_idx_1_stay_0p3_10_1.npy \
  --map_dir TPRO/map \
  --train_cache .tpro_cache/train.pkl \
  --test_cache  .tpro_cache/outliers_data_1_stay_0p3_10_1.pkl
```

**批量评测目录下全部 `outliers_data_*.npy`：**

```bash
python3 -m tpro_baseline.run_test_batch \
  --data_dir /autodl-tmp/opensource_release_20260511/src/project_v2/data/porto \
  --train train_data_init.npy \
  --map_dir TPRO/map
```

或：`bash scripts/run_tpro_porto.sh`

## 一条命令完成测试（MST-OATD 默认命名）

```bash
python3 -m tpro_baseline.run_test \
  --train  /你的路径/data/porto/train_data_init.npy \
  --test   /你的路径/data/porto/outliers_data_init_2_0.2_1.0.npy \
  --labels /你的路径/data/porto/outliers_idx_init_2_0.2_1.0.npy \
  --map_dir TPRO/map \
  --scores_out results/tpro_scores.npy
```

| 参数 | 含义 |
|------|------|
| `--train` | 正常训练轨迹（MST-OATD 预处理后的 `train_data_init.npy`） |
| `--test` | 你已注入异常的测试轨迹（`outliers_data_init_*.npy`） |
| `--labels` | 异常下标 `outliers_idx_*.npy`，或长度为测试集大小的 `0/1` 向量 |
| `--map_dir` | 本仓库 `TPRO/map`（含 `nodeOSM.txt`、`edgeOSM.txt`） |

输出示例：

```
PR-AUC: 0.xxxxxx
```

## 加速（可选）

首次会把网格轨迹 map-match 到路网，较慢。第二次可用缓存：

```bash
python -m tpro_baseline.run_test \
  --train ... --test ... --labels ... \
  --train_cache .tpro_cache/train.pkl \
  --test_cache  .tpro_cache/test.pkl
```

## 若测试集已是路网边序列

每行一条轨迹，逗号分隔 edge id：

```bash
python -m tpro_baseline.run_test \
  --train /path/to/train_data_init.npy \
  --train_edges /path/to/train_mm_edges.csv \
  --test  /path/to/dummy.npy \
  --test_edges /path/to/test_mm_edges.csv \
  --labels /path/to/labels.npy
```

（`--test` 在仅用 `--test_edges` 时可填任意占位；建议同时提供 `--test` 与真实 `labels`。）

## TPRO 超参

与原版 C++ 默认一致：`lon_blocks=10`, `lat_blocks=20`, `top_k=5`。
