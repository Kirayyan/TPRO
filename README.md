# TPRO 基线复现（Python）

在已生成的异常轨迹上运行 **TPRO** 检测，输出 **PR-AUC**（与 MST-OATD 相同）。

仅需本仓库中的 **Porto 路网**（`map/nodeOSM.txt`、`map/edgeOSM.txt`），无需在你的数据集里准备路网。

## 安装

```bash
pip install -r requirements.txt
```

## 数据路径示例（AutoDL）

```text
/autodl-tmp/opensource_release_20260511/src/project_v2/data/porto/
  train_data_init.npy
  outliers_data_1_stay_0p3_10_1.npy
  outliers_idx_1_stay_0p3_10_1.npy
  ...
```

## 单个异常类型

```bash
DATA=/autodl-tmp/opensource_release_20260511/src/project_v2/data/porto

python3 -m tpro_baseline.run_test \
  --train  ${DATA}/train_data_init.npy \
  --test   ${DATA}/outliers_data_1_stay_0p3_10_1.npy \
  --labels ${DATA}/outliers_idx_1_stay_0p3_10_1.npy \
  --map_dir map \
  --train_cache .tpro_cache/train.pkl \
  --test_cache  .tpro_cache/test_stay.pkl
```

## 批量评测

```bash
# 先查看训练集实际文件名
ls /autodl-tmp/opensource_release_20260511/src/project_v2/data/porto/train*.npy

python3 -m tpro_baseline.run_test_batch \
  --data_dir /autodl-tmp/opensource_release_20260511/src/project_v2/data/porto \
  --train train_data_1.npy \
  --map_dir map
```

若没有 `train_data_init.npy`，常见为 `train_data_1.npy`（与 `outliers_data_1_*` 对应）；脚本也会自动尝试匹配。

或：`bash scripts/run_tpro_porto.sh`

## 仓库结构

```text
map/                 # Porto 路网（TPRO 必需）
  nodeOSM.txt
  edgeOSM.txt
tpro_baseline/       # TPRO 算法与评测
requirements.txt
scripts/
```

## 成都（cd）— 不需要路网

与 MST-OATD 一样在 **网格轨迹** 上跑 TPRO（`--dataset cd` 自动开启，无需 `map/`）：

```bash
DATA=/root/autodl-tmp/opensource_release_20260511/src/project_v2/data/cd

python3 -m tpro_baseline.run_test_batch \
  --data_dir ${DATA} \
  --train train_data_init.npy \
  --dataset cd \
  --grid_only
```

单个文件：

```bash
python3 -m tpro_baseline.run_test \
  --dataset cd --grid_only \
  --train  ${DATA}/train_data_init.npy \
  --test   ${DATA}/outliers_data_xxx.npy \
  --labels ${DATA}/outliers_idx_xxx.npy
```

说明：这是 **网格版 TPRO**（与 MST-OATD 数据格式一致），不是路网 map-match 版；指标仍为 PR-AUC。

## 克隆建议（减小体积）

```bash
git clone --depth 1 -b cursor/tpro-mst-oatd-eval-f09f https://github.com/Kirayyan/TPRO.git
```
