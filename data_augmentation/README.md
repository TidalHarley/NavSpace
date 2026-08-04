# NavSpace 数据增强指南

本目录用于生成论文 **Stage B** 所需的 `aug_mix` 图像与标注（也兼容 baseline
`train_snav.py` Stage-2）。依赖：`habitat-sim`、根目录 `requirements-base.txt`，
以及 Qwen/DashScope（`DASHSCOPE_API_KEY`）。最终训练目录统一放在：

```text
snav_data/aug_mix/
├── vertical_perception/
├── environment_state/
├── spatial_relationship/
└── precise_movement/
```

每个可训练子目录里都需要有：

```text
annotations.json
<episode_id>/rgb/001.jpg ...
```

## 1. 环境

```bash
cd "$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

conda activate navspace   # or your habitat env
source data_augmentation/env_shim.sh
export DASHSCOPE_API_KEY=sk-xxxx
export R2RCE_TRAIN_JSON=/path/to/R2R_VLNCE_v1-3/train/train.json.gz
export MP3D_ROOT=/path/to/mp3d_scenes
```

`config.json` / `config.yaml` 使用 `${R2RCE_TRAIN_JSON}` / `${MP3D_ROOT}` 占位符，
加载时会展开环境变量。也可直接改成本地绝对路径。

## 2. Vertical Perception

做法：从 R2R 里筛选起点和终点有明显高度差的轨迹，用 Habitat 验证跨楼层，再用 Qwen 改写成包含楼层感知的指令。

```bash
OUTPUT_DIR=snav_data/aug_mix/vertical_perception \
DATASET_TAG=vertical_perception \
VIDEO_SUBDIR=vertical_perception \
bash data_augmentation/scripts/run_pipeline_vertical.sh
```

转成 Stage-2 训练格式：

```bash
python data_augmentation/scripts/convert_aug_to_sft.py \
  --folder snav_data/aug_mix/vertical_perception \
  --instructions-source data_augmentation/outputs/vertical_perception/custom_instructions.json
```

## 3. Environment State

做法：从 R2R 里筛选合适长度的轨迹，渲染轨迹帧，用 Qwen-VL 分析起点/终点可见和不可见物体，再套用 if/otherwise 类模板生成指令。

```bash
bash data_augmentation/scripts/run_pipeline_envstate.sh
```

该脚本会直接生成可训练数据：

```text
snav_data/aug_mix/environment_state/annotations.json
```

## 4. Spatial Relationship

做法：用正则从 R2R 指令中筛选空间关系表达，例如 left/right、between、beside、first/second/third 等；保留原始指令，不做改写。

```bash
python data_augmentation/spatial_relationship/1_filter.py

CUSTOM_INSTR=data_augmentation/outputs/spatial_relationship/custom_instructions.json \
OUTPUT_DIR=snav_data/aug_mix/spatial_relationship \
DATASET_TAG=spatial_relationship \
VIDEO_SUBDIR=spatial_relationship \
bash data_augmentation/scripts/run_render_aug.sh
```

转成 Stage-2 训练格式：

```bash
python data_augmentation/scripts/convert_aug_to_sft.py \
  --folder snav_data/aug_mix/spatial_relationship \
  --instructions-source data_augmentation/outputs/spatial_relationship/custom_instructions.json
```

## 5. Precise Movement

做法：在 MP3D 场景中随机采样起点/终点，用 Habitat 走最短路径，筛选 16-45 步且有转向的轨迹，再用规则模板生成精确移动指令。

```bash
python data_augmentation/precise_movement/run.py \
  --output-root snav_data/aug_mix/precise_movement
```

转成 Stage-2 训练格式：

```bash
python data_augmentation/scripts/convert_aug_to_sft.py \
  --folder snav_data/aug_mix/precise_movement
```

## 6. 准备 Stage-2 续训数据

检查哪些类别已经准备好：

```bash
for d in snav_data/aug_mix/*; do
  [ -f "$d/annotations.json" ] && echo "ready: $d"
done
```

如果只想用部分增强数据，手动指定 `VIDEO_FOLDERS`：

```bash
export VIDEO_FOLDERS=snav_data/aug_mix/vertical_perception,snav_data/aug_mix/precise_movement
```

如果不指定，`run_stage2_sft.sh` 会自动扫描 `snav_data/aug_mix/*/annotations.json`。

## 7. 接到论文训练（推荐）

把 `aug_mix` / `manual_98` 转成 hist8/future6 JSON，再跑 Stage B
（见 [`snav_training/README.md`](../snav_training/README.md)）：

```bash
python snav_training/scripts/build_hist8_future6.py \
  --aug-root snav_data/aug_mix \
  --out-dir train_data
# optional: --manual-root snav_data/manual_98

export PREV_STAGE_CHECKPOINT=/abs/path/to/vln_mix_ckpt
export IMAGE_FOLDER=$PWD/snav_data/aug_mix
bash snav_training/scripts/launch_paper_sft_stage_b.sh
```

## 8. Baseline Stage-2（`train_snav.py`，非论文权重链）

从 Stage-1 `train_snav.py` checkpoint 继续训 aug_mix（消融/冒烟用）：

```bash
export MODEL_PATH=/abs/path/to/snav_stage1_ckpt
export LLAVA_ROOT=/abs/path/to/LLaVA-or-StreamVLN
export OUTPUT_DIR=/abs/path/to/snav_stage2_aug
export VISION_TOWER_PATH=/abs/path/to/siglip-so400m-patch14-384
bash data_augmentation/scripts/run_stage2_sft.sh
```
