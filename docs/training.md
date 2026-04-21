# SNav 训练 / SNav Training

> 本文档为中英双语。 / This document is bilingual.
>
> - [中文版](#中文版)
> - [English version](#english-version)

---

## 中文版

`snav_training/` 现在已经包含一套**可以跑起来**的 SNav Stage-1 vanilla SFT 训练管线。
使用者拿到仓库后按 README 配好路径就能直接从 LLaVA-Video-7B-Qwen2 出发跑 SNav 的
Stage-1 监督微调。

> ⚠️ **重要提示：这只是 Stage-1 baseline。** 仓库里开源的训练脚本仅负责：
> 用 Habitat `GreedyGeodesicFollower` 渲染的专家轨迹 → LLaVA-Video-7B-Qwen2 SFT。
>
> 要复现论文中的完整 SNav 训练效果，还需要以下**本仓库不包含**的东西：
>
> 1. **通用 Video-QA 数据（mandatory for best results）** —— 与导航数据按 ~15 %
>   的比例混训，缓解灾难性遗忘，保留 LLaVA-Video 原有的视觉问答能力。脚本层面
>    已经支持 (`QA_JSON_PATHS` / `QA_VIDEO_ROOTS`)，但我们并不分发 QA 数据本身。
> 2. **高度 / 光影扰动**（height & lighting variation） —— 渲染期间随机扰动
>   相机高度和场景光照，用于提升模型对 embodiment 和外观变化的鲁棒性。当前
>    渲染脚本只使用固定相机高度（0.88 m）+ 默认 Habitat 光照。
> 3. **Data augmentation** —— 基于 LLM 的指令改写的data augmentation流程。

### 目录结构

```text
snav_training/
├── README.md                            # 训练模块使用说明（与本文档互补）
├── train_snav.py                        # 主训练入口（HF Trainer + DeepSpeed）
├── dataset_snav.py                      # 数据集 + collator，支持 QA 混训
├── configs/
│   ├── snav_data.yaml.template          # 可选 YAML，用于多数据源聚合
│   ├── deepspeed_zero2.json             # 默认 ZeRO-2 配置
│   └── deepspeed_zero3.json             # ZeRO-3 
├── data_generation/
│   ├── README.md                        # 渲染流程详述
│   ├── render_streamvln.py              # Habitat 渲染脚本（R2R-CE / RxR-CE / EnvDrop）
│   ├── run_render_r2rce.sh              # R2R-CE train split 渲染
│   ├── run_render_rxrce.sh              # RxR-CE train split 渲染
│   └── run_render_envdrop.sh            # EnvDrop 子集渲染
└── scripts/
    ├── run_snav_train.sh                # 单机多卡 SFT 启动脚本
    └── run_pipeline.sh                  # 渲染 → 训练 →（可选）评测 的编排脚本
```

### 环境要求

- conda 环境同评测：建议使用 `streamvln的相同虚拟环境`（带 Habitat-Sim 0.3.x + torch + transformers）。  
额外安装 `deepspeed` 即可训练。
- LLaVA 代码库：通过 `LLAVA_ROOT` 指向任意提供 `llava/` 子目录的 checkout；
StreamVLN 仓库自带的那份 `llava/` 就能直接用。
- Base 模型：`[lmms-lab/LLaVA-Video-7B-Qwen2](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2)`。
- 场景：HM3D 或 MP3D，`SCENES_ROOT` 指向即可，渲染器自动兼容两种布局。
- 数据集原始 JSON（不随仓库分发，自行从官方渠道下载）：
  - R2R-CE：`R2R_VLNCE_v1-3_preprocessed/train/train.json.gz`
  - RxR-CE：`RxR_VLNCE_v0/train/train_guide.json`
  - EnvDrop：`R2R_VLNCE_v1-3_preprocessed/envdrop/envdrop.json.gz`

### 端到端跑通

```bash
# 0. 激活环境
conda activate streamvln

# 1. 渲染专家轨迹（默认 --output_mode frames，即 StreamVLN 兼容布局）
export DATA_ROOT=/abs/path/to/snav_data
export SCENES_ROOT=/abs/path/to/scene_datasets
export R2RCE_TRAIN_JSON=/abs/path/to/train.json.gz
export RXRCE_TRAIN_JSON=/abs/path/to/train_guide.json
export ENVDROP_SOURCE_JSON=/abs/path/to/envdrop.json.gz

bash snav_training/data_generation/run_render_r2rce.sh
bash snav_training/data_generation/run_render_rxrce.sh
bash snav_training/data_generation/run_render_envdrop.sh

# 2. 启动 Stage-1 SFT
export LLAVA_ROOT=/abs/path/to/StreamVLN          # 任意含 llava/ 的 checkout
export MODEL_PATH=/abs/path/to/LLaVA-Video-7B-Qwen2
export VIDEO_FOLDERS=${DATA_ROOT}/r2rce,${DATA_ROOT}/rxrce,${DATA_ROOT}/envdrop
export OUTPUT_DIR=/abs/path/to/checkpoints/snav_stage1_vanilla

bash snav_training/scripts/run_snav_train.sh
```

#### 混入 Video-QA 数据

把 LLaVA-Video-178K 这类通用 Video-QA 数据按 15 % 的比例混进去，防止模型
在导航数据上过拟合而丢掉视觉理解能力：

```bash
export QA_JSON_PATHS=/abs/path/to/qa_train.json           # LLaVA SFT 格式
export QA_VIDEO_ROOTS=/abs/path/to/qa_videos              # 视频根目录
export QA_RATIO=0.15
bash snav_training/scripts/run_snav_train.sh
```

QA 样本由 `dataset_snav.VideoQADataset` 按需解码视频并与导航样本混合喂入 Trainer。

#### 一键串联（渲染 → 训练 → 评测）

```bash
export HM3D_BASE_PATH=/abs/path/to/hm3d_v0.2
bash snav_training/scripts/run_pipeline.sh
# 单独跳过某个阶段：SKIP_RENDER=1 / SKIP_TRAIN=1 / SKIP_EVAL=1
```

### 默认超参（匹配 SNav Stage-1 v1）


| 超参                      | 默认值                                 |
| ----------------------- | ----------------------------------- |
| Base model              | `LLaVA-Video-7B-Qwen2`              |
| Prompt 模板               | `qwen_1_5`                          |
| 学习率                     | `5e-5`（cosine，`warmup_ratio=0.075`） |
| Epoch                   | 1                                   |
| per-device batch / 梯度累积 | 1 / 12                              |
| 输入帧数                    | 16（首帧 + 当前帧 + 均匀采样）                 |
| 单步预测动作数 K               | 6                                   |
| 最大序列长度                  | 32 768                              |
| 精度                      | BF16 + `sdpa` attention             |
| 优化器                     | DeepSpeed ZeRO-2（可切 ZeRO-3）         |


所有超参均可通过 `run_snav_train.sh` 顶部的环境变量或 `train_snav.py --help` 看到的 CLI 重载。

### 与评测的衔接

Stage-1 SFT 产出的 checkpoint 可以直接丢给评测脚本：

```bash
python evaluation/eval_snav.py \
  --task environment_state \
  --hm3d-base-path /abs/path/to/hm3d_v0.2 \
  --model-path /abs/path/to/checkpoints/snav_stage1_vanilla \
  --model-name llava_qwen --conv-template qwen_1_5 \
  --attn-implementation sdpa
```

完整评测 CLI 见 `[docs/evaluation.md` §3](evaluation.md#3-snav-based-评测llavasnav-本地模型)。

---

## English version

`snav_training/` now ships a **runnable** SNav Stage-1 vanilla SFT pipeline. After
filling in the paths documented in the README you can kick off Stage-1 supervised
fine-tuning of `LLaVA-Video-7B-Qwen2` on SNav data out of the box.

> ⚠️ **Important: this is a Stage-1 baseline only.** The open-sourced training
> script covers exactly one thing: geodesic-follower expert trajectories rendered
> by Habitat-Sim → LLaVA-Video-7B-Qwen2 SFT.
>
> To reproduce the full SNav recipe from the paper you **also** need the
> following, none of which are shipped here:
>
> 1. **General Video-QA data** (strongly recommended) — mixed in at ~15 % to
>   mitigate catastrophic forgetting and preserve the base model's video
>    reasoning capability. The trainer already supports this via
>    `QA_JSON_PATHS` / `QA_VIDEO_ROOTS`, but we do not redistribute the QA
>    corpus itself (use LLaVA-Video-178K or any equivalent).
> 2. **Height & lighting variation** during rendering — jitter the camera
>   height and scene illumination to improve robustness to embodiment /
>    appearance changes. The current renderer uses a fixed camera height
>    (0.88 m) and Habitat's default lighting.
> 3. **Data augmentation** — LLM-based instruction rewriting.

### Layout

```text
snav_training/
├── README.md                            # module-level usage (complements this doc)
├── train_snav.py                        # main SFT entry (HF Trainer + DeepSpeed)
├── dataset_snav.py                      # dataset + collator, supports QA mixing
├── configs/
│   ├── snav_data.yaml.template          # optional YAML for multi-source aggregation
│   ├── deepspeed_zero2.json             # default ZeRO-2 config
│   └── deepspeed_zero3.json             # ZeRO-3 
├── data_generation/
│   ├── README.md                        # rendering details
│   ├── render_streamvln.py              # Habitat renderer (R2R-CE / RxR-CE / EnvDrop)
│   ├── run_render_r2rce.sh              # R2R-CE train split
│   ├── run_render_rxrce.sh              # RxR-CE train split
│   └── run_render_envdrop.sh            # EnvDrop subset
└── scripts/
    ├── run_snav_train.sh                # multi-GPU SFT launcher
    └── run_pipeline.sh                  # render → train → (optional) eval orchestrator
```

### Prerequisites

- Same conda env as evaluation: use `streamvln` (Habitat-Sim 0.3.x + torch +
transformers). Just `pip install deepspeed` to enable training.
- A LLaVA checkout: point `LLAVA_ROOT` at any directory that contains a
`llava/` package. The `llava/` copy shipped inside StreamVLN works as-is.
- Base model: `[lmms-lab/LLaVA-Video-7B-Qwen2](https://huggingface.co/lmms-lab/LLaVA-Video-7B-Qwen2)`.
- Scene assets: HM3D or MP3D under `SCENES_ROOT`; the renderer auto-detects.
- Raw dataset JSONs (NOT redistributed here — fetch from the official sources):
  - R2R-CE: `R2R_VLNCE_v1-3_preprocessed/train/train.json.gz`
  - RxR-CE: `RxR_VLNCE_v0/train/train_guide.json`
  - EnvDrop: `R2R_VLNCE_v1-3_preprocessed/envdrop/envdrop.json.gz`

### End-to-end usage

```bash
# 0. Activate the env
conda activate streamvln

# 1. Render expert trajectories (default --output_mode frames, StreamVLN layout)
export DATA_ROOT=/abs/path/to/snav_data
export SCENES_ROOT=/abs/path/to/scene_datasets
export R2RCE_TRAIN_JSON=/abs/path/to/train.json.gz
export RXRCE_TRAIN_JSON=/abs/path/to/train_guide.json
export ENVDROP_SOURCE_JSON=/abs/path/to/envdrop.json.gz

bash snav_training/data_generation/run_render_r2rce.sh
bash snav_training/data_generation/run_render_rxrce.sh
bash snav_training/data_generation/run_render_envdrop.sh

# 2. Launch Stage-1 SFT
export LLAVA_ROOT=/abs/path/to/StreamVLN          # any checkout with llava/
export MODEL_PATH=/abs/path/to/LLaVA-Video-7B-Qwen2
export VIDEO_FOLDERS=${DATA_ROOT}/r2rce,${DATA_ROOT}/rxrce,${DATA_ROOT}/envdrop
export OUTPUT_DIR=/abs/path/to/checkpoints/snav_stage1_vanilla

bash snav_training/scripts/run_snav_train.sh
```

#### Mix in Video-QA data (strongly recommended)

Add ~15 % general Video-QA samples (e.g. LLaVA-Video-178K format) to avoid
losing the base model's visual-QA capabilities:

```bash
export QA_JSON_PATHS=/abs/path/to/qa_train.json
export QA_VIDEO_ROOTS=/abs/path/to/qa_videos
export QA_RATIO=0.15
bash snav_training/scripts/run_snav_train.sh
```

QA samples are decoded on-the-fly by `dataset_snav.VideoQADataset` and
interleaved with navigation samples.

#### One-shot pipeline (render → train → eval)

```bash
export HM3D_BASE_PATH=/abs/path/to/hm3d_v0.2
bash snav_training/scripts/run_pipeline.sh
# Skip stages: SKIP_RENDER=1 / SKIP_TRAIN=1 / SKIP_EVAL=1
```

### Reproducibility defaults (matches SNav Stage-1 v1)


| Knob                          | Value                                 |
| ----------------------------- | ------------------------------------- |
| Base model                    | `LLaVA-Video-7B-Qwen2`                |
| Prompt version                | `qwen_1_5`                            |
| Learning rate                 | `5e-5` (cosine, `warmup_ratio=0.075`) |
| Epochs                        | 1                                     |
| Per-device batch / grad-accum | 1 / 12                                |
| Frames per sample             | 16 (first + current + uniform middle) |
| Future actions per chunk (K)  | 6                                     |
| Max seq length                | 32 768                                |
| Precision                     | BF16 + `sdpa` attention               |
| Optimizer                     | DeepSpeed ZeRO-2 (ZeRO-3 opt-in)      |


Everything is exposed as env vars at the top of `run_snav_train.sh` or CLI  
flags in `train_snav.py --help`.

### Hand-off to evaluation

The checkpoint produced by Stage-1 SFT can be handed directly to the evaluation
script:

```bash
python evaluation/eval_snav.py \
  --task environment_state \
  --hm3d-base-path /abs/path/to/hm3d_v0.2 \
  --model-path /abs/path/to/checkpoints/snav_stage1_vanilla \
  --model-name llava_qwen --conv-template qwen_1_5 \
  --attn-implementation sdpa
```

See `[docs/evaluation_en.md` §3](evaluation_en.md#3-snav-based-evaluation-llavasnav-local-models)
for the full CLI surface.