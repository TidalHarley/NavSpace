# SNav training

## 1. What this folder is for

Train SNav end-to-end:

- **Stage A** — VLN-mix SFT on LLaVA-Video-7B-Qwen2 (nav-only by default)  
- **Stage B** — paper final SFT on `aug_mix` + optional `manual_98` (hist=8, future=6) → SNav-7B  

Also includes Habitat renderers and a baseline Stage-1 trainer (`train_snav.py`) for ablations.  
Data under `snav_data/` / `train_data/` is **not** shipped.

> Paper-chain docs live **here**. [`docs/training.md`](../docs/training.md) only covers the
> older Stage-1 baseline (`train_snav.py`).

## 2. Layout

```
snav_training/
├── snav_llava/                 # trainer package (train_mem / model)
├── configs/                    # Stage A/B YAML + DeepSpeed
├── stage_a/                    # R2R/RxR collect + stepwise JSON builders
├── data_generation/            # baseline Stage-1 render (NOT paper Stage A)
├── scripts/
│   ├── prepare_stage_a_data.sh
│   ├── launch_vln_mix_stage_a.sh
│   ├── launch_paper_sft_stage_b.sh
│   ├── train_paper.sh
│   ├── build_hist8_future6.py
│   └── run_snav_train.sh       # baseline Stage-1 only
├── train_snav.py
└── dataset_snav.py
```

## 3. Prerequisites

| Need | Notes |
|------|--------|
| Conda env with `habitat-sim` + `habitat-lab` (VLN configs) | Stage A collect / aug render |
| `deepspeed`, `accelerate`, `transformers`, `torch` (CUDA-matched) | Stage A/B train |
| Optional `flash-attn` | Trainer defaults to flash-attn; set attn impl / install as needed |
| Local **LLaVA-Video-7B-Qwen2** + **SigLIP** | Launchers set `HF_HUB_OFFLINE=1` by default |
| Scenes / VLN-CE | See layout below |

Expected data layout (override with env vars):

```text
$NAVIGATION_ROOT/
  mp3d_scenes/                    # MP3D_ROOT — needs mp3d.scene_dataset_config.json
  R2R_VLNCE_v1-3/train/train.json.gz
  RxR_VLNCE_v0/train/train_guide.json.gz
  LLaVA-Video-7B-Qwen2/
  siglip-so400m-patch14-384/

<repo>/train_data/                # TRAIN_DATA_ROOT default — matches configs/*.yaml
```

Upstream sources (download yourself): Matterport3D, [VLN-CE](https://github.com/jacobkrantz/VLN-CE)
R2R-CE / RxR-CE releases, and the public LLaVA-Video / SigLIP checkpoints.

`requirements_paper.txt` is a **host freeze** (may contain private pins). Do not treat it as a
public one-shot install list.

## 4. End-to-end chain

```bash
cd /path/to/NavSpace
export NAVIGATION_ROOT=/path/to/mp3d_and_vlnce
export TRAIN_DATA_ROOT=$PWD/train_data           # default; matches configs/*.yaml
export LLM_VERSION=$NAVIGATION_ROOT/LLaVA-Video-7B-Qwen2
export VISION_MODEL_VERSION=$NAVIGATION_ROOT/siglip-so400m-patch14-384
export PYTHONPATH=$PWD/snav_training:$PYTHONPATH
```

**1. Prepare Stage-A data** (MP3D + R2R-CE + RxR-CE already on disk):

```bash
bash snav_training/scripts/prepare_stage_a_data.sh
# phases: all|collect|build|stopinstr ; smoke: R2R_TRAJ_NUM=2 RXR_TRAJ_NUM=2
```

Writes `$TRAIN_DATA_ROOT/training_data_*` frames and:
`r2r_stepwise_train_jupyter_full.json`, `rxr_stepwise_train_jupyter_full_en.json`,
plus stop-only and instruction-reconstruction companions.

**2. Stage A train** (nav-only YAML by default):

```bash
export IMAGE_FOLDER=$TRAIN_DATA_ROOT
bash snav_training/scripts/launch_vln_mix_stage_a.sh
# optional paper full mix (+ LLaVA-OE JSONs you prepare yourself):
# DATA_YAML=$PWD/snav_training/configs/train_llava_mix_stopdup.yaml \
# MID_RUN_NAME=r2r_rxr_llava_mix_stopdup bash snav_training/scripts/launch_vln_mix_stage_a.sh
```

**3. Stage-B images** via [`../data_augmentation/`](../data_augmentation/) → `snav_data/aug_mix`
(and optional `snav_data/manual_98`).

**4. Build Stage-B JSON + train:**

```bash
# aug only
python snav_training/scripts/build_hist8_future6.py \
  --aug-root snav_data/aug_mix \
  --out-dir train_data

# or aug + manual_98
python snav_training/scripts/build_hist8_future6.py \
  --aug-root snav_data/aug_mix \
  --manual-root snav_data/manual_98 \
  --out-dir train_data

export PREV_STAGE_CHECKPOINT=$TRAIN_DATA_ROOT/work_dirs/r2r_rxr_llava_mix_nav_only
export IMAGE_FOLDER=$PWD/snav_data/aug_mix
bash snav_training/scripts/launch_paper_sft_stage_b.sh
```

**5. Eval** with [`../evaluation/eval_snav.py`](../evaluation/eval_snav.py)
(defaults: 384×384, max-frames 16, HFOV 120; pass `--vision-tower-path`).

**Baseline Stage-1 only** (not the paper chain) — see [`data_generation/README.md`](data_generation/README.md):

```bash
export LLAVA_ROOT=... MODEL_PATH=... VIDEO_FOLDERS=... OUTPUT_DIR=...
bash snav_training/scripts/run_snav_train.sh
```
