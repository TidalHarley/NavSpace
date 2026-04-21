# SNav training (Stage-1 vanilla)

This directory contains the **vanilla SFT training recipe** for SNav, the
LLaVA-Video-style navigation policy that `evaluation/eval_snav.py` evaluates.

> **Scope.** Everything under `snav_training/` is limited to the **Stage-1
> vanilla supervised fine-tuning** (geodesic-follower expert trajectories +
> per-step action labels). It does **not** include any of the data-augmentation
> pipelines used in later stages (instruction rewriting with LLMs and etc). Those belong to a
> separate release and are intentionally out of scope here.

## What this pipeline does

1. **Data generation** — render R2R-CE / RxR-CE / (optionally) EnvDrop episodes
   through Habitat-Sim's `GreedyGeodesicFollower` into per-step RGB frames plus
   an `annotations.json` index (StreamVLN-compatible layout).
2. **Training** — full-parameter SFT of a LLaVA-Video-7B-Qwen2 base model on
   the rendered data via the in-repo `train_snav.py` entry (DeepSpeed ZeRO-2).
3. **Evaluation** (optional) — hand off the resulting checkpoint to
   `evaluation/eval_snav.py` to score on NavSpace.

## Layout

```
snav_training/
├── README.md                            (this file)
├── train_snav.py                        main SFT entry (used by run_snav_train.sh)
├── dataset_snav.py                      PyTorch dataset + collator for SNav / QA
├── configs/
│   ├── snav_data.yaml.template          optional YAML listing rendered folders
│   ├── deepspeed_zero2.json             DeepSpeed ZeRO-2 config (default)
│   └── deepspeed_zero3.json             ZeRO-3 with CPU offload (use for <80 GB GPUs)
├── data_generation/
│   ├── README.md                        data-generation guide
│   ├── render_streamvln.py              Python renderer (R2R-CE / RxR-CE / EnvDrop)
│   ├── run_render_r2rce.sh              R2R-CE train split
│   ├── run_render_rxrce.sh              RxR-CE train split
│   └── run_render_envdrop.sh            EnvDrop subset
└── scripts/
    ├── run_snav_train.sh                SNav Stage-1 vanilla SFT launcher
    └── run_pipeline.sh                  render → train → eval orchestrator
```

## Prerequisites

- One conda env with Habitat-Sim (0.3.x) **and** a LLaVA / DeepSpeed stack. The
  evaluation setup already uses `streamvln` for Habitat; the same env also
  works for training once `deepspeed` + `tensorboard` are installed
  (already included in the evaluation's `requirements-local-model.txt`).
- Scene assets (HM3D and/or MP3D) under `${SCENES_ROOT}`.
- A LLaVA codebase on `PYTHONPATH`. If you already have StreamVLN checked out,
  its `StreamVLN/` root ships the expected `llava/` package and can be reused;
  set `LLAVA_ROOT` to that directory.
- A base model for SFT, e.g. `lmms-lab/LLaVA-Video-7B-Qwen2`.

## End-to-end usage

### 1. Generate training data

```bash
export DATA_ROOT=/abs/path/to/snav_data
export SCENES_ROOT=/abs/path/to/scene_datasets
export R2RCE_TRAIN_JSON=/abs/path/to/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
export RXRCE_TRAIN_JSON=/abs/path/to/RxR_VLNCE_v0/train/train_guide.json
export ENVDROP_SOURCE_JSON=/abs/path/to/R2R_VLNCE_v1-3_preprocessed/envdrop/envdrop.json.gz

conda activate streamvln
bash data_generation/run_render_r2rce.sh
bash data_generation/run_render_rxrce.sh
bash data_generation/run_render_envdrop.sh
```

This produces `${DATA_ROOT}/{r2rce,rxrce,envdrop}/annotations.json` plus
per-episode `rgb/*.jpg` frames. See
[`data_generation/README.md`](data_generation/README.md) for the full flag
reference.

### 2. Launch SFT

```bash
export LLAVA_ROOT=/abs/path/to/StreamVLN        # any checkout providing llava/
export MODEL_PATH=/abs/path/to/LLaVA-Video-7B-Qwen2
export VIDEO_FOLDERS=${DATA_ROOT}/r2rce,${DATA_ROOT}/rxrce,${DATA_ROOT}/envdrop
export OUTPUT_DIR=/abs/path/to/checkpoints/snav_stage1_vanilla

bash scripts/run_snav_train.sh
```

Switch to ZeRO-3 with CPU offload for <80 GB GPUs:

```bash
export DEEPSPEED_CONFIG=$(pwd)/configs/deepspeed_zero3.json
bash scripts/run_snav_train.sh
```

Optional: mix a small fraction of general Video-QA data (LLaVA-Video-178K
format) to mitigate catastrophic forgetting:

```bash
export QA_JSON_PATHS=/abs/path/to/qa_train.json
export QA_VIDEO_ROOTS=/abs/path/to/qa_videos
export QA_RATIO=0.15
bash scripts/run_snav_train.sh
```

### 3. (Optional) Run NavSpace evaluation on the new checkpoint

```bash
export HM3D_BASE_PATH=/abs/path/to/hm3d_v0.2
EVAL_TASK=environment_state \
  SKIP_RENDER=1 SKIP_TRAIN=1 bash scripts/run_pipeline.sh
```

Or call `evaluation/eval_snav.py` directly — see `docs/evaluation.md` for the
full CLI surface.

## Reproducibility defaults (matches SNav Stage-1 v1)

| Knob | Value |
| --- | --- |
| Base model | `LLaVA-Video-7B-Qwen2` |
| Prompt version | `qwen_1_5` |
| Learning rate | `5e-5`, cosine schedule, `warmup_ratio=0.075` |
| Epochs | 1 |
| Per-device batch / grad-accum | 1 / 12 |
| Frames per sample | 16 |
| Future actions predicted per chunk | 6 |
| Max seq length | 32768 |
| Precision | BF16 + `sdpa` attention |
| Optimizer | DeepSpeed ZeRO-2 (ZeRO-3 available as opt-in) |

All knobs are exposed as env vars or CLI flags to `train_snav.py`; check the
top of `scripts/run_snav_train.sh` for the full list.

## What's explicitly not here

- No DAgger / trajectory-correction loop.
- No instruction rewriting via LLM (the SNav Stage-2 flow).
- No panorama augmentation, no landmark-enriched supervision.
- No evaluation-time beam search / rollback logic.

If you need those flows, stay tuned for the SNav Stage-2/3 release. For the
vanilla baseline above, everything in this folder is self-contained.
