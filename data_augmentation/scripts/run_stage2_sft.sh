#!/usr/bin/env bash
# Stage-2 aug-only SFT — continue training from a Stage-1 SNav checkpoint
# on ONLY the four augmented sub-folders inside snav_data/aug_mix/.
#
# Required env vars:
#   MODEL_PATH       Stage-1 SNav checkpoint (DeepSpeed output dir with HF
#                    files, e.g. /abs/path/to/snav_stage1_ckpt)
#   LLAVA_ROOT       LLaVA checkout providing `llava/` (e.g. StreamVLN root)
#   OUTPUT_DIR       where to write Stage-2 checkpoints + tb logs
#
# Optional:
#   VIDEO_FOLDERS       comma list of folders with annotations.json (SNav frames
#                       layout, i.e. {ep_tag}/rgb/{NNN:03d}.jpg).
#                       Default: all four pipelines under snav_data/aug_mix/
#   CONDA_ENV_NAME      default: navspace
#   SKIP_CONDA          1 to skip `conda activate` (use in Docker images that
#                       already provide python)
#   VISION_TOWER_PATH   local SigLIP dir; recommended for offline / Docker runs
#                       (avoids HuggingFace hub download for mm_vision_tower)
#   MODEL_NAME          default llava_qwen
#   CHUNK_STRIDE        default: 1 (step-wise samples per clip)
#   NUM_EPOCHS          default: 1
#   LR                  default: 2e-5 (smaller than Stage-1 5e-5 to avoid drift)
#   GRAD_ACCUM          default: 12
#   AUGMENT             default: 1 (dataset-side appearance jitter on)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DA_ROOT}/.." && pwd)"

: "${MODEL_PATH:?need MODEL_PATH=/abs/path/to/snav_stage1_ckpt}"
: "${LLAVA_ROOT:?need LLAVA_ROOT=/abs/path/to/LLaVA (or StreamVLN root)}"
: "${OUTPUT_DIR:?need OUTPUT_DIR=/abs/path/to/snav_stage2_aug_only}"

AUG_ROOT="${REPO_ROOT}/snav_data/aug_mix"
DEFAULT_FOLDERS=""
for sub in vertical_perception environment_state spatial_relationship precise_movement; do
  if [[ -f "${AUG_ROOT}/${sub}/annotations.json" ]]; then
    if [[ -z "$DEFAULT_FOLDERS" ]]; then
      DEFAULT_FOLDERS="${AUG_ROOT}/${sub}"
    else
      DEFAULT_FOLDERS="${DEFAULT_FOLDERS},${AUG_ROOT}/${sub}"
    fi
  fi
done
VIDEO_FOLDERS="${VIDEO_FOLDERS:-$DEFAULT_FOLDERS}"

if [[ -z "$VIDEO_FOLDERS" ]]; then
  echo "[run_stage2_sft] ERROR: no annotations.json found under $AUG_ROOT/*" >&2
  echo "  Run scripts/run_all_pipelines.sh + scripts/run_render_aug.sh," >&2
  echo "  then scripts/convert_aug_to_sft.py first." >&2
  exit 2
fi

export CONDA_ENV_NAME="${CONDA_ENV_NAME:-navspace}"
export SKIP_CONDA="${SKIP_CONDA:-0}"
export CHUNK_STRIDE="${CHUNK_STRIDE:-1}"
export NUM_EPOCHS="${NUM_EPOCHS:-1}"
export LR="${LR:-2e-5}"
export GRAD_ACCUM="${GRAD_ACCUM:-12}"
export AUGMENT="${AUGMENT:-1}"
export MODEL_NAME="${MODEL_NAME:-llava_qwen}"
export VISION_TOWER_PATH="${VISION_TOWER_PATH:-}"
export MODEL_PATH
export VIDEO_FOLDERS
export OUTPUT_DIR
export LLAVA_ROOT

echo "=============================================="
echo "  NavSpace Stage-2 aug-only SFT"
echo "=============================================="
echo "  base ckpt        : $MODEL_PATH"
echo "  model name       : $MODEL_NAME"
echo "  vision tower     : ${VISION_TOWER_PATH:-<from ckpt config>}"
echo "  llava root       : $LLAVA_ROOT"
echo "  video folders    : $VIDEO_FOLDERS"
echo "  output dir       : $OUTPUT_DIR"
echo "  conda env        : ${SKIP_CONDA:+SKIPPED }${CONDA_ENV_NAME}"
echo "  LR / Epochs      : $LR / $NUM_EPOCHS"
echo "  CHUNK_STRIDE     : $CHUNK_STRIDE"
echo "  AUGMENT          : $AUGMENT"
echo "----------------------------------------------"

STAGE1_LAUNCHER="${REPO_ROOT}/snav_training/scripts/run_snav_train.sh"
if [[ ! -f "$STAGE1_LAUNCHER" ]]; then
  STAGE1_LAUNCHER="${REPO_ROOT}/NavSpace/snav_training/scripts/run_snav_train.sh"
fi

if [[ ! -f "$STAGE1_LAUNCHER" ]]; then
  echo "[run_stage2_sft] ERROR: snav stage-1 launcher not found." >&2
  exit 2
fi

bash "$STAGE1_LAUNCHER"
