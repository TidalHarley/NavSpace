#!/usr/bin/env bash
# Stage B — paper SNav-7B final SFT (aug_mix + manual98, hist=8, future=6).
# Matches H200 run navspace_augmix_plus_manual98_1epoch_8gpu_20260611_retry1.
#
# Required:
#   PREV_STAGE_CHECKPOINT   Stage-A VLN-mix ckpt
# Build JSON first:
#   python snav_training/scripts/build_hist8_future6.py \
#     --aug-root snav_data/aug_mix --manual-root snav_data/manual_98 --out-dir train_data
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ST_ROOT}/.." && pwd)"

NAVIGATION_ROOT="${NAVIGATION_ROOT:-${REPO_ROOT}}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-${REPO_ROOT}/train_data}"
: "${PREV_STAGE_CHECKPOINT:?set PREV_STAGE_CHECKPOINT=/abs/path/to/stage_a_vln_mix_ckpt}"

LLM_VERSION="${LLM_VERSION:-${NAVIGATION_ROOT}/LLaVA-Video-7B-Qwen2}"
VISION_MODEL_VERSION="${VISION_MODEL_VERSION:-${NAVIGATION_ROOT}/siglip-so400m-patch14-384}"
DATA_YAML="${DATA_YAML:-${ST_ROOT}/configs/train_navspace_aug_mix_plus_manual98_hist8_future6.yaml}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${REPO_ROOT}/snav_data/aug_mix}"
MID_RUN_NAME="${MID_RUN_NAME:-navspace_augmix_plus_manual98_1epoch}"
LOG_DIR="${LOG_DIR:-${ST_ROOT}/logs}"
mkdir -p "$LOG_DIR" "${TRAIN_DATA_ROOT}/work_dirs"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NAVIGATION_ROOT TRAIN_DATA_ROOT LLM_VERSION VISION_MODEL_VERSION
export PREV_STAGE_CHECKPOINT DATA_YAML IMAGE_FOLDER MID_RUN_NAME
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-4}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-1}"
export SAVE_STEPS="${SAVE_STEPS:-5000}"
export SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-}"
export FRAMES_UPBOUND="${FRAMES_UPBOUND:-8}"
export MASTER_PORT="${MASTER_PORT:-30611}"
export REPORT_TO="${REPORT_TO:-wandb}"
export TORCH_COMPILE="${TORCH_COMPILE:-True}"
export TORCH_COMPILE_BACKEND="${TORCH_COMPILE_BACKEND:-inductor}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ST_ROOT}/configs/deepspeed_zero3_paper.json}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

if [[ ! -d "$IMAGE_FOLDER" ]]; then
  echo "[ERROR] IMAGE_FOLDER missing: $IMAGE_FOLDER" >&2
  echo "  Build/render aug_mix (+ manual_98) first; see data_augmentation/README.md" >&2
  exit 2
fi
if [[ ! -f "$DATA_YAML" ]]; then
  echo "[ERROR] DATA_YAML missing: $DATA_YAML" >&2
  exit 2
fi

if [[ "${SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-navspace}" 2>/dev/null || \
    echo "[WARN] conda env '${CONDA_ENV_NAME:-navspace}' not found; using current python"
fi

LOG="${LOG_DIR}/${MID_RUN_NAME}.log"
{
  echo "launch_time=$(date '+%F %T %Z')"
  echo "stage=B_paper_sft_augmix_manual98"
  echo "run_name=${MID_RUN_NAME}"
  echo "base_checkpoint=${PREV_STAGE_CHECKPOINT}"
  echo "data_yaml=${DATA_YAML}"
  echo "image_folder=${IMAGE_FOLDER}"
  echo "output_dir=${TRAIN_DATA_ROOT}/work_dirs/${MID_RUN_NAME}"
  echo "settings=epoch1 per_device_bs4 grad_accum1 lr1e-5 frames_upbound8 zero3_paper"
  echo "visible_gpus=${CUDA_VISIBLE_DEVICES}"
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true
} | tee "${LOG_DIR}/${MID_RUN_NAME}.launch.log"

bash "${SCRIPT_DIR}/train_paper.sh" >> "$LOG" 2>&1
echo "Done. Checkpoint -> ${TRAIN_DATA_ROOT}/work_dirs/${MID_RUN_NAME}"
