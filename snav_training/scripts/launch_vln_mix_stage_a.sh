#!/usr/bin/env bash
# Stage A — VLN-mix SFT (default: navigation-only recipe).
# Produces the intermediate checkpoint that Stage B initializes from.
#
# Required / typical env:
#   NAVIGATION_ROOT, LLM_VERSION, VISION_MODEL_VERSION, TRAIN_DATA_ROOT
# Optional: CUDA_VISIBLE_DEVICES, MID_RUN_NAME, CONDA_ENV_NAME, SKIP_CONDA=1
# Paper full mix (incl. LLaVA-OE):
#   DATA_YAML=.../train_llava_mix_stopdup.yaml MID_RUN_NAME=r2r_rxr_llava_mix_stopdup
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ST_ROOT}/.." && pwd)"

NAVIGATION_ROOT="${NAVIGATION_ROOT:-${REPO_ROOT}}"
# Keep default under the repo so configs/*.yaml ../../train_data paths resolve.
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-${REPO_ROOT}/train_data}"
LLM_VERSION="${LLM_VERSION:-${NAVIGATION_ROOT}/LLaVA-Video-7B-Qwen2}"
VISION_MODEL_VERSION="${VISION_MODEL_VERSION:-${NAVIGATION_ROOT}/siglip-so400m-patch14-384}"
PREV_STAGE_CHECKPOINT="${PREV_STAGE_CHECKPOINT:-${LLM_VERSION}}"
DATA_YAML="${DATA_YAML:-${ST_ROOT}/configs/train_llava_mix_nav_only.yaml}"
IMAGE_FOLDER="${IMAGE_FOLDER:-${TRAIN_DATA_ROOT}}"
MID_RUN_NAME="${MID_RUN_NAME:-r2r_rxr_llava_mix_nav_only}"
LOG_DIR="${LOG_DIR:-${ST_ROOT}/logs}"
mkdir -p "$LOG_DIR" "${TRAIN_DATA_ROOT}/work_dirs"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}"
export NAVIGATION_ROOT TRAIN_DATA_ROOT LLM_VERSION VISION_MODEL_VERSION
export PREV_STAGE_CHECKPOINT DATA_YAML IMAGE_FOLDER MID_RUN_NAME
export NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
export PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
export GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
export SAVE_STEPS="${SAVE_STEPS:-5000}"
export FRAMES_UPBOUND="${FRAMES_UPBOUND:-8}"
export MASTER_PORT="${MASTER_PORT:-30423}"
export REPORT_TO="${REPORT_TO:-wandb}"
export TORCH_COMPILE="${TORCH_COMPILE:-True}"
export TORCH_COMPILE_BACKEND="${TORCH_COMPILE_BACKEND:-inductor}"
export DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ST_ROOT}/configs/deepspeed_zero3_paper.json}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"

if [[ "${SKIP_CONDA:-0}" != "1" ]] && command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "${CONDA_ENV_NAME:-navspace}" 2>/dev/null || \
    echo "[WARN] conda env '${CONDA_ENV_NAME:-navspace}' not found; using current python"
fi

LOG="${LOG_DIR}/${MID_RUN_NAME}.log"
{
  echo "launch_time=$(date '+%F %T %Z')"
  echo "stage=A_vln_mix"
  echo "run_name=${MID_RUN_NAME}"
  echo "prev=${PREV_STAGE_CHECKPOINT}"
  echo "data_yaml=${DATA_YAML}"
  echo "image_folder=${IMAGE_FOLDER}"
  echo "output_dir=${TRAIN_DATA_ROOT}/work_dirs/${MID_RUN_NAME}"
} | tee "${LOG_DIR}/${MID_RUN_NAME}.launch.log"

bash "${SCRIPT_DIR}/train_paper.sh" >> "$LOG" 2>&1
echo "Done. Checkpoint -> ${TRAIN_DATA_ROOT}/work_dirs/${MID_RUN_NAME}"
