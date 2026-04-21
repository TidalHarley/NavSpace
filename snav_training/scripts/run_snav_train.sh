#!/usr/bin/env bash
# ==============================================================================
# SNav Stage-1 vanilla SFT launcher — multi-GPU DeepSpeed.
#
# Thin wrapper around the in-repo Python trainer `snav_training/train_snav.py`.
#
# Scope: Stage-1 vanilla SFT only. Data augmentation (instruction rewriting,
# panorama augmentation, DAgger, etc.) is NOT covered by this script.
#
# Required env vars:
#   LLAVA_ROOT      absolute path to a LLaVA checkout providing `llava/`
#                   (e.g. the `StreamVLN/` root — StreamVLN ships a LLaVA copy
#                   that is identical to the code we trained on).
#   MODEL_PATH      base LLaVA-Video model dir (e.g. LLaVA-Video-7B-Qwen2).
#   VIDEO_FOLDERS   comma-separated list of directories with `annotations.json`
#                   (produced by data_generation/run_render_*.sh).
#   OUTPUT_DIR      where checkpoints + tensorboard logs should be written.
#
# Optional env vars:
#   DEEPSPEED_CONFIG      default: ../configs/deepspeed_zero2.json
#   NUM_GPUS              default: autodetect via nvidia-smi
#   CONDA_ENV_NAME        default: streamvln
#   MASTER_PORT           default: randomised
#   HF_HOME               HuggingFace cache dir
#   ATTN_IMPL             default: sdpa (pass flash_attention_2 if installed)
#   NUM_FRAMES            default: 8
#   NUM_FUTURE_STEPS      default: 6
#   GRAD_ACCUM            default: 12
#   LR                    default: 5e-5
#   NUM_EPOCHS            default: 1
#   QA_JSON_PATHS         optional comma-separated QA JSONs (catastrophic-
#                         forgetting mitigation). Must be paired with
#                         QA_VIDEO_ROOTS (same length).
#   QA_VIDEO_ROOTS        see above.
#   QA_RATIO              default: 0.15
# ==============================================================================

set -euo pipefail
trap 'echo "[FATAL] Error at line $LINENO (exit $?). Command: $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
TRAIN_PY="${TRAIN_PY:-${REPO_ROOT}/train_snav.py}"

# ── Conda ──
CONDA_ENV_NAME="${CONDA_ENV_NAME:-streamvln}"
set +u
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME" 2>/dev/null || {
    echo "[WARN] conda env '${CONDA_ENV_NAME}' not found, continuing with current python"
  }
fi
set -u

# ── Required env vars ──
: "${LLAVA_ROOT:?need LLAVA_ROOT=/abs/path/to/LLaVA (or StreamVLN root)}"
: "${MODEL_PATH:?need MODEL_PATH=/abs/path/to/LLaVA-Video-7B-Qwen2}"
: "${VIDEO_FOLDERS:?need VIDEO_FOLDERS=/abs/r2rce,/abs/rxrce,/abs/envdrop}"
: "${OUTPUT_DIR:?need OUTPUT_DIR=/abs/path/to/snav_ckpt_out}"

# ── Optional overrides ──
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${REPO_ROOT}/configs/deepspeed_zero2.json}"
ATTN_IMPL="${ATTN_IMPL:-sdpa}"
NUM_FRAMES="${NUM_FRAMES:-16}"
NUM_FUTURE_STEPS="${NUM_FUTURE_STEPS:-6}"
GRAD_ACCUM="${GRAD_ACCUM:-12}"
LR="${LR:-5e-5}"
NUM_EPOCHS="${NUM_EPOCHS:-1}"
QA_JSON_PATHS="${QA_JSON_PATHS:-}"
QA_VIDEO_ROOTS="${QA_VIDEO_ROOTS:-}"
QA_RATIO="${QA_RATIO:-0.15}"

# ── HuggingFace cache (writable by default) ──
export HF_HOME="${HF_HOME:-${HOME}/.cache/huggingface}"
mkdir -p "${HF_HOME}/hub"

# ── Expose LLaVA on PYTHONPATH ──
export PYTHONPATH="${LLAVA_ROOT}:${PYTHONPATH:-}"
# train_snav.py also reads LLAVA_ROOT from env to be safe.
export LLAVA_ROOT

# ── GPU count ──
NUM_GPUS="${NUM_GPUS:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
if [[ "$NUM_GPUS" -lt 1 ]]; then NUM_GPUS=1; fi

# ── NCCL / CUDA knobs (safe defaults) ──
export NCCL_ASYNC_ERROR_HANDLING=1
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
SHM_AVAIL_KB=$(df --output=avail /dev/shm 2>/dev/null | tail -1 | tr -d ' ' || echo "0")
SHM_AVAIL_GB=$(( ${SHM_AVAIL_KB:-0} / 1048576 ))
if [[ "$SHM_AVAIL_GB" -lt 1 ]]; then
  export NCCL_SHM_DISABLE=1
fi

MASTER_PORT="${MASTER_PORT:-$((RANDOM % 1000 + 29500))}"

echo ""
echo "=========================================="
echo " NavSpace / SNav Stage-1 vanilla SFT"
echo "=========================================="
echo " LLaVA root      : $LLAVA_ROOT"
echo " Trainer script  : $TRAIN_PY"
echo " Base model      : $MODEL_PATH"
echo " Video folders   : $VIDEO_FOLDERS"
echo " Output dir      : $OUTPUT_DIR"
echo " DeepSpeed cfg   : $DEEPSPEED_CONFIG"
echo " GPUs            : $NUM_GPUS"
echo " Attn impl       : $ATTN_IMPL"
echo " Frames / Future : $NUM_FRAMES / $NUM_FUTURE_STEPS"
echo " LR / Epochs     : $LR / $NUM_EPOCHS (grad accum $GRAD_ACCUM)"
echo "=========================================="

[[ -f "$MODEL_PATH/config.json" ]] || { echo "[FATAL] Model not found: $MODEL_PATH"; exit 2; }
[[ -f "$DEEPSPEED_CONFIG" ]]     || { echo "[FATAL] DeepSpeed cfg not found: $DEEPSPEED_CONFIG"; exit 2; }
[[ -f "$TRAIN_PY" ]]             || { echo "[FATAL] Trainer not found: $TRAIN_PY"; exit 2; }

mkdir -p "$OUTPUT_DIR"

EXTRA_QA_ARGS=()
if [[ -n "$QA_JSON_PATHS" && -n "$QA_VIDEO_ROOTS" ]]; then
  EXTRA_QA_ARGS+=(--qa_json_paths "$QA_JSON_PATHS"
                  --qa_video_roots "$QA_VIDEO_ROOTS"
                  --qa_ratio "$QA_RATIO")
fi

deepspeed --num_gpus="$NUM_GPUS" \
  --master_port="$MASTER_PORT" \
  "$TRAIN_PY" \
  --model_path "$MODEL_PATH" \
  --video_folders "$VIDEO_FOLDERS" \
  --output_dir "$OUTPUT_DIR" \
  --deepspeed "$DEEPSPEED_CONFIG" \
  --num_frames "$NUM_FRAMES" \
  --num_future_steps "$NUM_FUTURE_STEPS" \
  --grad_accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --num_epochs "$NUM_EPOCHS" \
  --attn_implementation "$ATTN_IMPL" \
  "${EXTRA_QA_ARGS[@]}"

echo ""
echo "[DONE] SNav Stage-1 vanilla SFT complete!"
echo "  Checkpoint: $OUTPUT_DIR"
