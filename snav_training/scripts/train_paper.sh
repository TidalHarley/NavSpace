#!/bin/bash
# Paper SNav SFT entry (Path P). Used by Stage-A VLN-mix and Stage-B
# aug_mix+manual98 launchers. Runs snav_llava.train.train_mem via DeepSpeed.
#
# All key paths/hparams are overridable via environment variables.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ST_ROOT}/.." && pwd)"
cd "${ST_ROOT}"
export PYTHONPATH="${ST_ROOT}:${PYTHONPATH:-}"

NAVIGATION_ROOT="${NAVIGATION_ROOT:-${REPO_ROOT}}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-${REPO_ROOT}/train_data}"

if [[ -f /etc/network_turbo ]]; then
    # shellcheck disable=SC1091
    source /etc/network_turbo
fi

export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eth0}"
export NCCL_IB_DISABLE="${NCCL_IB_DISABLE:-1}"
export NCCL_NVLS_ENABLE="${NCCL_NVLS_ENABLE:-0}"

CUDNN_LIBRARY_DIR="${CUDNN_LIBRARY_DIR:-${CONDA_PREFIX:-}/lib/python3.10/site-packages/nvidia/cudnn/lib}"
if [[ -n "${CONDA_PREFIX:-}" && -d "${CUDNN_LIBRARY_DIR}" ]]; then
    case ":${LD_LIBRARY_PATH:-}:" in
        *":${CUDNN_LIBRARY_DIR}:"*) ;;
        *) export LD_LIBRARY_PATH="${CUDNN_LIBRARY_DIR}:${LD_LIBRARY_PATH:-}" ;;
    esac
fi

IMAGE_FOLDER="${IMAGE_FOLDER:-${TRAIN_DATA_ROOT}}"
VIDEO_FOLDER="${VIDEO_FOLDER:-}"
DATA_YAML="${DATA_YAML:-${ST_ROOT}/configs/train_llava_mix_nav_only.yaml}"
DEEPSPEED_CONFIG="${DEEPSPEED_CONFIG:-${ST_ROOT}/configs/deepspeed_zero3_paper.json}"

LLM_VERSION="${LLM_VERSION:-${NAVIGATION_ROOT}/LLaVA-Video-7B-Qwen2}"
VISION_MODEL_VERSION="${VISION_MODEL_VERSION:-${NAVIGATION_ROOT}/siglip-so400m-patch14-384}"
PROMPT_VERSION="${PROMPT_VERSION:-qwen_1_5}"
MID_RUN_NAME="${MID_RUN_NAME:-r2r_rxr_llava_mix_nav_only}"
PREV_STAGE_CHECKPOINT="${PREV_STAGE_CHECKPOINT:-${LLM_VERSION}}"
REPORT_TO="${REPORT_TO:-wandb}"
MASTER_PORT="${MASTER_PORT:-30000}"
TORCH_COMPILE="${TORCH_COMPILE:-True}"
TORCH_COMPILE_BACKEND="${TORCH_COMPILE_BACKEND:-inductor}"
NUM_TRAIN_EPOCHS="${NUM_TRAIN_EPOCHS:-1}"
PER_DEVICE_TRAIN_BATCH_SIZE="${PER_DEVICE_TRAIN_BATCH_SIZE:-2}"
GRADIENT_ACCUMULATION_STEPS="${GRADIENT_ACCUMULATION_STEPS:-2}"
SAVE_STEPS="${SAVE_STEPS:-5000}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-}"
FRAMES_UPBOUND="${FRAMES_UPBOUND:-8}"
LEARNING_RATE="${LEARNING_RATE:-1e-5}"

echo "DATA_YAML: ${DATA_YAML}"
echo "IMAGE_FOLDER: ${IMAGE_FOLDER}"
echo "NAVIGATION_ROOT: ${NAVIGATION_ROOT}"
echo "TRAIN_DATA_ROOT: ${TRAIN_DATA_ROOT}"
echo "LLM_VERSION: ${LLM_VERSION}"
echo "VISION_MODEL_VERSION: ${VISION_MODEL_VERSION}"
echo "PREV_STAGE_CHECKPOINT: ${PREV_STAGE_CHECKPOINT}"
echo "RUN_NAME: ${MID_RUN_NAME}"
echo "DEEPSPEED_CONFIG: ${DEEPSPEED_CONFIG}"
echo "NUM_TRAIN_EPOCHS: ${NUM_TRAIN_EPOCHS}"
echo "PER_DEVICE_TRAIN_BATCH_SIZE: ${PER_DEVICE_TRAIN_BATCH_SIZE}"
echo "GRADIENT_ACCUMULATION_STEPS: ${GRADIENT_ACCUMULATION_STEPS}"
echo "SAVE_STEPS: ${SAVE_STEPS}"
echo "LEARNING_RATE: ${LEARNING_RATE}"
if [[ -n "${SAVE_TOTAL_LIMIT}" ]]; then
    echo "SAVE_TOTAL_LIMIT: ${SAVE_TOTAL_LIMIT}"
else
    echo "SAVE_TOTAL_LIMIT: none"
fi
echo "FRAMES_UPBOUND: ${FRAMES_UPBOUND}"
echo "TORCH_COMPILE: ${TORCH_COMPILE}"

deepspeed_cmd=(
    deepspeed
    --master_port "${MASTER_PORT}"
    snav_llava/train/train_mem.py
    --deepspeed "${DEEPSPEED_CONFIG}"
    --model_name_or_path "${PREV_STAGE_CHECKPOINT}"
    --version "${PROMPT_VERSION}"
    --data_path "${DATA_YAML}"
    --image_folder "${IMAGE_FOLDER}"
    --video_folder "${VIDEO_FOLDER}"
    --mm_tunable_parts "mm_vision_tower,mm_mlp_adapter,mm_language_model"
    --mm_vision_tower_lr 2e-6
    --vision_tower "${VISION_MODEL_VERSION}"
    --mm_projector_type mlp2x_gelu
    --mm_vision_select_layer -2
    --mm_use_im_start_end False
    --mm_use_im_patch_token False
    --group_by_modality_length True
    --image_aspect_ratio anyres_max_9
    --image_grid_pinpoints "(1x1),...,(6x6)"
    --mm_patch_merge_type spatial_unpad
    --bf16 True
    --run_name "${MID_RUN_NAME}"
    --output_dir "${TRAIN_DATA_ROOT}/work_dirs/${MID_RUN_NAME}"
    --num_train_epochs "${NUM_TRAIN_EPOCHS}"
    --per_device_train_batch_size "${PER_DEVICE_TRAIN_BATCH_SIZE}"
    --per_device_eval_batch_size 4
    --gradient_accumulation_steps "${GRADIENT_ACCUMULATION_STEPS}"
    --evaluation_strategy no
    --save_strategy steps
    --save_steps "${SAVE_STEPS}"
    --learning_rate "${LEARNING_RATE}"
    --weight_decay 0.0
    --warmup_ratio 0.03
    --lr_scheduler_type cosine
    --logging_steps 1
    --tf32 True
    --model_max_length 32768
    --gradient_checkpointing True
    --dataloader_num_workers 2
    --lazy_preprocess True
    --report_to "${REPORT_TO}"
    --dataloader_drop_last True
    --frames_upbound "${FRAMES_UPBOUND}"
    --mm_newline_position grid
    --add_time_instruction True
    --force_sample False
    --mm_spatial_pool_stride 2
)

if [[ -n "${SAVE_TOTAL_LIMIT}" ]]; then
    deepspeed_cmd+=(
        --save_total_limit "${SAVE_TOTAL_LIMIT}"
    )
fi

if [[ "${TORCH_COMPILE}" == "True" || "${TORCH_COMPILE}" == "true" ]]; then
    deepspeed_cmd+=(
        --torch_compile True
        --torch_compile_backend "${TORCH_COMPILE_BACKEND}"
    )
fi

"${deepspeed_cmd[@]}"
