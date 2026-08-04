#!/usr/bin/env bash
# Render RxR-CE `train_guide` split (English only by default) into
# SNav-style per-step JPG + llava_annotations.json.
#
# Required env vars:
#   RXRCE_TRAIN_JSON   absolute path to RxR_VLNCE_v0/train/train_guide.json
#   SCENES_ROOT        root of scene assets
#   DATA_ROOT          where rendered data should land
#
# Optional:
#   LANG_FILTER        language code(s) to keep (default: en)
#   MAX_EPISODES       0 = all
#   CONDA_ENV_NAME     default: streamvln
#
# Domain randomization (see run_render_r2rce.sh for full description):
#   CAMERA_HEIGHT_JITTER, HFOV_JITTER, RESOLUTION_CHOICES,
#   NUM_RENDER_VARIANTS, RANDOMIZE_SEED

set -euo pipefail

export JAVA_HOME="${JAVA_HOME:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER_PY="${RENDER_PY:-${SCRIPT_DIR}/render_streamvln.py}"

# Habitat-Sim EGL/NVIDIA fix (see run_render_r2rce.sh for full rationale).
NVIDIA_GLX_LIB="${NVIDIA_GLX_LIB:-/lib/x86_64-linux-gnu/libGLX_nvidia.so.0}"
NVIDIA_GL_DISPATCH_LIB="${NVIDIA_GL_DISPATCH_LIB:-/lib/x86_64-linux-gnu/libGLdispatch.so.0}"
if [ -z "${LD_PRELOAD:-}" ] && [ -e "${NVIDIA_GLX_LIB}" ] && [ -e "${NVIDIA_GL_DISPATCH_LIB}" ]; then
  export LD_PRELOAD="${NVIDIA_GLX_LIB}:${NVIDIA_GL_DISPATCH_LIB}"
fi
NVIDIA_EGL_ICD="${NVIDIA_EGL_ICD:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
if [ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ] && [ -e "${NVIDIA_EGL_ICD}" ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_ICD}"
fi

CONDA_ENV_NAME="${CONDA_ENV_NAME:-streamvln}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME" 2>/dev/null || {
    echo "[WARN] conda env '${CONDA_ENV_NAME}' not found, continuing with current python"
  }
fi

: "${RXRCE_TRAIN_JSON:?need RXRCE_TRAIN_JSON=/abs/path/to/train_guide.json}"
: "${SCENES_ROOT:?need SCENES_ROOT=/abs/path/to/scene_datasets}"
DATA_ROOT="${DATA_ROOT:-$(pwd)/snav_data}"
MAX_EPISODES="${MAX_EPISODES:-0}"
LANG_FILTER="${LANG_FILTER:-en}"

CAMERA_HEIGHT_JITTER="${CAMERA_HEIGHT_JITTER:-0.0}"
HFOV_JITTER="${HFOV_JITTER:-0.0}"
RESOLUTION_CHOICES="${RESOLUTION_CHOICES:-}"
NUM_RENDER_VARIANTS="${NUM_RENDER_VARIANTS:-1}"
RANDOMIZE_SEED="${RANDOMIZE_SEED:-42}"
GPU_DEVICE_ID="${GPU_DEVICE_ID:-0}"

OUT_DIR="${DATA_ROOT}/rxrce"
LOG_PATH="${DATA_ROOT}/rxrce_render.log"

mkdir -p "$OUT_DIR" "$(dirname "$LOG_PATH")"

echo "=== Rendering RxR-CE train_guide (lang=${LANG_FILTER}, eval-aligned: 384×384, FOV=120°, turn=30°, h=1.5m) ==="
echo "  camera_height_jitter: $CAMERA_HEIGHT_JITTER m"
echo "  hfov_jitter         : $HFOV_JITTER °"
echo "  resolution_choices  : ${RESOLUTION_CHOICES:-—}"
echo "  num_render_variants : $NUM_RENDER_VARIANTS"
echo "  randomize_seed      : $RANDOMIZE_SEED"
echo "  gpu_device_id       : $GPU_DEVICE_ID  (-1=CPU/Mesa, 0=NVIDIA GPU 0)"
# Base values match evaluation/common.py::create_simulator. See run_render_r2rce.sh.
python "$RENDER_PY" \
  --data_json "$RXRCE_TRAIN_JSON" \
  --data_format rxr \
  --dataset_tag rxr \
  --scenes_root "$SCENES_ROOT" \
  --output_dir "$OUT_DIR" \
  --log_path "$LOG_PATH" \
  --output_mode "${OUTPUT_MODE:-frames}" \
  --video_subdir rxrce \
  --max_episodes "$MAX_EPISODES" \
  --max_steps 800 \
  --goal_radius 0.5 \
  --forward_step 0.25 \
  --turn_angle 30.0 \
  --width 384 --height 384 --hfov 120 \
  --camera_height 1.5 \
  --lang_filter "$LANG_FILTER" \
  --camera_height_jitter "$CAMERA_HEIGHT_JITTER" \
  --hfov_jitter "$HFOV_JITTER" \
  --resolution_choices "$RESOLUTION_CHOICES" \
  --num_render_variants "$NUM_RENDER_VARIANTS" \
  --randomize_seed "$RANDOMIZE_SEED" \
  --gpu_device_id "$GPU_DEVICE_ID" \
  "$@"

echo "Done. annotations -> $OUT_DIR/{annotations,llava_annotations}.json"
