#!/usr/bin/env bash
# Render R2R-CE `train` split into SNav-style per-step JPG + llava_annotations.json.
#
# Required env vars (export before running):
#   R2RCE_TRAIN_JSON   absolute path to R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
#   SCENES_ROOT        root of scene assets (HM3D and/or MP3D subfolders)
#   DATA_ROOT          where rendered data should land (default ./snav_data)
#
# Optional overrides:
#   MAX_EPISODES         0 = all
#   CONDA_ENV_NAME       conda env that has habitat-sim installed (default: streamvln)
#   RENDER_PY            path to render_streamvln.py (default: alongside this script)
#
# Domain randomization. All disabled by default to keep this wrapper backward
# compatible:
#   CAMERA_HEIGHT_JITTER   metres, e.g. 0.15 → height ~ U(1.35, 1.65)   [0]
#   HFOV_JITTER            degrees, e.g. 15 → hfov ~ U(75, 105)         [0]
#   RESOLUTION_CHOICES     comma list, e.g. "160,192,224,256,320"      [""]
#   NUM_RENDER_VARIANTS    K renders per episode (default 1)            [1]
#   RANDOMIZE_SEED         deterministic per-episode seed               [42]
#   GPU_DEVICE_ID          0 = first NVIDIA GPU (default; fast),         [0]
#                          1 = second GPU (use if 0 has a display),
#                          -1 = software OpenGL via Mesa (CPU; very slow,
#                               only useful as a fallback when EGL/CUDA
#                               interop is broken).
#
# Unknown CLI args are forwarded to render_streamvln.py.

set -euo pipefail

export JAVA_HOME="${JAVA_HOME:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER_PY="${RENDER_PY:-${SCRIPT_DIR}/render_streamvln.py}"

# ── Habitat-Sim EGL/NVIDIA fix (same trick NavSpace/el.sh uses for eval) ──
# Without this, the conda env's Mesa libGL/libEGL are loaded first and
# habitat-sim either crashes ("cannot retrieve OpenGL version: InvalidValue")
# or fails to find the CUDA-EGL extension on the GPU. Force-load the system
# NVIDIA GLX dispatch libs and point GLVND at /usr/share/glvnd's NVIDIA ICD.
NVIDIA_GLX_LIB="${NVIDIA_GLX_LIB:-/lib/x86_64-linux-gnu/libGLX_nvidia.so.0}"
NVIDIA_GL_DISPATCH_LIB="${NVIDIA_GL_DISPATCH_LIB:-/lib/x86_64-linux-gnu/libGLdispatch.so.0}"
if [ -z "${LD_PRELOAD:-}" ] && [ -e "${NVIDIA_GLX_LIB}" ] && [ -e "${NVIDIA_GL_DISPATCH_LIB}" ]; then
  export LD_PRELOAD="${NVIDIA_GLX_LIB}:${NVIDIA_GL_DISPATCH_LIB}"
fi
NVIDIA_EGL_ICD="${NVIDIA_EGL_ICD:-/usr/share/glvnd/egl_vendor.d/10_nvidia.json}"
if [ -z "${__EGL_VENDOR_LIBRARY_FILENAMES:-}" ] && [ -e "${NVIDIA_EGL_ICD}" ]; then
  export __EGL_VENDOR_LIBRARY_FILENAMES="${NVIDIA_EGL_ICD}"
fi

# ── Activate conda env (optional) ──
CONDA_ENV_NAME="${CONDA_ENV_NAME:-streamvln}"
if command -v conda >/dev/null 2>&1; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$CONDA_ENV_NAME" 2>/dev/null || {
    echo "[WARN] conda env '${CONDA_ENV_NAME}' not found, continuing with current python"
  }
fi

: "${R2RCE_TRAIN_JSON:?need R2RCE_TRAIN_JSON=/abs/path/to/train.json.gz}"
: "${SCENES_ROOT:?need SCENES_ROOT=/abs/path/to/scene_datasets}"
DATA_ROOT="${DATA_ROOT:-$(pwd)/snav_data}"
MAX_EPISODES="${MAX_EPISODES:-0}"

# ── Domain randomization knobs (default: off / vanilla) ──
CAMERA_HEIGHT_JITTER="${CAMERA_HEIGHT_JITTER:-0.0}"
HFOV_JITTER="${HFOV_JITTER:-0.0}"
RESOLUTION_CHOICES="${RESOLUTION_CHOICES:-}"
NUM_RENDER_VARIANTS="${NUM_RENDER_VARIANTS:-1}"
RANDOMIZE_SEED="${RANDOMIZE_SEED:-42}"
GPU_DEVICE_ID="${GPU_DEVICE_ID:-0}"

OUT_DIR="${DATA_ROOT}/r2rce"
LOG_PATH="${DATA_ROOT}/r2rce_render.log"

mkdir -p "$OUT_DIR" "$(dirname "$LOG_PATH")"

echo "=== Rendering R2R-CE train (eval-aligned: 384×384, FOV=120°, turn=30°, h=1.5m) ==="
echo "  Episodes JSON       : $R2RCE_TRAIN_JSON"
echo "  Scenes root         : $SCENES_ROOT"
echo "  Output dir          : $OUT_DIR"
echo "  camera_height_jitter: $CAMERA_HEIGHT_JITTER m"
echo "  hfov_jitter         : $HFOV_JITTER °"
echo "  resolution_choices  : ${RESOLUTION_CHOICES:-—}"
echo "  num_render_variants : $NUM_RENDER_VARIANTS"
echo "  randomize_seed      : $RANDOMIZE_SEED"
echo "  gpu_device_id       : $GPU_DEVICE_ID  (-1=CPU/Mesa, 0=NVIDIA GPU 0)"

# Base values match evaluation/common.py::create_simulator so that the
# model's "←/→" actions rotate by the same 30° at training and inference,
# and the camera sees the world from the same 1.5 m height.
python "$RENDER_PY" \
  --data_json "$R2RCE_TRAIN_JSON" \
  --data_format r2r \
  --dataset_tag r2r \
  --scenes_root "$SCENES_ROOT" \
  --output_dir "$OUT_DIR" \
  --log_path "$LOG_PATH" \
  --output_mode "${OUTPUT_MODE:-frames}" \
  --video_subdir r2rce \
  --max_episodes "$MAX_EPISODES" \
  --max_steps 500 \
  --goal_radius 0.5 \
  --forward_step 0.25 \
  --turn_angle 30.0 \
  --width 384 --height 384 --hfov 120 \
  --camera_height 1.5 \
  --camera_height_jitter "$CAMERA_HEIGHT_JITTER" \
  --hfov_jitter "$HFOV_JITTER" \
  --resolution_choices "$RESOLUTION_CHOICES" \
  --num_render_variants "$NUM_RENDER_VARIANTS" \
  --randomize_seed "$RANDOMIZE_SEED" \
  --gpu_device_id "$GPU_DEVICE_ID" \
  "$@"

echo "Done. annotations -> $OUT_DIR/{annotations,llava_annotations}.json"
