#!/usr/bin/env bash
# Render R2R-CE `train` split into SNav-style per-step JPG + llava_annotations.json.
#
# Required env vars (export before running):
#   R2RCE_TRAIN_JSON   absolute path to R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
#   SCENES_ROOT        root of scene assets (HM3D and/or MP3D subfolders)
#   DATA_ROOT          where rendered data should land (default ./snav_data)
#
# Optional overrides:
#   MAX_EPISODES       0 = all
#   CONDA_ENV_NAME     conda env that has habitat-sim installed (default: streamvln)
#   RENDER_PY          path to render_streamvln.py (default: alongside this script)
#
# Unknown CLI args are forwarded to render_streamvln.py.

set -euo pipefail

export JAVA_HOME="${JAVA_HOME:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER_PY="${RENDER_PY:-${SCRIPT_DIR}/render_streamvln.py}"

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

OUT_DIR="${DATA_ROOT}/r2rce"
LOG_PATH="${DATA_ROOT}/r2rce_render.log"

mkdir -p "$OUT_DIR" "$(dirname "$LOG_PATH")"

echo "=== Rendering R2R-CE train (SNav frames, FOV=120, 384x384) ==="
echo "  Episodes JSON : $R2RCE_TRAIN_JSON"
echo "  Scenes root   : $SCENES_ROOT"
echo "  Output dir    : $OUT_DIR"

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
  --turn_angle 15.0 \
  --width 384 --height 384 --hfov 120 \
  --camera_height 0.88 \
  "$@"

echo "Done. llava_annotations.json -> $OUT_DIR/llava_annotations.json"
