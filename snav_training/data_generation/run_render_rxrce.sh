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

set -euo pipefail

export JAVA_HOME="${JAVA_HOME:-}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RENDER_PY="${RENDER_PY:-${SCRIPT_DIR}/render_streamvln.py}"

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

OUT_DIR="${DATA_ROOT}/rxrce"
LOG_PATH="${DATA_ROOT}/rxrce_render.log"

mkdir -p "$OUT_DIR" "$(dirname "$LOG_PATH")"

echo "=== Rendering RxR-CE train_guide (lang=${LANG_FILTER}, SNav frames) ==="
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
  --turn_angle 15.0 \
  --width 384 --height 384 --hfov 120 \
  --camera_height 0.88 \
  --lang_filter "$LANG_FILTER" \
  "$@"

echo "Done. llava_annotations.json -> $OUT_DIR/llava_annotations.json"
