#!/usr/bin/env bash
# Render the merged vertical/env_state/spatial custom_instructions through
# snav_training/data_generation/render_streamvln.py into the SNav training
# layout (per-step JPGs + llava_annotations.json), inside
# snav_data/aug_mix/r2r_rewritten/.
#
# Precise Movement is NOT rendered here; precise_movement/run.py
# already produced its own snav_data/aug_mix/precise_movement/ output in
# the previous step.
#
# Requirements:
#   conda activate navspace39
#   source data_augmentation/env_shim.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DA_ROOT}/.." && pwd)"

cd "$REPO_ROOT"

R2RCE_TRAIN_JSON="${R2RCE_TRAIN_JSON:?need R2RCE_TRAIN_JSON=/abs/path/to/train.json.gz}"
SCENES_ROOT="${SCENES_ROOT:?need SCENES_ROOT=/abs/path/to/mp3d_or_hm3d}"
CUSTOM_INSTR="${CUSTOM_INSTR:-${DA_ROOT}/outputs/merged/custom_instructions.json}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/snav_data/aug_mix/r2r_rewritten}"
DATASET_TAG="${DATASET_TAG:-aug}"
VIDEO_SUBDIR="${VIDEO_SUBDIR:-aug_r2r}"
OUTPUT_MODE="${OUTPUT_MODE:-snav_frames}"
GPU_DEVICE_ID="${GPU_DEVICE_ID:-0}"
MAX_EPISODES="${MAX_EPISODES:-0}"

if [[ ! -f "$CUSTOM_INSTR" ]]; then
  echo "[run_render_aug] ERROR: $CUSTOM_INSTR not found." >&2
  echo "Run 'python data_augmentation/merge.py' first." >&2
  exit 2
fi
if [[ ! -f "$R2RCE_TRAIN_JSON" ]]; then
  echo "[run_render_aug] ERROR: R2RCE_TRAIN_JSON=$R2RCE_TRAIN_JSON not found." >&2
  exit 2
fi
if [[ ! -d "$SCENES_ROOT" ]]; then
  echo "[run_render_aug] ERROR: SCENES_ROOT=$SCENES_ROOT not found." >&2
  exit 2
fi

mkdir -p "$OUTPUT_DIR"

echo "=== render_streamvln.py | aug_r2r (vertical+env_state+spatial) ==="
echo "  custom instructions : $CUSTOM_INSTR"
echo "  output dir          : $OUTPUT_DIR"
echo "  dataset tag         : $DATASET_TAG"
echo "  video_subdir        : $VIDEO_SUBDIR"
echo "  output_mode         : $OUTPUT_MODE"

RENDER_PY="${REPO_ROOT}/NavSpace/snav_training/data_generation/render_streamvln.py"
if [[ ! -f "$RENDER_PY" ]]; then
  RENDER_PY="${REPO_ROOT}/snav_training/data_generation/render_streamvln.py"
fi

python "$RENDER_PY" \
  --data_json "$R2RCE_TRAIN_JSON" \
  --data_format r2r \
  --dataset_tag "$DATASET_TAG" \
  --scenes_root "$SCENES_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --output_mode "$OUTPUT_MODE" \
  --video_subdir "$VIDEO_SUBDIR" \
  --custom_instructions_json "$CUSTOM_INSTR" \
  --max_episodes "$MAX_EPISODES" \
  --max_steps 500 \
  --goal_radius 0.5 \
  --forward_step 0.25 \
  --turn_angle 30.0 \
  --width 384 --height 384 --hfov 120 \
  --camera_height 1.5 \
  --gpu_device_id "$GPU_DEVICE_ID"

echo "Done."
echo "  aug_r2r        -> $OUTPUT_DIR"
echo "  precise (sep.) -> ${REPO_ROOT}/snav_data/aug_mix/precise_movement"
