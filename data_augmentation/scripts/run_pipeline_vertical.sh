#!/usr/bin/env bash
# Full vertical_perception pipeline: filter -> verify -> render -> rewrite -> patch.
#
# Per the project requirements:
#   * we always RENDER FIRST with the original R2R instructions, so the
#     slow Habitat pass is independent of the Qwen rewrite;
#   * we then REWRITE LAST and PATCH the existing llava_annotations.json
#     in place, swapping the original instruction substring with the new
#     floor-aware one.
#   * VL stair detection is OFF by default — the semantic-region path is
#     used to decide whether an episode crosses a floor.
#
# Requirements before invocation:
#     conda activate navspace39
#     source data_augmentation/env_shim.sh           # only needed for render
#
# Optional knobs (override via env):
#     MAX_EPISODES=400            target verified episode count
#     STAIRS_CHECK=0              1 → also gate by Qwen-VL stair detection
#     RESTART=0                   1 → re-run from scratch (drops checkpoint)
#     OUTPUT_DIR=snav_data/aug_mix/r2r_rewritten
#     DATASET_TAG=aug_vert
#     VIDEO_SUBDIR=aug_vert
#     RENDER_MAX_EPISODES=0       (0 = render everything in the passthrough)
#     GPU_DEVICE_ID=0

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DA_ROOT}/.." && pwd)"

cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
MAX_EPISODES="${MAX_EPISODES:-400}"
STAIRS_CHECK="${STAIRS_CHECK:-0}"
RESTART="${RESTART:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/snav_data/aug_mix/r2r_rewritten}"
DATASET_TAG="${DATASET_TAG:-aug_vert}"
VIDEO_SUBDIR="${VIDEO_SUBDIR:-aug_vert}"
OUTPUT_MODE="${OUTPUT_MODE:-snav_frames}"
RENDER_MAX_EPISODES="${RENDER_MAX_EPISODES:-0}"
GPU_DEVICE_ID="${GPU_DEVICE_ID:-0}"
R2RCE_TRAIN_JSON="${R2RCE_TRAIN_JSON:?need R2RCE_TRAIN_JSON=/abs/path/to/train.json.gz}"
SCENES_ROOT="${SCENES_ROOT:?need SCENES_ROOT=/abs/path/to/mp3d_or_hm3d}"

VP_OUT="${DA_ROOT}/outputs/vertical_perception"
PASSTHROUGH="${VP_OUT}/passthrough_instructions.json"
CUSTOM_INSTR="${VP_OUT}/custom_instructions.json"

RESTART_FLAG=""
if [[ "$RESTART" == "1" ]]; then
  RESTART_FLAG="--restart"
fi

STAIRS_FLAG=""
if [[ "$STAIRS_CHECK" == "1" ]]; then
  STAIRS_FLAG="--enable-stairs-check"
fi

echo "=============================================="
echo "  Vertical Perception full pipeline"
echo "=============================================="
echo "  MAX_EPISODES        : $MAX_EPISODES"
echo "  STAIRS_CHECK        : $STAIRS_CHECK"
echo "  RESTART             : $RESTART"
echo "  OUTPUT_DIR          : $OUTPUT_DIR"
echo "  DATASET_TAG         : $DATASET_TAG"
echo "  VIDEO_SUBDIR        : $VIDEO_SUBDIR"
echo "  RENDER_MAX_EPISODES : $RENDER_MAX_EPISODES"
echo "  GPU_DEVICE_ID       : $GPU_DEVICE_ID"
echo "----------------------------------------------"

echo ">>> [1/5] 1_filter.py (R2R height filter)"
$PYTHON data_augmentation/vertical_perception/1_filter.py \
  --max-episodes "$MAX_EPISODES"

echo ">>> [2/5] 2_verify.py (Habitat semantic-floor verify)"
$PYTHON data_augmentation/vertical_perception/2_verify.py \
  --max-episodes "$MAX_EPISODES" $STAIRS_FLAG $RESTART_FLAG

echo ">>> [3/5] build_passthrough_instructions.py"
$PYTHON data_augmentation/scripts/build_passthrough_instructions.py \
  --verified "${VP_OUT}/verified.json" \
  --out "$PASSTHROUGH"

# ── 4. Render with ORIGINAL instructions so frames + LLaVA stub exist. ──
RENDER_PY="${REPO_ROOT}/snav_training/data_generation/render_streamvln.py"
if [[ ! -f "$RENDER_PY" ]]; then
  RENDER_PY="${REPO_ROOT}/NavSpace/snav_training/data_generation/render_streamvln.py"
fi

mkdir -p "$OUTPUT_DIR"
echo ">>> [4/5] render_streamvln.py (frames + original-text LLaVA stub)"
$PYTHON "$RENDER_PY" \
  --data_json "$R2RCE_TRAIN_JSON" \
  --data_format r2r \
  --dataset_tag "$DATASET_TAG" \
  --scenes_root "$SCENES_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --output_mode "$OUTPUT_MODE" \
  --video_subdir "$VIDEO_SUBDIR" \
  --custom_instructions_json "$PASSTHROUGH" \
  --max_episodes "$RENDER_MAX_EPISODES" \
  --max_steps 500 \
  --goal_radius 0.5 \
  --forward_step 0.25 \
  --turn_angle 30.0 \
  --width 384 --height 384 --hfov 120 \
  --camera_height 1.5 \
  --gpu_device_id "$GPU_DEVICE_ID"

# ── 5. Qwen rewrite + patch the rendered LLaVA annotations. ──
echo ">>> [5/5] 3_rewrite.py (Qwen) + patch_llava_instructions.py"
$PYTHON data_augmentation/vertical_perception/3_rewrite.py

LLAVA_PATH="${OUTPUT_DIR}/llava_annotations.json"
if [[ -f "$LLAVA_PATH" ]]; then
  $PYTHON data_augmentation/scripts/patch_llava_instructions.py \
    --llava "$LLAVA_PATH" \
    --originals "${VP_OUT}/verified.json" \
    --rewrites "$CUSTOM_INSTR"
else
  echo "WARN: $LLAVA_PATH not found — skip patch step."
fi

echo "Done."
echo "  frames + LLaVA : $OUTPUT_DIR"
echo "  custom_instr   : $CUSTOM_INSTR"
