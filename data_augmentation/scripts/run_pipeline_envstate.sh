#!/usr/bin/env bash
# Full environment_state pipeline:
#   1. filter R2R train by geodesic    → candidates.json
#   2. build passthrough instructions  → passthrough_instructions.json
#   3. render full trajectories        → snav_data/aug_mix/environment_state/
#   4. Qwen-VL analyze rendered frames → states.json
#   5. wrap into A/B/D/E conditionals  → custom_instructions.json
#   6. convert to SNav SFT format      → annotations.json + rgb/
#
# Each step is independent; rerun the orchestrator to skip already-done
# stages (renderer + Qwen analyzer are both resumable).
#
# Required:
#     conda activate navspace39
#     source data_augmentation/env_shim.sh         (only for the render step)
#     export DASHSCOPE_API_KEY=...                 (or set qwen.api_key in config.json)
#
# Optional knobs:
#     MAX_EPISODES=500       sample size from R2R train
#     RESTART=0              1 → drop existing checkpoints (states.jsonl etc.)
#     GPU_DEVICE_ID=0
#     OUTPUT_DIR=snav_data/aug_mix/environment_state
#     DATASET_TAG=env_state
#     VIDEO_SUBDIR=env_state

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DA_ROOT}/.." && pwd)"
cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"
MAX_EPISODES="${MAX_EPISODES:-500}"
RESTART="${RESTART:-0}"
GPU_DEVICE_ID="${GPU_DEVICE_ID:-0}"
OUTPUT_DIR="${OUTPUT_DIR:-${REPO_ROOT}/snav_data/aug_mix/environment_state}"
DATASET_TAG="${DATASET_TAG:-env_state}"
VIDEO_SUBDIR="${VIDEO_SUBDIR:-env_state}"
R2RCE_TRAIN_JSON="${R2RCE_TRAIN_JSON:?need R2RCE_TRAIN_JSON=/abs/path/to/train.json.gz}"
SCENES_ROOT="${SCENES_ROOT:?need SCENES_ROOT=/abs/path/to/mp3d_or_hm3d}"

ES_OUT="${DA_ROOT}/outputs/environment_state"
CANDIDATES="${ES_OUT}/candidates.json"
PASSTHROUGH="${ES_OUT}/passthrough_instructions.json"
CUSTOM_INSTR="${ES_OUT}/custom_instructions.json"

RESTART_FLAG=""
if [[ "$RESTART" == "1" ]]; then
  RESTART_FLAG="--restart"
fi

echo "=============================================="
echo "  Environment State full pipeline"
echo "=============================================="
echo "  MAX_EPISODES   : $MAX_EPISODES"
echo "  RESTART        : $RESTART"
echo "  OUTPUT_DIR     : $OUTPUT_DIR"
echo "  DATASET_TAG    : $DATASET_TAG / $VIDEO_SUBDIR"
echo "  GPU_DEVICE_ID  : $GPU_DEVICE_ID"
echo "----------------------------------------------"

echo ">>> [1/6] 1_filter.py (R2R geodesic filter + sample)"
$PYTHON data_augmentation/environment_state/1_filter.py \
  --max-episodes "$MAX_EPISODES"

echo ">>> [2/6] build_passthrough_instructions.py"
$PYTHON data_augmentation/scripts/build_passthrough_instructions.py \
  --verified "$CANDIDATES" \
  --out      "$PASSTHROUGH"

RENDER_PY="${REPO_ROOT}/snav_training/data_generation/render_streamvln.py"
if [[ ! -f "$RENDER_PY" ]]; then
  RENDER_PY="${REPO_ROOT}/NavSpace/snav_training/data_generation/render_streamvln.py"
fi
mkdir -p "$OUTPUT_DIR"

echo ">>> [3/6] render_streamvln.py (frames + original-text LLaVA stub)"
$PYTHON "$RENDER_PY" \
  --data_json "$R2RCE_TRAIN_JSON" \
  --data_format r2r \
  --dataset_tag "$DATASET_TAG" \
  --scenes_root "$SCENES_ROOT" \
  --output_dir "$OUTPUT_DIR" \
  --output_mode snav_frames \
  --video_subdir "$VIDEO_SUBDIR" \
  --custom_instructions_json "$PASSTHROUGH" \
  --max_episodes 0 \
  --max_steps 500 \
  --goal_radius 0.5 \
  --forward_step 0.25 \
  --turn_angle 30.0 \
  --width 384 --height 384 --hfov 120 \
  --camera_height 1.5 \
  --gpu_device_id "$GPU_DEVICE_ID"

echo ">>> [4/6] 2_analyze.py (Qwen-VL on rendered frames)"
$PYTHON data_augmentation/environment_state/2_analyze.py \
  --frames-root "$OUTPUT_DIR" $RESTART_FLAG

echo ">>> [5/6] 3_generate.py (wrap into A/B/D/E conditionals)"
$PYTHON data_augmentation/environment_state/3_generate.py

echo ">>> [6/6] convert_aug_to_sft.py (patch instructions + frames layout)"
$PYTHON data_augmentation/scripts/convert_aug_to_sft.py \
  --folder "$OUTPUT_DIR" \
  --instructions-source "$CUSTOM_INSTR"

echo "----------------------------------------------"
echo "Done. SFT-ready data at: $OUTPUT_DIR"
echo "Inspect:"
echo "  $ES_OUT/review.json   (per-line original vs rewritten)"
echo "  $ES_OUT/output.json   (full records incl. template + analysis)"
