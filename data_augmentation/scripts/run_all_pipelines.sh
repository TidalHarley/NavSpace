#!/usr/bin/env bash
# Run all four NavSpace Stage-2 data-augmentation pipelines in sequence.
#
# Requirements (set BEFORE invoking):
#   conda activate navspace39
#   source data_augmentation/env_shim.sh
#   export DASHSCOPE_API_KEY=sk-...
#
# Optional knobs (override via env):
#   SKIP_VERTICAL=1
#   SKIP_ENVSTATE=1
#   SKIP_SPATIAL=1
#   SKIP_PRECISE=1
#   STAIRS_CHECK=0          # 1 to skip Qwen-VL stair detection (fast debugging)
#   PRECISE_OUT_ROOT=/abs/path/to/snav_data/aug_mix/precise_movement
#
# All inner scripts are resumable; rerunning is safe.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DA_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${DA_ROOT}/.." && pwd)"

cd "$REPO_ROOT"

PYTHON="${PYTHON:-python}"

SKIP_VERTICAL="${SKIP_VERTICAL:-0}"
SKIP_ENVSTATE="${SKIP_ENVSTATE:-1}"   # env_state pipeline is still WIP
SKIP_SPATIAL="${SKIP_SPATIAL:-1}"     # spatial already rendered under aug_mix
SKIP_PRECISE="${SKIP_PRECISE:-1}"     # precise already rendered under aug_mix
STAIRS_CHECK="${STAIRS_CHECK:-0}"     # off by default — semantic API is trusted

if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "[run_all_pipelines] DASHSCOPE_API_KEY not set; will fall back to config.qwen.api_key."
fi

echo "=============================================="
echo "  NavSpace Stage-2 data augmentation pipelines"
echo "=============================================="
echo "  repo root        : $REPO_ROOT"
echo "  python           : $(which $PYTHON)"
echo "  STAIRS_CHECK     : $STAIRS_CHECK"
echo "  SKIP_VERTICAL    : $SKIP_VERTICAL"
echo "  SKIP_ENVSTATE    : $SKIP_ENVSTATE"
echo "  SKIP_SPATIAL     : $SKIP_SPATIAL"
echo "  SKIP_PRECISE     : $SKIP_PRECISE"
echo "----------------------------------------------"

# ── 1. Vertical Perception (R2R height filter + Habitat semantic API + Qwen rewrite) ──
if [[ "$SKIP_VERTICAL" != "1" ]]; then
  echo ">>> [1/4] vertical_perception"
  $PYTHON data_augmentation/vertical_perception/1_filter.py
  if [[ "$STAIRS_CHECK" == "1" ]]; then
    $PYTHON data_augmentation/vertical_perception/2_verify.py --enable-stairs-check
  else
    $PYTHON data_augmentation/vertical_perception/2_verify.py
  fi
  $PYTHON data_augmentation/vertical_perception/3_rewrite.py
fi

# ── 2. Environment State (R2R + first/last frame Qwen-VL + templates A-E) ──
if [[ "$SKIP_ENVSTATE" != "1" ]]; then
  echo ">>> [2/4] environment_state"
  $PYTHON data_augmentation/environment_state/1_filter.py
  $PYTHON data_augmentation/environment_state/2_analyze.py
  $PYTHON data_augmentation/environment_state/3_generate.py
fi

# ── 3. Spatial Relationship (regex filter only, no rewrite) ──
if [[ "$SKIP_SPATIAL" != "1" ]]; then
  echo ">>> [3/4] spatial_relationship"
  $PYTHON data_augmentation/spatial_relationship/1_filter.py
fi

# ── 4. Precise Movement (MP3D sample + rollout + rule + LLaVA render, one Habitat pass per scene) ──
if [[ "$SKIP_PRECISE" != "1" ]]; then
  echo ">>> [4/4] precise_movement"
  if [[ -n "${PRECISE_OUT_ROOT:-}" ]]; then
    $PYTHON data_augmentation/precise_movement/run.py --output-root "$PRECISE_OUT_ROOT"
  else
    $PYTHON data_augmentation/precise_movement/run.py
  fi
fi

echo ""
echo "=============================================="
echo "  All requested pipelines done."
echo "  Next: data_augmentation/merge.py"
echo "  then: scripts/run_render_aug.sh"
echo "=============================================="
