#!/usr/bin/env bash
# ==============================================================================
# End-to-end SNav Stage-1 vanilla pipeline: render → train → (optional) eval.
#
# Each stage can be skipped:
#   SKIP_RENDER=1 bash run_pipeline.sh
#   SKIP_TRAIN=1  bash run_pipeline.sh
#   SKIP_EVAL=1   bash run_pipeline.sh
#
# Scope: vanilla SFT only. Data augmentation is NOT handled here.
# See ../README.md for the full list of environment variables.
# ==============================================================================
set -euo pipefail
trap 'echo "[FATAL] Error at line $LINENO (exit $?). Command: $BASH_COMMAND" >&2' ERR

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
NAVSPACE_ROOT="$(cd "${REPO_ROOT}/.." && pwd)"

SKIP_RENDER="${SKIP_RENDER:-0}"
SKIP_TRAIN="${SKIP_TRAIN:-0}"
SKIP_EVAL="${SKIP_EVAL:-0}"

if [[ "$SKIP_RENDER" != "1" ]]; then
  echo ""
  echo ">>> [1/3] Rendering SNav training data ..."
  bash "${REPO_ROOT}/data_generation/run_render_r2rce.sh"
  bash "${REPO_ROOT}/data_generation/run_render_rxrce.sh"
  bash "${REPO_ROOT}/data_generation/run_render_envdrop.sh"
else
  echo "[INFO] SKIP_RENDER=1, skipping data rendering."
fi

if [[ "$SKIP_TRAIN" != "1" ]]; then
  echo ""
  echo ">>> [2/3] Launching SFT training ..."
  bash "${SCRIPT_DIR}/run_snav_train.sh"
else
  echo "[INFO] SKIP_TRAIN=1, skipping training."
fi

if [[ "$SKIP_EVAL" != "1" ]]; then
  echo ""
  echo ">>> [3/3] Running NavSpace evaluation ..."
  : "${OUTPUT_DIR:?need OUTPUT_DIR (the trained checkpoint root)}"
  : "${HM3D_BASE_PATH:?need HM3D_BASE_PATH=/abs/path/to/hm3d_v0.2}"
  TASK="${EVAL_TASK:-environment_state}"
  python "${NAVSPACE_ROOT}/evaluation/eval_snav.py" \
    --task "$TASK" \
    --hm3d-base-path "$HM3D_BASE_PATH" \
    --model-path "$OUTPUT_DIR" \
    --model-name llava_qwen \
    --conv-template qwen_1_5 \
    --max-frames-num 16 \
    --max-steps 70 \
    --attn-implementation sdpa
else
  echo "[INFO] SKIP_EVAL=1, skipping evaluation."
fi

echo ""
echo "[DONE] SNav pipeline finished."
