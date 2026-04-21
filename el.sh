#!/usr/bin/env bash

set -euo pipefail

if [ $# -lt 2 ]; then
  echo "Usage: bash el.sh <task> <hm3d_base_path> [profile]"
  echo "Example: bash el.sh environment_state /path/to/hm3d_v0.2 gemini-pro"
  exit 1
fi

TASK_NAME="$1"
HM3D_BASE_PATH="$2"
PROFILE="${3:-gemini-pro}"

cleanup() {
  echo "Terminating all evaluation processes..."
  pkill -f "evaluation/run_llm_eval.py --profile ${PROFILE}" || true
}

trap cleanup EXIT SIGINT

for i in {0..7}; do
  python evaluation/run_llm_eval.py \
    --profile "${PROFILE}" \
    --task "${TASK_NAME}" \
    --hm3d-base-path "${HM3D_BASE_PATH}" \
    --model-id "${i}" \
    --num-shards 8 &
done

wait
