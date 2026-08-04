#!/usr/bin/env bash
# Prepare Stage-A navigation training data (frames + stepwise JSONs).
#
# Pipeline:
#   1) Habitat collect  → training_data_r2r_tt / training_data_rxr_ce_tt
#   2) Build stepwise   → r2r_stepwise_*.json / rxr_stepwise_*_en.json
#   3) Stop-only + instruction-reconstruction side JSONs
#
# Prerequisites (download yourself):
#   MP3D scenes, R2R-CE, RxR-CE under $NAVIGATION_ROOT (or set MP3D_ROOT /
#   R2RCE_ROOT / RXRCE_ROOT). Requires habitat-sim + habitat-lab in the
#   active Python env.
#
# Usage:
#   export NAVIGATION_ROOT=/path/to/scenes_and_vlnce
#   export TRAIN_DATA_ROOT=$PWD/train_data               # default; matches configs/*.yaml
#   bash snav_training/scripts/prepare_stage_a_data.sh            # all
#   bash snav_training/scripts/prepare_stage_a_data.sh collect    # render only
#   bash snav_training/scripts/prepare_stage_a_data.sh build      # JSON only
#   bash snav_training/scripts/prepare_stage_a_data.sh stopinstr  # side JSONs
#
# Optional env:
#   R2R_NUM_WORKERS=8  RXR_NUM_WORKERS=8  BUILD_NUM_WORKERS=16
#   R2R_TRAJ_NUM=      RXR_TRAJ_NUM=       # cap for smoke tests
#   SKIP_R2R=1  SKIP_RXR=1
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ST_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
REPO_ROOT="$(cd "${ST_ROOT}/.." && pwd)"
STAGE_A="${ST_ROOT}/stage_a"

# Scenes / VLN-CE may live outside the repo; Stage-A JSONs default under
# <repo>/train_data so relative paths in configs/*.yaml resolve correctly.
NAVIGATION_ROOT="${NAVIGATION_ROOT:-${REPO_ROOT}}"
TRAIN_DATA_ROOT="${TRAIN_DATA_ROOT:-${REPO_ROOT}/train_data}"
export NAVIGATION_ROOT TRAIN_DATA_ROOT
export MP3D_ROOT="${MP3D_ROOT:-${NAVIGATION_ROOT}/mp3d_scenes}"
export R2RCE_ROOT="${R2RCE_ROOT:-${NAVIGATION_ROOT}/R2R_VLNCE_v1-3}"
export RXRCE_ROOT="${RXRCE_ROOT:-${NAVIGATION_ROOT}/RxR_VLNCE_v0}"
# paths.py also accepts R2R_ROOT / RXR_ROOT
export R2R_ROOT="${R2R_ROOT:-${R2RCE_ROOT}}"
export RXR_ROOT="${RXR_ROOT:-${RXRCE_ROOT}}"

R2R_NUM_WORKERS="${R2R_NUM_WORKERS:-8}"
RXR_NUM_WORKERS="${RXR_NUM_WORKERS:-8}"
BUILD_NUM_WORKERS="${BUILD_NUM_WORKERS:-16}"
PHASE="${1:-all}"

mkdir -p "${TRAIN_DATA_ROOT}"
cd "${STAGE_A}"

echo "[prepare_stage_a] NAVIGATION_ROOT=${NAVIGATION_ROOT}"
echo "[prepare_stage_a] TRAIN_DATA_ROOT=${TRAIN_DATA_ROOT}"
echo "[prepare_stage_a] MP3D_ROOT=${MP3D_ROOT}"
echo "[prepare_stage_a] R2RCE_ROOT=${R2RCE_ROOT}"
echo "[prepare_stage_a] RXRCE_ROOT=${RXRCE_ROOT}"
echo "[prepare_stage_a] phase=${PHASE}"

wait_workers_json() {
  local workers_json="$1"
  python3 - "$workers_json" <<'PY'
import json, os, sys, time
from pathlib import Path

path = Path(sys.argv[1])
payload = json.loads(path.read_text(encoding="utf-8"))
pids = [int(w["pid"]) for w in payload.get("workers", []) if "pid" in w]
if not pids:
    raise SystemExit(f"No worker pids in {path}")

print(f"[prepare_stage_a] waiting for {len(pids)} workers from {path}", flush=True)
pending = set(pids)
while pending:
    alive = set()
    for pid in pending:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            continue
        except PermissionError:
            alive.add(pid)
        else:
            alive.add(pid)
    pending = alive
    if pending:
        time.sleep(5)
print(f"[prepare_stage_a] all workers finished: {path}", flush=True)
PY
}

run_collect_r2r() {
  if [[ "${SKIP_R2R:-0}" == "1" ]]; then
    echo "[prepare_stage_a] SKIP_R2R=1 — skip R2R collect"
    return 0
  fi
  local extra=()
  if [[ -n "${R2R_TRAJ_NUM:-}" ]]; then
    extra+=(--traj-num "${R2R_TRAJ_NUM}")
  fi
  echo "[prepare_stage_a] collect R2R-CE train (${R2R_NUM_WORKERS} workers)"
  python3 collect_r2r_train_parallel.py \
    --mode launch \
    --num-workers "${R2R_NUM_WORKERS}" \
    --output-dir "${TRAIN_DATA_ROOT}/training_data_r2r_tt" \
    --runs-dir "${TRAIN_DATA_ROOT}/r2r_collect_runs" \
    "${extra[@]}"
  local latest
  latest="$(ls -1dt "${TRAIN_DATA_ROOT}/r2r_collect_runs"/run_* 2>/dev/null | head -1 || true)"
  if [[ -z "${latest}" || ! -f "${latest}/workers.json" ]]; then
    echo "[prepare_stage_a] ERROR: cannot find R2R collect workers.json" >&2
    exit 1
  fi
  wait_workers_json "${latest}/workers.json"
}

run_collect_rxr() {
  if [[ "${SKIP_RXR:-0}" == "1" ]]; then
    echo "[prepare_stage_a] SKIP_RXR=1 — skip RxR collect"
    return 0
  fi
  local extra=()
  if [[ -n "${RXR_TRAJ_NUM:-}" ]]; then
    extra+=(--traj-num "${RXR_TRAJ_NUM}")
  fi
  echo "[prepare_stage_a] collect RxR-CE train (${RXR_NUM_WORKERS} workers)"
  local pids=()
  local i
  for ((i = 0; i < RXR_NUM_WORKERS; i++)); do
    python3 collect_rxr_train.py \
      --num-workers "${RXR_NUM_WORKERS}" \
      --worker-index "${i}" \
      --output-dir "${TRAIN_DATA_ROOT}/training_data_rxr_ce_tt" \
      "${extra[@]}" \
      >"${TRAIN_DATA_ROOT}/rxr_collect_worker_${i}.log" 2>&1 &
    pids+=("$!")
  done
  local pid
  for pid in "${pids[@]}"; do
    wait "${pid}"
  done
  echo "[prepare_stage_a] RxR collect done"
}

run_build_r2r_json() {
  echo "[prepare_stage_a] build R2R stepwise JSON (${BUILD_NUM_WORKERS} workers)"
  python3 build_r2r_stepwise_json_parallel.py \
    --mode launch \
    --num-workers "${BUILD_NUM_WORKERS}" \
    --input-dir "${TRAIN_DATA_ROOT}/training_data_r2r_tt" \
    --image-root "${TRAIN_DATA_ROOT}" \
    --output-json "${TRAIN_DATA_ROOT}/r2r_stepwise_train_jupyter_full.json" \
    --runs-dir "${TRAIN_DATA_ROOT}/r2r_stepwise_runs"
  local latest
  latest="$(ls -1dt "${TRAIN_DATA_ROOT}/r2r_stepwise_runs"/run_* 2>/dev/null | head -1 || true)"
  if [[ -z "${latest}" || ! -f "${latest}/workers.json" ]]; then
    echo "[prepare_stage_a] ERROR: cannot find R2R stepwise workers.json" >&2
    exit 1
  fi
  wait_workers_json "${latest}/workers.json"
  python3 build_r2r_stepwise_json_parallel.py \
    --mode merge \
    --run-dir "${latest}" \
    --output-json "${TRAIN_DATA_ROOT}/r2r_stepwise_train_jupyter_full.json"
}

run_build_rxr_json() {
  echo "[prepare_stage_a] build RxR-en stepwise JSON (${BUILD_NUM_WORKERS} workers)"
  python3 build_rxr_stepwise_json_parallel.py \
    --mode launch \
    --num-workers "${BUILD_NUM_WORKERS}" \
    --input-dir "${TRAIN_DATA_ROOT}/training_data_rxr_ce_tt" \
    --image-root "${TRAIN_DATA_ROOT}" \
    --output-json "${TRAIN_DATA_ROOT}/rxr_stepwise_train_jupyter_full_en.json" \
    --runs-dir "${TRAIN_DATA_ROOT}/rxr_stepwise_runs" \
    --allowed-language-prefixes en
  local latest
  latest="$(ls -1dt "${TRAIN_DATA_ROOT}/rxr_stepwise_runs"/run_* 2>/dev/null | head -1 || true)"
  if [[ -z "${latest}" || ! -f "${latest}/workers.json" ]]; then
    echo "[prepare_stage_a] ERROR: cannot find RxR stepwise workers.json" >&2
    exit 1
  fi
  wait_workers_json "${latest}/workers.json"
  python3 build_rxr_stepwise_json_parallel.py \
    --mode merge \
    --run-dir "${latest}" \
    --output-json "${TRAIN_DATA_ROOT}/rxr_stepwise_train_jupyter_full_en.json"
}

run_stop_and_instr() {
  echo "[prepare_stage_a] build stop-only + instruction-reconstruction JSONs"
  python3 build_stop_only_subset.py \
    --input "${TRAIN_DATA_ROOT}/r2r_stepwise_train_jupyter_full.json" \
    --output "${TRAIN_DATA_ROOT}/r2r_stepwise_train_jupyter_full_stoponly.json" \
    --summary "${TRAIN_DATA_ROOT}/r2r_stepwise_train_jupyter_full_stoponly.summary.json"
  python3 build_stop_only_subset.py \
    --input "${TRAIN_DATA_ROOT}/rxr_stepwise_train_jupyter_full_en.json" \
    --output "${TRAIN_DATA_ROOT}/rxr_stepwise_train_jupyter_full_en_stoponly.json" \
    --summary "${TRAIN_DATA_ROOT}/rxr_stepwise_train_jupyter_full_en_stoponly.summary.json"
  python3 build_instruction_reconstruction_dataset.py \
    --r2r-input "${TRAIN_DATA_ROOT}/r2r_stepwise_train_jupyter_full.json" \
    --rxr-input "${TRAIN_DATA_ROOT}/rxr_stepwise_train_jupyter_full_en.json" \
    --r2r-output "${TRAIN_DATA_ROOT}/r2r_instruction_from_fullobs_jupyter.json" \
    --rxr-output "${TRAIN_DATA_ROOT}/rxr_instruction_from_fullobs_en_jupyter.json" \
    --mix-output "${TRAIN_DATA_ROOT}/r2r_rxr_instruction_from_fullobs_enmix_jupyter.json" \
    --summary-output "${TRAIN_DATA_ROOT}/instruction_from_fullobs_summary.json"
}

case "${PHASE}" in
  all)
    run_collect_r2r
    run_collect_rxr
    run_build_r2r_json
    run_build_rxr_json
    run_stop_and_instr
    ;;
  collect)
    run_collect_r2r
    run_collect_rxr
    ;;
  collect_r2r)
    run_collect_r2r
    ;;
  collect_rxr)
    run_collect_rxr
    ;;
  build)
    run_build_r2r_json
    run_build_rxr_json
    run_stop_and_instr
    ;;
  build_json)
    run_build_r2r_json
    run_build_rxr_json
    ;;
  stopinstr)
    run_stop_and_instr
    ;;
  *)
    echo "Unknown phase: ${PHASE}" >&2
    echo "Use: all|collect|collect_r2r|collect_rxr|build|build_json|stopinstr" >&2
    exit 2
    ;;
esac

echo "[prepare_stage_a] done. JSONs under ${TRAIN_DATA_ROOT}"
ls -1 "${TRAIN_DATA_ROOT}"/r2r_stepwise_train_jupyter_full*.json \
      "${TRAIN_DATA_ROOT}"/rxr_stepwise_train_jupyter_full_en*.json \
      "${TRAIN_DATA_ROOT}"/r2r_rxr_instruction_from_fullobs_enmix_jupyter.json \
      2>/dev/null || true
echo "Next: export IMAGE_FOLDER=${TRAIN_DATA_ROOT}"
echo "      bash snav_training/scripts/launch_vln_mix_stage_a.sh"
