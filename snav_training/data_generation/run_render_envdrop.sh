#!/usr/bin/env bash
# Render a sampled EnvDrop subset with SNav-style output.
#
# Required env vars:
#   ENVDROP_SOURCE_JSON   absolute path to the upstream envdrop.json.gz
#   SCENES_ROOT           root of scene assets
#   DATA_ROOT             where rendered data should land
#
# Optional:
#   SAMPLE_COUNT           default 20000 (size of the sampled subset)
#   SAMPLE_SEED            default 42
#   ENVDROP_SAMPLED_JSON   output path for the sampled subset (auto-created
#                          under ${DATA_ROOT}/envdrop_sampled_<N>_seed<S>.json.gz
#                          if unset)
#   CUSTOM_INSTR_JSON      optional LLaVA-style annotations file whose
#                          `instructions` overrides the original ones; set to
#                          empty string to use the dataset's own instructions
#   MAX_EPISODES           0 = all
#   CONDA_ENV_NAME         default: streamvln

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

: "${ENVDROP_SOURCE_JSON:?need ENVDROP_SOURCE_JSON=/abs/path/to/envdrop.json.gz}"
: "${SCENES_ROOT:?need SCENES_ROOT=/abs/path/to/scene_datasets}"
DATA_ROOT="${DATA_ROOT:-$(pwd)/snav_data}"
MAX_EPISODES="${MAX_EPISODES:-0}"
SAMPLE_COUNT="${SAMPLE_COUNT:-20000}"
SAMPLE_SEED="${SAMPLE_SEED:-42}"
ENVDROP_SAMPLED_JSON="${ENVDROP_SAMPLED_JSON:-${DATA_ROOT}/envdrop_sampled_${SAMPLE_COUNT}_seed${SAMPLE_SEED}.json.gz}"
CUSTOM_INSTR_JSON="${CUSTOM_INSTR_JSON-}"  # default empty → no override

OUT_DIR="${DATA_ROOT}/envdrop"
LOG_PATH="${DATA_ROOT}/envdrop_render.log"

mkdir -p "$OUT_DIR" "$(dirname "$ENVDROP_SAMPLED_JSON")" "$(dirname "$LOG_PATH")"

if [[ ! -f "$ENVDROP_SAMPLED_JSON" ]]; then
  echo "[INFO] Building sampled subset: ${SAMPLE_COUNT} episodes (seed=${SAMPLE_SEED})"
  python - "$ENVDROP_SOURCE_JSON" "$ENVDROP_SAMPLED_JSON" "$SAMPLE_COUNT" "$SAMPLE_SEED" <<'PY'
import gzip, json, os, random, sys
source_json, sampled_json, sample_count, sample_seed = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]))
with gzip.open(source_json, "rt", encoding="utf-8") as f:
    data = json.load(f)
episodes = data.get("episodes", [])
if not episodes:
    raise SystemExit(f"No episodes in {source_json}")
if sample_count <= 0 or sample_count >= len(episodes):
    sampled = episodes
else:
    rng = random.Random(sample_seed)
    sampled = rng.sample(episodes, sample_count)
os.makedirs(os.path.dirname(sampled_json), exist_ok=True)
with gzip.open(sampled_json, "wt", encoding="utf-8") as f:
    json.dump({"episodes": sampled}, f)
print(f"[INFO] sampled subset written: {sampled_json} (episodes={len(sampled)})")
PY
else
  echo "[INFO] Reusing existing sampled subset: $ENVDROP_SAMPLED_JSON"
fi

EXTRA_ARGS=()
if [[ -n "$CUSTOM_INSTR_JSON" ]]; then
  EXTRA_ARGS+=(--custom_instructions_json "$CUSTOM_INSTR_JSON")
  echo "[INFO] Using custom instruction overrides from $CUSTOM_INSTR_JSON"
fi

echo "=== Rendering EnvDrop sampled subset (SNav frames, FOV=120, 384x384) ==="
python "$RENDER_PY" \
  --data_json "$ENVDROP_SAMPLED_JSON" \
  --data_format r2r \
  --dataset_tag envdrop \
  --scenes_root "$SCENES_ROOT" \
  --output_dir "$OUT_DIR" \
  --log_path "$LOG_PATH" \
  --output_mode "${OUTPUT_MODE:-frames}" \
  --video_subdir envdrop \
  --max_episodes "$MAX_EPISODES" \
  --max_steps 500 \
  --goal_radius 0.5 \
  --forward_step 0.25 \
  --turn_angle 15.0 \
  --width 384 --height 384 --hfov 120 \
  --camera_height 0.88 \
  "${EXTRA_ARGS[@]}" \
  "$@"

echo "Done. llava_annotations.json -> $OUT_DIR/llava_annotations.json"
