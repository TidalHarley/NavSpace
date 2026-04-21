# NavSpace Evaluation Guide

This document describes how to use the scripts under `evaluation/` to run the three
evaluation routes supported on the NavSpace benchmark:

1. **LLM based** — call online multimodal APIs such as OpenAI / ChatAnywhere / DashScope / Zhipu.
2. **SNav based** — load a LLaVA / SNav-style local video-navigation model.
3. **StreamVLN based** — load a local streaming VLN model from the StreamVLN family.

All three routes share the same Habitat environment, the same NavSpace dataset and the same
result file format, which makes cross-method comparison straightforward.

> Reference: the overall Habitat-Sim / Habitat-Lab / HM3D / MP3D deployment recipe closely
> follows the one used by VLN-CE ([https://github.com/jacobkrantz/VLN-CE](https://github.com/jacobkrantz/VLN-CE)). The most useful
> commands are also inlined in the deployment section below.

---

## 0. Script inventory

The tree below lists every evaluation-related file inside `NavSpace-main` with a short
description. Entries marked `[ENTRY]` are CLI entry points that can be invoked via
`python ... .py` directly.

```text
NavSpace-main/
├── evaluation/                            # evaluation package
│   ├── __init__.py                        # marks `evaluation/` as a Python package
│   ├── config.py                          # task -> dataset map; LLM profile defaults; LlmEvalConfig dataclass
│   ├── common.py                          # shared utilities: data/scene/image/metric/simulator/key/resume helpers
│   ├── prompts.py                         # navigation prompt template (with video frame timestamps)
│   ├── providers.py                       # LLM backends: OpenAI-compatible & Zhipu GLM
│   ├── simulation.py                      # LLM-route Habitat-Sim rollout loop
│   ├── run_llm_eval.py          [ENTRY]   # LLM API evaluation (main entry)
│   ├── eval_snav.py             [ENTRY]   # SNav / LLaVA-style local model evaluation
│   └── eval_streamvln.py        [ENTRY]   # StreamVLN-style local model evaluation
├── tools/                                 # auxiliary utilities
│   ├── smoke_test.py            [ENTRY]   # offline self-check (no Habitat / HM3D / live API needed)
│   ├── merge_results.py         [ENTRY]   # merge shard result JSONs, re-sort and summarize metrics
│   ├── llm_client.py            [ENTRY]   # image -> description data-augmentation utility
│   ├── llm_client2.py           [ENTRY]   # zh-en translation utility
│   └── text_prompt.py                     # prompt templates used by the data-aug scripts
├── gpt_eval.py                  [ENTRY]   # backward-compatible wrapper that forwards to run_llm_eval
├── el.sh                        [ENTRY]   # one-shot 8-shard parallel LLM evaluation (bash)
├── docs/
│   ├── evaluation.md                      # Chinese version of this guide
│   └── evaluation_en.md                   # this document (English)
├── NavSpace-Datasets/                     # the six NavSpace subtasks
│   ├── Environment State/envstate_vln.json
│   ├── Space Structure/spacestructure_vln.json
│   ├── Precise Movement/precisemove_vln.json
│   ├── Viewpoint Shifting/viewpointsft_vln.json
│   ├── Vertical Perception/verticalpercep_vln.json
│   ├── Spatial Relationship/spatialrel_vln.json
│   └── validate_dataset_integrity.py      # dataset sanity checker
├── requirements-base.txt                  # base deps (numpy/opencv/Pillow/filelock)
├── requirements-llm.txt                   # LLM route (openai + zai)
└── requirements-local-model.txt           # local-model route (torch/transformers/decord/...)
```

### 0.1 Call graph at a glance

```text
gpt_eval.py   ─┐
el.sh         ─┼──► evaluation/run_llm_eval.py ──► evaluation/simulation.py
               │                                     │
               │                                     ├─► evaluation/common.py     (simulator + metrics + image utils)
               │                                     └─► evaluation/providers.py  ──► evaluation/prompts.py
               │
               ├──► evaluation/eval_snav.py       ──► evaluation/common.py (+ external LLaVA code)
               └──► evaluation/eval_streamvln.py  ──► evaluation/common.py (+ external StreamVLN code)

tools/merge_results.py  ──► evaluation/common.py
tools/smoke_test.py     ──► evaluation/* (offline self-check, does not depend on habitat_sim)
```

### 0.2 Per-file description


| File                                                        | Role                                                                                                                                                                                                                                        | When it is used                                          |
| ----------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `evaluation/__init__.py`                                    | marks `evaluation/` as a Python package                                                                                                                                                                                                     | any `from evaluation.* import ...`                       |
| `evaluation/config.py`                                      | task → `*_vln.json` map, per-profile defaults (model, base_url, resolution, retry, ...) and the `LlmEvalConfig` dataclass                                                                                                                   | before starting an LLM evaluation                        |
| `evaluation/common.py`                                      | shared utilities: locked JSON I/O, gzip/JSON dataset loader, HM3D scene-path resolver, quaternion fix-up, Habitat-Sim setup, RGB/Depth extraction, video frame sampling, action parsing, result append / resume, SR/NE/OS/SPL summarization | all three routes                                         |
| `evaluation/prompts.py`                                     | builds the canonical navigation prompt (with timestamps, history frames, action enumeration)                                                                                                                                                | called right before `providers.py` sends a request       |
| `evaluation/providers.py`                                   | wraps the OpenAI-compatible and Zhipu GLM backends behind a unified retry / back-off / rate-limit layer                                                                                                                                     | every LLM inference step                                 |
| `evaluation/simulation.py`                                  | LLM-route rollout loop: load episode → build scene → feed LLM → parse action → step → collect metrics                                                                                                                                       | the only module `run_llm_eval.py` drives                 |
| `evaluation/run_llm_eval.py`                                | LLM evaluation CLI: parse args, merge profile, build `LlmEvalConfig`, configure logging, run the loop; supports `--dry-run`                                                                                                                 | **main entry** (LLM route)                               |
| `evaluation/eval_snav.py`                                   | adapter for SNav / LLaVA-style local models: load weights → reuse `common.py` for simulation/metrics → feed frames to the model                                                                                                             | **main entry** (SNav route)                              |
| `evaluation/eval_streamvln.py`                              | adapter for StreamVLN-style local models: additionally handles depth, pose, intrinsics and token caching, calling `StreamVLNForCausalLM.generate`                                                                                           | **main entry** (StreamVLN route)                         |
| `gpt_eval.py`                                               | backward-compatible wrapper; internally `from evaluation.run_llm_eval import main`                                                                                                                                                          | legacy invocations `python gpt_eval.py`                  |
| `el.sh`                                                     | bash one-shot script that launches 8 shards in parallel                                                                                                                                                                                     | fast parallel LLM eval                                   |
| `tools/smoke_test.py`                                       | 16-item offline self-check (byte-compile, imports, dataset schema, resolution consistency, metric math, prompt/action parsing, ...)                                                                                                         | before open-sourcing or after migrating to a new machine |
| `tools/merge_results.py`                                    | merges shard result files, re-orders episodes, prints global SR/NE/OS/SPL                                                                                                                                                                   | after sharded runs finish                                |
| `tools/llm_client.py` / `llm_client2.py` / `text_prompt.py` | offline data-augmentation / translation / prompt-template utilities                                                                                                                                                                         | content-generation flows outside of evaluation           |


> **Invocation convention**: the three entry scripts each install a `sys.path` fallback at
> the top of the file, so both `python evaluation/run_llm_eval.py ...` and
> `python -m evaluation.run_llm_eval ...` work. The former is recommended for clarity.

### 0.3 Unified resolution

All three routes now use a unified **224×224** image resolution — both for the Habitat
renderer and for the `encode_resize` sent to the online API. If you want to experiment with
336 or 384 locally, simply override the defaults on the command line, e.g.
`--frame-width 336 --frame-height 336 --encode-resize 336`.

### 0.4 Relation to VLN-CE / official R2R-CE instruction format

The NavSpace evaluation JSON files (`*_vln.json` under each subtask) follow the **VLN-CE-style
continuous-navigation episode convention**: each episode bundles a natural-language
`instruction`, a `scene_id`, start pose, `goals`, and optional reference material. This is
**aligned with** how episodes are structured for **R2R-CE** in the official
[VLN-CE](https://github.com/jacobkrantz/VLN-CE) project—NavSpace extends the *task set* with
six spatial-intelligence splits, but the **field layout stays in the same family** as
standard R2R-CE / Habitat rollout code expects.

Therefore, **any model pipeline that already evaluates on the official R2R-CE instruction and
episode format** can, in principle, be wired to NavSpace **the same way**, after plugging in
this repository’s evaluation adapters (simulation loop, action parsing, optional multi-frame /
video prompts). If something you need for evaluation is not documented here (Habitat versions,
raw R2R-CE field details, environment recipes, …), consult the VLN-CE repository:
[https://github.com/jacobkrantz/VLN-CE](https://github.com/jacobkrantz/VLN-CE).

---

## 1. Deployment (shared across all three routes)

### 1.1 System prerequisites

- Python ≥ 3.9 (3.9–3.10 are the smoothest; Habitat-Sim wheel support on 3.11+ is platform-dependent)
- Linux x86_64 or macOS (on Windows, run Habitat inside WSL2)
- A CUDA-capable GPU (only needed for the local-model routes)

### 1.2 Create a conda environment

```bash
conda create -n navspace python=3.10 -y
conda activate navspace
```

### 1.3 Install Habitat-Sim

The official prebuilt conda-forge binaries are recommended (this is the same channel used by VLN-CE):

```bash
conda install -c aihabitat -c conda-forge habitat-sim=0.2.5 withbullet headless -y
```

Notes:

- `withbullet` enables collision detection; `headless` is for servers / no-display setups.
- If you want windowed rendering, replace `headless` with `with-cuda` (matching your CUDA version).
- Pin the version to match Habitat-Lab; the NavSpace dataset is debugged on `habitat-sim 0.2.x`.

Quick self-check:

```bash
python -c "import habitat_sim; print(habitat_sim.__version__)"
```

### 1.4 Install Habitat-Lab (optional)

Install `habitat-lab` if you plan to reuse its episode structures, pathfinder APIs, etc.:

```bash
git clone --branch v0.2.5 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -r requirements.txt
pip install -e habitat-lab
cd ..
```

> The NavSpace evaluation scripts only depend on `habitat_sim`, **not** on `habitat-lab`, but
> installing lab makes it easier to extend the training / evaluation flow later.

### 1.5 Install NavSpace dependencies

```bash
cd NavSpace-main
pip install -r requirements-base.txt
# LLM route
pip install -r requirements-llm.txt
# Local-model route (SNav / StreamVLN, either one)
pip install -r requirements-local-model.txt
```

### 1.6 Download scene assets

NavSpace is built on HM3D, and the data-augmentation pipeline is built on MP3D. The two
datasets can be downloaded independently.

#### 1.6.1 HM3D v0.2

1. Apply for HM3D access at [https://matterport.com/habitat-matterport-3d-research-dataset](https://matterport.com/habitat-matterport-3d-research-dataset) or on the `ai-habitat` official page. A short form is required; turnaround is normally quick.
2. Use `matterport_download.py` / the official CLI to download v0.2.
3. After extraction the directory structure should be:

```text
/path/to/hm3d_v0.2/
├── train/
│   ├── 00000-kfPV7w3FaU5/
│   │   ├── kfPV7w3FaU5.basis.glb
│   │   ├── kfPV7w3FaU5.basis.navmesh
│   │   └── ...
│   └── ...
├── val/
└── test/
```

1. Point every evaluation command at this directory with `--hm3d-base-path /path/to/hm3d_v0.2`.

> Internally, `resolve_scene_path` (in `evaluation/common.py`) searches for `*.basis.glb` by
> scene name under `train/`, `val/` and `test/` in that order, so even if the dataset
> `scene_id` still holds an absolute path from the original machine, it will be relocated
> automatically on your host.

#### 1.6.2 Matterport3D (MP3D, optional)

Only required if you run the MP3D variant or extend your own annotations:

```bash
# The official downloader requires Python 2.7.
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

After extraction, keep the VLN-CE layout `data/scene_datasets/mp3d/{scene}/{scene}.glb`.

> The three scripts under `evaluation/` currently only exercise HM3D. To switch to MP3D,
> point `--hm3d-base-path` at the MP3D root and extend `resolve_scene_path` accordingly.

### 1.7 Validate dataset integrity

The six tasks' `*_vln.json` files ship with the repository. From the repo root:

```bash
python NavSpace-Datasets/validate_dataset_integrity.py
```

This quickly checks episode counts, required fields and scene-id format legality.

### 1.8 The six NavSpace subtasks


| `--task` value         | Dataset file                                                    | Task                             |
| ---------------------- | --------------------------------------------------------------- | -------------------------------- |
| `environment_state`    | `NavSpace-Datasets/Environment State/envstate_vln.json`         | Environment-state discrimination |
| `space_structure`      | `NavSpace-Datasets/Space Structure/spacestructure_vln.json`     | Spatial-structure understanding  |
| `precise_movement`     | `NavSpace-Datasets/Precise Movement/precisemove_vln.json`       | Precise movement                 |
| `viewpoint_shifting`   | `NavSpace-Datasets/Viewpoint Shifting/viewpointsft_vln.json`    | Viewpoint shifting               |
| `vertical_perception`  | `NavSpace-Datasets/Vertical Perception/verticalpercep_vln.json` | Vertical-space perception        |
| `spatial_relationship` | `NavSpace-Datasets/Spatial Relationship/spatialrel_vln.json`    | Spatial-relation reasoning       |


You can also pass `--trajectory-path /abs/path/to/custom.json` to bypass the preset map.

### 1.9 API-key configuration

Resolution order (highest priority first):

1. `--api-key <YOUR_KEY>` (most direct)
2. `--api-key-env <ENV_NAME>` + environment variable (e.g. `export OPENAI_API_KEY=sk-xxx`)
3. `--api-key-file /path/to/key.json` (content: `{"api_key": "..."}`)

Each built-in profile has a default env-variable name:

- `gemini-pro` / `gemini-flash` → `OPENAI_API_KEY` (ChatAnywhere proxy endpoint)
- `qwen72b` → `DASHSCOPE_API_KEY`
- `glm4.5v` / `glm-4.1v-thinking-flash` → `ZHIPU_API_KEY`

---

## 2. LLM-based evaluation

**Entry script**: `evaluation/run_llm_eval.py`.
**Core loop**: `evaluation/simulation.py` → `evaluation/providers.py` → online API.

### 2.1 Built-in profiles


| Profile                   | Provider          | Model                     | Default endpoint                                    |
| ------------------------- | ----------------- | ------------------------- | --------------------------------------------------- |
| `gemini-pro`              | openai_compatible | `gemini-2.5-pro`          | `https://api.chatanywhere.tech/v1`                  |
| `gemini-flash`            | openai_compatible | `gemini-2.5-flash`        | same as above                                       |
| `qwen72b`                 | openai_compatible | `qwen2.5-vl-72b-instruct` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `glm4.5v`                 | zhipu             | `glm-4.5v`                | Zhipu official SDK                                  |
| `glm-4.1v-thinking-flash` | zhipu             | `glm-4.1v-thinking-flash` | Zhipu official SDK                                  |


Override any of these with `--provider`, `--model`, `--base-url`, `--frame-width`,
`--encode-resize`, etc.

Where to get API keys:

- Zhipu: [https://bigmodel.cn/](https://bigmodel.cn/)
- Qwen (DashScope / Alibaba Cloud): [https://cn.aliyun.com/](https://cn.aliyun.com/)
- ChatAnywhere (used as an OpenAI-compatible proxy for GPT / Gemini): [https://chatanywhere.apifox.cn/](https://chatanywhere.apifox.cn/)
- If you hold an official OpenAI key, just point `--base-url` / `--model` at it directly.

### 2.2 Quick start

Gemini (via ChatAnywhere):

```bash
export OPENAI_API_KEY=sk-xxxxx
python evaluation/run_llm_eval.py \
  --profile gemini-pro \
  --task environment_state \
  --hm3d-base-path /path/to/hm3d_v0.2
```

DashScope / Qwen:

```bash
export DASHSCOPE_API_KEY=sk-xxxxx
python evaluation/run_llm_eval.py \
  --profile qwen72b \
  --task space_structure \
  --hm3d-base-path /path/to/hm3d_v0.2
```

Zhipu GLM:

```bash
export ZHIPU_API_KEY=xxxxx.xxxxx
python evaluation/run_llm_eval.py \
  --profile glm4.5v \
  --task vertical_perception \
  --hm3d-base-path /path/to/hm3d_v0.2
```

Custom third-party OpenAI-compatible endpoint:

```bash
python evaluation/run_llm_eval.py \
  --profile gemini-pro \
  --provider openai_compatible \
  --model your-model-name \
  --base-url https://your-endpoint.example/v1 \
  --api-key sk-xxxx \
  --task spatial_relationship \
  --hm3d-base-path /path/to/hm3d_v0.2
```

### 2.3 Single-node 8-way sharding

The shard rule is the familiar `traj_idx % num_shards == model_id`.

A single shard:

```bash
python evaluation/run_llm_eval.py \
  --profile gemini-pro \
  --task environment_state \
  --hm3d-base-path /path/to/hm3d_v0.2 \
  --model-id 0 \
  --num-shards 8
```

Eight parallel shards via bash:

```bash
bash el.sh environment_state /path/to/hm3d_v0.2 gemini-pro
```

`el.sh` launches 8 background processes with different `--model-id` values and `pkill`s them on exit.

### 2.4 Resume

Re-running with the same `--profile --task` automatically skips episodes already present in `outputs/llm/<xxx>.json`.

Resume from a different result file:

```bash
python evaluation/run_llm_eval.py \
  --profile gemini-pro \
  --task environment_state \
  --hm3d-base-path /path/to/hm3d_v0.2 \
  --resume-from outputs/llm/older_results.json
```

### 2.5 Output & metrics

Default output directory: `outputs/llm/`. Default files:

- `llm_eval_<profile>_<task>.json` — results
- `llm_eval_<profile>_<task>.log` — run log

Result format:

```json
[
  {
    "Starting from the bedroom, ... stop in the bedroom.": {
      "success": 1,
      "nav_error": 0.82,
      "os": 1,
      "shortest_path_length": 6.99,
      "actual_path_length": 7.75
    }
  }
]
```

Metric semantics:

- `success`: did the agent stop inside `goal_radius`?
- `nav_error`: Euclidean distance to the goal at termination (meters).
- `os` (Oracle Success): did the agent ever enter `goal_radius` at any step during the episode?
- `shortest_path_length`: geodesic distance provided by the dataset.
- `actual_path_length`: accumulated length of executed `move_forward` actions (0.25 m per step).
- `SPL` = `success × shortest / max(actual, shortest)`, aggregated inside `summarize_results`.

### 2.6 Merge results

```bash
python tools/merge_results.py \
  --input outputs/llm/shard0.json outputs/llm/shard1.json ... \
  --trajectory-path "NavSpace-Datasets/Environment State/envstate_vln.json" \
  --output outputs/llm/merged.json
```

The merge script re-orders episodes by dataset order and prints global SR/NE/OS/SPL.

---

## 3. SNav-based evaluation (LLaVA/SNav local models)

**Entry script**: `evaluation/eval_snav.py`.
**Core loop**: the script's own rollout, reusing `common.py` for simulation and metrics.

### 3.1 Extra dependencies (not shipped here)

- The LLaVA / SNav codebase (providing `llava.model.builder.load_pretrained_model` and friends). If you already keep StreamVLN locally, its bundled `StreamVLN/llava/` subtree is the same LLaVA code and can be reused directly.
- Your SNav checkpoint directory (`--model-path`).
- A compatible `torch` + `transformers` (already in `requirements-local-model.txt`).
- Point HuggingFace cache to a writable directory:

```bash
export HF_HOME=/your/writable/hf_cache
```

Put the LLaVA code on `PYTHONPATH`, e.g.:

```bash
# Option A: a standalone LLaVA checkout
export PYTHONPATH=/path/to/LLaVA:$PYTHONPATH

# Option B (recommended, stays consistent with SNav training): reuse the one shipped with StreamVLN
export PYTHONPATH=/path/to/StreamVLN:$PYTHONPATH
```

### 3.2 Example

```bash
python evaluation/eval_snav.py \
  --task vertical_perception \
  --hm3d-base-path /path/to/hm3d_v0.2 \
  --model-path /path/to/your_snav_checkpoint \
  --model-name llava_qwen \
  --conv-template qwen_1_5 \
  --max-frames-num 16 \
  --max-steps 70 \
  --attn-implementation sdpa
```

Common flags:

- `--device` / `--device-map`: default `cuda` / `auto`.
- `--frame-width` / `--frame-height`: default 224 (unified across all routes).
- `--model-id` / `--num-shards`: same sharding convention as the LLM route.
- `--resume-from`: resume from another result file.
- `--attn-implementation`: `sdpa` (default, no flash-attn needed) / `flash_attention_2` (requires `flash-attn`) / `eager`.

### 3.3 Output

Default output directory: `outputs/snav/`. Default files:

- `snav_<task>.json`
- `snav_<task>.log`

The result schema is identical to the LLM route, so `tools/merge_results.py` can merge them directly.

---

## 4. StreamVLN-based evaluation

**Entry script**: `evaluation/eval_streamvln.py`.
**Core loop**: the script's own rollout plus `common.py` utilities (with `include_depth=True`).

### 4.1 Extra dependencies (not shipped here)

The following modules must be importable via `PYTHONPATH`:

- `model.stream_video_vln.StreamVLNForCausalLM` (lives at `StreamVLN/streamvln/model/` in the official repo)
- `utils.utils` (provides `DEFAULT_IMAGE_TOKEN` / `DEFAULT_MEMORY_TOKEN` / `DEFAULT_VIDEO_TOKEN` / `IMAGE_TOKEN_INDEX` / `MEMORY_TOKEN_INDEX` **and** `dict_to_cuda`)
- `llava.`* (transitive dep of StreamVLN, lives at `StreamVLN/llava/`)
- `depth_camera_filtering.filter_depth` (pip package)
- A StreamVLN checkpoint directory (`--model-path`)

> **PYTHONPATH layout.** `StreamVLN/streamvln` exposes `model/` and `utils/`; `StreamVLN/` exposes `llava/`. Add **both**:
>
> ```bash
> export PYTHONPATH=/path/to/StreamVLN:/path/to/StreamVLN/streamvln:$PYTHONPATH
> ```

### 4.2 Example

```bash
conda activate streamvln
export PYTHONPATH=/path/to/StreamVLN:/path/to/StreamVLN/streamvln:$PYTHONPATH
export HF_HOME=/your/writable/hf_cache

python evaluation/eval_streamvln.py \
  --task vertical_perception \
  --hm3d-base-path /path/to/hm3d_v0.2 \
  --model-path /path/to/your_streamvln_model \
  --max-steps 150 \
  --attn-implementation sdpa
```

### 4.3 Implementation notes

- Every step collects RGB + Depth, runs `filter_depth` for temporal filtering, then converts to a `uint16` depth map for the model.
- The agent pose is turned into a 4×4 homogeneous matrix and multiplied by `get_axis_align_matrix()` to align coordinate systems before being fed to the model.
- `model.reset_for_env(traj_idx)` is invoked on `stop` or every 32 steps to reset the KV cache, mirroring the original memory-window behaviour.
- `actual_path_length` accumulates `0.25` per `move_forward`, producing a correct SPL.

### 4.4 Output

Default output directory: `outputs/streamvln/`. Default files:

- `streamvln_<task>.json`
- `streamvln_<task>.log`

---

## 5. Troubleshooting


| Symptom                                                                             | Cause & fix                                                                                                                                                                                                                                                                                                                                                  |
| ----------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `ModuleNotFoundError: habitat_sim`                                                  | Re-install habitat-sim as in §1.3; double-check the active conda env. In our setup the working env is `streamvln` (`habitat-sim 0.3.3` + `habitat-lab 0.3.3` + `torch` + `transformers`).                                                                                                                                                                    |
| `Unable to resolve HM3D scene path for scene_id=...`                                | The dataset carries a legacy machine path and your HM3D root is missing that scene; inspect `--hm3d-base-path` and confirm HM3D is extracted under `train/val/test`. For custom datasets you can simply set `scene_id` to an absolute path pointing at the `.glb`; `resolve_scene_path` returns the file as-is when it exists, **so MP3D scenes also work**. |
| `API key not found`                                                                 | Provide the key through one of the three mechanisms in §1.9; verify the variable really landed in the current shell.                                                                                                                                                                                                                                         |
| `Zhipu rate limited, sleeping ...`                                                  | Normal back-off log; keep it running, or increase `--initial-retry-delay` / `--max-retries` if needed.                                                                                                                                                                                                                                                       |
| `FlashAttention2 has been toggled on, but ... flash_attn seems to be not installed` | `--attn-implementation` is now a CLI flag defaulting to `sdpa`; only pass `flash_attention_2` if `flash-attn` is actually installed.                                                                                                                                                                                                                         |
| `AttributeError: 'dict' object has no attribute 'to_dict'` (SNav checkpoint load)   | Legacy LlaVA-Qwen checkpoints store `text_config` / `vision_config` as raw dicts. `eval_snav.py` already passes `overwrite_config={"text_config": None, "vision_config": None}`. Removing that line will reproduce the crash.                                                                                                                                |
| `ImportError: cannot import name 'dict_to_cuda' from 'utils.dist'`                  | StreamVLN's official repo keeps `dict_to_cuda` in `utils.utils`; `eval_streamvln.py` imports it from there. Reverting the import path reproduces the error.                                                                                                                                                                                                  |
| `There was a problem when trying to write in your cache folder`                     | Set `HF_HOME` to a writable directory (the legacy `TRANSFORMERS_CACHE` is deprecated).                                                                                                                                                                                                                                                                       |
| `Please install petrel_client to Client.`                                           | Harmless log from StreamVLN's optional OSS client.                                                                                                                                                                                                                                                                                                           |
| Want to plug in a custom dataset                                                    | Pass `--trajectory-path /abs/path/custom.json`; the file must be either `{"episodes": [...]}` or a bare list, and each episode must carry at least `instruction.instruction_text`, `scene_id`, `start_position`, `start_rotation`, `goals[0].position` and `info.geodesic_distance`.                                                                         |


---

## 6. Pre-release verification

Even before you install habitat-sim or download HM3D, you can cover a large part of the
repository through the three offline helpers below.

### 6.1 One-shot offline self-check: `tools/smoke_test.py`

```bash
python tools/smoke_test.py
```

This script does **not** depend on `habitat_sim`, HM3D or any live API key — only the
packages in `requirements-base.txt`. It runs 16 ordered checks:

1. `byte-compile all Python sources` — every `.py` file compiles without syntax errors.
2. `import evaluation.* modules` — `evaluation.common / config / prompts / providers` import cleanly.
3. `build CLI parsers for all three entry scripts` — `argparse` for the three entry points is healthy.
4. `gpt_eval.py wrapper re-exports main` — the legacy wrapper still forwards to the new impl.
5. `TASK_DATASET_MAP covers every task` — all six task names are present; missing files produce a `WARN`, not a failure.
6. `episode schema is consistent across tasks` — the first few episodes of every task carry `scene_id / start_position / start_rotation / goals / info / instruction`.
7. `scene_id values are parseable for resolver` — `resolve_scene_path` can extract candidate names from each `scene_id`.
8. `LLM profiles share the expected resolution` — every profile in `PROFILE_DEFAULTS` carries the expected `frame_width / frame_height / encode_resize` (default 224).
9. `local-model CLIs use the expected resolution` — `eval_snav / eval_streamvln` default resolutions agree.
10. `action extraction parses known verbs` — `extract_actions` correctly parses `Move forward / Turn left / ...`.
11. `image pipeline end-to-end` — `ensure_size_bgr → encode_image_b64 → process_images_as_video` shapes and base64 outputs are correct.
12. `metrics math matches expectations` — `summarize_results` produces the expected SR/OS/SPL on synthetic inputs.
13. `navigation prompt contains required tokens` — `build_navigation_prompt` returns a string containing both the instruction and the action keywords.
14. `resume index round-trip is stable` — resume index keys are consistent.
15. `provider dispatch rejects unknown providers cleanly` — `infer_actions` raises a clear error for unknown providers (without any network call).
16. `requirements-*.txt are non-empty` — the three requirements files exist and are non-empty.

Optional flags:

```bash
python tools/smoke_test.py --max-episodes 64 --expected-resolution 224
```

### 6.2 LLM entry `--dry-run` (optionally probe the real API)

`evaluation/run_llm_eval.py` ships a dry-run mode that validates configuration without launching Habitat:

```bash
python evaluation/run_llm_eval.py \
  --profile gemini-pro \
  --task environment_state \
  --hm3d-base-path /path/does/not/exist \
  --dry-run
```

It:

- prints the merged profile, provider, model, base_url and output_dir,
- confirms the API key is resolvable (prints only the last 4 characters),
- loads the target task's `*_vln.json` and prints the episode count,
- resolves the HM3D path for the first 3 episodes' `scene_id` (missing paths log `NOT FOUND` without crashing),
- does not import `habitat_sim` and does not run any rollout.

To additionally probe API connectivity (fires a tiny text-only request, no images / no image quota):

```bash
python evaluation/run_llm_eval.py \
  --profile gemini-pro \
  --task environment_state \
  --hm3d-base-path /path/does/not/exist \
  --dry-run --dry-run-probe-api
```

### 6.3 Dataset integrity

```bash
python NavSpace-Datasets/validate_dataset_integrity.py
```

The original per-episode structural checker — complementary to `smoke_test.py`.

### 6.4 Tiny end-to-end smoke test on a real HM3D host

Once habitat-sim is installed on the target server, run a minimal end-to-end check with a
tiny `--max-steps`:

```bash
python evaluation/run_llm_eval.py \
  --profile gemini-pro \
  --task environment_state \
  --hm3d-base-path /path/to/hm3d_v0.2 \
  --max-steps 5 \
  --num-shards 64 --model-id 0   # 64-way shard, only run shard 0 => barely any episodes
```

This consumes a tiny amount of API quota but confirms end-to-end: scene renders → LLM
request succeeds → action parses → result is written to disk.

Combining §6.1 + §6.2 + §6.4 you effectively never need to re-download HM3D locally.

### 6.5 MP3D-only smoke test for SNav / StreamVLN

If you have MP3D but no HM3D, you can still validate both local-model routes on a single
episode.

1. Sample a navigable start / goal pair from an MP3D scene with `habitat_sim`'s `PathFinder`
  and dump a 1-episode JSON:

```python
# /tmp/make_mp3d_mini.py
import json, habitat_sim

SCENE = "/path/to/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
sim_cfg = habitat_sim.SimulatorConfiguration()
sim_cfg.scene_id = SCENE
agent = habitat_sim.agent.AgentConfiguration()
rgb = habitat_sim.CameraSensorSpec()
rgb.uuid = "color_sensor"; rgb.resolution = [224, 224]; rgb.position = [0, 1.5, 0]
agent.sensor_specifications = [rgb]
sim = habitat_sim.Simulator(habitat_sim.Configuration(sim_cfg, [agent]))

path = habitat_sim.ShortestPath()
for _ in range(300):
    path.requested_start = sim.pathfinder.get_random_navigable_point()
    path.requested_end   = sim.pathfinder.get_random_navigable_point()
    if sim.pathfinder.find_path(path) and 2.5 < path.geodesic_distance < 4.0:
        break

episode = {
    "episode_id": 0, "trajectory_id": 0, "scene_id": SCENE,
    "start_position": list(map(float, path.requested_start)),
    "start_rotation": [1.0, 0.0, 0.0, 0.0],
    "info": {"geodesic_distance": float(path.geodesic_distance)},
    "goals": [{"position": list(map(float, path.requested_end)), "radius": 3.0}],
    "instruction": {"instruction_text": "Walk forward and explore the living room."},
    "reference_path": [list(path.requested_start), list(path.requested_end)],
}
json.dump([episode], open("/tmp/mp3d_mini_vln.json", "w"), indent=2)
```

1. Point `--trajectory-path` at the JSON; `--hm3d-base-path` can be any existing directory
  because an absolute `scene_id` is returned as-is:

```bash
# StreamVLN
python evaluation/eval_streamvln.py \
  --task environment_state \
  --trajectory-path /tmp/mp3d_mini_vln.json \
  --hm3d-base-path /tmp \
  --model-path /path/to/streamvln_ckpt \
  --max-steps 3 --num-shards 1 --model-id 0 \
  --output-dir outputs/_smoke_streamvln \
  --attn-implementation sdpa

# SNav (LLaVA-Qwen-style checkpoint)
python evaluation/eval_snav.py \
  --task environment_state \
  --trajectory-path /tmp/mp3d_mini_vln.json \
  --hm3d-base-path /tmp \
  --model-path /path/to/snav_ckpt \
  --model-name llava_qwen --conv-template qwen_1_5 \
  --max-frames-num 8 --max-steps 3 \
  --num-shards 1 --model-id 0 \
  --output-dir outputs/_smoke_snav \
  --attn-implementation sdpa
```

Expected: both finish within 2–5 minutes, print `Eval Results: Success:... Nav Error:... OS:... SPL:...`,
and produce the corresponding `*_environment_state.json` result file.

---

## 7. Refs and Links

- Habitat official docs: [https://aihabitat.org/](https://aihabitat.org/)
- VLN-CE (Habitat + R2R/RxR continuous navigation): [https://github.com/jacobkrantz/VLN-CE](https://github.com/jacobkrantz/VLN-CE)
- HM3D v0.2 download & usage: [https://aihabitat.org/datasets/hm3d/](https://aihabitat.org/datasets/hm3d/)
- Matterport3D dataset: [https://niessner.github.io/Matterport/](https://niessner.github.io/Matterport/)
- Qwen / Alibaba Tongyi Qianwen: [https://cn.aliyun.com/](https://cn.aliyun.com/)
- Zhipu (GLM): [https://bigmodel.cn/](https://bigmodel.cn/)
- ChatAnywhere: [https://chatanywhere.apifox.cn/](https://chatanywhere.apifox.cn/)
- StreamVLN official repo: [https://github.com/InternRobotics/StreamVLN](https://github.com/InternRobotics/StreamVLN)

All NavSpace evaluation entry points depend only on Habitat-Sim and the NavSpace dataset,
so once §1 is complete the three routes share the exact same environment and can be run
against the exact same data.