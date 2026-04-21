# NavSpace 数据标注流水线 / Annotation Pipeline

> 本文档为中英双语。 / This document is bilingual.
>
> - [中文版](#中文版)
> - [English version](#english-version)

---

## 中文版

`annotation_pipeline/` 目录提供了一个基于 **Flask + Flask-SocketIO + Habitat-Sim** 的轻量级数据标注前后端框架，直接在浏览器里驾驶 Habitat 里的 agent，同时把每一步动作、位姿、图像、指令保存成 NavSpace 官方格式的 `*_vln.json`。

> ⚠️ **关于 LLM 辅助标注按钮**：本仓库中的 `annotation_pipeline/llm_client.py` **只是一个最小可运行的示例**，默认走 OpenAI-compatible 协议（ChatAnywhere + `gpt-4o-mini`），用来演示"把当前帧送给视觉大模型 → 拿到中文描述 → 回传给标注员"的完整链路。**具体使用时请根据自身情况选择**：可以换成 Qwen / GLM / Gemini / 自建本地 VLM，或者直接在 `_call_llm_api` 里接入你团队已有的描述模型；也可以不启用这个按钮（不设 `NAVSPACE_ANNO_API_KEY` 时它会自动禁用）。**标注流程本身不依赖 LLM**，LLM 只是给标注员的一个提示源。

### 1. 目录结构

```text
NavSpace-main/
├── run_annotation_server.sh             # 从任意路径启动（内部 cd 到仓库根目录）
└── annotation_pipeline/
    ├── __init__.py
    ├── websocket_annotation_server.py   # Flask + SocketIO 主程序，含 HTML 前端
    └── llm_client.py                    # 可选：图像 → 文本描述的 LLM 辅助按钮
```

### 2. 主要功能

- **第一人称导航**：用键盘在 Habitat-Sim 的 MP3D / HM3D 场景里控制 agent 前后左右、上下抬头。
- **多场景切换**：一键随机切到 HM3D 中任意其它场景。
- **两段式标注流程**：
  1. *熟悉期* — 必须先在当前场景里移动至少 `MIN_FAMILIARIZATION_STEPS`（默认 200）步，标注按钮才会解锁。
  2. *标注期* — 点 **Start Recording** 开始记录轨迹，走完点 **Stop Recording**，再写下指令文本并 **Submit**。
- **LLM 辅助描述（可选）**：点 *Trigger LLM Analysis* 把当前帧发到 OpenAI-compatible 视觉模型，返回一段中文场景描述，帮助标注员写指令。
- **自动续写**：所有轨迹以 NavSpace 兼容的 `*_vln.json` 格式写到 `trajectories.json`，程序重启后会继续递增 `episode_id`。
- **实时日志**：前端同时显示后端的 DEBUG/INFO/ERROR 消息，便于调试。

### 3. 部署步骤

#### 3.1 Python 环境

推荐沿用评测用的同一个 conda 环境（见 `[docs/evaluation.md](evaluation.md)`），因为都需要 Habitat-Sim。

```bash
conda create -n navspace python=3.10 -y
conda activate navspace
conda install -c aihabitat -c conda-forge habitat-sim=0.2.5 withbullet headless -y
```

> 如果宿主机是带显示器的工作站，可以把 `headless` 改成 `with-cuda`；服务器/无显示器场景保留 `headless` 即可。

#### 3.2 安装前后端依赖

```bash
cd NavSpace-main
pip install -r requirements-base.txt        # numpy / opencv / pillow ...
pip install -r requirements-annotation.txt  # Flask / Flask-SocketIO / openai (LLM 辅助按钮)
```

> Windows 下推荐先在 WSL2 里跑，避免 Habitat-Sim 的平台编译问题。

#### 3.3 准备 HM3D  / MP3D 数据

和评测一致，如果准备的是HM3D数据，参考 `docs/evaluation.md` [§1.6.1](evaluation.md#161-hm3d-v02)。下载完成后目录形如：

```text
/path/to/hm3d_v0.2/
├── train/ ...
├── val/   ...
└── test/  ...
```

注：目前的程序指向的是MP3D数据（MP3D train set, 90 scenes total），如果需要改为HM3D数据(HM3D train set, 900 scenes total)，需要自己修改路径，但是修改的接口已经在程序里给出。

#### 3.4 配置环境变量


| 变量名                                        | 作用                                                                                                                                                                         | 默认值                                                         |
| ------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| `NAVSPACE_SCENE_PATHS`                     | **（可选，调试用）** 逗号分隔的 **单个场景文件** 绝对路径（`.glb`），不依赖自动扫描。目录里暂时找不到场景时，可先指定一个已下载的 mesh 验证前后端                                                                                       | 未设置                                                         |
| `NAVSPACE_MP3D_ROOT`                       | **（可选）** Matterport3D 场景根目录，与 [VLN-CE](https://github.com/jacobkrantz/VLN-CE) 相同布局：`mp3d/<scene_id>/<scene_id>.glb`。若设置且目录下存在 `.glb`，**优先于 HM3D** 加载场景；也支持对子目录做递归查找 `.glb` | 未设置                                                         |
| `NAVSPACE_HM3D_BASE_PATH`                  | HM3D 根目录（未设置 `NAVSPACE_MP3D_ROOT` 或 MP3D 下无可用 `.glb` 时使用）                                                                                                                  | `/LingXi/habitat-data-0.2.5/scenes/hm3d_v0.2`（仅作历史后备，请务必覆盖） |
| `NAVSPACE_MIN_FAMILIARIZATION_STEPS`       | 正式标注前必须完成的步数                                                                                                                                                               | `200`                                                       |
| `NAVSPACE_TRAJECTORY_LOG`                  | 轨迹 JSON 输出路径                                                                                                                                                               | `trajectories.json`                                         |
| `NAVSPACE_HOST`                            | Web 服务监听地址                                                                                                                                                                 | `0.0.0.0`                                                   |
| `NAVSPACE_PORT`                            | Web 服务监听端口                                                                                                                                                                 | `8000`                                                      |
| `NAVSPACE_ANNO_API_KEY` / `OPENAI_API_KEY` | LLM 辅助按钮用的 API Key                                                                                                                                                         | 未设置时会禁用该按钮                                                  |
| `NAVSPACE_ANNO_BASE_URL`                   | LLM 端点                                                                                                                                                                     | `https://api.chatanywhere.tech/v1`                          |
| `NAVSPACE_ANNO_MODEL`                      | 视觉模型名                                                                                                                                                                      | `gpt-4o-mini`                                               |
| `NAVSPACE_ANNO_PROMPT`                     | LLM 提示词                                                                                                                                                                    | 中文场景描述，见 `llm_client.py`                                    |


示例（**仅 HM3D**）：

```bash
export NAVSPACE_HM3D_BASE_PATH=/data/hm3d_v0.2
export NAVSPACE_MIN_FAMILIARIZATION_STEPS=200
export NAVSPACE_PORT=8000
# 可选：启用 LLM 辅助
export NAVSPACE_ANNO_API_KEY=sk-xxxxxx
```

示例（**仅 MP3D**）：

```bash
export NAVSPACE_MP3D_ROOT=/path/to/scene_datasets/mp3d
export NAVSPACE_MIN_FAMILIARIZATION_STEPS=200
export NAVSPACE_PORT=8000
python annotation_pipeline/websocket_annotation_server.py
```

若 `NAVSPACE_MP3D_ROOT` 已设置但该目录下 **没有** 找到任何 `.glb`，服务会自动 **回退** 到扫描 `NAVSPACE_HM3D_BASE_PATH`（请仍保证其一可用）。

#### 3.5 启动服务

**工作目录**：必须在 **本仓库根目录** `NavSpace-main` 下启动，否则会出现  
`can't open file '.../annotation_pipeline/websocket_annotation_server.py'`（你在其它目录执行了相对路径）。

推荐任选其一：

```bash
# 方式 A：先 cd 再运行（最常见）
cd /path/to/NavSpace-main
python annotation_pipeline/websocket_annotation_server.py

# 方式 B：从任意目录用绝对路径调用 Python（不依赖当前目录）
python /path/to/NavSpace-main/annotation_pipeline/websocket_annotation_server.py

# 方式 C：使用仓库自带的启动脚本（内部会 cd 到 NavSpace-main）
bash /path/to/NavSpace-main/run_annotation_server.sh
```

控制台会输出：

```
🚀 启动 Flask-SocketIO 服务器...
🌐 请在浏览器中访问: http://localhost:8000
```

用 Chromium 内核浏览器打开 `http://<your-host>:8000` 即可。

### 4. 前端操作说明


| 控件                           | 含义                                         |
| ---------------------------- | ------------------------------------------ |
| 键盘 `W/S/A/D`                 | 前进 / 后退 / 左转 30° / 右转 30°                  |
| 键盘 `Q/E` 或 `↑/↓`             | 抬头 / 低头 30°                                |
| 按钮 **Switch Scene**          | 随机切换到另一个 HM3D 场景；未提交的记录会自动另存为 "incomplete" |
| 按钮 **Start Recording**       | 开始记录一条新 episode（只有熟悉 ≥ 200 步才解锁）           |
| 按钮 **Stop Recording**        | 停止记录，将当前位置作为 goal                          |
| 输入框 + **Submit Instruction** | 写下指令文本并提交，正式落盘一条 episode                   |
| 按钮 **Trigger LLM Analysis**  | 把当前帧发给大模型，返回一段中文描述（可选）                     |
| 日志面板                         | 实时显示后端 `logger` 消息                         |


**熟悉期强约束**：直到你在同一场景内累计移动满 `MIN_FAMILIARIZATION_STEPS` 步，`Start Recording` / `Submit Instruction` 都会被后端拒绝并返回友好的提示。这是为了避免标注员"一上来就开录"导致空指令或者指令与场景严重脱节。

### 5. 输出格式

所有 episode 都写入 `trajectories.json`，结构与评测代码消费的 `*_vln.json` 完全一致，这个格式也是官方VLN-CE的格式，即任何在VLN-CE上能够评测的模型都应该能够在NavSpace上进行评测：

```json
{
  "episodes": [
    {
      "episode_id": 1,
      "trajectory_id": 1,
      "scene_id": "/abs/path/to/xxxx.basis.glb",
      "start_position": [x, y, z],
      "start_rotation": [w, x, y, z],
      "info": {"geodesic_distance": 6.21},
      "goals": [{"position": [x, y, z], "radius": 0.2}],
      "instruction": {"instruction_text": "Walk forward, turn right at the hallway ..."},
      "reference_path": [[x, y, z], ...],
      "action_sequence": ["forward", "forward", "right", "forward", "stop"]
    }
  ]
}
```

- `reference_path`：每一步执行后 agent 的 `position`。
- `action_sequence`：导航动作序列，最后一步固定为 `stop`。
- `info.geodesic_distance`：后端调用 `habitat_sim.pathfinder.find_path` 算出的测地线距离；失败时退化到欧氏距离。

> **建议**：把 `trajectories.json` 放到 `NavSpace-Datasets/` 下某个子目录里，然后直接用 `evaluation/run_llm_eval.py --trajectory-path ...` 评测。

### 6. 常见问题


| 现象                                                                        | 原因 / 处理                                                                                                                                                                                                           |
| ------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `can't open file '...annotation_pipeline/websocket_annotation_server.py'` | 当前目录不是 `NavSpace-main`。请先 `cd` 到仓库根目录，或使用 **绝对路径** 调用 `python`，或运行 `bash run_annotation_server.sh`                                                                                                                |
| 后端报错 `No scene paths available to initialize simulator`                   | **没有任何 .glb 被扫描到**。请确认已下载 HM3D 或 MP3D，并设置 `NAVSPACE_HM3D_BASE_PATH` 或 `NAVSPACE_MP3D_ROOT` 指向正确根目录；或先用 `**export NAVSPACE_SCENE_PATHS=/绝对路径/某个场景.glb`** 指定单个文件验证。MP3D 常见路径形如 `.../mp3d/<scene_id>/<scene_id>.glb` |
| 启动时报 *未找到任何有效的 HM3D 场景文件*                                                 | `NAVSPACE_HM3D_BASE_PATH` 指错了，或 HM3D 没有解压在 `train/val/test` 三个子目录                                                                                                                                                 |
| 浏览器能打开页面但画面不刷新                                                            | 检查是否被反向代理挡住了 WebSocket；前端 `io(..., { transports: ['websocket', 'polling'] })` 可自动回落                                                                                                                               |
| 点 Start Recording 一直被拒绝                                                   | 你还没走够 `MIN_FAMILIARIZATION_STEPS` 步；或者上一条 episode 还在 `pending_submission` 状态，先把它提交掉                                                                                                                               |
| *错误: 未检测到 NAVSPACE_ANNO_API_KEY*                                          | 没配 LLM key；其它功能不受影响，只是 LLM 辅助按钮不可用                                                                                                                                                                                |
| Habitat-Sim 报 OSMesa 相关错误                                                 | 无显示器服务器要装 OSMesa 或 EGL，或者换成 `gpu_device_id=0` 的有显卡渲染                                                                                                                                                              |


---

## English version

`annotation_pipeline/` is a lightweight annotation front-end + back-end built on **Flask + Flask-SocketIO + Habitat-Sim**. You drive a Habitat agent from your browser and every action / pose / frame / instruction is written to disk in the NavSpace `*_vln.json` format.

> ⚠️ **About the LLM-assist button**: `annotation_pipeline/llm_client.py` is **only a minimal working example** — it targets an OpenAI-compatible endpoint (ChatAnywhere + `gpt-4o-mini` by default) just to demonstrate the full "current frame → vision LLM → Chinese caption → annotator" loop. **Choose the backend that fits your own deployment**: you may swap in Qwen / GLM / Gemini / a locally served VLM, or wire `_call_llm_api` to whatever captioning model your team already owns. You can also disable the button entirely — leaving `NAVSPACE_ANNO_API_KEY` unset automatically hides it. **The annotation pipeline itself does not depend on any LLM**; the LLM only supplies an optional hint for the annotator.

### 1. Layout

```text
NavSpace-main/
├── run_annotation_server.sh             # launcher: cds to repo root then starts Python
└── annotation_pipeline/
    ├── __init__.py
    ├── websocket_annotation_server.py   # Flask + SocketIO server with embedded HTML UI
    └── llm_client.py                    # optional image-to-text helper for the "LLM" button
```

### 2. Key features

- **First-person teleoperation** in HM3D scenes via keyboard.
- **Scene switcher** to randomly jump to another HM3D scene.
- **Two-phase annotation**:
  1. *Familiarization* — the annotator must move at least `MIN_FAMILIARIZATION_STEPS` (default: 200) steps in the current scene before the annotation buttons unlock.
  2. *Recording* — click **Start Recording**, drive the agent to the goal, click **Stop Recording**, type the instruction, click **Submit**.
- **Optional LLM helper**: *Trigger LLM Analysis* sends the current frame to an OpenAI-compatible vision model and returns a short textual description.
- **Resumable logging**: all episodes are appended to `trajectories.json` in a NavSpace-compatible format; `episode_id` keeps incrementing across restarts.
- **Live log panel** mirroring the server-side logger.

### 3. Deployment

#### 3.1 Python env

Reuse the same conda environment you created for evaluation (see `[docs/evaluation_en.md](evaluation_en.md)`):

```bash
conda create -n navspace python=3.10 -y
conda activate navspace
conda install -c aihabitat -c conda-forge habitat-sim=0.2.5 withbullet headless -y
```

> Use `with-cuda` instead of `headless` if you run on a workstation with a display.

#### 3.2 Install web deps

```bash
cd NavSpace-main
pip install -r requirements-base.txt
pip install -r requirements-annotation.txt   # Flask / Flask-SocketIO / openai (LLM helper)
```

> On Windows, use WSL2 to avoid Habitat-Sim build issues.

#### 3.3 Prepare MP3D / HM3D

Identical to the evaluation guide — see `[docs/evaluation_en.md` §1.6.1](evaluation_en.md#161-hm3d-v02).

#### 3.4 Configure environment variables


| Variable                                   | Purpose                                                                                                                                                                                             | Default                                         |
| ------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------- |
| `NAVSPACE_SCENE_PATHS`                     | **(optional)** Comma-separated **absolute** paths to individual `.glb` mesh files (skips auto-scan). Use when datasets are incomplete or to smoke-test the UI                                       | unset                                           |
| `NAVSPACE_MP3D_ROOT`                       | **(optional)** Matterport3D root with the same layout as VLN-CE: `mp3d/<scene_id>/<scene_id>.glb`. If set and non-empty, **takes precedence over HM3D**; also scans subdirs recursively for `*.glb` | unset                                           |
| `NAVSPACE_HM3D_BASE_PATH`                  | HM3D root (used when `NAVSPACE_MP3D_ROOT` is unset or contains no `.glb`)                                                                                                                           | legacy path (please override)                   |
| `NAVSPACE_MIN_FAMILIARIZATION_STEPS`       | Steps required before annotation unlocks                                                                                                                                                            | `200`                                           |
| `NAVSPACE_TRAJECTORY_LOG`                  | Output JSON path                                                                                                                                                                                    | `trajectories.json`                             |
| `NAVSPACE_HOST`                            | Web host                                                                                                                                                                                            | `0.0.0.0`                                       |
| `NAVSPACE_PORT`                            | Web port                                                                                                                                                                                            | `8000`                                          |
| `NAVSPACE_ANNO_API_KEY` / `OPENAI_API_KEY` | API key for the LLM helper                                                                                                                                                                          | disabled when unset                             |
| `NAVSPACE_ANNO_BASE_URL`                   | LLM endpoint                                                                                                                                                                                        | `https://api.chatanywhere.tech/v1`              |
| `NAVSPACE_ANNO_MODEL`                      | Vision model name                                                                                                                                                                                   | `gpt-4o-mini`                                   |
| `NAVSPACE_ANNO_PROMPT`                     | LLM prompt template                                                                                                                                                                                 | short Chinese description (see `llm_client.py`) |


Example (HM3D only):

```bash
export NAVSPACE_HM3D_BASE_PATH=/data/hm3d_v0.2
export NAVSPACE_MIN_FAMILIARIZATION_STEPS=200
export NAVSPACE_PORT=8000
# Optional: enable the LLM helper button.
export NAVSPACE_ANNO_API_KEY=sk-xxxxxx
```

Example (MP3D only, same scene layout as VLN-CE):

```bash
export NAVSPACE_MP3D_ROOT=/path/to/scene_datasets/mp3d
export NAVSPACE_MIN_FAMILIARIZATION_STEPS=200
export NAVSPACE_PORT=8000
python annotation_pipeline/websocket_annotation_server.py
```

If `NAVSPACE_MP3D_ROOT` is set but no `.glb` files are found, the server **falls back** to scanning `NAVSPACE_HM3D_BASE_PATH`—ensure at least one path is valid.

#### 3.5 Launch

**Working directory**: you must start the server from the `**NavSpace-main` repository root**. Otherwise Python raises  
`can't open file '.../annotation_pipeline/websocket_annotation_server.py'` (you used a relative path from the wrong folder).

Pick one:

```bash
# A: cd first (most common)
cd /path/to/NavSpace-main
python annotation_pipeline/websocket_annotation_server.py

# B: absolute path to the script (works from any cwd)
python /path/to/NavSpace-main/annotation_pipeline/websocket_annotation_server.py

# C: bundled launcher (cd's into NavSpace-main for you)
bash /path/to/NavSpace-main/run_annotation_server.sh
```

You should see:

```
🚀 Launching Flask-SocketIO server...
🌐 Open http://localhost:8000 in your browser
```

Open `http://<your-host>:8000` in a modern browser.

### 4. UI cheatsheet


| Control                           | Meaning                                                                                   |
| --------------------------------- | ----------------------------------------------------------------------------------------- |
| `W/S/A/D`                         | forward / back / turn left 30° / turn right 30°                                           |
| `Q/E` or `↑/↓`                    | look up / look down 30°                                                                   |
| **Switch Scene**                  | jump to a random different HM3D scene; unfinished episodes are auto-saved as "incomplete" |
| **Start Recording**               | begin a new episode (locked until `MIN_FAMILIARIZATION_STEPS` steps)                      |
| **Stop Recording**                | freeze the trajectory; current pose becomes the goal                                      |
| Text box + **Submit Instruction** | write the instruction and commit the episode to disk                                      |
| **Trigger LLM Analysis**          | send the current frame to a vision LLM (optional)                                         |
| Log panel                         | live tail of the server-side logger                                                       |


**Familiarization gate**: `Start Recording` and `Submit Instruction` are rejected by the backend with a friendly message until the annotator has moved `MIN_FAMILIARIZATION_STEPS` steps in the current scene. This prevents blindly-annotated trajectories and encourages the annotator to first build a mental map of the scene.

### 5. Output format

Every submitted episode is appended to `trajectories.json` under the shape consumed by the evaluation scripts:

```json
{
  "episodes": [
    {
      "episode_id": 1,
      "trajectory_id": 1,
      "scene_id": "/abs/path/to/xxxx.basis.glb",
      "start_position": [x, y, z],
      "start_rotation": [w, x, y, z],
      "info": {"geodesic_distance": 6.21},
      "goals": [{"position": [x, y, z], "radius": 0.2}],
      "instruction": {"instruction_text": "Walk forward, turn right at the hallway ..."},
      "reference_path": [[x, y, z], ...],
      "action_sequence": ["forward", "forward", "right", "forward", "stop"]
    }
  ]
}
```

- `reference_path` stores the agent `position` after each action.
- `action_sequence` is the recorded action list, terminated by `stop`.
- `info.geodesic_distance` is computed via `habitat_sim.pathfinder.find_path`; falls back to Euclidean distance if the pathfinder is not available.

> **Tip**: drop the generated `trajectories.json` into a `NavSpace-Datasets/` subfolder and feed it to `evaluation/run_llm_eval.py --trajectory-path ...` to evaluate your freshly annotated split.

### 6. Troubleshooting


| Symptom                                                                   | Cause / fix                                                                                                                                                                                   |
| ------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `can't open file '...annotation_pipeline/websocket_annotation_server.py'` | Current directory is not `NavSpace-main`. `cd` into the repo root, call `python` with an **absolute** path, or run `bash run_annotation_server.sh`                                            |
| `No scene paths available to initialize simulator`                        | No `.glb` files were discovered. Set `NAVSPACE_HM3D_BASE_PATH` or `NAVSPACE_MP3D_ROOT` to a valid dataset root, or set `**NAVSPACE_SCENE_PATHS=/abs/path/to/one_scene.glb`** for a quick test |
| *No valid HM3D scene files found at startup*                              | `NAVSPACE_HM3D_BASE_PATH` points to the wrong directory or HM3D is not split into `train/val/test`                                                                                            |
| Page opens but frame never updates                                        | A reverse proxy is blocking WebSocket; the client falls back to long-polling via `io(..., { transports: ['websocket', 'polling'] })`                                                          |
| *Start Recording* keeps being rejected                                    | Either familiarization is not done, or the previous episode is still `pending_submission` — submit it first                                                                                   |
| *Error: NAVSPACE_ANNO_API_KEY not set*                                    | Only the LLM helper button is disabled; everything else still works                                                                                                                           |
| OSMesa/EGL errors from Habitat-Sim                                        | Headless rendering needs OSMesa/EGL, or switch to GPU rendering (`gpu_device_id=0`)                                                                                                           |


