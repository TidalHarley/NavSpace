# NavSpace 评测指南

本文档说明如何使用 `evaluation/` 目录下的脚本，在 NavSpace 基准上完成三类评测：

1. **LLM based**：调用 OpenAI / ChatAnywhere / DashScope / Zhipu 等在线多模态 API。
2. **SNav based**：在本地加载 LLaVA / SNav 风格的视频导航模型。
3. **StreamVLN based**：在本地加载 StreamVLN 系列的流式视觉语言导航模型。

三条路线共用同一套 Habitat 环境、同一份 NavSpace 数据集、同一种结果文件格式，便于对比。

> 参考：Habitat-Sim、Habitat-Lab 与 HM3D/MP3D 的通用部署思路可参考 VLN-CE 项目（<https://github.com/jacobkrantz/VLN-CE>），本文也在部署章节中汇总了同类命令。

---

## 0. 脚本清单

下面以目录树的方式列出 `NavSpace-main` 中与评测相关的每个文件和它的用途。带 `[ENTRY]` 标记的文件是可以直接 `python ... .py` 调用的 CLI 入口。

```text
NavSpace-main/
├── evaluation/                            # 评测主模块
│   ├── __init__.py                        # 把 evaluation/ 注册为 Python 包
│   ├── config.py                          # 任务 → 数据集映射；LLM Profile 默认值；LlmEvalConfig 数据类
│   ├── common.py                          # 公共工具：数据/场景/图像/指标/仿真器/API Key/断点续跑
│   ├── prompts.py                         # 导航 Prompt 模板（含视频帧时间戳拼接）
│   ├── providers.py                       # LLM 调用后端：OpenAI-compatible 与 Zhipu GLM
│   ├── simulation.py                      # LLM 路线的 Habitat-Sim rollout 主循环
│   ├── run_llm_eval.py          [ENTRY]   # LLM API 评测（主入口）
│   ├── eval_snav.py             [ENTRY]   # SNav / LLaVA 风格本地模型评测
│   └── eval_streamvln.py        [ENTRY]   # StreamVLN 风格本地模型评测
├── tools/                                 # 辅助工具
│   ├── smoke_test.py            [ENTRY]   # 离线自检（不需要 Habitat / HM3D / 真实 API）
│   ├── merge_results.py         [ENTRY]   # 合并多个 shard 的结果 JSON，重新排序并汇总指标
│   ├── llm_client.py            [ENTRY]   # 图像 → 描述的数据增广小工具
│   ├── llm_client2.py           [ENTRY]   # 中英翻译小工具
│   └── text_prompt.py                     # 数据增广用的 Prompt 模板集合
├── gpt_eval.py                  [ENTRY]   # 旧入口的向后兼容 wrapper，内部转发到 run_llm_eval
├── el.sh                        [ENTRY]   # 一键并行 8 分片 LLM 评测（bash）
├── docs/
│   ├── evaluation.md                      # 本文（中文）
│   └── evaluation_en.md                   # 英文版
├── NavSpace-Datasets/                     # 六类任务的 *_vln.json 数据
│   ├── Environment State/envstate_vln.json
│   ├── Space Structure/spacestructure_vln.json
│   ├── Precise Movement/precisemove_vln.json
│   ├── Viewpoint Shifting/viewpointsft_vln.json
│   ├── Vertical Perception/verticalpercep_vln.json
│   ├── Spatial Relationship/spatialrel_vln.json
│   └── validate_dataset_integrity.py      # 数据集自检脚本
├── requirements-base.txt                  # 基础依赖（numpy/opencv/Pillow/filelock）
├── requirements-llm.txt                   # LLM 路线（openai + zai）
└── requirements-local-model.txt           # 本地模型路线（torch/transformers/decord 等）
```

### 0.1 调用关系简述

```text
gpt_eval.py   ─┐
el.sh         ─┼──► evaluation/run_llm_eval.py ──► evaluation/simulation.py
               │                                     │
               │                                     ├─► evaluation/common.py     （仿真器 + 指标 + 图像工具）
               │                                     └─► evaluation/providers.py  ──► evaluation/prompts.py
               │
               ├──► evaluation/eval_snav.py       ──► evaluation/common.py （+ 外部 LLaVA 代码）
               └──► evaluation/eval_streamvln.py  ──► evaluation/common.py （+ 外部 StreamVLN 代码）

tools/merge_results.py  ──► evaluation/common.py
tools/smoke_test.py     ──► evaluation/* （离线自检，不依赖 habitat_sim）
```

### 0.2 逐个文件说明

| 文件 | 角色 | 何时用到 |
| --- | --- | --- |
| `evaluation/__init__.py` | 把 `evaluation/` 标记为 Python 包 | 任何 `from evaluation.* import ...` |
| `evaluation/config.py` | 维护任务名 → `*_vln.json` 的映射，以及每个 LLM Profile（模型、base_url、分辨率、重试等）的默认值，并定义 `LlmEvalConfig` | LLM 评测启动前 |
| `evaluation/common.py` | 公共工具：JSON 加锁读写、gzip/JSON 数据加载、HM3D 场景路径解析、quaternion 修正、Habitat-Sim 初始化、RGB/Depth 抽帧、视频帧采样、动作解析、结果追加/断点续跑、SR/NE/OS/SPL 汇总 | 三条评测路线都会调用 |
| `evaluation/prompts.py` | 生成标准化的导航 Prompt（含时间戳、历史帧、动作枚举） | `providers.py` 请求前拼 prompt |
| `evaluation/providers.py` | 封装 OpenAI-compatible 与 Zhipu GLM 两类后端，统一重试/退避/限流 | LLM 评测每一步推理 |
| `evaluation/simulation.py` | LLM 评测 rollout 主循环：加载 episode → 建场景 → 喂 LLM → 解析动作 → 执行动作 → 收集指标 | `run_llm_eval.py` 唯一会调用的核心 |
| `evaluation/run_llm_eval.py` | LLM 评测 CLI：解析参数、合并 Profile、构建 `LlmEvalConfig`、配置日志、调用主循环；支持 `--dry-run` | **主要入口**（LLM 路线） |
| `evaluation/eval_snav.py` | 适配 SNav / LLaVA 系列本地模型：装载权重 → 沿用 `common.py` 的仿真/指标 → 把视频帧交给模型生成动作 | **主要入口**（SNav 路线） |
| `evaluation/eval_streamvln.py` | 适配 StreamVLN 系列本地模型：额外处理深度、位姿、内参与 token 缓存，并调用 `StreamVLNForCausalLM.generate` | **主要入口**（StreamVLN 路线） |
| `gpt_eval.py` | 老脚本的向后兼容 wrapper，内部 `from evaluation.run_llm_eval import main` | 旧代码继续用 `python gpt_eval.py` 时 |
| `el.sh` | bash 一键并行脚本，把 LLM 评测切成 8 个分片后台同时跑 | 分片并行加速 |
| `tools/smoke_test.py` | 离线自检：byte-compile、导入检查、数据集 schema、分辨率一致性、指标数学、prompt/动作解析等 16 项 | 开源前或换机器后 |
| `tools/merge_results.py` | 合并多个 shard 的结果文件并重排，打印全局 SR/NE/OS/SPL | 分片结束后 |
| `tools/llm_client.py` / `llm_client2.py` / `text_prompt.py` | 数据增广 / 翻译 / Prompt 模板等离线小工具 | 评测之外的内容生产流程 |

> **运行方式约定**：三个入口脚本已经在文件顶部加了 `sys.path` 兜底，因此既可以用 `python evaluation/run_llm_eval.py ...`，也可以用 `python -m evaluation.run_llm_eval ...`，推荐前者（更直观）。

### 0.3 统一的分辨率约定

所有评测路线现在统一使用 **224×224** 作为 Habitat 渲染分辨率与 API 请求时的 `encode_resize` 尺寸。如果你想在本地实验 336/384，只需要在命令行里传 `--frame-width 336 --frame-height 336 --encode-resize 336` 覆盖默认值即可；

### 0.4 与 VLN-CE / 官方 R2R-CE 指令格式的关系

NavSpace 评测使用的 episode JSON（各子任务下的 `*_vln.json`）遵循 **VLN-CE 风格的连续环境 episode 约定**：每条样本包含自然语言 `instruction`、场景标识 `scene_id`、起始位姿、目标 `goals`、以及（可选）参考路径等字段，与 [VLN-CE](https://github.com/jacobkrantz/VLN-CE) 项目中 **R2R-CE** 数据在 Habitat 中 rollout 所需的信息组织方式 **一致或可直接对齐**（NavSpace 在六类空间智能子任务上扩展了指令与场景，但 **数据字段形态与官方 R2R-CE 评测管线兼容同一类解析逻辑**）。

因此，**任何已经能在官方 R2R-CE 指令与 episode 格式上完成评测的模型**，在接入本仓库 `evaluation/` 中的仿真循环、动作解析与（可选）多帧 / 视频 prompt 拼接后，**理论上可以用同样的方式**在 NavSpace benchmark 上评测。若评测所需的额外说明（例如 Habitat 版本、原始 R2R-CE 数据字段、环境配置细节等）在本仓库中未展开，可在 VLN-CE 官方仓库及其文档中查阅：<https://github.com/jacobkrantz/VLN-CE>。

---

## 1. 部署指南（三类评测通用）

### 1.1 系统前置

- Python ≥ 3.9（3.9–3.10 兼容性最好；Habitat-Sim 预编译包对 3.11+ 支持视平台而定）
- Linux x86_64 或 macOS（Windows 请在 WSL2 中运行 Habitat）
- 一块支持 CUDA 的 GPU（仅本地模型路线需要）

### 1.2 创建 Conda 环境

```bash
conda create -n navspace python=3.10 -y
conda activate navspace
```

### 1.3 安装 Habitat-Sim

官方推荐 conda-forge 二进制包（与 VLN-CE 使用的是同一套发布渠道）：

```bash
conda install -c aihabitat -c conda-forge habitat-sim=0.2.5 withbullet headless -y
```

要点：

- `withbullet` 打开碰撞检测；`headless` 适合无显示器/服务器场景。
- 如果要在本机带显示器渲染，把 `headless` 换成 `with-cuda`（需要匹配的 CUDA 版本）。
- 版本号请对齐 Habitat-Lab 的版本；NavSpace 的数据集默认在 `habitat-sim 0.2.x` 上调试。

安装完成后用下面的命令做快速自检：

```bash
python -c "import habitat_sim; print(habitat_sim.__version__)"
```

### 1.4 安装 Habitat-Lab（可选）

如果需要使用 `habitat` 的 episode 结构、pathfinder 等更完整的 API，装 lab：

```bash
git clone --branch v0.2.5 https://github.com/facebookresearch/habitat-lab.git
cd habitat-lab
pip install -r requirements.txt
pip install -e habitat-lab
cd ..
```

> NavSpace 的评测脚本只依赖 `habitat_sim`，**不强制**要求 `habitat-lab`；但后续要自己扩展训练/评测流程时装上更方便。

### 1.5 安装 NavSpace 依赖

```bash
cd NavSpace-main
pip install -r requirements-base.txt
# LLM 路线
pip install -r requirements-llm.txt
# 本地模型路线（SNav / StreamVLN 任一）
pip install -r requirements-local-model.txt
```

### 1.6 下载场景资源

NavSpace 数据集基于 HM3D 构建，Data Augmentation全部基于 MP3D 进行；两者都可以独立下载使用。

#### 1.6.1 HM3D v0.2

1. 在 <https://matterport.com/habitat-matterport-3d-research-dataset> 或 `ai-habitat` 官方指南页申请 HM3D 访问权，此处需要填写相应申请表，反馈时间不会太长。
2. 使用 `matterport_download.py` / 官方 CLI 下载 v0.2 数据。
3. 解压后目录结构应为：

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

4. 在所有评测命令里用 `--hm3d-base-path /path/to/hm3d_v0.2` 指向这一层目录即可。

> 内部 `resolve_scene_path`（`evaluation/common.py`）会依次在 `train/`、`val/`、`test/` 下按 scene name 查找 `*.basis.glb`，因此即使 dataset 里的 `scene_id` 是历史机器上的绝对路径，也能被自动重定位。

#### 1.6.2 Matterport3D（MP3D，可选）

仅在你要跑 MP3D 变体或扩展自己的标注时才需要：

```bash
# 注意需要 Python 2.7 执行官方下载脚本
python download_mp.py --task habitat -o data/scene_datasets/mp3d/
```

解压后保持 VLN-CE 推荐的 `data/scene_datasets/mp3d/{scene}/{scene}.glb` 目录结构即可。

> 当前 `evaluation/` 中的三个评测脚本只用到 HM3D；如果把 MP3D 接进来，只需要把 `--hm3d-base-path` 指向 MP3D 的根目录，并相应拓展 `resolve_scene_path`。

### 1.7 验证数据集完整性

NavSpace 仓库已经随附六个任务的 `*_vln.json`，在仓库根部执行：

```bash
python NavSpace-Datasets/validate_dataset_integrity.py
```

可以快速检查 episode 数量、字段完整性和场景 ID 是否格式合法。

### 1.8 NavSpace六类任务清单

| `--task` 参数 | 对应数据集文件 | 任务类型 |
| --- | --- | --- |
| `environment_state` | `NavSpace-Datasets/Environment State/envstate_vln.json` | 环境状态判别 |
| `space_structure` | `NavSpace-Datasets/Space Structure/spacestructure_vln.json` | 空间结构理解 |
| `precise_movement` | `NavSpace-Datasets/Precise Movement/precisemove_vln.json` | 精确动作 |
| `viewpoint_shifting` | `NavSpace-Datasets/Viewpoint Shifting/viewpointsft_vln.json` | 视角转换 |
| `vertical_perception` | `NavSpace-Datasets/Vertical Perception/verticalpercep_vln.json` | 垂直空间感知 |
| `spatial_relationship` | `NavSpace-Datasets/Spatial Relationship/spatialrel_vln.json` | 空间关系推理 |

也可以用 `--trajectory-path /abs/path/to/custom.json` 跳过预设映射。

### 1.9 API Key 配置

优先级从高到低：

1. `--api-key <YOUR_KEY>`（最直接）
2. `--api-key-env <ENV_NAME>` + 环境变量（例如 `export OPENAI_API_KEY=sk-xxx`）
3. `--api-key-file /path/to/key.json`（里面 `{"api_key": "..."}`）

内置 Profile 会自动选择默认环境变量名：

- `gemini-pro` / `gemini-flash` → `OPENAI_API_KEY`（ChatAnywhere 代理端点）
- `qwen72b` → `DASHSCOPE_API_KEY`
- `glm4.5v` / `glm-4.1v-thinking-flash` → `ZHIPU_API_KEY`

---

## 2. LLM based 评测

**入口脚本**：`evaluation/run_llm_eval.py`。
**核心循环**：`evaluation/simulation.py` → `evaluation/providers.py` → 在线 API。

### 2.1 内置 Profile

| Profile | Provider | 模型 | 默认端点 |
| --- | --- | --- | --- |
| `gemini-pro` | openai_compatible | `gemini-2.5-pro` | `https://api.chatanywhere.tech/v1` |
| `gemini-flash` | openai_compatible | `gemini-2.5-flash` | 同上 |
| `qwen72b` | openai_compatible | `qwen2.5-vl-72b-instruct` | `https://dashscope.aliyuncs.com/compatible-mode/v1` |
| `glm4.5v` | zhipu | `glm-4.5v` | Zhipu 官方 SDK |
| `glm-4.1v-thinking-flash` | zhipu | `glm-4.1v-thinking-flash` | Zhipu 官方 SDK |

可以用 `--provider`、`--model`、`--base-url`、`--frame-width`、`--encode-resize` 等参数临时覆盖。
常见 API 获取入口：

- Zhipu：<https://bigmodel.cn/>
- Qwen（DashScope/阿里云）：<https://cn.aliyun.com/>
- ChatAnywhere（用于 GPT / Gemini 等 OpenAI-compatible 接口）：<https://chatanywhere.apifox.cn/>
- 如有 OpenAI 官方 API Key，也可通过 `--base-url` / `--model` 参数直接适配。

### 2.2 快速开始

Gemini（ChatAnywhere 通道）：

```bash
export OPENAI_API_KEY=sk-xxxxx
python evaluation/run_llm_eval.py \
  --profile gemini-pro \
  --task environment_state \
  --hm3d-base-path /path/to/hm3d_v0.2
```

DashScope / Qwen：

```bash
export DASHSCOPE_API_KEY=sk-xxxxx
python evaluation/run_llm_eval.py \
  --profile qwen72b \
  --task space_structure \
  --hm3d-base-path /path/to/hm3d_v0.2
```

Zhipu GLM：

```bash
export ZHIPU_API_KEY=xxxxx.xxxxx
python evaluation/run_llm_eval.py \
  --profile glm4.5v \
  --task vertical_perception \
  --hm3d-base-path /path/to/hm3d_v0.2
```

自定义第三方 OpenAI 协议端点：

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

### 2.3 单机八分片并行

脚本沿用 `traj_idx % num_shards == model_id` 的切片规则。

单个分片：

```bash
python evaluation/run_llm_eval.py \
  --profile gemini-pro \
  --task environment_state \
  --hm3d-base-path /path/to/hm3d_v0.2 \
  --model-id 0 \
  --num-shards 8
```

八分片一键并行（bash）：

```bash
bash el.sh environment_state /path/to/hm3d_v0.2 gemini-pro
```

`el.sh` 内部会以不同 `--model-id` 启动 8 个后台进程，并在脚本退出时自动 `pkill` 清理。

### 2.4 断点续跑

只要重新执行相同 `--profile --task`，脚本就会自动跳过 `outputs/llm/<xxx>.json` 里已完成的 episode。

如果想从另一份结果复用：

```bash
python evaluation/run_llm_eval.py \
  --profile gemini-pro \
  --task environment_state \
  --hm3d-base-path /path/to/hm3d_v0.2 \
  --resume-from outputs/llm/older_results.json
```

### 2.5 输出 & 指标

默认输出目录：`outputs/llm/`，默认文件：

- `llm_eval_<profile>_<task>.json`：结果
- `llm_eval_<profile>_<task>.log`：运行日志

结果格式：

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

指标含义：

- `success`：episode 结束时是否进入 `goal_radius`。
- `nav_error`：结束时与目标点的欧氏距离（米）。
- `os`（Oracle Success）：episode 过程中是否**任意一步**进入过 `goal_radius`。
- `shortest_path_length`：数据集中提供的 geodesic distance。
- `actual_path_length`：实际执行 `move_forward` 累积的行走长度（每步 0.25 m）。
- `SPL` = `success × shortest / max(actual, shortest)`，在 `summarize_results` 中汇总计算。

### 2.6 合并结果

```bash
python tools/merge_results.py \
  --input outputs/llm/shard0.json outputs/llm/shard1.json ... \
  --trajectory-path "NavSpace-Datasets/Environment State/envstate_vln.json" \
  --output outputs/llm/merged.json
```

合并脚本会按数据集顺序对齐 episode，并打印全局 SR/NE/OS/SPL。

---

## 3. SNav based 评测（LLaVA/SNav 本地模型）

**入口脚本**：`evaluation/eval_snav.py`。
**核心循环**：脚本内独立的 rollout + `common.py` 的仿真与指标工具。

### 3.1 额外依赖（本仓库不随附）

- LLaVA / SNav 代码库（提供 `llava.model.builder.load_pretrained_model` 等入口）。如果你已经在用 StreamVLN 官方仓库，它的 `StreamVLN/llava/` 子目录就是同一套 LLaVA 代码，可以直接复用。
- 你的 SNav 权重目录（`--model-path`）。
- 合适的 `torch` + `transformers`（已在 `requirements-local-model.txt` 里）。
- 推荐把 HuggingFace 缓存显式指向一个可写目录：

```bash
export HF_HOME=/your/writable/hf_cache
```

把 LLaVA 代码路径加入 `PYTHONPATH`，例如：

```bash
# 情况 A：独立 LLaVA 代码库
export PYTHONPATH=/path/to/LLaVA:$PYTHONPATH

# 情况 B：复用 StreamVLN 附带的 LLaVA（推荐，和 SNav 训练保持一致）
export PYTHONPATH=/path/to/StreamVLN:$PYTHONPATH
```

### 3.2 运行示例

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

常用参数：

- `--device` / `--device-map`：默认 `cuda` / `auto`。
- `--frame-width` / `--frame-height`：默认 224。
- `--model-id` / `--num-shards`：和 LLM 路线一致的分片切分。
- `--resume-from`：复用另一份结果文件做断点续跑。

### 3.3 输出

默认输出目录：`outputs/snav/`，默认文件：

- `snav_<task>.json`
- `snav_<task>.log`

结果字段与 LLM 路线完全一致，可以直接用 `tools/merge_results.py` 合并。

---

## 4. StreamVLN based 评测

**入口脚本**：`evaluation/eval_streamvln.py`。
**核心循环**：脚本内独立的 rollout + `common.py` 的仿真与指标工具（开启 `include_depth=True`）。

### 4.1 额外依赖（本仓库不随附）

以下模块必须能通过 `PYTHONPATH` 被导入：

- `model.stream_video_vln.StreamVLNForCausalLM`（位于 StreamVLN 官方仓库 `StreamVLN/streamvln/model/`）
- `utils.utils`（含 `DEFAULT_IMAGE_TOKEN`/`DEFAULT_MEMORY_TOKEN`/`DEFAULT_VIDEO_TOKEN`/`IMAGE_TOKEN_INDEX`/`MEMORY_TOKEN_INDEX` 以及 `dict_to_cuda`）
- `llava.*`（StreamVLN 内部依赖，位于 `StreamVLN/llava/`）
- `depth_camera_filtering.filter_depth`（pip 包）
- StreamVLN 权重目录（`--model-path`）

> **PYTHONPATH 组合**：`StreamVLN/streamvln` 暴露 `model/` 和 `utils/`；`StreamVLN/` 根目录暴露 `llava/`。两者都要加上：
>
> ```bash
> export PYTHONPATH=/path/to/StreamVLN:/path/to/StreamVLN/streamvln:$PYTHONPATH
> ```

### 4.2 运行示例

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

### 4.3 实现要点

- 该脚本在每一步都会同时采集 RGB + Depth，并用 `filter_depth` 做时域滤波，再转成 `uint16` 深度图送入模型。
- agent 位姿会转成 4×4 homogeneous matrix，乘上 `get_axis_align_matrix()` 做坐标对齐后传给模型。
- `model.reset_for_env(traj_idx)` 会在 stop 时或每 32 步触发一次 KV 缓存重置，以复刻原作者的 memory 窗口行为。
- `actual_path_length` 会随 `move_forward` 增加 0.25 m/步，从而得到正确的 SPL。

### 4.4 输出

默认输出目录：`outputs/streamvln/`，默认文件：

- `streamvln_<task>.json`
- `streamvln_<task>.log`

---

## 5. 常见问题

| 现象 | 原因 & 处理 |
| --- | --- |
| `ModuleNotFoundError: habitat_sim` | 按 §1.3 重新安装 habitat-sim；确认当前 conda 环境正确（我们实测用的是 `streamvln` 环境，里面有 `habitat-sim 0.3.3` + `habitat-lab 0.3.3` + `torch` + `transformers`）。 |
| `Unable to resolve HM3D scene path for scene_id=...` | 数据里仍是历史机器路径，且本机 HM3D 目录里没有对应场景；检查 `--hm3d-base-path`，或 HM3D 是否解压在 `train/val/test` 三个子目录里。如果是自定义数据，可直接把 `scene_id` 写成指向 `.glb` 的绝对路径，`resolve_scene_path` 在文件存在时会直接返回它，**因此 MP3D 场景也能用**。 |
| `API key not found` | 按 §1.9 的三种方式之一提供密钥；检查环境变量是否真的导入当前终端。 |
| `Zhipu rate limited, sleeping ...` | 正常退避日志，保持运行即可；必要时调大 `--initial-retry-delay` 或 `--max-retries`。 |
| `FlashAttention2 has been toggled on, but ... flash_attn seems to be not installed` | 本仓库已经把 `--attn-implementation` 做成 CLI 参数，默认 `sdpa`；如有需要再显式传 `flash_attention_2`（要求本机装好 `flash-attn`）。 |
| `AttributeError: 'dict' object has no attribute 'to_dict'`（SNav 加载 ckpt 时） | 历史 LlaVA-Qwen checkpoint 里 `text_config/vision_config` 是裸 dict。`eval_snav.py` 在 `load_pretrained_model` 里加了 `overwrite_config={"text_config": None, "vision_config": None}` 自动修；如果你把这行删了就会复现。 |
| `ImportError: cannot import name 'dict_to_cuda' from 'utils.dist'` | StreamVLN 官方仓库把 `dict_to_cuda` 放在 `utils.utils`，我们的 `eval_streamvln.py` 已经改成从 `utils.utils` 导入；如自行改回去会复现。 |
| `There was a problem when trying to write in your cache folder` | HuggingFace 默认缓存目录不可写；`export HF_HOME=/your/writable/hf_cache` 即可（别再用已经弃用的 `TRANSFORMERS_CACHE`）。 |
| `Please install petrel_client to Client.` | StreamVLN 内部可选的 OSS 客户端提示，离线评测可忽略。 |
| 想换一份自定义数据 | 用 `--trajectory-path /abs/path/custom.json`，该文件要么是 `{"episodes": [...]}`，要么是直接的 list；字段需至少包含 `instruction.instruction_text`、`scene_id`、`start_position`、`start_rotation`、`goals[0].position`、`info.geodesic_distance`。 |

---

## 6. 开源前的快速验证

完成代码修改、但还没装 habitat-sim / 没下载 HM3D 时，可以靠下面三种离线手段尽量多地覆盖仓库：

### 6.1 一键离线自检：`tools/smoke_test.py`

```bash
python tools/smoke_test.py
```

这个脚本**不依赖** `habitat_sim`、HM3D 场景或任何真实 API Key，只需要 `requirements-base.txt` 里的基础包就能跑。它会按顺序执行 16 项检查：

1. `byte-compile all Python sources` —— 全部 Python 文件可编译，不会有语法错误。
2. `import evaluation.* modules` —— `evaluation.common/config/prompts/providers` 能正常导入。
3. `build CLI parsers for all three entry scripts` —— 三个入口脚本的 `argparse` 构建正常、参数齐全。
4. `gpt_eval.py wrapper re-exports main` —— 老入口仍然能转发到新实现。
5. `TASK_DATASET_MAP covers every task` —— 六个任务名全部存在，缺失文件会以 WARN 提示而不打断测试。
6. `episode schema is consistent across tasks` —— 每个任务的前若干条 episode 都有 `scene_id/start_position/start_rotation/goals/info/instruction`。
7. `scene_id values are parseable for resolver` —— `resolve_scene_path` 能从每条 `scene_id` 中提取候选名。
8. `LLM profiles share the expected resolution` —— `PROFILE_DEFAULTS` 中每个 Profile 的 `frame_width/frame_height/encode_resize` 都是期望分辨率（默认 224）。
9. `local-model CLIs use the expected resolution` —— `eval_snav/eval_streamvln` 的 CLI 默认分辨率一致。
10. `action extraction parses known verbs` —— `extract_actions` 正确解析 `Move forward / Turn left / ...`。
11. `image pipeline end-to-end` —— `ensure_size_bgr → encode_image_b64 → process_images_as_video` 整条图像链路的形状和 base64 输出都正确。
12. `metrics math matches expectations` —— `summarize_results` 的 SR/OS/SPL 数值在构造数据上完全正确。
13. `navigation prompt contains required tokens` —— `build_navigation_prompt` 拼出的内容含指令和动作关键字。
14. `resume index round-trip is stable` —— 断点续跑的索引键一致。
15. `provider dispatch rejects unknown providers cleanly` —— `infer_actions` 对未知 provider 报错清晰（不真的发网络请求）。
16. `requirements-*.txt are non-empty` —— 三个依赖文件都存在且非空。

可选参数：

```bash
python tools/smoke_test.py --max-episodes 64 --expected-resolution 224
```

### 6.2 LLM 入口的 `--dry-run`（可选调真实 API）

`evaluation/run_llm_eval.py` 内置了一个只校验配置、不启动 Habitat 的干跑模式：

```bash
python evaluation/run_llm_eval.py \
  --profile gemini-pro \
  --task environment_state \
  --hm3d-base-path /path/does/not/exist \
  --dry-run
```

它会：

- 打印合并后的 profile、provider、model、base_url、output_dir；
- 确认 API Key 能被解析（只打印尾 4 位）；
- 加载对应任务的 `*_vln.json` 并打印 episode 数；
- 试着用前 3 条 episode 的 `scene_id` 去解析 HM3D 路径（路径不存在也不会 crash，只会打 `NOT FOUND`）；
- 不会加载 `habitat_sim`，也不会真的跑 rollout。

如果想**顺便**验证 API Key 能连通（只发一条极小的文本请求，不消耗图片额度）：

```bash
python evaluation/run_llm_eval.py \
  --profile gemini-pro \
  --task environment_state \
  --hm3d-base-path /path/does/not/exist \
  --dry-run --dry-run-probe-api
```

### 6.3 数据集自检

```bash
python NavSpace-Datasets/validate_dataset_integrity.py
```

原仓库提供的 episode 级别完整性检查，和 `smoke_test.py` 互补。

### 6.4 拥有HM3D环境的检测

在目标服务器上装完 `habitat-sim` 后，可以用很小的 `--max-steps` 做一次端到端冒烟测试：

```bash
python evaluation/run_llm_eval.py \
  --profile gemini-pro \
  --task environment_state \
  --hm3d-base-path /path/to/hm3d_v0.2 \
  --max-steps 5 \
  --num-shards 64 --model-id 0   # 64 分片只跑第 0 片 => 只算一两个 episode
```

这样只会消耗极少的 API 配额就能确认：场景能渲染 → LLM 请求能发 → 动作能解析 → 结果能写盘。

组合 §6.1 + §6.2 + §6.4，你基本不需要在本地再下载一次 HM3D。

### 6.5 MP3D 上的 SNav / StreamVLN 冒烟测试

如果你只有 MP3D、没有 HM3D，也能在 1 条 episode 上快速验证两条本地模型评测链路。

1. 用 habitat_sim 的 `PathFinder` 从 MP3D 场景里采一个可导航的起点/终点对，写成 1-episode 的 JSON：

```python
# /tmp/make_mp3d_mini.py
import json, habitat_sim, numpy as np

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

2. 用 `--trajectory-path` 指向它，`--hm3d-base-path` 随便填个存在的目录即可（因为 `scene_id` 是绝对路径会直接返回）：

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

预期：两条都能在 2–5 分钟内完成，输出里会看到 `Eval Results: Success:... Nav Error:... OS:... SPL:...` 以及对应的 `*_environment_state.json` 结果文件。

---

## 7. 参考

- Habitat 官方文档：<https://aihabitat.org/>
- VLN-CE（Habitat + R2R/RxR 的连续空间导航）：<https://github.com/jacobkrantz/VLN-CE>
- HM3D v0.2 下载与使用说明：<https://aihabitat.org/datasets/hm3d/>
- Matterport3D 数据集：<https://niessner.github.io/Matterport/>
- Qwen阿里通义千问官网：<https://cn.aliyun.com/>
- GLM智谱官网: <https://bigmodel.cn/>
- ChatAnywhere官网: <https://chatanywhere.apifox.cn/>
- StreamVLN官方仓库: <https://github.com/InternRobotics/StreamVLN>

NavSpace 的所有评测入口都只依赖 Habitat-Sim 与 NavSpace-Datasets 本身，因此一旦完成 §1 的部署，这三类评测都能在同一个环境里复用同一组数据。
