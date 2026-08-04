# Stage-1 数据渲染（baseline，不是论文 Stage A）

这里负责把 R2R-CE、RxR-CE 和 EnvDrop 的 episode JSON 渲染成 **baseline**
`train_snav.py` 用的帧数据（`GreedyGeodesicFollower`）。

论文 Stage A 的 collect/build 在 [`../stage_a/`](../stage_a/) 与
`scripts/prepare_stage_a_data.sh`，不要用本目录替代。

## 1. 如何开始

先进入有 `habitat-sim` 的环境：

```bash
conda activate your_virtual_env
```

然后进到渲染目录，并把本地路径配好：

```bash
cd snav_training/data_generation

export DATA_ROOT=/abs/path/to/snav_data      # 输出根目录，默认 ./snav_data
export SCENES_ROOT=/abs/path/to/scene_datasets  # HM3D 和/或 MP3D 场景根目录

export R2RCE_TRAIN_JSON=/abs/path/to/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz
export RXRCE_TRAIN_JSON=/abs/path/to/RxR_VLNCE_v0/train/train_guide.json
export ENVDROP_SOURCE_JSON=/abs/path/to/R2R_VLNCE_v1-3_preprocessed/envdrop/envdrop.json.gz
```

三个数据集分别渲染，建议一个一个跑。日志会写到 `${DATA_ROOT}/*_render.log`：

```bash
bash run_render_r2rce.sh
bash run_render_rxrce.sh
bash run_render_envdrop.sh
```

第一次跑可以先做个小测试：

```bash
bash run_render_r2rce.sh --max_episodes 5
```

渲染完成后，训练脚本需要知道这几个数据目录：

```bash
export VIDEO_FOLDERS="${DATA_ROOT}/r2rce,${DATA_ROOT}/rxrce,${DATA_ROOT}/envdrop"
```

---

## 2. 渲染产物说明

默认输出是 `frames` 模式，三个 wrapper 都是这个设置。训练时真正读取的是 **`annotations.json`**，不是 `llava_annotations.json`。

```
${DATA_ROOT}/
├── r2rce/
│   ├── annotations.json
│   └── images/<scan>_r2r_<episode_id>/rgb/001.jpg ...
├── rxrce/
│   ├── annotations.json
│   └── images/<scan>_rxr_<episode_id>/rgb/...
└── envdrop/
    ├── annotations.json
    └── images/<scan>_envdrop_<episode_id>/rgb/...
```

`annotations.json` 里的一条记录对应一个 episode，里面有 instruction 列表和完整的 GT action 序列。对应的 RGB 帧放在 `images/.../rgb/` 下，按 step 编号。

下面这些渲染参数已经在 wrapper 里固定住了，目的是让训练视角和评测视角一致。一般不要改，除非你也同步改评测设置。

| 项 | 值 |
|----|-----|
| 分辨率 | 384×384 |
| 水平 FOV | 120° |
| 相机高度 | 1.5 m |
| 前进步长 | 0.25 m |
| 转向角 | **30°** |
| R2R / EnvDrop `max_steps` | 500 |
| RxR `max_steps` | 800 |

三个 wrapper 的输入略有不同：

| 脚本 | 输入 | 说明 |
|------|------|------|
| `run_render_r2rce.sh` | `R2RCE_TRAIN_JSON` | R2R-CE train |
| `run_render_rxrce.sh` | `RXRCE_TRAIN_JSON` | RxR-CE train_guide，默认 `LANG_FILTER=en` |
| `run_render_envdrop.sh` | `ENVDROP_SOURCE_JSON` | 先从全量 **随机抽 20000** 条（`SAMPLE_COUNT` / `SAMPLE_SEED`），再渲染 |

全量渲染会比较占空间，三套数据加起来通常是数百 GB 级，具体取决于 episode 数量和轨迹长度。

---

## 3. 必要补充

`SCENES_ROOT` 下需要能找到 HM3D 或 MP3D 场景。两种格式都支持，渲染器会自己匹配：

- HM3D：`hm3d_v0.2/{train,val,test}/<scene>/<scene>.basis.glb`
- MP3D：`mp3d/<scene>/<scene>.glb`

渲染支持断点续跑。已经写进 `annotations.json` 的 episode 会被跳过；如果想强制重渲某条，删掉对应的 `images/...` 目录和 annotation 记录再跑。

wrapper 里已经处理了 NVIDIA EGL 相关环境变量。正常情况下用 GPU 渲染即可；如果机器没有可用 GPU，可以设 `GPU_DEVICE_ID=-1` 走 CPU/Mesa，但会非常慢。

如果想在渲染阶段做几何随机化，可以打开下面这些环境变量。默认都是关闭的。

```bash
export CAMERA_HEIGHT_JITTER=0.15
export HFOV_JITTER=15
export RESOLUTION_CHOICES="256,320,384"
export NUM_RENDER_VARIANTS=2   # K>1 时目录名带 _v1, _v2 ...
bash run_render_r2rce.sh
```

这里的随机化只改相机高度、FOV、分辨率这类几何因素。亮度、模糊、噪声等外观增强不在这里做，而是在训练时通过 `AUGMENT=1` 开启。

本仓库训练不需要其他输出模式。如果要给外部 LLaVA SFT 流程用，可以看 `render_streamvln.py --output_mode video|snav_frames`，这两种模式会写 `llava_annotations.json`。

也可以绕过 wrapper 直接调 Python，参数以 `render_streamvln.py --help` 为准：

```bash
python render_streamvln.py --data_json ... --scenes_root ... --output_dir ... --output_mode frames
```
