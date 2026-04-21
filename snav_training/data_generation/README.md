# Data generation

This folder turns raw VLN-CE / RxR-CE / EnvDrop episode files into the
**SNav-style training format** consumed by `scripts/run_snav_train.sh`.

For every episode we let Habitat-Sim's `GreedyGeodesicFollower` follow the
shortest path to the goal, dumping (by default, `--output_mode frames` — the
StreamVLN-compatible layout consumed by `train_snav.py`):

- `images/<scan>_<tag>_<id>/rgb/<step>.jpg` — one RGB frame per navigation
step (384×384, FOV 120°).
- `annotations.json` — one entry **per episode**, with the instruction(s)
and the full ground-truth action list.

An alternative `--output_mode snav_frames` is available if you want the
LLaVA-SFT style `llava_annotations.json` (one sample per step) instead. It's
there for compatibility with external LLaVA training pipelines and is NOT
needed for the in-repo trainer.

Only HM3D **or** MP3D scenes are needed. The renderer picks whichever
directory layout you point `--scenes_root` at.

> This pipeline only produces the vanilla SNav Stage-1 training data
> (expert trajectories rendered with the geodesic follower). It does **not**
> cover any data-augmentation flows (instruction rewriting, panorama
> augmentation, etc.) — those are deliberately out of scope here.

## Layout

```
snav_training/data_generation/
├── README.md                      (this file)
├── render_streamvln.py            Python renderer (R2R-CE / RxR-CE / EnvDrop)
├── run_render_r2rce.sh            R2R-CE train split (bash wrapper)
├── run_render_rxrce.sh            RxR-CE train split
└── run_render_envdrop.sh          EnvDrop subset
```

Output tree after running all three wrappers (default `frames` mode):

```
${DATA_ROOT}/                      (defaults to ./snav_data)
├── r2rce/
│   ├── annotations.json
│   └── images/<scan>_r2r_<id>/rgb/*.jpg
├── rxrce/
│   ├── annotations.json
│   └── images/<scan>_rxr_<id>/rgb/*.jpg
└── envdrop/
    ├── annotations.json
    └── images/<scan>_envdrop_<id>/rgb/*.jpg
```

Pass these three directory paths as `VIDEO_FOLDERS=...` when launching
`scripts/run_snav_train.sh`.

## Prerequisites

- `habitat-sim ≥ 0.3.3` (tested with 0.3.3 inside the `streamvln` conda env).
- Scene assets under `${SCENES_ROOT}`:
  - HM3D: `${SCENES_ROOT}/hm3d_v0.2/{train,val,test}/<scene>/<scene>.basis.glb`
  - MP3D: `${SCENES_ROOT}/mp3d/<scene>/<scene>.glb`
  - The renderer falls back on either layout automatically.
- Episode JSONs (NOT shipped here — grab from the official releases):
  - R2R-CE: `R2R_VLNCE_v1-3_preprocessed/train/train.json.gz`
  - RxR-CE: `RxR_VLNCE_v0/train/train_guide.json`
  - EnvDrop: `R2R_VLNCE_v1-3_preprocessed/envdrop/envdrop.json.gz`

## Quick start

```bash
# 1. Activate the env that has habitat-sim installed.
conda activate streamvln

# 2. Point environment variables at your local paths (see each script for
#    the full list; the defaults below are what you typically need).
export DATA_ROOT=/abs/path/to/snav_data
export SCENES_ROOT=/abs/path/to/scene_datasets
export R2RCE_TRAIN_JSON=/abs/path/to/train.json.gz
export RXRCE_TRAIN_JSON=/abs/path/to/train_guide.json
export ENVDROP_SOURCE_JSON=/abs/path/to/envdrop.json.gz

# 3. Render (run them one at a time, each uses ≈40–60 GB of disk).
bash run_render_r2rce.sh
bash run_render_rxrce.sh
bash run_render_envdrop.sh
```

### Common flags

All three wrappers forward unknown arguments to `render_streamvln.py`, so you
can pass e.g. `--max_episodes 20` for a quick smoke test:

```bash
bash run_render_r2rce.sh --max_episodes 20
```

The renderer's own CLI exposes (non-exhaustive list):


| Flag                         | Default  | Meaning                                                       |
| ---------------------------- | -------- | ------------------------------------------------------------- |
| `--width / --height`         | 384      | Rendered RGB resolution                                       |
| `--hfov`                     | 120      | Camera horizontal FOV (SNav convention)                       |
| `--camera_height`            | 0.88     | Camera height in metres                                       |
| `--forward_step`             | 0.25     | Forward step size in metres                                   |
| `--turn_angle`               | 15.0     | Turn angle in degrees                                         |
| `--max_steps`                | 500      | Episode length cap                                            |
| `--goal_radius`              | 0.5      | Success radius in metres                                      |
| `--output_mode`              | `frames` | `frames` (StreamVLN style, default) / `video` / `snav_frames` |
| `--custom_instructions_json` | —        | Overwrite episode instructions (EnvDrop uses this)            |
| `--lang_filter`              | —        | Keep only a subset of languages (RxR-CE typically `en`)       |


## Notes

- The renderer is deterministic given the same episode JSON + seed; if you  
re-run it only episodes missing from disk will be rendered.
- `render_streamvln.py` has no dependencies beyond `habitat-sim`,
`opencv-python`, `Pillow`, and `tqdm`.

