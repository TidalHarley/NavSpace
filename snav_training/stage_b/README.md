# Stage-B episode release (`stage_b_episodes.json`)

One self-contained, R2R-CE compatible file describing **every** episode in the
SNav Stage-B training set (`aug_mix` + `manual_98`). With this file and your own
MP3D copy you can regenerate the training data that SNav-7B was fine-tuned on —
no LLM API calls, no join against `R2R_VLNCE_v1-3`, no access to our machines.

We ship the episode list rather than the rendered JPGs because every Stage-B
frame is derived from Matterport3D, whose Terms of Use each user signs
individually. This is the same policy as `NavSpace-Datasets/`.

## Quick start

```bash
# 1. Re-render every episode into snav_data/
python snav_training/scripts/render_stage_b.py \
  --episodes snav_training/stage_b/stage_b_episodes.json \
  --scenes-root /path/to/mp3d \
  --out-root snav_data

# 2. Build the hist8/future6 training JSON
python snav_training/scripts/build_hist8_future6.py \
  --aug-root snav_data/aug_mix \
  --manual-root snav_data/manual_98 \
  --out-dir train_data

# 3. Train (see ../README.md for the checkpoint prerequisites)
export PREV_STAGE_CHECKPOINT=/path/to/stage_a_ckpt
export IMAGE_FOLDER=$PWD/snav_data/aug_mix
bash snav_training/scripts/launch_paper_sft_stage_b.sh
```

Add `--max-episodes 3` to `render_stage_b.py` for a fast smoke test, or
`--dry-run` to print the underlying render commands without executing them.

## Contents

| Task | Episodes | Training rows | `dataset_tag` | Render mode |
| --- | --- | --- | --- | --- |
| `environment_state` | 500 (+409 STOP siblings) | 25,827 | `env_state` | follower |
| `spatial_relationship` | 500 | 26,629 | `aug` | follower |
| `vertical_perception` | 178 | 8,902 | `aug_vert` | follower |
| `precise_movement` | 400 | 12,886 | `precise` | resample |
| `manual_98` | 98 | 5,966 | `manual` | replay |
| **Total** | **1,676** | **80,210** | | |

80,210 is the exact row count SNav-7B trained on: the recorded run did
2,506 optimiser steps at an effective batch of 32 (4 per device × 8 GPUs),
i.e. 80,192 samples seen, with the trailing 18 dropped by
`dataloader_drop_last`.

## Schema

Each entry in `episodes` is a standard R2R-CE episode, so
`render_streamvln.py --data_format r2r` reads it directly:

```json
{
  "episode_id": 6913,
  "trajectory_id": 6913,
  "scene_id": "mp3d/r1Q1Z4BcV1o/r1Q1Z4BcV1o.glb",
  "start_position": [-1.87, 0.14, 6.02],
  "start_rotation": [0.0, 0.7071, 0.0, 0.7071],
  "goals": [{"position": [2.41, 0.14, 1.55], "radius": 0.5}],
  "instruction": {"instruction_text": "Starting from the hallway, if there is ..."},
  "navspace": { "...": "see below" }
}
```

`instruction_text` is the **final** trained text, including the Qwen-generated
conditionals for `environment_state` and the rewrites for
`vertical_perception`. Nothing needs to be re-generated.

The `navspace` block carries what plain R2R-CE cannot express:

| Field | Meaning |
| --- | --- |
| `task` | which Stage-B split the episode belongs to |
| `dataset_tag` | makes the output dir name: `{scan}_{dataset_tag}_{episode_id:06d}` |
| `ep_tag` | that directory name, precomputed |
| `render_mode` | `follower` \| `replay` \| `resample` (see below) |
| `reference_actions` | the action ids in the published data — use to verify a re-render |
| `reference_num_frames` | expected JPG count for the episode |
| `action_sequence` | human action verbs; `replay` episodes only |
| `balance_variants` | the `_stay` STOP siblings, with their own instruction text |
| `segments` / `geodesic_distance` | procedural motion spec; `resample` episodes only |
| `excluded_from_training` | set when the annotation is unusable, so row counts stay consistent |

`render_params` at the top level records the camera and actuation settings the
published frames were rendered with (384×384, 120° HFOV, camera height 1.5 m,
0.25 m forward step, 30° turns). Changing them changes the pixels.

### Render modes

**`follower`** (1,178 episodes) — a `GreedyGeodesicFollower` walks to
`goals[0].position`. Deterministic given the scene, the navmesh and the render
params, so this reproduces the published frames exactly. Note that
`reference_path` is *not* used by the renderer; only the goal is.

**`replay`** (98 episodes, `manual_98`) — replays the human
`navspace.action_sequence` step by step instead of pathfinding. Also exact.

**`resample`** (400 episodes, `precise_movement`) — **not frame-identical.**
These episodes were sampled with `pathfinder.get_random_navigable_point()`,
which draws from habitat-sim's internal RNG, and their start poses were never
persisted. They cannot be replayed. `render_stage_b.py` prints the regeneration
command instead:

```bash
python data_augmentation/precise_movement/run.py \
  --output-root snav_data/aug_mix/precise_movement --seed 42
python data_augmentation/scripts/convert_aug_to_sft.py \
  --folder snav_data/aug_mix/precise_movement
```

The sampler now seeds the pathfinder per scene, so a given `--seed` always
yields the same episode set — but that set is a *different draw* from the
paper's. The task is procedural (turn N degrees, advance M metres, with the
instruction generated from the sampled actions), so a fresh draw is
distributionally equivalent; it is simply not the same 400 trajectories.
The entries in this file record the paper's 400 for reference.

### Balanced STOP siblings

Every `environment_state` template ships the walk-to-goal trajectory, so
without a counterweight the split contains no `(instruction, STOP)` pair and
the model never learns to honour the IF clause. `4_balance_stay.py` fixes this
by emitting 409 sibling rows with the IF/OTHERWISE branches swapped, `actions =
[-1, 0]`, and the base episode's frames reused.

Those siblings are published verbatim under `balance_variants` and applied by
`render_stage_b.py`. Re-deriving them with `4_balance_stay.py` would work too,
but that script shuffles its input before picking which episodes get a sibling,
so a different annotation ordering yields a different subset.

## Provenance

The file was assembled from the artifacts of the original paper run:

- **Instructions and action labels** — the `annotations.json` of each rendered
  split, i.e. the files the training JSON was built from. Not the pipelines'
  intermediate LLM output, so no Qwen call is needed to reproduce them.
- **Geometry** — `environment_state` and `vertical_perception` from their
  pipeline records, `spatial_relationship` and `manual_98` from the R2R-CE train
  split and the annotation server's `trajectories.json` respectively. All of it
  baked in, so nothing has to be joined at render time.
- **`ep_tag`** — recomputed as `{scan}_{dataset_tag}_{episode_id:06d}` from the
  published geometry and checked against the original directory name for all
  1,676 episodes.

## Verifying a re-render

`render_stage_b.py` compares every rendered episode's frame count against
`reference_num_frames` and reports mismatches. For a stricter check, compare
`reference_actions` against the `actions` in the regenerated
`annotations.json` — they should be identical for `follower` and `replay`
episodes.

If a handful of episodes differ, you are probably on a different habitat-sim
version; if most differ, check the scene assets and `render_params`. Our
reference environment is habitat-sim 0.3.3 with MP3D `.glb` + sibling
`.navmesh`.

After `build_hist8_future6.py` you should see:

```
wrote merged 80210 = aug 74244 + manual 5966
```

minus the `precise_movement` contribution if you skipped that task, and with a
small delta on `precise_movement` since it is a fresh draw.
