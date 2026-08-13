#!/usr/bin/env python3
"""Re-render the SNav Stage-B training set from the released episode list.

Takes ``snav_training/stage_b/stage_b_episodes.json`` — one fused R2R-CE style
file covering every Stage-B episode — plus your own MP3D copy, and rebuilds the
``snav_data/`` tree that ``build_hist8_future6.py`` consumes:

    snav_data/aug_mix/{task}/{ep_tag}/rgb/NNN.jpg
    snav_data/aug_mix/{task}/annotations.json
    snav_data/manual_98/{ep_tag}/rgb/NNN.jpg
    snav_data/manual_98/annotations.json

Each episode carries a ``navspace.render_mode`` that selects the path:

``follower``  geodesic follower to ``goals[0].position`` — the three R2R-derived
              tasks. Reproduces the published frames exactly.
``replay``    replays a human ``action_sequence`` — ``manual_98``. Reproduces the
              published frames exactly.
``resample``  ``precise_movement``. Start poses were drawn from habitat-sim's
              internal RNG during the original run and never persisted, so these
              cannot be replayed; they are regenerated from a fixed seed instead.
              Statistically equivalent, not frame-identical to the paper draw.

Usage:
    python snav_training/scripts/render_stage_b.py \
        --episodes snav_training/stage_b/stage_b_episodes.json \
        --scenes-root /path/to/mp3d \
        --out-root snav_data
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
RENDERER = REPO_ROOT / "snav_training/data_generation/render_streamvln.py"
MANUAL_CONVERTER = REPO_ROOT / "data_augmentation/scripts/convert_manual_to_sft.py"
PRECISE_RUNNER = REPO_ROOT / "data_augmentation/precise_movement/run.py"

AUG_TASKS = ("environment_state", "spatial_relationship", "vertical_perception")


def load(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def scan_of(scene_id: str) -> str:
    return scene_id.replace("\\", "/").split("/")[-2]


def abs_scene(scene_id: str, scenes_root: Path) -> Path:
    """Resolve a released relative ``mp3d/<scan>/<scan>.glb`` against a local root."""
    scan = scan_of(scene_id)
    for cand in (scenes_root / scan / f"{scan}.glb",
                 scenes_root / scene_id.replace("mp3d/", ""),
                 scenes_root / scene_id):
        if cand.is_file():
            return cand
    raise FileNotFoundError(
        f"scene {scan} not found under {scenes_root} — expected "
        f"{scenes_root / scan / f'{scan}.glb'}")


def run(cmd: list[str], *, dry: bool) -> None:
    printable = " ".join(str(c) for c in cmd)
    print(f"    $ {printable}", flush=True)
    if dry:
        return
    subprocess.run([str(c) for c in cmd], check=True, cwd=REPO_ROOT)


# ── follower tasks ──────────────────────────────────────────────────────────
def render_follower_task(
    task: str,
    episodes: list[dict],
    *,
    params: dict,
    scenes_root: Path,
    out_root: Path,
    gpu: int,
    dry: bool,
) -> Path:
    """Render one aug task and normalise into the Stage-B layout."""
    tag = episodes[0]["navspace"]["dataset_tag"]
    dest = out_root / "aug_mix" / task
    dest.mkdir(parents=True, exist_ok=True)

    staging = Path(tempfile.mkdtemp(prefix=f"stageb_{task}_"))
    ep_json = staging / "episodes.json"
    with open(ep_json, "w", encoding="utf-8") as f:
        json.dump({"episodes": episodes}, f, ensure_ascii=False)

    print(f"  [{task}] {len(episodes)} episodes, dataset_tag={tag}")
    try:
        run([
            sys.executable, RENDERER,
            "--data_json", ep_json,
            "--data_format", "r2r",
            "--dataset_tag", tag,
            "--scenes_root", scenes_root,
            "--output_dir", staging / "render",
            "--output_mode", "frames",
            "--width", params["width"],
            "--height", params["height"],
            "--hfov", params["hfov"],
            "--camera_height", params["camera_height"],
            "--forward_step", params["forward_step"],
            "--turn_angle", params["turn_angle"],
            "--goal_radius", params["goal_radius"],
            "--max_steps", params["max_steps"],
            "--gpu_device_id", gpu,
        ], dry=dry)
        if dry:
            return dest

        # frames mode writes render/images/{ep_tag}/rgb/NNN.jpg and annotations
        # rows keyed by int episode_id with video="images/{ep_tag}". Stage-B
        # wants the episode dirs at the task root and both id and video set to
        # the ep_tag.
        images = staging / "render" / "images"
        rows: list[dict[str, Any]] = []
        for row in load(staging / "render" / "annotations.json"):
            ep_tag = os.path.basename(str(row.get("video") or row.get("id")))
            src = images / ep_tag
            if not src.is_dir():
                print(f"    [warn] missing frames for {ep_tag}")
                continue
            dst = dest / ep_tag
            if dst.exists():
                shutil.rmtree(dst)
            shutil.move(str(src), str(dst))
            rows.append({
                "id": ep_tag,
                "video": ep_tag,
                "instructions": row["instructions"],
                "actions": row["actions"],
                "render_params": {},
            })
        write_annotations(dest, rows)
        print(f"    -> {len(rows)} episodes in {dest}")
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return dest


def write_annotations(folder: Path, rows: list[dict]) -> None:
    with open(folder / "annotations.json", "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


# ── manual replay ───────────────────────────────────────────────────────────
def render_manual(
    episodes: list[dict],
    *,
    params: dict,
    scenes_root: Path,
    out_root: Path,
    gpu: int,
    dry: bool,
) -> Path:
    dest = out_root / "manual_98"
    dest.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="stageb_manual_"))
    traj = staging / "trajectories.json"

    # convert_manual_to_sft resolves scene_id by os.path.isfile, so the released
    # relative path has to be expanded against the local MP3D root here.
    payload = {"episodes": [
        {
            "episode_id": e["episode_id"],
            "trajectory_id": e.get("trajectory_id", e["episode_id"]),
            "scene_id": str(abs_scene(e["scene_id"], scenes_root)),
            "start_position": e["start_position"],
            "start_rotation": e["start_rotation"],
            "goals": e.get("goals") or [],
            "reference_path": e.get("reference_path"),
            "instruction": {"instruction_text":
                            e["instruction"]["instruction_text"]},
            "action_sequence": e["navspace"]["action_sequence"],
        }
        for e in episodes
    ]}
    with open(traj, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False)

    print(f"  [manual_98] {len(episodes)} episodes, replaying action_sequence")
    try:
        run([
            sys.executable, MANUAL_CONVERTER,
            "--input", traj,
            "--output", dest,
            "--width", params["width"],
            "--height", params["height"],
            "--hfov", params["hfov"],
            "--forward-step", params["forward_step"],
            "--turn-angle", params["turn_angle"],
            "--camera-height", params["camera_height"],
            "--task-tag", episodes[0]["navspace"]["dataset_tag"],
            "--gpu-device-id", gpu,
        ], dry=dry)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return dest


# ── balanced STOP siblings ──────────────────────────────────────────────────
def apply_balance_variants(episodes: list[dict], out_root: Path) -> int:
    """Append the released ``_stay`` rows to their task's annotations.json.

    The published variants are used verbatim rather than re-derived by
    4_balance_stay.py, whose selection depends on input ordering.
    """
    added = 0
    by_task: dict[str, list[dict]] = {}
    for e in episodes:
        ns = e["navspace"]
        for v in ns.get("balance_variants") or []:
            by_task.setdefault(ns["task"], []).append({
                "id": v["id"],
                "video": v["reuses_frames_of"],
                "instructions": [v["instruction_text"]],
                "actions": v["actions"],
                "render_params": {},
                "source_episode": v["reuses_frames_of"],
                "balance_tag": v.get("balance_tag"),
            })

    for task, rows in by_task.items():
        anno = out_root / "aug_mix" / task / "annotations.json"
        if not anno.is_file():
            print(f"  [warn] {anno} missing — skipping {len(rows)} variants")
            continue
        existing = load(anno)
        have = {r.get("id") for r in existing}
        fresh = [r for r in rows if r["id"] not in have]
        rendered = {r.get("video") for r in existing}
        fresh = [r for r in fresh if r["video"] in rendered]
        existing.extend(fresh)
        write_annotations(anno.parent, existing)
        added += len(fresh)
        print(f"  [{task}] +{len(fresh)} balanced STOP siblings "
              f"-> {len(existing)} annotations")
    return added


# ── verification ────────────────────────────────────────────────────────────
def verify(episodes: list[dict], out_root: Path) -> int:
    """Compare rendered frame counts against the published reference counts."""
    print("\n== verification ==")
    problems = 0
    for e in episodes:
        ns = e["navspace"]
        task = ns["task"]
        if ns["render_mode"] == "resample":
            continue
        base = out_root / ("manual_98" if task == "manual_98"
                           else f"aug_mix/{task}")
        rgb = base / ns["ep_tag"] / "rgb"
        n = len(list(rgb.iterdir())) if rgb.is_dir() else -1
        want = ns.get("reference_num_frames")
        if n != want:
            problems += 1
            if problems <= 10:
                print(f"  [diff] {ns['ep_tag']}: rendered {n} frames, "
                      f"reference {want}")
    if problems:
        print(f"  {problems} episodes differ from the reference frame counts.")
        print("  A small number usually means a habitat-sim version difference; "
              "a large number means the scene assets or render params differ.")
    else:
        print("  every rendered episode matches its reference frame count")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--episodes", type=Path,
                    default=REPO_ROOT / "snav_training/stage_b/stage_b_episodes.json")
    ap.add_argument("--scenes-root", type=Path, required=True,
                    help="local MP3D root containing <scan>/<scan>.glb + .navmesh")
    ap.add_argument("--out-root", type=Path, default=REPO_ROOT / "snav_data")
    ap.add_argument("--tasks", nargs="+",
                    default=list(AUG_TASKS) + ["manual_98", "precise_movement"])
    ap.add_argument("--max-episodes", type=int, default=0,
                    help="per-task cap, for smoke tests (0 = all)")
    ap.add_argument("--gpu-device-id", type=int, default=0)
    ap.add_argument("--skip-verify", action="store_true")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the commands without rendering")
    args = ap.parse_args()

    payload = load(args.episodes)
    params = payload["render_params"]
    episodes = payload["episodes"]
    out_root = args.out_root.resolve()
    print(f"episode list : {args.episodes}  ({len(episodes)} episodes)")
    print(f"scenes root  : {args.scenes_root}")
    print(f"output root  : {out_root}")
    print(f"render params: {params}\n")

    by_task: dict[str, list[dict]] = {}
    for e in episodes:
        by_task.setdefault(e["navspace"]["task"], []).append(e)

    selected: list[dict] = []
    for task in args.tasks:
        eps = by_task.get(task)
        if not eps:
            print(f"  [skip] no episodes for task {task}")
            continue
        if args.max_episodes:
            eps = eps[: args.max_episodes]
        selected.extend(eps)

        mode = eps[0]["navspace"]["render_mode"]
        if mode == "follower":
            render_follower_task(
                task, eps, params=params, scenes_root=args.scenes_root,
                out_root=out_root, gpu=args.gpu_device_id, dry=args.dry_run)
        elif mode == "replay":
            render_manual(
                eps, params=params, scenes_root=args.scenes_root,
                out_root=out_root, gpu=args.gpu_device_id, dry=args.dry_run)
        elif mode == "resample":
            print(f"  [{task}] {len(eps)} episodes are resample-only.")
            print("    Start poses were never persisted, so these are "
                  "regenerated from a fixed seed instead of replayed:")
            print(f"    $ {sys.executable} {PRECISE_RUNNER} "
                  f"--output-root {out_root / 'aug_mix' / task} --seed 42")
            print(f"    $ {sys.executable} "
                  f"{REPO_ROOT / 'data_augmentation/scripts/convert_aug_to_sft.py'} "
                  f"--folder {out_root / 'aug_mix' / task}")

    if not args.dry_run:
        print("\n== balanced STOP siblings ==")
        apply_balance_variants(selected, out_root)

    rc = 0
    if not args.dry_run and not args.skip_verify:
        rc = 1 if verify(selected, out_root) else 0

    print("\n== next step ==")
    print(f"  python snav_training/scripts/build_hist8_future6.py \\")
    print(f"    --aug-root {out_root / 'aug_mix'} \\")
    print(f"    --manual-root {out_root / 'manual_98'} \\")
    print(f"    --out-dir train_data")
    print(f"  expected rows: {payload.get('expected_training_rows')}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
