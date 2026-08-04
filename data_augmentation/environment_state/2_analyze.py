#!/usr/bin/env python3
"""Step 2 — Run Qwen-VL on already-rendered trajectories.

For each candidate episode, picks N=cfg.environment_state.frames_for_vl
frames (first / middle / last) from the rendered snav_frames output and
asks Qwen-VL to extract a decisively-true observable condition AND a
decisively-false unobservable object, suitable for IF/OTHERWISE wrapping.

Inputs
------
* ``data_augmentation/outputs/environment_state/candidates.json``
* rendered frames at ``<frames_root>/<ep_tag>/<NNN>.jpg``
  (snav_frames output mode, *before* convert_aug_to_sft.py runs)

Outputs
-------
* ``states.jsonl`` (resumable per-episode checkpoint)
* ``states.json``  (final aggregated)

Decisive-condition prompt enforces:
  - ``observable_objects`` items are 100% clearly visible in the given frames
  - ``unobservable_objects`` items are decisively-not-here (clearly do not
    belong in the room types shown — never marginal cases)
  - the model produces one ``condition_phrase`` (positive predicate built
    from an observable) and one ``decisive_unobs_object``
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_augmentation.common import (  # noqa: E402
    checkpoint_append,
    checkpoint_iter,
    checkpoint_load_ids,
    load_config,
    load_json,
    pipeline_output_dir,
    qwen_json,
    save_json,
)


ANALYZE_SYSTEM = (
    "You analyze indoor navigation frames to build IF/OTHERWISE "
    "instructions. Return strict JSON only — no prose, no markdown fences."
)

ANALYZE_PROMPT = """You are given {n_frames} RGB frames captured by an agent
following a navigation trajectory in an indoor scene.  Frame 1 is the
STARTING view, frame {n_frames} is the FINAL view, frames in between are
intermediate viewpoints along the way.

Your job is to extract concise English phrases for building an
IF/OTHERWISE navigation instruction in NavSpace style.  YOU MUST think
for every individual scene — DO NOT copy fixed example items.

  RULES
  - ``observable_objects`` must be objects you can CLEARLY see in the
    provided frames — not guesses, not "probably there".  Each entry
    should be a 3-7 word noun phrase that names a salient object and
    optionally its support or colour, e.g. ``a wooden dining table``,
    ``a beige sofa in the living room``, ``a framed painting above the
    bed``, ``a stainless-steel fridge``.  Provide 3-6 items.

  - ``unobservable_objects`` must be items that DEFINITELY DO NOT
    belong in this particular scene.  Look at the rooms involved and
    pick objects that would be unmistakably out of place in those
    rooms — VARY YOUR CHOICES according to the scene.  Examples of
    valid out-of-place items by scene type:
        * bedroom / hallway → an industrial conveyor belt, a chemistry
          fume hood, an MRI scanner, a tractor tire, a basketball hoop
        * bathroom          → a grand piano, a snowmobile, a barbecue
          grill, a hay bale
        * kitchen           → a hospital bed, a surfboard, a satellite
          dish, a pinball machine
        * living room       → a forklift, an industrial sewing machine,
          an x-ray machine
    DO NOT default to ``a pool table`` or ``gym equipment`` for every
    scene.  Choose freshly each time.  Provide 2-4 items.

  - ``condition_phrase`` is a single POSITIVE predicate built from one
    ``observable_objects`` entry, with the structure ``there is/are
    <noun> in the <room>`` or ``there is/are <noun> on the <support>``.
    Examples:
        "there is a beige sofa in the living room"
        "there are framed paintings on the wall"
    Avoid spatial-only phrases like "on the right" — use room or
    support nouns.

  - ``decisive_unobs_object`` is one short noun phrase chosen from
    ``unobservable_objects`` that a human would immediately recognise
    as not belonging to THIS scene.  Pick a different item for
    different episodes.

Original navigation hint (may help disambiguate the rooms): {instruction}

Return JSON exactly with these fields and nothing else:
{{
  "start_room": "lower-cased room type, e.g. bedroom / kitchen / living room / hallway / bathroom",
  "end_room":   "lower-cased room type",
  "landmarks":  ["short landmark phrases visible in any frame"],
  "observable_objects":   ["...", "...", "..."],
  "unobservable_objects": ["...", "..."],
  "condition_phrase":     "there is/are <noun> in the <room|support>",
  "decisive_unobs_object": "<short noun phrase>"
}}
"""


def _load_frame(folder: Path, idx: int) -> np.ndarray | None:
    fp = folder / f"{idx:03d}.jpg"
    if not fp.exists():
        return None
    return np.array(Image.open(fp).convert("RGB"))


def _select_frames(ep_dir: Path, n: int) -> list[np.ndarray]:
    jpgs = sorted(p for p in ep_dir.iterdir() if p.suffix == ".jpg" and p.is_file())
    if not jpgs:
        return []
    total = len(jpgs)
    if n >= total:
        idxs = list(range(total))
    elif n == 1:
        idxs = [0]
    elif n == 2:
        idxs = [0, total - 1]
    else:
        step = (total - 1) / (n - 1)
        idxs = sorted({int(round(step * i)) for i in range(n)})
        if idxs[-1] != total - 1:
            idxs[-1] = total - 1
    frames: list[np.ndarray] = []
    for i in idxs:
        arr = np.array(Image.open(jpgs[i]).convert("RGB"))
        frames.append(arr)
    return frames


def _scan_from_scene_id(scene_id: str) -> str:
    return scene_id.replace("\\", "/").split("/")[-2]


def _ep_tag(scene_id: str, video_subdir: str, episode_id: int) -> str:
    scan = _scan_from_scene_id(scene_id)
    return f"{scan}_{video_subdir}_{int(episode_id):06d}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        default=str(Path(__file__).resolve().parents[1] / "config.json"))
    parser.add_argument("--input", default=None,
                        help="candidates.json (defaults to outputs/environment_state/candidates.json)")
    parser.add_argument("--frames-root", default=None,
                        help="Directory containing rendered <ep_tag>/<NNN>.jpg "
                             "(defaults to snav_data/aug_mix/environment_state).")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--restart", action="store_true",
                        help="Ignore existing states.jsonl checkpoint and start over.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    ecfg = cfg["environment_state"]
    out_dir = pipeline_output_dir(cfg, "environment_state")
    input_path = Path(args.input) if args.input else out_dir / "candidates.json"
    payload = load_json(input_path)
    records = payload["records"] if isinstance(payload, dict) else payload
    if args.max_episodes:
        records = records[: args.max_episodes]

    frames_root = Path(args.frames_root) if args.frames_root else (
        Path(ROOT) / ecfg.get("output_dir_rel", "snav_data/aug_mix/environment_state")
    )
    video_subdir = ecfg.get("video_subdir", "env_state")
    n_frames = int(ecfg.get("frames_for_vl", 3))

    checkpoint_path = out_dir / "states.jsonl"
    final_path = out_dir / "states.json"
    if args.restart and checkpoint_path.exists():
        checkpoint_path.unlink()
    done_ids = checkpoint_load_ids(checkpoint_path)
    print(f"Resuming with {len(done_ids)} episodes already analyzed.")
    print(f"frames_root = {frames_root}  video_subdir = {video_subdir}  n_frames = {n_frames}")

    skip_missing = skip_qwen = ok = 0

    for rec in records:
        ep_id = int(rec["episode_id"])
        if ep_id in done_ids:
            continue
        ep_tag = _ep_tag(rec["scene_id"], video_subdir, ep_id)
        ep_dir = frames_root / ep_tag
        if not ep_dir.is_dir():
            skip_missing += 1
            continue
        frames = _select_frames(ep_dir, n_frames)
        if len(frames) < 2:
            skip_missing += 1
            continue

        try:
            analysis = qwen_json(
                cfg,
                ANALYZE_PROMPT.format(
                    n_frames=len(frames),
                    instruction=rec["original_instruction"],
                ),
                system=ANALYZE_SYSTEM,
                images=frames,
                model=cfg["qwen"]["vision_model"],
            )
        except Exception as exc:  # noqa: BLE001
            skip_qwen += 1
            print(f"  Qwen-VL failed for ep {ep_id}: {exc}")
            continue

        record = {
            "episode_id": ep_id,
            "scene_id": rec["scene_id"],
            "ep_tag": ep_tag,
            "start_position": rec["start_position"],
            "start_rotation": rec["start_rotation"],
            "goal_position": rec["goal_position"],
            "original_instruction": rec["original_instruction"],
            "analysis": analysis,
        }
        checkpoint_append(checkpoint_path, record)
        done_ids.add(ep_id)
        ok += 1
        if ok % 10 == 0:
            print(f"  analysed {ok} new episodes (total done={len(done_ids)})")

    records_out = list(checkpoint_iter(checkpoint_path))
    save_json({
        "pipeline": "environment_state",
        "count": len(records_out),
        "records": records_out,
    }, final_path)
    print(f"Saved {len(records_out)} analyzed records -> {final_path}")
    print(f"  ok={ok}  skip_missing_frames={skip_missing}  skip_qwen_error={skip_qwen}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
