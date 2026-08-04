#!/usr/bin/env python3
"""Convert an aug_mix folder from ``snav_frames`` to SNav stage-1 ``frames`` layout.

The render pipeline writes ``llava_annotations.json`` (LLaVA SFT format,
one entry per step) and flat ``{ep_tag}/{NNN}.jpg`` frames.  The SNav
trainer (``snav_training/dataset_snav.py``) only loads
``annotations.json`` (one entry per episode, with a full ``actions``
list) and frames at ``{ep_tag}/rgb/{NNN+1:03d}.jpg``.

This converter does the migration **in place** without re-rendering:

  1. Group LLaVA entries by episode_id and sort by step.
  2. Recover the raw ``actions`` list:
        actions = [-1, gpt(step=0)[0], gpt(step=1)[0], ..., gpt(step=N-2)[0]]
     (gpt at step i predicts 6 future actions; its first element is the
      action taken from frame i to frame i+1.  The dataset loader will
      append a trailing ``Stop`` action itself.)
  3. Pick the instruction:
        --instructions-source verified.json / custom_instructions.json:
            inject per-episode instruction by id.
        otherwise: extract from the LLaVA prompt text in conversations[0].
  4. Move ``{ep_tag}/000.jpg .. {N-1}.jpg`` → ``{ep_tag}/rgb/001.jpg .. N.jpg``.
  5. Write ``annotations.json`` (one entry per episode).
  6. Rename the old ``llava_annotations.json`` → ``.bak`` (unless --no-backup).

Usage::

    python data_augmentation/scripts/convert_aug_to_sft.py \\
        --folder snav_data/aug_mix/r2r_rewritten \\
        --instructions-source data_augmentation/outputs/vertical_perception/custom_instructions.json

    python data_augmentation/scripts/convert_aug_to_sft.py \\
        --folder snav_data/aug_mix/precise_movement   # instruction extracted from llava prompt

    python data_augmentation/scripts/convert_aug_to_sft.py \\
        --folder snav_data/aug_mix/spatial_relationship
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path


_EPID_RE = re.compile(r"_(\d+)_step_(\d+)$")
_INSTR_RE = re.compile(r"given the instruction:\s*(.*?);\s*\n", re.DOTALL)
_ACTION_MAP = {
    "Move forward": 1,
    "Turn left": 2,
    "Turn right": 3,
    "Stop": 0,
}


def _load_instruction_overrides(path: str | None) -> dict[int, str]:
    if not path or not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    out: dict[int, str] = {}
    if isinstance(payload, dict) and "records" in payload:
        for rec in payload["records"]:
            eid = rec.get("episode_id") or rec.get("id")
            if eid is None:
                continue
            instr = rec.get("instruction") or rec.get("original_instruction")
            if isinstance(instr, list):
                instr = instr[0] if instr else None
            if instr:
                out[int(eid)] = instr.strip()
        return out
    for item in payload:
        eid = item.get("id") or item.get("episode_id")
        if eid is None:
            continue
        instructions = item.get("instructions") or []
        if isinstance(instructions, str):
            instructions = [instructions]
        if instructions:
            out[int(eid)] = instructions[0].strip()
    return out


def _extract_instruction_from_prompt(prompt: str) -> str:
    m = _INSTR_RE.search(prompt or "")
    return m.group(1).strip() if m else ""


def _gpt_first_action(gpt_value: str) -> int:
    text = gpt_value.replace("Final Answer:", "").strip()
    first = text.split(",", 1)[0].strip()
    return _ACTION_MAP.get(first, 0)


def _ep_tag_from_video(video_field: str) -> str:
    return os.path.basename(video_field.rstrip("/"))


def _move_jpgs_to_rgb(ep_dir: Path) -> int:
    """Move 000.jpg, 001.jpg, ... → rgb/001.jpg, 002.jpg, ... Return new frame count."""
    rgb_dir = ep_dir / "rgb"
    if rgb_dir.is_dir() and any(rgb_dir.iterdir()):
        return sum(1 for _ in rgb_dir.iterdir() if _.suffix == ".jpg")
    rgb_dir.mkdir(parents=True, exist_ok=True)
    jpgs = sorted(p for p in ep_dir.iterdir() if p.suffix == ".jpg" and p.is_file())
    for p in jpgs:
        try:
            base = int(p.stem)
        except ValueError:
            continue
        new_name = f"{base + 1:03d}.jpg"
        p.rename(rgb_dir / new_name)
    return sum(1 for _ in rgb_dir.iterdir() if _.suffix == ".jpg")


def convert(folder: Path, instructions_override: dict[int, str],
             *, no_backup: bool, dry_run: bool) -> None:
    llava_path = folder / "llava_annotations.json"
    if not llava_path.exists():
        raise FileNotFoundError(f"{llava_path} not found.")

    with llava_path.open("r", encoding="utf-8") as fh:
        entries = json.load(fh)

    by_ep: dict[int, list[dict]] = defaultdict(list)
    for entry in entries:
        m = _EPID_RE.search(entry.get("id", ""))
        if not m:
            continue
        ep_id = int(m.group(1))
        step = int(m.group(2))
        by_ep[ep_id].append({"step": step, "entry": entry})

    annotations: list[dict] = []
    stats = {"episodes": 0, "frames": 0, "instr_from_override": 0,
              "instr_from_prompt": 0, "skipped_missing_frames": 0}

    for ep_id, steps in by_ep.items():
        steps.sort(key=lambda x: x["step"])
        if not steps:
            continue
        first_entry = steps[0]["entry"]
        ep_tag = _ep_tag_from_video(first_entry.get("video", ""))
        if not ep_tag:
            continue
        ep_dir = folder / ep_tag
        if not ep_dir.is_dir():
            stats["skipped_missing_frames"] += 1
            continue

        # ── instruction selection ──
        if ep_id in instructions_override:
            instruction = instructions_override[ep_id]
            stats["instr_from_override"] += 1
        else:
            instruction = _extract_instruction_from_prompt(
                first_entry["conversations"][0]["value"]
            )
            stats["instr_from_prompt"] += 1

        # ── actions recovery ──
        N = len(steps)
        actions = [-1]
        for i in range(N - 1):
            actions.append(_gpt_first_action(steps[i]["entry"]["conversations"][1]["value"]))

        annotations.append({
            "id": ep_tag,
            "video": ep_tag,
            "instructions": [instruction],
            "actions": actions,
            "render_params": {},
        })
        stats["episodes"] += 1
        stats["frames"] += N

        if not dry_run:
            moved = _move_jpgs_to_rgb(ep_dir)
            if moved != N:
                print(f"  WARN ep {ep_tag}: expected {N} frames, found {moved} after move")

    if dry_run:
        print(f"[dry-run] would convert {stats['episodes']} episodes, {stats['frames']} frames")
        return

    out_path = folder / "annotations.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(annotations, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {out_path}  ({len(annotations)} episodes, "
          f"{stats['frames']} frames total)")
    print(f"  instr_from_override : {stats['instr_from_override']}")
    print(f"  instr_from_prompt   : {stats['instr_from_prompt']}")
    print(f"  skipped_missing_dir : {stats['skipped_missing_frames']}")

    if not no_backup:
        bak = llava_path.with_suffix(llava_path.suffix + ".bak")
        llava_path.rename(bak)
        print(f"Old llava annotations moved to {bak}")
    else:
        llava_path.unlink()
        print(f"Removed {llava_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--folder", required=True,
                        help="aug_mix subfolder containing llava_annotations.json "
                             "and {ep_tag}/{NNN}.jpg frame directories.")
    parser.add_argument("--instructions-source", default=None,
                        help="Optional JSON of {id, instructions} or {records: "
                             "[{episode_id, instruction}]} used to override the "
                             "instruction field per episode.")
    parser.add_argument("--no-backup", action="store_true",
                        help="Delete (instead of .bak) the old llava_annotations.json.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print stats without writing or moving files.")
    args = parser.parse_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        print(f"Folder not found: {folder}", file=sys.stderr)
        return 1

    overrides = _load_instruction_overrides(args.instructions_source)
    if overrides:
        print(f"Loaded {len(overrides)} instruction overrides from "
              f"{args.instructions_source}")

    convert(folder, overrides, no_backup=args.no_backup, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    sys.exit(main())
