#!/usr/bin/env python3
"""Replace the instruction substring inside an existing llava_annotations.json.

Given:
  * ``--llava`` : llava_annotations.json that was rendered with the
                  ORIGINAL R2R text in each human turn,
  * ``--originals`` : verified.json containing ``original_instruction``
                       per episode_id (used to locate the exact substring),
  * ``--rewrites`` : merged custom_instructions.json (or any
                      ``[{"id": int, "instructions": [str]}]`` list) with
                      the new floor-aware instruction,

this script rewrites the conversation's human ``value`` field by replacing
the original instruction substring with the rewritten one.  All other
fields (video, video_nframes, gpt action target, etc.) are untouched.

Episode_id is recovered from the ``id`` field, which the renderer formats
as ``{video_subdir}_{ep_id:06d}_step_{step_idx}`` (see
``render_streamvln.py``).  When the substring is not found (e.g. the
renderer wrapped the text differently) we additionally try a fuzzy
prefix match: locate any phrase starting with the first 12 characters of
the original and ending at the next "; \n" boundary.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path


_EPID_RE = re.compile(r"_([0-9]{6})_step_")


def _episode_id_from_entry_id(entry_id: str) -> int | None:
    m = _EPID_RE.search(entry_id)
    if not m:
        return None
    return int(m.group(1))


def _load_originals(path: str) -> dict[int, str]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    records = payload.get("records", payload if isinstance(payload, list) else [])
    out: dict[int, str] = {}
    for rec in records:
        eid = rec.get("episode_id")
        if eid is None:
            continue
        original = rec.get("original_instruction") or rec.get("instruction")
        if not original:
            continue
        out[int(eid)] = original.strip()
    return out


def _load_rewrites(path: str) -> dict[int, str]:
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    if isinstance(payload, dict) and "records" in payload:
        records = payload["records"]
        out: dict[int, str] = {}
        for rec in records:
            eid = rec.get("episode_id") or rec.get("id")
            if eid is None:
                continue
            inst = rec.get("instruction") or (rec.get("instructions") or [None])[0]
            if inst:
                out[int(eid)] = inst.strip()
        return out
    out2: dict[int, str] = {}
    for item in payload:
        eid = item.get("id")
        if eid is None:
            continue
        instructions = item.get("instructions") or []
        if instructions:
            out2[int(eid)] = instructions[0].strip()
    return out2


def _swap_in_text(value: str, original: str, new: str) -> tuple[str, str]:
    """Return (new_value, mode). Mode is exact / fuzzy / unchanged."""
    if not value:
        return value, "unchanged"
    if original in value:
        return value.replace(original, new, 1), "exact"

    # Fuzzy: prompts wrap the instruction in
    # "given the instruction: <INSTR>;\n            You are given ..."
    m = re.search(r"given the instruction:\s*(.*?);\s*\n", value, re.DOTALL)
    if m and m.group(1).strip()[:20] == original.strip()[:20]:
        head, tail = value[: m.start(1)], value[m.end(1):]
        return f"{head}{new}{tail}", "fuzzy"
    return value, "unchanged"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--llava", required=True,
                        help="llava_annotations.json to patch in-place "
                             "(a backup is written alongside).")
    parser.add_argument("--originals", required=True,
                        help="verified.json with original R2R texts.")
    parser.add_argument("--rewrites", required=True,
                        help="custom_instructions.json or output.json with "
                             "the rewritten instructions.")
    parser.add_argument("--no-backup", action="store_true")
    args = parser.parse_args()

    llava_path = Path(args.llava)
    originals = _load_originals(args.originals)
    rewrites = _load_rewrites(args.rewrites)

    with llava_path.open("r", encoding="utf-8") as fh:
        entries = json.load(fh)

    if not args.no_backup:
        backup = llava_path.with_suffix(llava_path.suffix + ".bak")
        backup.write_text(json.dumps(entries, ensure_ascii=False, indent=2))
        print(f"Backup -> {backup}")

    stats = {"patched_exact": 0, "patched_fuzzy": 0, "missing_eid": 0,
             "no_rewrite": 0, "no_original": 0, "no_match": 0}
    for entry in entries:
        eid = _episode_id_from_entry_id(entry.get("id", ""))
        if eid is None:
            stats["missing_eid"] += 1
            continue
        if eid not in rewrites:
            stats["no_rewrite"] += 1
            continue
        if eid not in originals:
            stats["no_original"] += 1
            continue
        original = originals[eid]
        new = rewrites[eid]
        for turn in entry.get("conversations", []):
            if turn.get("from") != "human":
                continue
            new_value, mode = _swap_in_text(turn.get("value", ""), original, new)
            if mode == "exact":
                turn["value"] = new_value
                stats["patched_exact"] += 1
            elif mode == "fuzzy":
                turn["value"] = new_value
                stats["patched_fuzzy"] += 1
            else:
                stats["no_match"] += 1

    with llava_path.open("w", encoding="utf-8") as fh:
        json.dump(entries, fh, indent=2, ensure_ascii=False)

    print(f"Patched {llava_path}")
    for k, v in stats.items():
        print(f"  {k:18s} {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
