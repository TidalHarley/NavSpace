#!/usr/bin/env python3
"""Merge per-pipeline custom_instructions into one master JSON.

Notes:

- ``precise_movement`` is rendered directly by ``precise_movement/run.py``
  into ``snav_data/aug_mix/precise_movement/`` and does NOT enter the merge.
- ``spatial_relationship`` is also rendered directly into
  ``snav_data/aug_mix/spatial_relationship/`` and does NOT enter the merge
  (its instructions are the *original* R2R sentences — the renderer can
  consume R2R train directly without a custom-instructions file).
- ``environment_state`` is still under construction and is NOT in the
  default priority either; once it lands you can pass it via
  ``--priority vertical_perception environment_state``.

Therefore the merge currently produces a file that only contains
``vertical_perception`` rewrites; ``run_render_aug.sh`` will then render
them under ``snav_data/aug_mix/r2r_rewritten/``.

Collision policy:
  * default: priority order (first hit wins, one instruction per episode_id)
  * ``--merge-instructions``: keep ALL rewrites for the same id as a list.
"""

from __future__ import annotations

import argparse
import sys
from collections import OrderedDict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
REPO_ROOT = ROOT.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from data_augmentation.common import load_config, load_json, save_json  # noqa: E402

DEFAULT_PRIORITY = ["vertical_perception"]


def _load_pipeline_custom(path: Path) -> list[dict]:
    if not path.exists():
        return []
    data = load_json(path)
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        records = data.get("records", [])
        out = []
        for rec in records:
            instr = rec.get("instruction")
            if isinstance(instr, list):
                out.append({"id": rec["episode_id"], "instructions": instr})
            elif instr:
                out.append({"id": rec["episode_id"], "instructions": [instr]})
        return out
    return []


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(ROOT / "config.json"))
    parser.add_argument("--priority", nargs="*", default=DEFAULT_PRIORITY,
                        help="Pipeline order; the first pipeline that has a "
                             "rewrite for an episode wins (unless "
                             "--merge-instructions).")
    parser.add_argument("--merge-instructions", action="store_true",
                        help="Keep ALL rewrites per episode_id, concatenated "
                             "into the 'instructions' list.")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    output_root = REPO_ROOT / cfg["paths"]["output_root"]

    merged: "OrderedDict[int, list[str]]" = OrderedDict()
    sources: dict[int, list[str]] = {}

    for pipeline in args.priority:
        path = output_root / pipeline / "custom_instructions.json"
        items = _load_pipeline_custom(path)
        print(f"{pipeline}: {len(items)} items <- {path}")
        for item in items:
            eid = int(item["id"])
            instructions = list(item.get("instructions") or [])
            if not instructions:
                continue
            if eid in merged:
                if args.merge_instructions:
                    for instr in instructions:
                        if instr not in merged[eid]:
                            merged[eid].append(instr)
                    sources[eid].append(pipeline)
                # else: keep the first (higher priority) entry
                continue
            merged[eid] = instructions
            sources[eid] = [pipeline]

    out_records = [{"id": eid, "instructions": instrs} for eid, instrs in merged.items()]

    out_path = Path(args.out) if args.out else output_root / "merged" / "custom_instructions.json"
    save_json(out_records, out_path)

    sources_path = out_path.parent / "sources.json"
    save_json(
        {str(eid): srcs for eid, srcs in sources.items()},
        sources_path,
    )
    print(f"Merged {len(out_records)} episodes -> {out_path}")
    print(f"Source map -> {sources_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
