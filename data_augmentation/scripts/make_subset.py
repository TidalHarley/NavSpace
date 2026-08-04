#!/usr/bin/env python3
"""Create a deterministic subset of an aug_mix folder via symlinks.

We don't copy any frames — just symlink each clip directory and write a
trimmed ``annotations.json`` so the dataset code (which resolves
``<vf>/<video>/rgb/``) keeps working.

Usage:
    python data_augmentation/scripts/make_subset.py \
        --source snav_data/aug_mix/precise_movement \
        --target snav_data/aug_mix_subset/precise_movement_200 \
        --n 200 --seed 0
"""
from __future__ import annotations

import argparse
import json
import os
import random
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--source", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    src = Path(args.source).resolve()
    tgt = Path(args.target).resolve()
    items = json.loads((src / "annotations.json").read_text())
    if args.n > len(items):
        raise SystemExit(f"requested {args.n} > available {len(items)}")

    rng = random.Random(args.seed)
    rng.shuffle(items)
    keep = items[: args.n]

    tgt.mkdir(parents=True, exist_ok=True)
    (tgt / "annotations.json").write_text(json.dumps(keep, indent=2, ensure_ascii=False))

    # IMPORTANT: write *relative* symlinks so the layout still resolves
    # when the host dir is mounted into a docker container at a different
    # absolute path. We assume `target` and `source` live under a common
    # ancestor (e.g. both under snav_data/).
    linked = 0
    for it in keep:
        clip = it["video"]
        link = tgt / clip
        if link.exists() or link.is_symlink():
            continue
        rel = os.path.relpath(src / clip, tgt)
        os.symlink(rel, link)
        linked += 1

    print(f"[done] subset {args.n}/{len(items)} -> {tgt}  (created {linked} symlinks)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
