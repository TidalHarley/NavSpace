#!/usr/bin/env python3
"""Build a custom_instructions.json that just passes the ORIGINAL R2R text through.

Usage:
    python data_augmentation/scripts/build_passthrough_instructions.py \\
        --verified data_augmentation/outputs/vertical_perception/verified.json \\
        --out data_augmentation/outputs/vertical_perception/passthrough_instructions.json

The renderer (snav_training/data_generation/render_streamvln.py) uses the
``--custom_instructions_json`` flag for two things at once:

  1. it filters R2R episodes down to the listed ``id``s, and
  2. it replaces each episode's instruction with the one(s) provided.

We want to RENDER FIRST (slow, GPU-bound) using the original R2R text,
then REWRITE AFTER (fast, Qwen API) and patch the LLaVA annotations.  To
make the first render step only touch the verified set without yet
changing any text, this helper emits a "passthrough" custom_instructions
file: same episode ids, but each ``instructions`` list contains the
original R2R instruction unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verified", required=True,
                        help="Path to verified.json (output of 2_verify.py).")
    parser.add_argument("--out", required=True,
                        help="Where to write the passthrough JSON.")
    args = parser.parse_args()

    with open(args.verified, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    records = payload.get("records", payload if isinstance(payload, list) else [])

    out = [
        {"id": int(rec["episode_id"]),
         "instructions": [rec["original_instruction"].strip()]}
        for rec in records
        if rec.get("original_instruction")
    ]

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"Wrote {len(out)} passthrough instructions -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
