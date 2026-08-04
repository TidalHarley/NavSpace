#!/usr/bin/env python3
"""Step 4 — Inject STAY-action variants so the model learns to actually
honour the IF clause.

`3_generate.py` produces four templates (A/B/D/E). All four currently
ship the SAME 'walk-to-destination' trajectory:

* POS templates (A/B): the chosen object IS visible -> condition TRUE
  -> branch = walk to destination. Action = full trajectory.
* NEG templates (D/E): the chosen object is NOT visible -> condition
  FALSE -> 'otherwise' branch = walk to destination. Action = full
  trajectory.

In every case the LABEL is the full walk-to-B sequence. The model never
sees a single (instruction, action) pair where the correct answer is
"stay here / STOP". Empirically this means SFT on env_state will not
improve — and may regress — the model's ability to evaluate the IF
predicate at eval time.

This script generates **paired sibling examples** that swap the branches
so the correct action is STOP. We keep only the start frame (rgb/000.jpg)
and emit actions = [-1, 0]. The new instructions:

* POS-stay (mirror of A/B):
    "Starting from the {start_room}, if {cond_true}, stop where you are. "
    "Otherwise, {orig}."
    -> condition TRUE because cond_true uses a visible object
    -> action = STOP.

* NEG-walk-mirror (mirror of D/E):
    "If there is {unobs_obj} in the {start_room}, {orig}; "
    "otherwise, stop where you are."
    -> condition FALSE because unobs_obj is not visible
    -> action = STOP.

The siblings are appended to the original annotations.json so the dataset
contains a balanced mix of (walk-to-B) and (stop) labels.

Usage:
    python data_augmentation/environment_state/4_balance_stay.py \
        --annotations snav_data/aug_mix/environment_state/annotations.json \
        --review      data_augmentation/outputs/environment_state/review.json \
        --target-stay-ratio 0.45
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _decap(t: str) -> str:
    t = t.lstrip()
    return t[0].lower() + t[1:] if t else t


def _strip_punct(t: str) -> str:
    return re.sub(r"[.!?\s]+$", "", t.strip())


def _build_pos_stay(start_room: str, cond_true: str, orig: str) -> str:
    return (f"Starting from the {start_room}, if {cond_true}, "
            f"stop where you are. Otherwise, {_decap(_strip_punct(orig))}.")


def _build_neg_walk(unobs_phrase: str, start_room: str, orig: str) -> str:
    return (f"If there is {unobs_phrase} in the {start_room}, "
            f"{_decap(_strip_punct(orig))}; otherwise, stop where you are.")


# Re-parse the rewritten instruction emitted by 3_generate.py to recover
# the conditional / object phrasing we need for the mirror sentence.
COND_TRUE_RE_A = re.compile(
    r"^Starting from the (?P<room>[^,]+), if (?P<cond>[^,]+?),",
    re.I)
COND_TRUE_RE_B = re.compile(r"^If (?P<cond>[^,]+?),", re.I)
COND_FALSE_RE_D = re.compile(
    r"^If there is (?P<obj>.+?) in the (?P<room>[^,]+?),", re.I)
COND_FALSE_RE_E = re.compile(
    r"^From where you are right now, if there is (?P<obj>.+?), "
    r"stop in the (?P<room>[^,]+?),", re.I)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--annotations", required=True,
                    help="snav_data/aug_mix/environment_state/annotations.json")
    ap.add_argument("--review", default="",
                    help="optional review.json from 3_generate.py "
                         "(provides template/start_room — speeds parsing)")
    ap.add_argument("--target-stay-ratio", type=float, default=0.45,
                    help="approximate stay/total ratio after balancing")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    anno_path = Path(args.annotations)
    items: list[dict[str, Any]] = json.loads(anno_path.read_text())
    orig_n = len(items)

    # Build a lookup ep_id -> review entry so we know template / start_room
    # without re-parsing. Falls back to regex parsing if review missing.
    review_map: dict[str, dict[str, Any]] = {}
    if args.review:
        rev_path = Path(args.review)
        if rev_path.exists():
            for r in json.loads(rev_path.read_text()):
                review_map[str(r.get("episode_id"))] = r

    import random
    rng = random.Random(args.seed)
    rng.shuffle(items)

    target_stay = int(orig_n * args.target_stay_ratio / (1 - args.target_stay_ratio))
    added = 0
    by_kind = {"pos_stay": 0, "neg_walk": 0}

    new_items: list[dict[str, Any]] = []
    for it in items:
        if added >= target_stay:
            break
        instr = it["instructions"][0]
        ep_tag = it["video"]                 # e.g. "<scene>_env_state_<id>"
        # Recover original R2R body. The fastest way: take the text inside
        # the comma-separated `orig` portion of the rewritten instr.
        # We use review_map when available, else fall back to a regex.
        rev = review_map.get(str(it.get("id")), {})
        template = rev.get("template", "")
        start_room = (rev.get("start_room") or "").strip().lower() or "room"
        orig_body = (rev.get("original") or "").strip()

        if not template:
            if m := COND_TRUE_RE_A.search(instr):
                template = "A_starting_if"
                start_room = m.group("room").strip().lower()
            elif COND_TRUE_RE_B.search(instr):
                template = "B_if_then"
            elif COND_FALSE_RE_D.search(instr):
                template = "D_if_stay"
            elif COND_FALSE_RE_E.search(instr):
                template = "E_from_here"

        # Try to recover cond_true / unobs_phrase + orig_body from the
        # rewritten instruction. Body lives BETWEEN the IF clause and the
        # OTHERWISE clause (e.g.
        # "Starting from the hallway, if X, <body>. Otherwise, stop in the hallway.").
        cond_true, unobs_phrase = None, None
        parsed_body = None
        if template == "A_starting_if":
            m = re.search(
                r"Starting from the (?P<room>[^,]+), if (?P<cond>[^,]+?),\s*"
                r"(?P<body>.+?)(?:\s*\.\s*Otherwise[, ].*)?$",
                instr, re.I | re.S)
            if m:
                cond_true = m.group("cond").strip()
                start_room = m.group("room").strip().lower()
                parsed_body = m.group("body").strip(" .")
        elif template == "B_if_then":
            m = re.search(
                r"^If (?P<cond>[^,]+?),\s*(?P<body>.+?)(?:;\s*otherwise[, ].*)?$",
                instr, re.I | re.S)
            if m:
                cond_true = m.group("cond").strip()
                parsed_body = m.group("body").strip(" .")
        elif template == "D_if_stay":
            m = re.search(
                r"^If there is (?P<obj>.+?) in the (?P<room>[^,]+?), stop where you are;\s*"
                r"otherwise,\s*(?P<body>.+?)\.?$", instr, re.I | re.S)
            if m:
                unobs_phrase = m.group("obj").strip()
                start_room = m.group("room").strip().lower()
                parsed_body = m.group("body").strip(" .")
        elif template == "E_from_here":
            m = re.search(
                r"^From where you are right now, if there is (?P<obj>.+?), "
                r"stop in the (?P<room>[^,]+?);\s*otherwise,\s*(?P<body>.+?)\.?$",
                instr, re.I | re.S)
            if m:
                unobs_phrase = m.group("obj").strip()
                start_room = m.group("room").strip().lower()
                parsed_body = m.group("body").strip(" .")

        if not orig_body:
            orig_body = parsed_body or instr

        if template in {"A_starting_if", "B_if_then"} and cond_true:
            new_instr = _build_pos_stay(start_room, cond_true, orig_body)
            kind = "pos_stay"
        elif template in {"D_if_stay", "E_from_here"} and unobs_phrase:
            new_instr = _build_neg_walk(unobs_phrase, start_room, orig_body)
            kind = "neg_walk"
        else:
            continue

        new_id = f"{it['id']}_stay"
        new_video = it["video"]            # re-use existing rgb/ dir
        new_items.append({
            "id": new_id,
            "video": new_video,
            "instructions": [new_instr],
            # init + STOP. Dataset_snav builds frames from rgb/000.. only
            # up to len(actions)-1, so it will use rgb/000.jpg as the only
            # visual context, which is exactly the start frame.
            "actions": [-1, 0],
            "render_params": it.get("render_params", {}),
            "source_episode": it["id"],
            "balance_tag": kind,
        })
        by_kind[kind] += 1
        added += 1

    if not new_items:
        print("[warn] no stay siblings generated — instructions could not be parsed.")
        return 1

    items.extend(new_items)
    backup = anno_path.with_suffix(anno_path.suffix + ".prebalance.bak")
    if not backup.exists():
        backup.write_text(json.dumps(json.loads(anno_path.read_text()),
                                      indent=2, ensure_ascii=False))
        print(f"[backup] wrote {backup}")
    anno_path.write_text(json.dumps(items, indent=2, ensure_ascii=False))

    print(f"[done] base={orig_n}  added={added}  total={len(items)}  by_kind={by_kind}")
    print(f"       stay/total = {added/len(items)*100:.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
