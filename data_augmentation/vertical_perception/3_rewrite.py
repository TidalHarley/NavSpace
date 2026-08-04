#!/usr/bin/env python3
"""Replace vague stair phrases inside R2R instructions with concrete floor mentions.
"""

from __future__ import annotations

import argparse
import json as _json
import random
import re
import sys
from pathlib import Path
from typing import Any, Optional

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_augmentation.common import (  # noqa: E402
    load_config,
    load_json,
    ordinal_word,
    pipeline_output_dir,
    qwen_call,
    save_json,
    to_custom_instructions,
    token_overlap_ratio,
)


# ─── stair-phrase regex (used by the deterministic fallback) ───────────

_VERBS = r"(?:walk|go|head|move|come|continue|proceed|do)"
_STAIRS = r"(?:stair(?:s|case|way)?|steps?)"
_FLIGHT = r"(?:flight|set)\s+of\s+stairs?"

_STAIR_VERB_PATTERNS = [
    rf"\b{_VERBS}\s+(?:up|down)\s+(?:the\s+|another\s+|a\s+)?{_FLIGHT}\b",
    rf"\b{_VERBS}\s+(?:up|down)\s+(?:the\s+)?(?:\w+\s+){{0,3}}{_STAIRS}\b",
    rf"\b{_VERBS}\s+(?:upstairs|downstairs)\b",
    r"\b(?:ascend|descend)(?:\s+(?:the\s+)?(?:stair(?:s|case|way)?|steps?))?\b",
    rf"\btake\s+(?:the\s+)?{_STAIRS}(?:\s+(?:up|down))?\b",
    rf"\b(?:up|down)\s+(?:the\s+)?(?:\w+\s+){{0,2}}{_STAIRS}\b",
]
_STAIR_VERB_RE = re.compile("|".join(_STAIR_VERB_PATTERNS), re.IGNORECASE)
_BARE_DIRECTION_RE = re.compile(r"\b(upstairs|downstairs)\b", re.IGNORECASE)


def _find_stair_span(text: str) -> Optional[tuple[int, int, str]]:
    m = _STAIR_VERB_RE.search(text)
    if m:
        return m.start(), m.end(), m.group(0)
    m = _BARE_DIRECTION_RE.search(text)
    if m:
        return m.start(), m.end(), m.group(0)
    return None


# ─── floor-phrase generators (target wording) ──────────────────────────

def _ord(n: int) -> str:
    return ordinal_word(int(n))


def _floor_phrase_absolute(end_one: int, direction: str,
                            *, capitalise: bool, rng: random.Random) -> str:
    end_ord = _ord(end_one)
    if direction == "up":
        templates = [
            f"go up to the {end_ord} floor",
            f"head up to the {end_ord} floor",
            f"go to the {end_ord} floor",
        ]
    elif direction == "down":
        templates = [
            f"go down to the {end_ord} floor",
            f"head down to the {end_ord} floor",
            f"go to the {end_ord} floor",
        ]
    else:
        templates = [f"go to the {end_ord} floor"]
    phrase = rng.choice(templates)
    return phrase[0].upper() + phrase[1:] if capitalise else phrase


def _floor_phrase_fuzzy(end_one: int, total: int, direction: str,
                         *, capitalise: bool, rng: random.Random) -> Optional[str]:
    end_word: Optional[str] = None
    if end_one == total:
        end_word = rng.choice(["the top floor", "the topmost floor"])
    elif end_one == 1:
        end_word = rng.choice(["the bottom floor", "the ground floor",
                                 "the lowest floor"])
    if end_word is None:
        return None
    if direction == "up":
        templates = [f"go up to {end_word}", f"head up to {end_word}",
                      f"go to {end_word}"]
    elif direction == "down":
        templates = [f"go down to {end_word}", f"head down to {end_word}",
                      f"go to {end_word}"]
    else:
        templates = [f"go to {end_word}"]
    phrase = rng.choice(templates)
    return phrase[0].upper() + phrase[1:] if capitalise else phrase


def _floor_phrase_updown(direction: str, *, capitalise: bool,
                          rng: random.Random) -> str:
    word = "upstairs" if direction == "up" else "downstairs"
    templates = [f"go {word}", f"head {word}"]
    phrase = rng.choice(templates)
    return phrase[0].upper() + phrase[1:] if capitalise else phrase


# ─── deterministic fallback substitution ───────────────────────────────

def _rule_substitute(meta: dict[str, Any], original: str, style: str,
                      rng: random.Random) -> tuple[str, str]:
    span = _find_stair_span(original)
    if span is None:
        return original, "passthrough"
    start_idx, end_idx, _matched = span

    end_one = int(meta["end_level_index"]) + 1
    total = int(meta["total_levels"])
    direction = meta["direction"]
    cap = (
        start_idx == 0
        or (start_idx >= 2 and original[start_idx - 2: start_idx] in (". ", "? ", "! "))
    )

    if style == "fuzzy":
        replacement = _floor_phrase_fuzzy(end_one, total, direction,
                                            capitalise=cap, rng=rng)
        if replacement is None:
            replacement = _floor_phrase_absolute(end_one, direction,
                                                  capitalise=cap, rng=rng)
    elif style == "updown":
        replacement = _floor_phrase_updown(direction, capitalise=cap, rng=rng)
    else:
        replacement = _floor_phrase_absolute(end_one, direction,
                                              capitalise=cap, rng=rng)

    new_text = original[:start_idx] + replacement + original[end_idx:]
    return new_text, "replaced"


# ─── LLM substitution ──────────────────────────────────────────────────

REWRITE_SYSTEM = (
    "You substitute ONE vague vertical-movement phrase inside an indoor "
    "navigation instruction with an explicit floor descriptor in the "
    "style requested by the user. "
    "You MUST NOT add any new sentence before or after the original. "
    "You MUST NOT change anything outside the single matched phrase: "
    "every landmark, room name, object, turn, verb and end action stays "
    "verbatim. You MUST delete the stair / upstairs / downstairs wording "
    "you replace — do NOT keep both the stair phrase AND the new floor "
    "descriptor. If the original contains NO vague stair / upstairs / "
    "downstairs phrase at all, output the original verbatim. "
    "Output exactly the rewritten instruction. No quotes, no preamble, "
    "no markdown."
)

REWRITE_PROMPT = """Rewrite the navigation instruction by substituting ONE vague stair / upstairs / downstairs phrase with a benchmark-style floor descriptor. Make NO other edits.

【Floor metadata】
- start_floor (1-indexed) : {start_ord}
- end_floor   (1-indexed) : {end_ord}
- total_floors            : {total_floors}
- direction               : {direction}

【Required style】 {style}
 - absolute → use "the {end_ord} floor" with one of the verbs
              "go to" / "head to" / "go up to" / "head up to" /
              "go down to" / "head down to" / "head up" / "head down".
              Benchmark examples:
                "Please head up to the second floor and wait for me by the sink."
                "Please go to the first floor, stop on the right side of the long sofa, and wait for me."
                "Head to the second floor, by the wooden stairs and the bike, and wait for me."
 - fuzzy    → use "the top floor" / "the topmost floor" (only when end
              IS the top floor) OR "the bottom floor" / "the ground floor"
              / "the lowest floor" (only when end IS floor 1). If neither
              applies, fall back to absolute wording.
              Benchmark examples:
                "Please go to the bottom floor of the house, stop behind the brown sofa, and face the TV."
                "Could you head up to the top floor and wait for me by the shoe cabinet?"
                "Please go to the lowest level foyer near the end of the carpeted area."
 - updown   → use bare "upstairs" / "downstairs" with the verbs
              "go" or "head". Do NOT mention any specific floor number.
              Benchmark examples:
                "Go downstairs, walk into the solid wood-style bedroom, and stop by the window side of the bed."
                "Please go upstairs to the white bedroom, go into the bathroom, and stop next to the toilet."

【The vague phrase you SHOULD replace looks like one of these】
- "walk up the stairs", "go up the stairs", "head up the stairs",
- "walk up the carpet/wooden/etc stairs", "walk up the staircase",
- "walk up the flight of stairs", "walk up another flight of stairs",
- "walk down the stairs", "go down the stairs", "head down the stairs",
- "walk upstairs", "go upstairs", "walk downstairs", "go downstairs",
- "ascend the stairs" / "descend the stairs",
- "take the stairs (up | down)",
- bare "upstairs" / "downstairs" used as a verb,
- "do down the stairs" (typo for "go down"),
- "go up the stairwell", "walk up the stairwell".

【How to rewrite — examples】
1. style=absolute, end_floor=2
   Original: "Walk up the stairs and turn right. Walk through the door and stop just inside the bedroom."
   ✅      : "Go up to the second floor and turn right. Walk through the door and stop just inside the bedroom."
   ❌      : "Please head up to the second floor. Walk up the stairs and turn right. ..."  (does NOT delete the stair phrase)

2. style=absolute, end_floor=1
   Original: "Walk down the hallway to the overlook, go down the stairs to the right, wait halfway down the stairs."
   ✅      : "Walk down the hallway to the overlook, go to the first floor to the right, wait halfway down the stairs."

3. style=fuzzy, end_floor=2/2
   Original: "Walk up the stairs and stop at the top."
   ✅      : "Go to the top floor and stop at the top."

4. style=fuzzy, end_floor=1/4
   Original: "Walk down the stairs and stop beside the portrait."
   ✅      : "Go to the bottom floor and stop beside the portrait."

5. style=updown
   Original: "Walk up the stairs. Make a right at the open doorway. Make a left and walk into the room."
   ✅      : "Go upstairs. Make a right at the open doorway. Make a left and walk into the room."

6. style=absolute, end_floor=2
   Original: "Walk upstairs to the first archway on the left and wait outside the bathroom."
   ✅      : "Go to the second floor, to the first archway on the left and wait outside the bathroom."

7. NO stair phrase at all:
   Original: "Stop in front of the painting on the wall and wait."
   ✅      : "Stop in front of the painting on the wall and wait."  (verbatim, no change)

【Hard rules】
1. Replace the FIRST occurrence of a vague stair phrase. Leave any later occurrences alone (those usually serve as landmarks like "halfway up the stairs").
2. Do NOT prepend or append any new sentence. Do NOT add "Please go to the second floor." in front when the original already names a stair clause.
3. Delete the matched stair phrase from the output — never keep BOTH "Go to the second floor" AND "walk up the stairs" together.
4. If the original contains NO vague stair / upstairs / downstairs phrase, output the original verbatim.
5. Preserve every other word, punctuation, landmark, room, object, turn, and end action exactly as written.
6. Output one instruction. No quotes. No preamble. No markdown.

【Original instruction】
{original}

Output the rewritten instruction only.
"""


_FLOOR_PHRASE_RE = re.compile(
    r"\b(?:the\s+)?(?:first|second|third|fourth|fifth|sixth|"
    r"top|bottom|ground|topmost|lowest)(?:\s|-)floor\b|"
    r"\b(?:upstairs|downstairs)\b",
    re.IGNORECASE,
)


def _strip(text: str) -> str:
    out = (text or "").strip().strip('"').strip("'")
    if out.startswith("```"):
        out = out.strip("`").strip()
    return out


def _validate(rewritten: str, original: str) -> bool:
    if not rewritten:
        return False
    if len(rewritten) > len(original) + 80:
        return False
    if token_overlap_ratio(original, rewritten) < 0.6:
        return False
    bad_openers = ("starting from", "from the", "you will")
    if rewritten.lower().startswith(bad_openers):
        return False
    return True


def _choose_style(meta: dict[str, Any], styles: list[str],
                   rng: random.Random) -> str:
    start_one = int(meta["start_level_index"]) + 1
    end_one = int(meta["end_level_index"]) + 1
    total = int(meta["total_levels"])
    eligible = list(styles)
    if "fuzzy" in eligible and not (
        start_one in (1, total) or end_one in (1, total)
    ):
        eligible = [s for s in eligible if s != "fuzzy"]
    if not eligible:
        eligible = ["absolute"]
    return rng.choice(eligible)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.json"))
    parser.add_argument("--input", default=None)
    parser.add_argument("--no-qwen", action="store_true",
                        help="Skip the Qwen API; rule-based fallback only (dry-run).")
    parser.add_argument("--max-records", type=int, default=None,
                        help="Limit the number of records processed.")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--styles", default="absolute,fuzzy,updown",
                        help="Comma list of styles to sample from "
                             "(default: absolute,fuzzy,updown).")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = pipeline_output_dir(cfg, "vertical_perception")
    input_path = Path(args.input) if args.input else out_dir / "verified.json"
    payload = load_json(input_path)
    records = payload.get("records", [])
    if args.max_records:
        records = records[: args.max_records]

    styles = [s.strip() for s in args.styles.split(",") if s.strip()]
    rng = random.Random(args.seed)
    outputs: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    stats = {"qwen_ok": 0, "qwen_failed_sanity": 0, "qwen_error": 0,
              "rule_replaced": 0, "passthrough": 0}

    for rec in records:
        meta = rec["floor_meta"]
        if int(meta["start_level_index"]) == int(meta["end_level_index"]):
            continue
        style = _choose_style(meta, styles, rng)
        original = rec["original_instruction"].strip()

        instruction = original
        via = "passthrough"

        if not args.no_qwen:
            prompt = REWRITE_PROMPT.format(
                start_ord=_ord(int(meta["start_level_index"]) + 1),
                end_ord=_ord(int(meta["end_level_index"]) + 1),
                total_floors=int(meta["total_levels"]),
                direction=meta["direction"],
                style=style,
                original=original,
            )
            try:
                raw = qwen_call(
                    cfg, prompt=prompt, system=REWRITE_SYSTEM,
                    model=cfg["qwen"]["text_model"],
                )
                cand = _strip(raw)
                if _validate(cand, original):
                    instruction = cand
                    via = "qwen"
                    stats["qwen_ok"] += 1
                else:
                    new_text, mode = _rule_substitute(meta, original, style, rng)
                    instruction = new_text
                    via = "rule" if mode == "replaced" else "passthrough"
                    stats["qwen_failed_sanity"] += 1
                    stats["rule_replaced" if via == "rule" else "passthrough"] += 1
            except Exception as exc:  # noqa: BLE001
                print(f"Qwen error for ep {rec['episode_id']}: {exc}; using rule")
                new_text, mode = _rule_substitute(meta, original, style, rng)
                instruction = new_text
                via = "rule" if mode == "replaced" else "passthrough"
                stats["qwen_error"] += 1
                stats["rule_replaced" if via == "rule" else "passthrough"] += 1
        else:
            new_text, mode = _rule_substitute(meta, original, style, rng)
            instruction = new_text
            via = "rule" if mode == "replaced" else "passthrough"
            stats["rule_replaced" if via == "rule" else "passthrough"] += 1

        outputs.append(
            {
                "episode_id": rec["episode_id"],
                "scene_id": rec["scene_id"],
                "original_instruction": original,
                "instruction": instruction,
                "style": style,
                "floor_meta": meta,
                "via": via,
                "pipeline": "vertical_perception",
            }
        )
        review.append(
            {
                "episode_id": rec["episode_id"],
                "style": style,
                "via": via,
                "start_floor": int(meta["start_level_index"]) + 1,
                "end_floor": int(meta["end_level_index"]) + 1,
                "total_floors": int(meta["total_levels"]),
                "original": original,
                "rewritten": instruction,
            }
        )

    save_json(
        {"pipeline": "vertical_perception", "count": len(outputs), "records": outputs},
        out_dir / "output.json",
    )
    custom = to_custom_instructions(outputs)
    save_json(custom, out_dir / "custom_instructions.json")

    review_path = out_dir / "review.json"
    with review_path.open("w", encoding="utf-8") as fh:
        fh.write("[\n")
        for i, item in enumerate(review):
            fh.write("  " + _json.dumps(item, ensure_ascii=False))
            fh.write(",\n" if i < len(review) - 1 else "\n")
        fh.write("]\n")

    print(
        f"Generated {len(outputs)} instructions  "
        f"(qwen_ok={stats['qwen_ok']}, "
        f"qwen_failed_sanity={stats['qwen_failed_sanity']}, "
        f"qwen_error={stats['qwen_error']}, "
        f"rule_replaced={stats['rule_replaced']}, "
        f"passthrough={stats['passthrough']})"
    )
    print(f"  output            -> {out_dir / 'output.json'}")
    print(f"  custom_instr      -> {out_dir / 'custom_instructions.json'}")
    print(f"  review (compare)  -> {review_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
