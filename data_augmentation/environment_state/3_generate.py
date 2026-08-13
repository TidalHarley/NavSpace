#!/usr/bin/env python3
"""Step 3 — Wrap each R2R instruction in an IF/OTHERWISE conditional.

The trajectory and its frames already exist on disk; here we only build
the new instruction text and emit a ``custom_instructions.json`` that
``convert_aug_to_sft.py`` will inject into ``annotations.json``.

Templates (proportions match cfg.environment_state.template_weights):

  A_starting_if (≈35% positive)
      "Starting from the {start_room}, if {cond_true}, {orig}. "
      "Otherwise, stop in the {start_room}."

  B_if_then     (≈35% positive)
      "If {cond_true}, {orig}; otherwise, return to your starting point."

  D_if_stay     (≈15% negative — IF is decisively false → otherwise = orig)
      "If there is {unobs_obj} in the {start_room}, stop where you are; "
      "otherwise, {orig}"

  E_from_here   (≈15% negative)
      "From where you are right now, if there is {unobs_obj}, "
      "stop in the {start_room}; otherwise, {orig}"

In every template the ORIGINAL R2R instruction text is kept verbatim
(just lower-cased + initial article tweaks so the resulting sentence is
grammatically smooth).  Trajectory frames and actions are unchanged, so
the rendered data is automatically (instruction, actions) consistent —
the agent must follow the trajectory whenever the IF resolves to
"execute the original instruction".

Outputs:
  outputs/environment_state/output.json
  outputs/environment_state/custom_instructions.json
  outputs/environment_state/review.json   (per-line original vs rewritten)
"""

from __future__ import annotations

import argparse
import random
import re
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data_augmentation.common import (  # noqa: E402
    load_config,
    load_json,
    pipeline_output_dir,
    save_json,
    to_custom_instructions,
)


# ─── helpers ────────────────────────────────────────────────────────────────

_SENTENCE_END = re.compile(r"[.!?]+\s*$")


def _strip_trailing_punct(text: str) -> str:
    return re.sub(r"[.!?\s]+$", "", text.strip())


def _decap_first(text: str) -> str:
    text = text.lstrip()
    if not text:
        return text
    return text[0].lower() + text[1:]


def _ensure_period(text: str) -> str:
    text = text.strip()
    return text if _SENTENCE_END.search(text) else text + "."


def _normalize_room(room: str | None, fallback: str = "room") -> str:
    if not room:
        return fallback
    room = room.strip().lower()
    # drop leading articles
    room = re.sub(r"^(the|a|an)\s+", "", room)
    return room or fallback


_ARTICLE_RE = re.compile(r"^(a|an|the|some)\s+", re.I)


def _strip_article(noun: str) -> str:
    return _ARTICLE_RE.sub("", noun.strip())


def _normalize_unobs(unobs: str | None, fallback: str = "pool table") -> str:
    noun = (unobs or fallback).strip().lower()
    noun = noun.strip(".,;:!?")
    if not noun:
        return fallback
    bare = _strip_article(noun)
    # singular → "a / an + noun"; lists → as-is.
    if bare.startswith(("two ", "three ", "four ", "five ", "several ", "many ")) \
            or bare.endswith("s") and not bare.endswith(("ss", "is", "us", "ws")):
        return bare
    article = "an" if bare[:1] in "aeiou" else "a"
    return f"{article} {bare}"


def _normalize_cond(cond: str | None, observable: list[str] | None,
                    start_room: str) -> str:
    """Make a positive ``there is/are ... in/on the ...`` predicate."""
    if cond:
        text = cond.strip().rstrip(".")
        if re.match(r"^(there\s+is|there\s+are|both|the\b)", text, re.I):
            return _decap_first(text)
    # fallback: build from first observable
    if observable:
        item = observable[0].strip().rstrip(".")
        if item:
            return f"there is {item}"
    return f"there is something in the {start_room}"


# ─── templates ───────────────────────────────────────────────────────────────

def _build_A(start_room: str, cond: str, orig_body: str) -> str:
    return (f"Starting from the {start_room}, if {cond}, {_decap_first(orig_body)}. "
            f"Otherwise, stop in the {start_room}.")


def _build_B(cond: str, orig_body: str) -> str:
    return (f"If {cond}, {_decap_first(orig_body)}; "
            f"otherwise, return to your starting point.")


def _build_D(unobs_phrase: str, start_room: str, orig_body: str) -> str:
    return (f"If there is {unobs_phrase} in the {start_room}, stop where you are; "
            f"otherwise, {_decap_first(orig_body)}.")


def _build_E(unobs_phrase: str, start_room: str, orig_body: str) -> str:
    return (f"From where you are right now, if there is {unobs_phrase}, "
            f"stop in the {start_room}; otherwise, {_decap_first(orig_body)}.")


TEMPLATE_TAG_POS = ("A_starting_if", "B_if_then")
TEMPLATE_TAG_NEG = ("D_if_stay", "E_from_here")


def _choose_template(rng: random.Random, weights: dict[str, float]) -> str:
    keys = list(weights.keys())
    unknown = [k for k in keys if k not in TEMPLATE_TAG_POS + TEMPLATE_TAG_NEG]
    if unknown:
        # A key that no _build_* handles would silently fall through to
        # template E, skewing the whole instruction mix without any error.
        raise SystemExit(
            f"config environment_state.template_weights has unknown templates: "
            f"{unknown}. Supported: {list(TEMPLATE_TAG_POS + TEMPLATE_TAG_NEG)}")
    vals = [float(weights[k]) for k in keys]
    return rng.choices(keys, weights=vals, k=1)[0]


def _build_one(template: str, analysis: dict[str, Any], orig: str,
                rng: random.Random) -> str:
    start_room = _normalize_room(analysis.get("start_room"))
    observable = analysis.get("observable_objects") or []
    unobservable = analysis.get("unobservable_objects") or []

    orig_body = _strip_trailing_punct(orig)

    if template in TEMPLATE_TAG_POS:
        cond = _normalize_cond(analysis.get("condition_phrase"),
                                observable, start_room)
        if template == "A_starting_if":
            return _build_A(start_room, cond, orig_body)
        return _build_B(cond, orig_body)

    # negative-branch templates: pick a decisive unobservable.  Prefer
    # the LLM's hand-picked ``decisive_unobs_object`` but fall back to a
    # random sample from the full ``unobservable_objects`` list so we
    # don't over-concentrate on a single phrase (Qwen tends to repeat
    # "a snowmobile" across many scenes — random sampling spreads the
    # IF condition lexicon).
    decisive = analysis.get("decisive_unobs_object")
    candidates = [u for u in (unobservable or []) if u]
    if decisive and decisive not in candidates:
        candidates.append(decisive)
    unobs = rng.choice(candidates) if candidates else None
    unobs_phrase = _normalize_unobs(unobs)
    if template == "D_if_stay":
        return _build_D(unobs_phrase, start_room, orig_body)
    return _build_E(unobs_phrase, start_room, orig_body)


# ─── main ────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",
                        default=str(Path(__file__).resolve().parents[1] / "config.json"))
    parser.add_argument("--input", default=None,
                        help="states.json from step 2.")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config(args.config)
    ecfg = cfg["environment_state"]
    out_dir = pipeline_output_dir(cfg, "environment_state")
    input_path = Path(args.input) if args.input else out_dir / "states.json"
    payload = load_json(input_path)
    records = payload["records"] if isinstance(payload, dict) else payload

    weights = ecfg["template_weights"]
    rng = random.Random(args.seed)

    outputs: list[dict[str, Any]] = []
    review: list[dict[str, Any]] = []
    by_template: dict[str, int] = {}

    for rec in records:
        analysis = rec.get("analysis", {})
        orig = rec.get("original_instruction", "").strip()
        if not orig:
            continue

        template = _choose_template(rng, weights)
        # If no decisive_unobs/unobservable available, fallback to positive template
        if template in TEMPLATE_TAG_NEG:
            unobs = analysis.get("decisive_unobs_object") or (
                (analysis.get("unobservable_objects") or [None])[0]
            )
            if not unobs:
                template = rng.choice(TEMPLATE_TAG_POS)

        new_instr = _build_one(template, analysis, orig, rng)
        new_instr = _ensure_period(new_instr)

        by_template[template] = by_template.get(template, 0) + 1
        outputs.append({
            "episode_id": rec["episode_id"],
            "scene_id": rec.get("scene_id"),
            "template": template,
            "original_instruction": orig,
            "instruction": new_instr,
            "analysis": analysis,
            "pipeline": "environment_state",
        })
        review.append({
            "episode_id": rec["episode_id"],
            "template": template,
            "start_room": _normalize_room(analysis.get("start_room")),
            "original": orig,
            "rewritten": new_instr,
        })

    save_json({"pipeline": "environment_state", "count": len(outputs),
                "by_template": by_template, "records": outputs},
               out_dir / "output.json")
    save_json(to_custom_instructions(outputs),
               out_dir / "custom_instructions.json")
    save_json(review, out_dir / "review.json")

    print(f"Generated {len(outputs)} instructions in {out_dir}")
    print("  by_template counts:")
    for tag, n in sorted(by_template.items()):
        print(f"    {tag:14s}: {n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
