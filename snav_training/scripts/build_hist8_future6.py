#!/usr/bin/env python3
"""Build hist8/future6 SFT JSON from SNav-layout annotations.

Converts per-episode ``annotations.json`` folders under ``aug_mix`` (and
optionally ``manual_98``) into the multi-image conversation format used by
``snav_llava/train/train_mem.py``, matching the H200 paper recipe:

  train_data/navspace_aug_mix_plus_manual98_hist8_future6.json
  (~80210 rows = aug_mix + manual98 when using the published splits)

Image paths are relative to ``IMAGE_FOLDER`` (= ``snav_data/aug_mix``):
  - aug rows:   ``{category}/{video}/rgb/{NNN}.jpg``
  - manual rows: ``../manual_98/{video}/rgb/{NNN}.jpg``

Usage (from NavSpace repo root)::

    python snav_training/scripts/build_hist8_future6.py \\
        --aug-root snav_data/aug_mix \\
        --manual-root snav_data/manual_98 \\
        --out-dir train_data \\
        --yaml-out snav_training/configs/train_navspace_aug_mix_plus_manual98_hist8_future6.yaml
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

PROMPT_TMPL = (
    "<image>\n"
    "You are an autonomous indoor navigation agent. You observe the environment "
    "through sequential RGB frames and follow natural language instructions to "
    "reach a goal location. At each decision step, predict the next 6 low-level "
    "actions from: FORWARD (↑) moves 25 cm, TURN LEFT (←) rotates 30°, "
    "TURN RIGHT (→) rotates 30°, STOP ends navigation.\n\n"
    "These frames show your navigation history. Instruction: {instruction}\n"
    "Predict the next 6 actions."
)

ACTION = {1: "↑", 2: "←", 3: "→", 0: "STOP", -1: "STOP"}

DEFAULT_AUG_TASKS = (
    "environment_state",
    "precise_movement",
    "spatial_relationship",
    "vertical_perception",
)


def _load_annotations(folder: Path) -> list[dict]:
    ann_path = folder / "annotations.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"missing annotations.json under {folder}")
    data = json.loads(ann_path.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        for key in ("records", "episodes", "annotations"):
            if isinstance(data.get(key), list):
                data = data[key]
                break
    if not isinstance(data, list):
        raise ValueError(f"unsupported annotations payload in {ann_path}")
    return data


def _instruction(item: dict) -> str | None:
    instrs = item.get("instructions") or []
    if isinstance(instrs, str):
        instrs = [instrs]
    if not instrs:
        return None
    text = str(instrs[0]).strip()
    if not text or text.lower() in {"no", "none", "n/a"}:
        return None
    return text


def _rel_image(prefix: str, video: str, frame_idx_1based: int) -> str:
    return f"{prefix}/{video}/rgb/{frame_idx_1based:03d}.jpg"


def _build_history(prefix: str, video: str, step_idx: int, n_frames: int, hist: int) -> list[str]:
    # step_idx is 0-based along the actions list; frame index is 1-based rgb/NNN.jpg
    cur = min(step_idx + 1, n_frames)
    start = max(1, cur - hist + 1)
    frames = [_rel_image(prefix, video, i) for i in range(start, cur + 1)]
    while len(frames) < hist:
        frames.append(frames[-1] if frames else _rel_image(prefix, video, 1))
    return frames[-hist:]


def _future_actions(actions: list[int], step_idx: int, future: int) -> str:
    toks: list[str] = []
    for j in range(future):
        pos = step_idx + 1 + j
        aid = int(actions[pos]) if pos < len(actions) else 0
        if aid < 0:
            aid = 0
        toks.append(ACTION.get(aid, "STOP"))
    return "".join(toks)


def build_rows_from_folder(
    folder: Path,
    *,
    image_prefix: str,
    id_template: str,
    data_source: str,
    hist: int,
    future: int,
) -> tuple[list[dict], list[dict]]:
    """Return (rows, skipped).

    ``id_template`` must contain ``{ep_id}`` and ``{step:04d}``.
    """
    rows: list[dict] = []
    skipped: list[dict] = []
    for item in _load_annotations(folder):
        ep_id = str(item.get("id") or item.get("video") or "").strip()
        video = str(item.get("video") or ep_id).strip()
        if not ep_id or not video:
            skipped.append({"id": ep_id or "<missing>", "reason": "missing_id"})
            continue
        instruction = _instruction(item)
        if instruction is None:
            skipped.append({"id": ep_id, "reason": "bad_instruction", "instruction": item.get("instructions")})
            continue
        actions = [int(x) for x in item.get("actions", [])]
        if len(actions) < 2:
            skipped.append({"id": ep_id, "reason": "too_short"})
            continue
        n_frames = len(actions)
        for step_idx in range(len(actions)):
            rows.append(
                {
                    "id": id_template.format(ep_id=ep_id, step=step_idx),
                    "image": _build_history(image_prefix, video, step_idx, n_frames, hist),
                    "conversations": [
                        {"from": "human", "value": PROMPT_TMPL.format(instruction=instruction)},
                        {"from": "gpt", "value": "Final Answer: " + _future_actions(actions, step_idx, future)},
                    ],
                    "data_source": data_source,
                }
            )
    return rows, skipped


def _validate_sample(rows: list[dict], image_folder: Path, sample_limit: int = 1500) -> list[str]:
    if not rows:
        return ["no rows"]
    missing: list[str] = []
    checks = rows[:80] + rows[-80:]
    if len(rows) > sample_limit:
        step = max(1, len(rows) // sample_limit)
        checks += rows[::step][:sample_limit]
    seen: set[str] = set()
    for row in checks:
        for img in row["image"]:
            if img in seen:
                continue
            seen.add(img)
            path = (image_folder / img).resolve()
            if not path.is_file():
                missing.append(str(path))
                if len(missing) >= 20:
                    return missing
    return missing


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--aug-root", type=Path, required=True, help="snav_data/aug_mix")
    ap.add_argument("--manual-root", type=Path, default=None, help="snav_data/manual_98 (optional)")
    ap.add_argument("--tasks", nargs="+", default=list(DEFAULT_AUG_TASKS))
    ap.add_argument("--hist", type=int, default=8)
    ap.add_argument("--future", type=int, default=6)
    ap.add_argument("--out-dir", type=Path, required=True, help="train_data output directory")
    ap.add_argument(
        "--merged-name",
        default="navspace_aug_mix_plus_manual98_hist8_future6.json",
    )
    ap.add_argument(
        "--aug-name",
        default="navspace_aug_mix_correctnav_hist8_future6.json",
    )
    ap.add_argument(
        "--manual-name",
        default="navspace_manual_98_correctnav_hist8_future6.json",
    )
    ap.add_argument(
        "--yaml-out",
        type=Path,
        default=None,
        help="optional YAML pointing at the merged JSON (snav_training data_path)",
    )
    ap.add_argument("--skip-path-check", action="store_true")
    ap.add_argument("--tag", default=None, help="suffix for data_source strings (default: today's UTC date)")
    args = ap.parse_args()

    tag = args.tag or datetime.now(timezone.utc).strftime("%Y%m%d")
    aug_root = args.aug_root.resolve()
    out_dir = args.out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    aug_rows: list[dict] = []
    all_skipped: list[dict] = []
    by_task: dict[str, int] = {}
    for task in args.tasks:
        folder = aug_root / task
        if not folder.is_dir():
            print(f"[skip] missing task folder: {folder}")
            continue
        rows, skipped = build_rows_from_folder(
            folder,
            image_prefix=task,
            id_template=f"navspace_{task}_{{ep_id}}_ins00_step{{step:04d}}",
            data_source=f"navspace_aug_mix_{task}_hist{args.hist}_future{args.future}_{tag}",
            hist=args.hist,
            future=args.future,
        )
        aug_rows.extend(rows)
        by_task[task] = len(rows)
        all_skipped.extend({"task": task, **s} for s in skipped)
        print(f"[aug] {task}: {len(rows)} rows (skipped {len(skipped)})")

    aug_out = out_dir / args.aug_name
    aug_out.write_text(json.dumps(aug_rows, ensure_ascii=False), encoding="utf-8")
    print(f"wrote aug {len(aug_rows)} -> {aug_out}")

    manual_rows: list[dict] = []
    manual_traj_total = 0
    if args.manual_root is not None:
        manual_root = args.manual_root.resolve()
        anns = _load_annotations(manual_root)
        manual_traj_total = len(anns)
        manual_rows, skipped = build_rows_from_folder(
            manual_root,
            image_prefix="../manual_98",
            id_template="navspace_manual98_{ep_id}_step{step:04d}",
            data_source=f"navspace_manual98_human_hist{args.hist}_future{args.future}_{tag}",
            hist=args.hist,
            future=args.future,
        )
        all_skipped.extend({"task": "manual98", **s} for s in skipped)
        manual_out = out_dir / args.manual_name
        manual_out.write_text(json.dumps(manual_rows, ensure_ascii=False), encoding="utf-8")
        used = manual_traj_total - sum(1 for s in skipped if s.get("reason") == "bad_instruction")
        print(f"wrote manual {len(manual_rows)} -> {manual_out} (traj≈{used}/{manual_traj_total})")

    merged = aug_rows + manual_rows
    ids = [r["id"] for r in merged]
    if len(ids) != len(set(ids)):
        raise SystemExit(f"duplicate row ids: {len(ids) - len(set(ids))}")

    if not args.skip_path_check:
        missing = _validate_sample(merged, aug_root)
        if missing:
            raise SystemExit("missing images (sample):\n" + "\n".join(missing[:20]))

    merged_out = out_dir / args.merged_name
    merged_out.write_text(json.dumps(merged, ensure_ascii=False), encoding="utf-8")
    print(f"wrote merged {len(merged)} = aug {len(aug_rows)} + manual {len(manual_rows)} -> {merged_out}")

    yaml_out = args.yaml_out
    if yaml_out is None:
        yaml_out = (
            Path(__file__).resolve().parents[1]
            / "configs"
            / "train_navspace_aug_mix_plus_manual98_hist8_future6.yaml"
        )
    else:
        yaml_out = yaml_out.resolve()
    rel = os.path.relpath(merged_out, yaml_out.parent)
    yaml_out.write_text(
        f"datasets:\n  - json_path: {rel}\n    sampling_strategy: all\n",
        encoding="utf-8",
    )
    print(f"wrote yaml -> {yaml_out}")

    img_lens = Counter(len(r["image"]) for r in merged)
    report = {
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S %Z"),
        "aug_root": str(aug_root),
        "manual_root": str(args.manual_root.resolve()) if args.manual_root else None,
        "aug_json": str(aug_out),
        "manual_json": str(out_dir / args.manual_name) if manual_rows else None,
        "merged_json": str(merged_out),
        "yaml": str(yaml_out),
        "image_folder_for_training": str(aug_root),
        "hist": args.hist,
        "future": args.future,
        "aug_rows": len(aug_rows),
        "manual_rows": len(manual_rows),
        "merged_rows": len(merged),
        "aug_by_task": by_task,
        "manual_trajectories_total": manual_traj_total,
        "manual_trajectories_skipped": sum(1 for s in all_skipped if s.get("task") == "manual98"),
        "skipped": all_skipped[:50],
        "image_len_counts_merged": {str(k): v for k, v in sorted(img_lens.items())},
        "sample_first": merged[0] if merged else None,
        "sample_last": merged[-1] if merged else None,
    }
    report_path = merged_out.with_suffix(".report.json")
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"wrote report -> {report_path}")


if __name__ == "__main__":
    main()
