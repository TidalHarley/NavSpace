#!/usr/bin/env python3
"""Build a Stop-only subset JSON from a large training JSON array.

This keeps the original training code untouched. We instead create an
extra JSON containing only samples whose target action sequence includes
`Stop`, then reference both the original JSON and the Stop-only JSON in
the training YAML so those cases are seen twice.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Iterator


def iter_json_array(path: Path, chunk_size: int = 1 << 20) -> Iterator[dict]:
    decoder = json.JSONDecoder()
    with path.open("r", encoding="utf-8") as f:
        buffer = ""
        index = 0
        started = False
        eof = False

        while True:
            if not eof and len(buffer) - index < chunk_size // 2:
                chunk = f.read(chunk_size)
                if chunk:
                    buffer = buffer[index:] + chunk
                    index = 0
                else:
                    buffer = buffer[index:]
                    index = 0
                    eof = True

            while index < len(buffer) and buffer[index].isspace():
                index += 1

            if not started:
                if index >= len(buffer):
                    if eof:
                        raise ValueError(f"Empty JSON array: {path}")
                    continue
                if buffer[index] != "[":
                    raise ValueError(f"Expected '[' at start of {path}")
                started = True
                index += 1
                continue

            while index < len(buffer) and buffer[index].isspace():
                index += 1

            if index >= len(buffer):
                if eof:
                    raise ValueError(f"Unexpected EOF while parsing {path}")
                continue

            if buffer[index] == "]":
                return

            try:
                obj, next_index = decoder.raw_decode(buffer, index)
            except json.JSONDecodeError:
                if eof:
                    raise
                chunk = f.read(chunk_size)
                if not chunk:
                    eof = True
                else:
                    buffer += chunk
                continue

            yield obj
            index = next_index

            while True:
                while index < len(buffer) and buffer[index].isspace():
                    index += 1
                if index >= len(buffer):
                    break
                if buffer[index] == ",":
                    index += 1
                    break
                if buffer[index] == "]":
                    return
                raise ValueError(
                    f"Expected ',' or ']' after JSON item in {path}, got {buffer[index]!r}"
                )


def get_target_text(sample: dict) -> str:
    conversations = sample.get("conversations") or []
    if len(conversations) < 2:
        return ""
    return str(conversations[1].get("value", ""))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, default=None)
    parser.add_argument("--match", type=str, default="Stop")
    parser.add_argument("--progress-every", type=int, default=100000)
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0

    temp_output = args.output.with_name(f"{args.output.name}.{os.getpid()}.tmp")

    with temp_output.open("w", encoding="utf-8") as out:
        out.write("[\n")
        first = True
        for sample in iter_json_array(args.input):
            total += 1
            if args.match not in get_target_text(sample):
                matched = False
            else:
                matched = True
                if not first:
                    out.write(",\n")
                json.dump(sample, out, ensure_ascii=False)
                first = False
                kept += 1

            if args.progress_every > 0 and total % args.progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "progress",
                            "input": str(args.input),
                            "total_samples": total,
                            "kept_samples": kept,
                            "latest_matched": matched,
                        },
                        ensure_ascii=False,
                    ),
                    flush=True,
                )
        out.write("\n]\n")

    os.replace(temp_output, args.output)

    summary = {
        "input": str(args.input),
        "output": str(args.output),
        "match": args.match,
        "total_samples": total,
        "kept_samples": kept,
    }
    print(json.dumps(summary, ensure_ascii=False))

    if args.summary is not None:
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
