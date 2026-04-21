from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.common import (
    build_resume_index,
    load_dataset_file,
    locked_dump_json,
    locked_load_json,
    summarize_results,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Merge NavSpace evaluation result files.")
    parser.add_argument("--input", nargs="+", required=True, help="Input result JSON files.")
    parser.add_argument("--trajectory-path", required=True, help="Dataset file used for ordering.")
    parser.add_argument("--output", required=True, help="Merged output JSON path.")
    args = parser.parse_args(argv)

    merged_index = {}
    for path_str in args.input:
        merged_index.update(build_resume_index(locked_load_json(Path(path_str).resolve())))

    ordered_results = []
    for episode in load_dataset_file(Path(args.trajectory_path).resolve()):
        instruction = episode["instruction"]["instruction_text"]
        shortest_path = float(episode.get("info", {}).get("geodesic_distance", 0.0))
        key = (instruction, shortest_path)
        if key in merged_index:
            ordered_results.append({instruction: merged_index[key]})

    output_path = Path(args.output).resolve()
    locked_dump_json(ordered_results, output_path)
    summary = summarize_results(ordered_results)
    print(f"Merged {len(ordered_results)} results into {output_path}")
    print(
        f"SR={summary['sr']:.3f} NE={summary['ne']:.3f} "
        f"OS={summary['os']:.3f} SPL={summary['spl']:.3f}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
