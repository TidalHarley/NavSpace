#!/usr/bin/env python3
import argparse
import json
import os
import re
import textwrap
from pathlib import Path

try:
    import orjson
except ImportError:
    orjson = None


SCRIPT_DIR = Path(__file__).resolve().parent
from paths import TRAIN_DATA_ROOT  # noqa: E402

DEFAULT_R2R_INPUT = TRAIN_DATA_ROOT / "r2r_stepwise_train_jupyter_full.json"
DEFAULT_RXR_INPUT = TRAIN_DATA_ROOT / "rxr_stepwise_train_jupyter_full_en.json"
DEFAULT_R2R_OUTPUT = TRAIN_DATA_ROOT / "r2r_instruction_from_fullobs_jupyter.json"
DEFAULT_RXR_OUTPUT = TRAIN_DATA_ROOT / "rxr_instruction_from_fullobs_en_jupyter.json"
DEFAULT_MIX_OUTPUT = TRAIN_DATA_ROOT / "r2r_rxr_instruction_from_fullobs_enmix_jupyter.json"
DEFAULT_SUMMARY_OUTPUT = TRAIN_DATA_ROOT / "instruction_from_fullobs_summary.json"

INSTRUCTION_PREFIX = "You are navigating in an indoor environment given the instruction: "
INSTRUCTION_PATTERN = re.compile(
    r"You are navigating in an indoor environment given the instruction:\s*(.*?)\s*;\s*\r?\n\s*You are given the observation history of previous steps you have taken;",
    re.DOTALL,
)
ALL_STOP_ANSWER = "Final Answer: Stop,Stop,Stop,Stop,Stop,Stop"


def build_instruction_prompt() -> str:
    prompt = """
    You are given the complete observation history of an indoor navigation trajectory from start to finish.
    Based only on these observations, reconstruct the original navigation instruction that best matches this trajectory.
    You should:
    1) describe the route from the starting position to the final goal based on the observations;
    2) mention key turns, landmarks, and destination cues when they are useful;
    3) produce a single navigation instruction that could guide another agent along the same trajectory.
    Notice that:
    1) Your answer should be one navigation instruction, not an action list.
    2) Do not mention frame numbers or refer to the images explicitly.
    3) Stay faithful to the observations and do not invent unsupported details.
    ----
    Starting below, you should strictly follow this format:
    Final Answer: The navigation instruction
    """
    return textwrap.dedent(prompt).strip()


def iter_json_object_strings(path: Path):
    with open(path, "r", encoding="utf-8", buffering=1 << 20) as f:
        capture = False
        in_string = False
        escape = False
        brace_depth = 0
        current_chars = []
        while True:
            chunk = f.read(1 << 20)
            if not chunk:
                break
            for ch in chunk:
                if not capture:
                    if ch != "{":
                        continue
                    capture = True
                    in_string = False
                    escape = False
                    brace_depth = 1
                    current_chars = ["{"]
                    continue

                current_chars.append(ch)
                if in_string:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_string = False
                    continue

                if ch == '"':
                    in_string = True
                elif ch == "{":
                    brace_depth += 1
                elif ch == "}":
                    brace_depth -= 1
                    if brace_depth == 0:
                        yield "".join(current_chars)
                        capture = False
                        current_chars = []

        if capture:
            raise ValueError(f"{path} ended with an incomplete JSON object")


def loads_json(payload: str):
    if orjson is not None:
        return orjson.loads(payload)
    return json.loads(payload)


def iter_json_array(path: Path):
    for item_text in iter_json_object_strings(path):
        yield loads_json(item_text)


def write_output_json(path: Path, payload_iter) -> int:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    count = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write("[\n")
        first = True
        for item in payload_iter:
            if not first:
                f.write(",\n")
            json.dump(item, f, ensure_ascii=False)
            first = False
            count += 1
        f.write("\n]\n")
    tmp_path.replace(path)
    return count


def parse_final_answer_actions(answer_text: str):
    answer = str(answer_text or "")
    if "Final Answer:" in answer:
        answer = answer.split("Final Answer:", 1)[1]
    return [part.strip() for part in answer.split(",") if part.strip()]


def is_all_stop_sample(item) -> bool:
    conversations = item.get("conversations") or []
    if len(conversations) < 2:
        return False
    actions = parse_final_answer_actions(conversations[1].get("value", ""))
    return bool(actions) and all(action == "Stop" for action in actions)


def extract_instruction_from_prompt(prompt: str) -> str:
    prompt = str(prompt or "")
    if INSTRUCTION_PREFIX not in prompt:
        raise ValueError("instruction prefix not found")
    match = INSTRUCTION_PATTERN.search(prompt)
    if match is None:
        raise ValueError("instruction suffix not found")
    instruction = match.group(1).strip()
    if not instruction:
        raise ValueError("empty instruction")
    return instruction


def image_tokens_for(images) -> str:
    image_count = len(images or [])
    if image_count <= 0:
        raise ValueError("sample has no images")
    return "\n".join(["<image>"] * image_count)


def build_output_item(item, prompt_template: str):
    conversations = item["conversations"]
    images = item["image"]
    instruction = extract_instruction_from_prompt(conversations[0]["value"])
    source = item.get("data_source", "unknown")
    if source == "r2r":
        data_source = "r2r_instruction_reconstruction"
    elif source == "rxr":
        data_source = "rxr_instruction_reconstruction"
    else:
        data_source = f"{source}_instruction_reconstruction"
    return {
        "id": f"{item['id']}_instrrec",
        "conversations": [
            {"from": "human", "value": f"{image_tokens_for(images)}\n{prompt_template}"},
            {"from": "gpt", "value": f"Final Answer: {instruction}"},
        ],
        "image": images,
        "data_source": data_source,
    }


def convert_dataset(input_path: Path, output_path: Path, prompt_template: str, limit: int = 0):
    kept_count = 0
    total_count = 0
    first_output = None

    def payload_iter():
        nonlocal kept_count, total_count, first_output
        for item_text in iter_json_object_strings(input_path):
            total_count += 1
            if ALL_STOP_ANSWER not in item_text:
                continue
            item = loads_json(item_text)
            if not is_all_stop_sample(item):
                continue
            output_item = build_output_item(item, prompt_template)
            if first_output is None:
                first_output = output_item
            kept_count += 1
            yield output_item
            if limit > 0 and kept_count >= limit:
                break

    written_count = write_output_json(output_path, payload_iter())
    return {
        "input_json": str(input_path),
        "output_json": str(output_path),
        "total_samples_seen": total_count,
        "selected_terminal_all_stop_samples": kept_count,
        "written_samples": written_count,
        "first_output_id": None if first_output is None else first_output["id"],
        "first_output_image_count": None if first_output is None else len(first_output["image"]),
    }


def merge_datasets(output_path: Path, input_paths):
    written_count = write_output_json(
        output_path,
        (
            item
            for input_path in input_paths
            for item in iter_json_array(input_path)
        ),
    )
    return {
        "output_json": str(output_path),
        "merged_input_jsons": [str(path) for path in input_paths],
        "written_samples": written_count,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--r2r-input", type=Path, default=DEFAULT_R2R_INPUT)
    parser.add_argument("--rxr-input", type=Path, default=DEFAULT_RXR_INPUT)
    parser.add_argument("--r2r-output", type=Path, default=DEFAULT_R2R_OUTPUT)
    parser.add_argument("--rxr-output", type=Path, default=DEFAULT_RXR_OUTPUT)
    parser.add_argument("--mix-output", type=Path, default=DEFAULT_MIX_OUTPUT)
    parser.add_argument("--summary-output", type=Path, default=DEFAULT_SUMMARY_OUTPUT)
    parser.add_argument("--limit-per-dataset", type=int, default=0)
    parser.add_argument("--skip-r2r", action="store_true")
    parser.add_argument("--skip-rxr", action="store_true")
    parser.add_argument("--skip-mix", action="store_true")
    args = parser.parse_args()

    prompt_template = build_instruction_prompt()
    summary = {
        "prompt_template": prompt_template,
        "datasets": {},
    }

    generated_outputs = []
    if not args.skip_r2r:
        args.r2r_output.parent.mkdir(parents=True, exist_ok=True)
        summary["datasets"]["r2r"] = convert_dataset(
            input_path=args.r2r_input,
            output_path=args.r2r_output,
            prompt_template=prompt_template,
            limit=args.limit_per_dataset,
        )
        generated_outputs.append(args.r2r_output)

    if not args.skip_rxr:
        args.rxr_output.parent.mkdir(parents=True, exist_ok=True)
        summary["datasets"]["rxr_en"] = convert_dataset(
            input_path=args.rxr_input,
            output_path=args.rxr_output,
            prompt_template=prompt_template,
            limit=args.limit_per_dataset,
        )
        generated_outputs.append(args.rxr_output)

    if not args.skip_mix and generated_outputs:
        args.mix_output.parent.mkdir(parents=True, exist_ok=True)
        summary["datasets"]["mixed_en"] = merge_datasets(
            output_path=args.mix_output,
            input_paths=generated_outputs,
        )

    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    tmp_summary = args.summary_output.with_suffix(args.summary_output.suffix + ".tmp")
    with open(tmp_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
        f.write("\n")
    tmp_summary.replace(args.summary_output)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
