"""Offline smoke test for the NavSpace evaluation stack.

This script validates as much of the evaluation repository as possible
WITHOUT requiring Habitat-Sim, HM3D/MP3D scenes, GPU weights, or any
live API key. Run it once before open-sourcing to catch import issues,
broken CLI contracts, inconsistent resolutions, dataset path typos, or
malformed episode records.

Usage:
    python tools/smoke_test.py
    python tools/smoke_test.py --max-episodes 32      # sample more episodes
    python tools/smoke_test.py --expected-resolution 224
"""

from __future__ import annotations

import argparse
import compileall
import importlib
import io
import json
import sys
import traceback
from contextlib import redirect_stdout
from pathlib import Path
from typing import Callable

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


_OK = "[  OK  ]"
_FAIL = "[ FAIL ]"
_WARN = "[ WARN ]"


class SmokeTest:
    def __init__(self, expected_resolution: int, max_episodes: int) -> None:
        self.expected_resolution = expected_resolution
        self.max_episodes = max_episodes
        self.failures: list[str] = []
        self.warnings: list[str] = []

    def run(self, label: str, func: Callable[[], None]) -> None:
        try:
            func()
        except AssertionError as exc:
            self.failures.append(f"{label}: {exc}")
            print(f"{_FAIL} {label}: {exc}")
            return
        except Exception as exc:  # noqa: BLE001
            self.failures.append(f"{label}: {exc!r}")
            print(f"{_FAIL} {label}: {exc!r}")
            traceback.print_exc()
            return
        print(f"{_OK} {label}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        print(f"{_WARN} {message}")

    def summary(self) -> int:
        print()
        print("=" * 60)
        print(f"Warnings: {len(self.warnings)}")
        print(f"Failures: {len(self.failures)}")
        if self.failures:
            print("\nFAILURES:")
            for item in self.failures:
                print(f"  - {item}")
            return 1
        print("\nAll offline checks passed. You are ready to run live evaluation.")
        return 0


def check_compileall() -> None:
    with redirect_stdout(io.StringIO()):
        ok = compileall.compile_dir(str(_REPO_ROOT / "evaluation"), quiet=1, force=True)
        ok &= compileall.compile_dir(str(_REPO_ROOT / "tools"), quiet=1, force=True)
        ok &= compileall.compile_file(str(_REPO_ROOT / "gpt_eval.py"), quiet=1, force=True)
        if (_REPO_ROOT / "snav_training").exists():
            ok &= compileall.compile_dir(str(_REPO_ROOT / "snav_training"), quiet=1, force=True)
    assert ok, "compileall detected syntax errors; re-run manually for details."


def check_core_imports() -> None:
    for module_name in [
        "evaluation",
        "evaluation.common",
        "evaluation.config",
        "evaluation.prompts",
        "evaluation.providers",
    ]:
        importlib.import_module(module_name)


def check_entry_parsers() -> None:
    from evaluation.run_llm_eval import build_parser as build_llm_parser
    from evaluation.eval_snav import build_parser as build_snav_parser
    from evaluation.eval_streamvln import build_parser as build_streamvln_parser

    for name, builder in [
        ("run_llm_eval", build_llm_parser),
        ("eval_snav", build_snav_parser),
        ("eval_streamvln", build_streamvln_parser),
    ]:
        parser = builder()
        assert parser is not None, f"{name} build_parser returned None"
        help_text = parser.format_help()
        assert "--hm3d-base-path" in help_text, (
            f"{name} CLI is missing --hm3d-base-path"
        )


def check_wrapper_gpt_eval() -> None:
    import gpt_eval  # noqa: F401

    assert hasattr(gpt_eval, "main"), "gpt_eval.py must re-export main()"


def check_task_dataset_map(tester: SmokeTest) -> Callable[[], None]:
    def _check() -> None:
        from evaluation.config import TASK_DATASET_MAP

        required_tasks = {
            "environment_state",
            "space_structure",
            "precise_movement",
            "viewpoint_shifting",
            "vertical_perception",
            "spatial_relationship",
        }
        assert required_tasks.issubset(TASK_DATASET_MAP.keys()), (
            f"TASK_DATASET_MAP missing tasks: {required_tasks - set(TASK_DATASET_MAP)}"
        )
        for task, relative in TASK_DATASET_MAP.items():
            absolute = (_REPO_ROOT / relative).resolve()
            if not absolute.exists():
                tester.warn(
                    f"Task dataset file missing on disk: {task} -> {absolute}. "
                    "This is only fatal at evaluation time."
                )

    return _check


def check_episode_schema(tester: SmokeTest) -> Callable[[], None]:
    def _check() -> None:
        from evaluation.common import load_dataset_file
        from evaluation.config import TASK_DATASET_MAP

        required_fields = ("episode_id", "scene_id", "start_position", "start_rotation", "goals", "info", "instruction")
        for task, relative in TASK_DATASET_MAP.items():
            path = (_REPO_ROOT / relative).resolve()
            if not path.exists():
                continue
            episodes = load_dataset_file(path)
            assert isinstance(episodes, list) and episodes, f"{task}: empty episode list"
            sample = episodes[: tester.max_episodes]
            for idx, episode in enumerate(sample):
                for field in required_fields:
                    assert field in episode, (
                        f"{task} episode #{idx} missing field '{field}'"
                    )
                assert isinstance(episode["start_position"], list) and len(episode["start_position"]) == 3, (
                    f"{task} episode #{idx} start_position must have length 3"
                )
                assert isinstance(episode["start_rotation"], list) and len(episode["start_rotation"]) == 4, (
                    f"{task} episode #{idx} start_rotation must have length 4"
                )
                goals = episode["goals"]
                assert goals and "position" in goals[0], (
                    f"{task} episode #{idx} missing goals[0].position"
                )
                assert "instruction_text" in episode["instruction"], (
                    f"{task} episode #{idx} missing instruction.instruction_text"
                )

    return _check


def check_scene_id_reachable() -> None:
    from evaluation.common import _scene_name_candidates
    from evaluation.config import TASK_DATASET_MAP
    from evaluation.common import load_dataset_file

    for task, relative in TASK_DATASET_MAP.items():
        path = (_REPO_ROOT / relative).resolve()
        if not path.exists():
            continue
        episodes = load_dataset_file(path)
        for episode in episodes[:5]:
            candidates = _scene_name_candidates(episode["scene_id"])
            assert candidates, (
                f"{task}: unable to derive scene candidates from scene_id={episode['scene_id']}"
            )


def check_profiles_consistency(expected_resolution: int) -> Callable[[], None]:
    def _check() -> None:
        from evaluation.config import PROFILE_DEFAULTS

        assert PROFILE_DEFAULTS, "No LLM profiles are defined"
        required_keys = {
            "provider",
            "model",
            "base_url",
            "key_env",
            "frame_width",
            "frame_height",
            "encode_resize",
            "max_retries",
            "initial_retry_delay",
            "use_system_message",
        }
        for name, profile in PROFILE_DEFAULTS.items():
            missing = required_keys - profile.keys()
            assert not missing, f"Profile '{name}' missing keys: {missing}"
            assert profile["provider"] in {"openai_compatible", "zhipu"}, (
                f"Profile '{name}' has unsupported provider: {profile['provider']}"
            )
            assert profile["frame_width"] == expected_resolution, (
                f"Profile '{name}' frame_width={profile['frame_width']} "
                f"but expected {expected_resolution}"
            )
            assert profile["frame_height"] == expected_resolution, (
                f"Profile '{name}' frame_height={profile['frame_height']} "
                f"but expected {expected_resolution}"
            )
            assert profile["encode_resize"] == expected_resolution, (
                f"Profile '{name}' encode_resize={profile['encode_resize']} "
                f"but expected {expected_resolution}"
            )

    return _check


def check_entry_resolution_defaults(expected_resolution: int) -> Callable[[], None]:
    def _check() -> None:
        from evaluation.eval_snav import build_parser as build_snav_parser
        from evaluation.eval_streamvln import build_parser as build_streamvln_parser

        for name, parser in [
            ("eval_snav", build_snav_parser()),
            ("eval_streamvln", build_streamvln_parser()),
        ]:
            defaults = {a.dest: a.default for a in parser._actions}
            assert defaults.get("frame_width") == expected_resolution, (
                f"{name} --frame-width default={defaults.get('frame_width')} "
                f"but expected {expected_resolution}"
            )
            assert defaults.get("frame_height") == expected_resolution, (
                f"{name} --frame-height default={defaults.get('frame_height')} "
                f"but expected {expected_resolution}"
            )

    return _check


def check_action_extraction() -> None:
    from evaluation.common import extract_actions

    assert extract_actions("Move Forward, Turn Left") == ["move_forward", "turn_left"]
    assert extract_actions("forward forward stop", max_actions=5) == [
        "move_forward",
        "move_forward",
        "stop",
    ]
    assert extract_actions("no actions here") == []


def check_image_pipeline(expected_resolution: int) -> None:
    import numpy as np

    from evaluation.common import encode_image_b64, ensure_size_bgr, process_images_as_video

    rgb_input = np.random.randint(0, 255, size=(320, 320, 3), dtype=np.uint8)
    bgr = ensure_size_bgr(rgb_input, expected_resolution, expected_resolution)
    assert bgr.shape == (expected_resolution, expected_resolution, 3), (
        f"ensure_size_bgr produced shape {bgr.shape}"
    )

    b64 = encode_image_b64(bgr, resize_to=expected_resolution)
    assert isinstance(b64, str) and len(b64) > 0

    images = [bgr for _ in range(20)]
    sampled, time_str, duration = process_images_as_video(
        images, original_fps=1.0, max_frames_num=8, target_fps=1.0
    )
    assert sampled.shape[0] == 8, f"Unexpected frame count: {sampled.shape[0]}"
    assert sampled.shape[1:3] == (expected_resolution, expected_resolution)
    assert duration == 20.0
    assert "s" in time_str


def check_metrics_math() -> None:
    from evaluation.common import summarize_results

    items = [
        {"inst-1": {"success": 1, "nav_error": 0.4, "os": 1, "shortest_path_length": 5.0, "actual_path_length": 5.0}},
        {"inst-2": {"success": 0, "nav_error": 3.2, "os": 1, "shortest_path_length": 4.0, "actual_path_length": 6.0}},
        {"inst-3": {"success": 1, "nav_error": 0.1, "os": 1, "shortest_path_length": 3.0, "actual_path_length": 6.0}},
    ]
    summary = summarize_results(items)
    assert summary["count"] == 3
    assert abs(summary["sr"] - 2 / 3) < 1e-6
    assert abs(summary["os"] - 1.0) < 1e-6
    # SPL: [1*5/5, 0, 1*3/6] / 3 = (1 + 0 + 0.5) / 3
    assert abs(summary["spl"] - (1.0 + 0.0 + 0.5) / 3) < 1e-6


def check_prompt_format() -> None:
    from evaluation.prompts import DEFAULT_SYSTEM_PROMPT, build_navigation_prompt

    prompt = build_navigation_prompt(
        instruction="go to the kitchen",
        frame_time="0.00s,1.00s",
        video_time=2.0,
        num_frames=2,
        future_steps=6,
    )
    assert "go to the kitchen" in prompt
    assert "Move forward" in prompt or "move forward" in prompt.lower()
    assert isinstance(DEFAULT_SYSTEM_PROMPT, str) and DEFAULT_SYSTEM_PROMPT.strip()


def check_resume_index_roundtrip() -> None:
    from evaluation.common import build_resume_index

    items = [
        {"inst-A": {"success": 1, "shortest_path_length": 3.0}},
        {"inst-B": {"success": 0, "shortest_path_length": 4.5}},
    ]
    index = build_resume_index(items)
    assert ("inst-A", 3.0) in index
    assert ("inst-B", 4.5) in index


def check_provider_dispatch_dryrun() -> None:
    """Validate provider dispatch logic without sending any HTTP request."""
    from evaluation.providers import ProviderRequest, infer_actions

    request = ProviderRequest(
        provider="unknown_provider",
        model="whatever",
        api_key="placeholder",
        base_url=None,
        max_retries=1,
        initial_retry_delay=0.0,
        use_system_message=False,
    )
    import numpy as np

    try:
        infer_actions(
            request,
            images=np.zeros((1, 4, 4, 3), dtype=np.uint8),
            instruction="test",
            frame_time="0.00s",
            video_time=0.0,
        )
    except ValueError as exc:
        assert "Unsupported provider" in str(exc)
    else:
        raise AssertionError("infer_actions must raise ValueError for unknown providers")


def check_requirement_files() -> None:
    for name in ("requirements-base.txt", "requirements-llm.txt", "requirements-local-model.txt"):
        path = _REPO_ROOT / name
        assert path.exists(), f"missing {name}"
        content = path.read_text(encoding="utf-8").strip()
        assert content, f"{name} is empty"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-resolution", type=int, default=224)
    parser.add_argument("--max-episodes", type=int, default=8,
                        help="Number of episodes per task validated for schema.")
    args = parser.parse_args(argv)

    tester = SmokeTest(
        expected_resolution=args.expected_resolution,
        max_episodes=args.max_episodes,
    )

    print(f"Running NavSpace offline smoke tests (expected_resolution={args.expected_resolution})")
    print("=" * 60)
    tester.run("byte-compile all Python sources", check_compileall)
    tester.run("import evaluation.* modules", check_core_imports)
    tester.run("build CLI parsers for all three entry scripts", check_entry_parsers)
    tester.run("gpt_eval.py wrapper re-exports main", check_wrapper_gpt_eval)
    tester.run("TASK_DATASET_MAP covers every task", check_task_dataset_map(tester))
    tester.run("episode schema is consistent across tasks", check_episode_schema(tester))
    tester.run("scene_id values are parseable for resolver", check_scene_id_reachable)
    tester.run("LLM profiles share the expected resolution",
               check_profiles_consistency(args.expected_resolution))
    tester.run("local-model CLIs use the expected resolution",
               check_entry_resolution_defaults(args.expected_resolution))
    tester.run("action extraction parses known verbs", check_action_extraction)
    tester.run("image pipeline end-to-end (resize + base64 + video sampling)",
               lambda: check_image_pipeline(args.expected_resolution))
    tester.run("metrics math matches expectations", check_metrics_math)
    tester.run("navigation prompt contains required tokens", check_prompt_format)
    tester.run("resume index round-trip is stable", check_resume_index_roundtrip)
    tester.run("provider dispatch rejects unknown providers cleanly",
               check_provider_dispatch_dryrun)
    tester.run("requirements-*.txt are non-empty", check_requirement_files)

    return tester.summary()


if __name__ == "__main__":
    raise SystemExit(main())
