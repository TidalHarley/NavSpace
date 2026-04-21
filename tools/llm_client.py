from __future__ import annotations

import argparse
import base64
import sys
from pathlib import Path

from openai import OpenAI

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.common import load_api_key
from tools.text_prompt import get_task_prompt


def call_llm_api(
    image_path: Path,
    model: str,
    api_key: str,
    base_url: str,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    image_b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    messages = [
        {
            "role": "system",
            "content": "你是专业的图像分析助手，请基于用户意图客观、简洁地描述图像并指出重要细节。",
        },
        {
            "role": "user",
            "content": [
                {"type": "text", "text": get_task_prompt()},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        },
    ]
    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Small utility for image-based prompt generation.")
    parser.add_argument("image_path")
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--base-url", default="https://api.chatanywhere.tech/v1")
    parser.add_argument("--api-key")
    parser.add_argument("--api-key-env", default="OPENAI_API_KEY")
    parser.add_argument("--api-key-file")
    args = parser.parse_args(argv)

    api_key = load_api_key(
        direct_key=args.api_key,
        env_name=args.api_key_env,
        key_file=Path(args.api_key_file).resolve() if args.api_key_file else None,
    )
    print(call_llm_api(Path(args.image_path).resolve(), args.model, api_key, args.base_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
