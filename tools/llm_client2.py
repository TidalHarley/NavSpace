from __future__ import annotations

import argparse
import sys
from pathlib import Path

from openai import OpenAI

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from evaluation.common import load_api_key
from tools.text_prompt import get_task_prompt_translate


def call_llm_api(
    user_text: str,
    model: str,
    api_key: str,
    base_url: str,
) -> str:
    client = OpenAI(api_key=api_key, base_url=base_url)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "你是专业的中英翻译助手。"},
            {"role": "user", "content": f"{get_task_prompt_translate()}\n\n输入：{user_text}"},
        ],
    )
    return response.choices[0].message.content or ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Small utility for prompt translation.")
    parser.add_argument("text")
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
    print(call_llm_api(args.text, args.model, api_key, args.base_url))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
