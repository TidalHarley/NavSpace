"""Minimal image-to-text LLM helper used by the annotation web UI.

This module is intentionally lightweight so the annotation pipeline can be open-sourced
without leaking any private endpoint / API key. It calls an OpenAI-compatible vision
endpoint (default: ChatAnywhere) and returns a short Chinese description of the image.

Environment variables
---------------------
- ``NAVSPACE_ANNO_API_KEY`` / ``OPENAI_API_KEY`` : API key (required)
- ``NAVSPACE_ANNO_BASE_URL`` : base URL, default ``https://api.chatanywhere.tech/v1``
- ``NAVSPACE_ANNO_MODEL``    : model name, default ``gpt-4o-mini``
- ``NAVSPACE_ANNO_PROMPT``   : prompt template, default: short Chinese description

The function signature intentionally mirrors the original private helper used by the
annotation server so the server code stays unchanged.
"""

from __future__ import annotations

import os
from typing import Optional


_DEFAULT_PROMPT = (
    "请你用一段简洁的中文描述下面这张室内场景的主要布局、可见物体和潜在的导航线索，"
    "不要超过 120 字。"
)


def _resolve_api_key() -> Optional[str]:
    for key in ("NAVSPACE_ANNO_API_KEY", "OPENAI_API_KEY"):
        value = os.environ.get(key)
        if value:
            return value
    return None


def call_llm_api(image_base64: str) -> str:
    """Send a single base64-encoded image to a vision LLM and return the text reply.

    The return value is either the model's text response, or a string that starts
    with ``"错误:"`` when an error occurs. The annotation server treats the latter
    as a user-facing error message.
    """

    api_key = _resolve_api_key()
    if not api_key:
        return (
            "错误: 未检测到 NAVSPACE_ANNO_API_KEY / OPENAI_API_KEY 环境变量，"
            "请先配置再使用大模型辅助按钮。"
        )

    try:
        from openai import OpenAI  # lazy import so the file is importable without the SDK
    except Exception as exc:  # pragma: no cover - import error path
        return f"错误: 未安装 openai SDK ({exc})，请先 `pip install openai`."

    base_url = os.environ.get("NAVSPACE_ANNO_BASE_URL", "https://api.chatanywhere.tech/v1")
    model = os.environ.get("NAVSPACE_ANNO_MODEL", "gpt-4o-mini")
    prompt = os.environ.get("NAVSPACE_ANNO_PROMPT", _DEFAULT_PROMPT)

    # Accept both raw base64 strings and already-prefixed data URIs.
    if image_base64.startswith("data:"):
        data_url = image_base64
    else:
        data_url = f"data:image/jpeg;base64,{image_base64}"

    try:
        client = OpenAI(api_key=api_key, base_url=base_url)
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                }
            ],
            max_tokens=256,
            temperature=0.2,
        )
        text = (resp.choices[0].message.content or "").strip()
        if not text:
            return "错误: 大模型返回为空。"
        return text
    except Exception as exc:  # pragma: no cover - network / auth errors
        return f"错误: 调用大模型失败 ({exc})."
