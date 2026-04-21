from __future__ import annotations

import logging
import random
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from evaluation.common import encode_image_b64
from evaluation.prompts import DEFAULT_SYSTEM_PROMPT, build_navigation_prompt


LOGGER = logging.getLogger(__name__)


@dataclass
class ProviderRequest:
    provider: str
    model: str
    api_key: str
    base_url: Optional[str]
    max_retries: int
    initial_retry_delay: float
    use_system_message: bool
    system_prompt: str = DEFAULT_SYSTEM_PROMPT
    encode_resize: Optional[int] = None


def _extract_text_content(response) -> str:
    if not getattr(response, "choices", None):
        return ""
    message = response.choices[0].message
    content = getattr(message, "content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                parts.append(part["text"])
        return "\n".join(parts)
    return ""


def _build_messages(
    images: np.ndarray,
    prompt_text: str,
    encode_resize: Optional[int],
    use_system_message: bool,
    system_prompt: str,
) -> list[dict]:
    content_parts = []
    for image in images:
        image_b64 = encode_image_b64(image, resize_to=encode_resize)
        content_parts.append(
            {
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"},
            }
        )
    content_parts.append({"type": "text", "text": prompt_text})

    messages = []
    if use_system_message:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": content_parts})
    return messages


def _call_openai_compatible(request: ProviderRequest, messages: list[dict]) -> str:
    from openai import OpenAI

    client = OpenAI(api_key=request.api_key, base_url=request.base_url)
    retry_delay = request.initial_retry_delay
    for attempt in range(request.max_retries):
        try:
            response = client.chat.completions.create(model=request.model, messages=messages)
            content = _extract_text_content(response)
            if content:
                return content
        except Exception as exc:  # noqa: BLE001
            if attempt == request.max_retries - 1:
                raise RuntimeError(f"API call failed after retries: {exc}") from exc
            LOGGER.warning("API call failed, retry %s/%s: %s", attempt + 1, request.max_retries, exc)
            time.sleep(retry_delay)
            retry_delay *= 2
    raise RuntimeError("API call returned empty content after retries.")


def _call_zhipu(request: ProviderRequest, messages: list[dict]) -> str:
    from openai import RateLimitError
    from zai import ZhipuAiClient

    client = ZhipuAiClient(api_key=request.api_key)
    for attempt in range(request.max_retries):
        try:
            time.sleep(request.initial_retry_delay)
            response = client.chat.completions.create(model=request.model, messages=messages)
            content = _extract_text_content(response)
            if content:
                return content
        except RateLimitError as exc:
            if attempt == request.max_retries - 1:
                raise RuntimeError("Zhipu rate limit exceeded after retries.") from exc
            wait_time = (18 * 3**attempt) + random.uniform(0, 2)
            LOGGER.warning("Zhipu rate limited, sleeping %.2fs", wait_time)
            time.sleep(wait_time)
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(f"Zhipu API call failed: {exc}") from exc
    raise RuntimeError("Zhipu API call returned empty content after retries.")


def infer_actions(
    request: ProviderRequest,
    images: np.ndarray,
    instruction: str,
    frame_time: str,
    video_time: float,
    future_steps: int = 6,
) -> str:
    prompt_text = build_navigation_prompt(
        instruction=instruction,
        frame_time=frame_time,
        video_time=video_time,
        num_frames=len(images),
        future_steps=future_steps,
    )
    messages = _build_messages(
        images=images,
        prompt_text=prompt_text,
        encode_resize=request.encode_resize,
        use_system_message=request.use_system_message,
        system_prompt=request.system_prompt,
    )
    if request.provider == "openai_compatible":
        return _call_openai_compatible(request, messages)
    if request.provider == "zhipu":
        return _call_zhipu(request, messages)
    raise ValueError(f"Unsupported provider: {request.provider}")
