from __future__ import annotations

import base64
from typing import Any

from ..core.types import GenerationRequest
from .gemini_openai_adapter import GeminiOpenAIAdapter


class GeminiZaiAdapter(GeminiOpenAIAdapter):
    """带有额外参数的 Zai 风格 OpenAI 适配器。"""

    def _build_payload(self, request: GenerationRequest) -> dict[str, Any]:
        """构建请求载荷。"""
        message_content: list[dict] = [{"type": "text", "text": request.prompt}]

        for image in request.images:
            b64_data = base64.b64encode(image.data).decode("utf-8")
            message_content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": f"data:{image.mime_type};base64,{b64_data}"},
                }
            )

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": message_content}],
            "stream": False,
        }

        params: dict[str, Any] = {}
        if request.aspect_ratio and not request.images:
            params["image_aspect_ratio"] = request.aspect_ratio
        if request.resolution:
            params["image_resolution"] = request.resolution
        if params:
            payload["params"] = params

        return payload
