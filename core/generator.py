from __future__ import annotations

from astrbot.api import logger

from ..adapter import GeminiAdapter, GeminiOpenAIAdapter, GeminiZaiAdapter
from .types import (
    AdapterConfig,
    AdapterType,
    GenerationRequest,
    GenerationResult,
    ImageData,
)
from .utils import convert_images_batch


class ImageGenerator:
    """Adapter orchestrator responsible for dispatching generation requests."""

    def __init__(self, adapter_config: AdapterConfig):
        self.adapter_config = adapter_config
        self.adapter = self._create_adapter(adapter_config)

    def _create_adapter(self, config: AdapterConfig):
        adapter_map: dict[AdapterType, type] = {
            AdapterType.GEMINI: GeminiAdapter,
            AdapterType.GEMINI_OPENAI: GeminiOpenAIAdapter,
            AdapterType.GEMINI_ZAI: GeminiZaiAdapter,
        }

        adapter_cls = adapter_map.get(config.type)
        if not adapter_cls:
            raise ValueError(f"Unsupported adapter type: {config.type}")
        return adapter_cls(config)

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        if not self.adapter:
            return GenerationResult(images=None, error="适配器未初始化")

        # 先将参考图批量转换成兼容格式，再调用下游适配器
        converted_images: list[ImageData] = []
        if request.images:
            converted_images = await convert_images_batch(request.images)

        patched_request = GenerationRequest(
            prompt=request.prompt,
            images=converted_images,
            aspect_ratio=request.aspect_ratio,
            resolution=request.resolution,
            task_id=request.task_id,
        )

        try:
            return await self.adapter.generate(patched_request)
        except Exception as exc:  # noqa: BLE001
            logger.error(f"[ImageGen] Generation failed: {exc}", exc_info=True)
            return GenerationResult(images=None, error=str(exc))

    def update_model(self, model: str) -> None:
        if self.adapter:
            self.adapter.update_model(model)

    async def close(self) -> None:
        if self.adapter:
            await self.adapter.close()
