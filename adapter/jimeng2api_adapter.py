from __future__ import annotations

import asyncio
import base64

import aiohttp

from astrbot.api import logger

from ..core.base_adapter import BaseImageAdapter
from ..core.types import GenerationRequest, GenerationResult, ImageCapability


class Jimeng2APIAdapter(BaseImageAdapter):
    """Jimeng2API 图像生成适配器。"""

    def get_capabilities(self) -> ImageCapability:
        """获取适配器支持的功能。"""
        return (
            ImageCapability.TEXT_TO_IMAGE
            | ImageCapability.IMAGE_TO_IMAGE
            | ImageCapability.RESOLUTION
            | ImageCapability.ASPECT_RATIO
        )

    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """执行生图逻辑。"""
        if not self.api_keys:
            return GenerationResult(images=None, error="未配置 API Key")

        last_error = "未配置 API Key"
        for attempt in range(self.max_retry_attempts):
            if attempt:
                logger.info(
                    f"[ImageGen] 重试 Jimeng2API 适配器 ({attempt + 1}/{self.max_retry_attempts})"
                )

            images, err = await self._generate_once(request)
            if images is not None:
                return GenerationResult(images=images, error=None)

            last_error = err or "生成失败"
            if attempt < self.max_retry_attempts - 1:
                self._rotate_api_key()
                if (attempt + 1) % max(1, len(self.api_keys)) == 0:
                    await asyncio.sleep(
                        min(2 ** ((attempt + 1) // len(self.api_keys)), 10)
                    )

        return GenerationResult(images=None, error=f"重试失败: {last_error}")

    async def _generate_once(
        self, request: GenerationRequest
    ) -> tuple[list[bytes] | None, str | None]:
        """执行单次生图请求。"""
        session = self._get_session()

        base_url = self.base_url or "http://localhost:5100"
        headers = {
            "Authorization": f"Bearer {self._get_current_api_key()}",
        }

        try:
            if request.images:
                # 图生图
                url = f"{base_url.rstrip('/')}/v1/images/compositions"

                # 使用 multipart/form-data
                data = aiohttp.FormData()
                data.add_field("model", self.model or "jimeng-4.5")
                data.add_field("prompt", request.prompt)
                if request.aspect_ratio:
                    if request.aspect_ratio == "自动":
                        data.add_field("intelligent_ratio", "true")
                    else:
                        data.add_field("ratio", request.aspect_ratio)
                if request.resolution:
                    data.add_field("resolution", request.resolution.lower())

                for img in request.images:
                    data.add_field(
                        "images",
                        img.data,
                        filename=f"image.{img.mime_type.split('/')[-1]}",
                        content_type=img.mime_type,
                    )

                async with session.post(
                    url,
                    data=data,
                    headers=headers,
                    proxy=self.proxy,
                    timeout=self.timeout,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(
                            f"[ImageGen] Jimeng2API Compositions 错误 ({resp.status}): {error_text}"
                        )
                        return None, f"API 错误 ({resp.status})"

                    data_json = await resp.json()
                    return await self._extract_images(data_json)
            else:
                # 文生图
                url = f"{base_url.rstrip('/')}/v1/images/generations"
                headers["Content-Type"] = "application/json"

                payload = {
                    "model": self.model or "jimeng-4.5",
                    "prompt": request.prompt,
                    "response_format": "url",  # 默认使用 url，然后下载
                }
                if request.aspect_ratio:
                    if request.aspect_ratio == "自动":
                        payload["intelligent_ratio"] = True
                    else:
                        payload["ratio"] = request.aspect_ratio
                if request.resolution:
                    payload["resolution"] = request.resolution.lower()

                async with session.post(
                    url,
                    json=payload,
                    headers=headers,
                    proxy=self.proxy,
                    timeout=self.timeout,
                ) as resp:
                    if resp.status != 200:
                        error_text = await resp.text()
                        logger.error(
                            f"[ImageGen] Jimeng2API Generations 错误 ({resp.status}): {error_text}"
                        )
                        return None, f"API 错误 ({resp.status})"

                    data_json = await resp.json()
                    return await self._extract_images(data_json)

        except Exception as e:
            logger.error(f"[ImageGen] Jimeng2API 请求异常: {e}")
            return None, str(e)

    async def _extract_images(
        self, response: dict
    ) -> tuple[list[bytes] | None, str | None]:
        """从响应中提取图片数据。"""
        if "data" not in response:
            return None, "响应中未找到 data 字段"

        images = []
        for item in response["data"]:
            if "b64_json" in item:
                images.append(base64.b64decode(item["b64_json"]))
            elif "url" in item:
                async with self._get_session().get(
                    item["url"], proxy=self.proxy
                ) as resp:
                    if resp.status == 200:
                        images.append(await resp.read())

        if not images:
            return None, "未找到有效的图片数据"

        return images, None
