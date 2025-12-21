from __future__ import annotations

import asyncio
from collections.abc import Iterable
from io import BytesIO

from PIL import Image

from astrbot.api import logger

from .types import ImageData

SUPPORTED_IMAGE_FORMATS = {
    "image/png",
    "image/jpeg",
    "image/webp",
    "image/heic",
    "image/heif",
}

ALLOWED_ASPECT_RATIOS = {
    "1:1",
    "2:3",
    "3:2",
    "3:4",
    "4:3",
    "4:5",
    "5:4",
    "9:16",
    "16:9",
    "21:9",
    "自动",
}

ALLOWED_RESOLUTIONS = {"1K", "2K", "4K"}


def detect_mime_type(data: bytes) -> str:
    """Best-effort MIME detection from magic numbers."""

    if data.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        return "image/png"
    if data.startswith(b"GIF87a") or data.startswith(b"GIF89a"):
        return "image/gif"
    if len(data) > 12 and data[4:8] == b"ftyp":
        brand = data[8:12]
        if brand in (b"heic", b"heix", b"heim", b"heis"):
            return "image/heic"
        if brand in (b"mif1", b"msf1", b"heif"):
            return "image/heif"
    if data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        return "image/webp"
    return "application/octet-stream"


def _sync_convert_image_format(image_data: bytes, mime_type: str) -> ImageData:
    """Convert unsupported images to JPEG synchronously."""

    try:
        img = Image.open(BytesIO(image_data))

        if img.mode in ("RGBA", "LA", "P"):
            background = Image.new("RGB", img.size, (255, 255, 255))
            if img.mode == "P":
                img = img.convert("RGBA")
            elif img.mode == "LA":
                img = img.convert("RGBA")
            background.paste(img, mask=img.split()[3])
            img = background

        output = BytesIO()
        img.save(output, format="JPEG", quality=95)
        logger.debug("[ImageGen] Converted image to JPEG")
        return ImageData(data=output.getvalue(), mime_type="image/jpeg")
    except Exception as exc:  # noqa: BLE001
        logger.error(f"[ImageGen] Failed to convert image: {exc}")
        return ImageData(data=image_data, mime_type=mime_type)


async def convert_image_format(image_data: bytes, mime_type: str) -> ImageData:
    """Convert an image if its MIME type is unsupported."""

    real_mime = detect_mime_type(image_data)
    if real_mime in SUPPORTED_IMAGE_FORMATS:
        return ImageData(data=image_data, mime_type=real_mime)
    logger.info(f"[ImageGen] Converting image format: {mime_type} -> image/jpeg")
    return await asyncio.to_thread(_sync_convert_image_format, image_data, mime_type)


async def convert_images_batch(images: Iterable[ImageData]) -> list[ImageData]:
    """Convert a batch of images in parallel."""

    tasks = [convert_image_format(img.data, img.mime_type) for img in images]
    return await asyncio.gather(*tasks)


def validate_aspect_ratio(value: str | None) -> str | None:
    """Validate aspect ratio from the allowed set."""

    if value is None:
        return None
    return value if value in ALLOWED_ASPECT_RATIOS else None


def validate_resolution(value: str | None) -> str | None:
    """Validate resolution from the allowed set."""

    if value is None:
        return None
    return value if value in ALLOWED_RESOLUTIONS else None
