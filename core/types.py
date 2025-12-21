from __future__ import annotations

import enum
from dataclasses import dataclass, field


class AdapterType(str, enum.Enum):
    """Supported image generation adapter types."""

    GEMINI = "gemini"
    GEMINI_OPENAI = "gemini(OpenAI)"
    GEMINI_ZAI = "gemini(Zai)"


@dataclass
class AdapterMetadata:
    """Metadata about an adapter's capabilities."""

    name: str
    supports_aspect_ratio: bool = True
    supports_resolution: bool = True


@dataclass
class AdapterConfig:
    """Configuration required to construct an adapter."""

    type: AdapterType = AdapterType.GEMINI
    base_url: str | None = None
    api_keys: list[str] = field(default_factory=list)
    model: str = ""
    available_models: list[str] = field(default_factory=list)
    provider_id: str | None = None
    proxy: str | None = None
    timeout: int = 180
    max_retry_attempts: int = 3
    safety_settings: str | None = None


@dataclass
class ImageData:
    """Image bytes with an associated MIME type."""

    data: bytes
    mime_type: str


@dataclass
class GenerationRequest:
    """User-facing generation request."""

    prompt: str
    images: list[ImageData] = field(default_factory=list)
    aspect_ratio: str | None = None
    resolution: str | None = None
    task_id: str | None = None


@dataclass
class GenerationResult:
    """Result of a generation attempt."""

    images: list[bytes] | None = None
    error: str | None = None
