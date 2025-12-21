from __future__ import annotations

import abc

import aiohttp

from astrbot.api import logger

from .types import AdapterConfig, GenerationRequest, GenerationResult


class BaseImageAdapter(abc.ABC):
    """Base class for image generation adapters."""

    def __init__(self, config: AdapterConfig):
        self.config = config
        self.api_keys = config.api_keys or []
        self.current_key_index = 0
        self.base_url = (config.base_url or "").rstrip("/")
        self.model = config.model
        self.proxy = config.proxy
        self.timeout = config.timeout
        self.max_retry_attempts = max(1, config.max_retry_attempts)
        self.safety_settings = config.safety_settings
        self._session: aiohttp.ClientSession | None = None

    async def close(self) -> None:
        """Close underlying HTTP session."""

        if self._session and not self._session.closed:
            await self._session.close()
        self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session

    def _get_current_api_key(self) -> str:
        if not self.api_keys:
            return ""
        return self.api_keys[self.current_key_index % len(self.api_keys)]

    def _rotate_api_key(self) -> None:
        if len(self.api_keys) > 1:
            self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
            logger.info(f"[ImageGen] Rotate API key -> index {self.current_key_index}")

    def update_model(self, model: str) -> None:
        self.model = model

    @abc.abstractmethod
    async def generate(self, request: GenerationRequest) -> GenerationResult:
        """Generate images for the given request."""
