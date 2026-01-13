"""
插件配置管理模块
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from astrbot.api import logger
from astrbot.core.config.astrbot_config import AstrBotConfig

from .types import AdapterConfig, AdapterType


@dataclass
class UsageSettings:
    """用户使用限制设置。"""

    rate_limit_seconds: int = 0
    enable_daily_limit: bool = False
    daily_limit_count: int = 10
    max_image_size_mb: int = 10


@dataclass
class CacheSettings:
    """缓存设置。"""

    max_cache_count: int = 100
    cleanup_interval_hours: int = 24


@dataclass
class GenerationSettings:
    """生成设置。"""

    default_aspect_ratio: str = "自动"
    default_resolution: str = "1K"
    max_concurrent_tasks: int = 3
    show_generation_info: bool = False
    show_model_info: bool = False


@dataclass
class PluginConfig:
    """完整的插件配置。"""

    adapter_config: AdapterConfig | None = None
    usage_settings: UsageSettings = field(default_factory=UsageSettings)
    cache_settings: CacheSettings = field(default_factory=CacheSettings)
    generation_settings: GenerationSettings = field(default_factory=GenerationSettings)
    presets: dict[str, Any] = field(default_factory=dict)
    enable_llm_tool: bool = True


class ConfigManager:
    """插件配置管理器。"""

    def __init__(self, config: AstrBotConfig):
        self._config = config
        self._plugin_config: PluginConfig = PluginConfig()
        self._all_provider_configs: list[AdapterConfig] = []  # 保存所有供应商配置
        self.load()

    def load(self) -> PluginConfig:
        """加载并解析插件配置。"""
        gen_cfg = self._config.get("generation", {})
        user_limits_cfg = self._config.get("user_limits", {})
        cache_cfg = self._config.get("cache", {})
        api_providers_raw = self._config.get("api_providers", [])

        self._plugin_config.enable_llm_tool = self._config.get("enable_llm_tool", True)

        # 1. 收集所有供应商配置
        all_provider_configs: list[AdapterConfig] = []
        for provider_item in api_providers_raw:
            if not isinstance(provider_item, dict):
                continue

            adapter_type_str = provider_item.get("__template_key")
            if not adapter_type_str:
                continue

            try:
                adapter_type = AdapterType(adapter_type_str)
            except ValueError:
                logger.warning(f"[ImageGen] 忽略未知适配器类型: {adapter_type_str}")
                continue

            name = provider_item.get("name", "")
            base_url = (provider_item.get("base_url") or "").strip()
            api_keys = [k for k in provider_item.get("api_keys", []) if k]
            available_models = provider_item.get("available_models") or []
            proxy = (provider_item.get("proxy") or "").strip() or None

            all_provider_configs.append(
                AdapterConfig(
                    type=adapter_type,
                    name=name,
                    base_url=self._clean_base_url(base_url),
                    api_keys=api_keys,
                    available_models=available_models,
                    proxy=proxy,
                    timeout=gen_cfg.get("timeout", 180),
                    max_retry_attempts=gen_cfg.get("max_retry_attempts", 3),
                )
            )

        # 保存所有供应商配置供后续使用
        self._all_provider_configs = all_provider_configs

        # 2. 获取当前选择的模型
        model_setting = gen_cfg.get("model", "")

        # 3. 匹配当前适配器
        matched_config = None
        current_model = ""

        if "/" in model_setting:
            try:
                target_provider_name, target_model = model_setting.split("/", 1)
                for cfg in all_provider_configs:
                    if cfg.name == target_provider_name:
                        matched_config = cfg
                        current_model = target_model
                        break
            except ValueError:
                logger.warning(
                    f"[ImageGen] 模型设置格式错误: {model_setting}，期望格式为 '供应商/模型'"
                )

        # 如果没匹配到（或者没设置），取第一个可用的
        if not matched_config and all_provider_configs:
            matched_config = all_provider_configs[0]
            current_model = (
                matched_config.available_models[0]
                if matched_config.available_models
                else ""
            )
            logger.info(
                f"[ImageGen] 未匹配到当前模型配置，默认使用: {matched_config.name}/{current_model}"
            )

        if matched_config:
            self._plugin_config.adapter_config = matched_config
            self._plugin_config.adapter_config.model = current_model
            # 将所有可用模型汇总，供切换指令使用，格式为 "供应商名称/模型名称"
            all_available_models = []
            for cfg in all_provider_configs:
                for m in cfg.available_models:
                    all_available_models.append(f"{cfg.name}/{m}")
            self._plugin_config.adapter_config.available_models = all_available_models
        else:
            self._plugin_config.adapter_config = None
            logger.error("[ImageGen] 未找到任何有效的生图模型配置")

        # 用户限制设置
        self._plugin_config.usage_settings = UsageSettings(
            rate_limit_seconds=max(0, user_limits_cfg.get("rate_limit_seconds", 0)),
            max_image_size_mb=max(1, user_limits_cfg.get("max_image_size_mb", 10)),
            enable_daily_limit=user_limits_cfg.get("enable_daily_limit", False),
            daily_limit_count=max(1, user_limits_cfg.get("daily_limit_count", 10)),
        )

        # 缓存设置
        self._plugin_config.cache_settings = CacheSettings(
            max_cache_count=max(1, cache_cfg.get("max_cache_count", 100)),
            cleanup_interval_hours=max(1, cache_cfg.get("cleanup_interval_hours", 24)),
        )

        # 生成设置
        self._plugin_config.generation_settings = GenerationSettings(
            default_aspect_ratio=gen_cfg.get("default_aspect_ratio", "自动"),
            default_resolution=gen_cfg.get("default_resolution", "1K"),
            max_concurrent_tasks=max(1, gen_cfg.get("max_concurrent_tasks", 3)),
            show_generation_info=gen_cfg.get("show_generation_info", False),
            show_model_info=gen_cfg.get("show_model_info", False),
        )

        # 预设
        self._plugin_config.presets = self._load_presets(
            self._config.get("presets", [])
        )

        return self._plugin_config

    def reload(self) -> PluginConfig:
        """重新加载配置。"""
        return self.load()

    def _clean_base_url(self, url: str) -> str:
        """清理 Base URL，移除末尾的 /v1*"""
        if not url:
            return ""
        url = url.rstrip("/")
        if "/v1" in url:
            url = url.split("/v1", 1)[0]
        return url.rstrip("/")

    def _load_presets(self, presets_config: list[Any]) -> dict[str, Any]:
        """加载预设配置。"""
        presets: dict[str, Any] = {}
        if not isinstance(presets_config, list):
            return presets

        for preset_str in presets_config:
            if isinstance(preset_str, str) and ":" in preset_str:
                name, prompt = preset_str.split(":", 1)
                if name.strip() and prompt.strip():
                    presets[name.strip()] = prompt.strip()
        return presets

    def save_model_setting(self, model: str) -> None:
        """保存模型设置。"""
        self._config.setdefault("generation", {})["model"] = model
        self._config.save_config()

    def save_preset(self, name: str, content: str) -> None:
        """保存预设。"""
        self._plugin_config.presets[name] = content
        self._config["presets"] = [
            f"{k}:{v}" for k, v in self._plugin_config.presets.items()
        ]
        self._config.save_config()

    def delete_preset(self, name: str) -> bool:
        """删除预设，返回是否成功。"""
        if name in self._plugin_config.presets:
            del self._plugin_config.presets[name]
            self._config["presets"] = [
                f"{k}:{v}" for k, v in self._plugin_config.presets.items()
            ]
            self._config.save_config()
            return True
        return False

    # ---------------------- 便捷属性访问 ----------------------
    @property
    def adapter_config(self) -> AdapterConfig | None:
        """获取适配器配置。"""
        return self._plugin_config.adapter_config

    @property
    def presets(self) -> dict[str, Any]:
        """获取预设字典。"""
        return self._plugin_config.presets

    @property
    def enable_llm_tool(self) -> bool:
        """是否启用 LLM 工具。"""
        return self._plugin_config.enable_llm_tool

    @property
    def default_aspect_ratio(self) -> str:
        """默认宽高比。"""
        return self._plugin_config.generation_settings.default_aspect_ratio

    @property
    def default_resolution(self) -> str:
        """默认分辨率。"""
        return self._plugin_config.generation_settings.default_resolution

    @property
    def max_concurrent_tasks(self) -> int:
        """最大并发任务数。"""
        return self._plugin_config.generation_settings.max_concurrent_tasks

    @property
    def show_generation_info(self) -> bool:
        """是否显示生成信息。"""
        return self._plugin_config.generation_settings.show_generation_info

    @property
    def show_model_info(self) -> bool:
        """是否显示模型信息。"""
        return self._plugin_config.generation_settings.show_model_info

    @property
    def usage_settings(self) -> UsageSettings:
        """用户使用限制设置。"""
        return self._plugin_config.usage_settings

    @property
    def cache_settings(self) -> CacheSettings:
        """缓存设置。"""
        return self._plugin_config.cache_settings

    # ---------------------- 供应商查询方法 ----------------------
    def has_provider_type(self, adapter_type: AdapterType) -> bool:
        """检查配置中是否包含指定类型的供应商。

        Args:
            adapter_type: 要检查的适配器类型。

        Returns:
            如果配置中包含该类型的供应商则返回 True，否则返回 False。
        """
        return any(cfg.type == adapter_type for cfg in self._all_provider_configs)

    def get_provider_config(self, adapter_type: AdapterType) -> AdapterConfig | None:
        """获取指定类型的供应商配置。

        Args:
            adapter_type: 要获取的适配器类型。

        Returns:
            匹配的供应商配置，如果没有则返回 None。
        """
        for cfg in self._all_provider_configs:
            if cfg.type == adapter_type:
                return cfg
        return None
