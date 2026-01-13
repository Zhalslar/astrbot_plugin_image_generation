"""
AstrBot 图像生成插件主模块

重构后的精简版本，核心逻辑已拆分到 core/ 目录下的各个模块中。
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import Coroutine
from typing import Any

from astrbot.api import logger
from astrbot.api.event import AstrMessageEvent, MessageChain, filter
from astrbot.api.star import Context, Star
from astrbot.core.config.astrbot_config import AstrBotConfig
from astrbot.core.star.star_tools import StarTools

from .core.config_manager import ConfigManager
from .core.generator import ImageGenerator
from .core.image_processor import ImageProcessor
from .core.llm_tool import ImageGenerationTool, adjust_tool_parameters
from .core.task_manager import TaskManager
from .core.types import GenerationRequest, ImageCapability, ImageData
from .core.usage_manager import UsageManager
from .core.utils import mask_sensitive, validate_aspect_ratio, validate_resolution


class ImageGenerationPlugin(Star):
    """图像生成插件主类"""

    def __init__(self, context: Context, config: AstrBotConfig):
        super().__init__(context)
        self.context = context

        # 数据目录配置
        self.data_dir = StarTools.get_data_dir()
        self.cache_dir = self.data_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 初始化配置管理器
        self.config_manager = ConfigManager(config)

        # 初始化使用数据管理器
        self.usage_manager = UsageManager(
            str(self.data_dir), self.config_manager.usage_settings
        )

        # 初始化图片处理器
        self.image_processor = ImageProcessor(
            str(self.cache_dir),
            self.config_manager.usage_settings.max_image_size_mb,
            self.config_manager.cache_settings.max_cache_count,
        )

        # 初始化任务管理器
        self.task_manager = TaskManager()

        # 初始化生成器
        self.generator: ImageGenerator | None = None
        self.semaphore: asyncio.Semaphore | None = None


    # ---------------------- 生命周期 ----------------------

    async def initialize(self):
        """插件加载时调用"""
        if self.config_manager.adapter_config:
            self.generator = ImageGenerator(self.config_manager.adapter_config)
            self.semaphore = asyncio.Semaphore(self.config_manager.max_concurrent_tasks)
        else:
            logger.error("[ImageGen] 适配器配置加载失败，插件未初始化")

        # 注册 LLM 工具
        if self.config_manager.enable_llm_tool and self.generator:
            tool = ImageGenerationTool(plugin=self)
            self._adjust_tool_parameters(tool)
            self.context.add_llm_tools(tool)
            logger.info("[ImageGen] 已注册图像生成工具")

        # 配置定时任务
        self._setup_tasks()

        # 执行启动任务（在后台异步执行）
        self.task_manager.create_task(self.task_manager.run_startup_tasks())

        logger.info(
            f"[ImageGen] 插件加载完成，模型: {self.config_manager.adapter_config.model if self.config_manager.adapter_config else '未知'}"
        )

    async def terminate(self):
        """插件卸载时调用"""
        try:
            if self.generator:
                await self.generator.close()
            await self.task_manager.cancel_all()
            logger.info("[ImageGen] 插件已卸载")
        except Exception as exc:
            logger.error(f"[ImageGen] 卸载清理出错: {exc}")

    # ---------------------- 内部工具 ----------------------

    def _setup_tasks(self) -> None:
        """配置并启动定时任务。"""
        # 1. 缓存清理任务
        self.task_manager.start_loop_task(
            name="cache_cleanup",
            coro_func=self.image_processor.cleanup_cache,
            interval_seconds=self.config_manager.cache_settings.cleanup_interval_hours
            * 3600,
            run_immediately=True,
        )

        # 2. Jimeng2API 自动领积分任务
        self._setup_jimeng_token_task()

    def _setup_jimeng_token_task(self) -> None:
        """配置即梦自动领积分任务。

        该任务会：
        1. 在插件启动时执行一次（通过启动任务）
        2. 每天日期变更时自动执行（通过每日任务）

        注意：只要配置中包含即梦渠道，就会启用该任务，
        无论当前使用的是哪个渠道。
        """
        from .adapter.jimeng2api_adapter import Jimeng2APIAdapter
        from .core.types import AdapterType

        # 检查配置中是否包含即梦渠道（而非检查当前适配器）
        jimeng_config = self.config_manager.get_provider_config(AdapterType.JIMENG2API)
        if not jimeng_config:
            return

        # 创建专门用于任务的即梦适配器实例
        jimeng_adapter = Jimeng2APIAdapter(jimeng_config)

        # 1. 注册为启动任务，插件启动时执行一次
        self.task_manager.register_startup_task(
            name="jimeng_token_receive",
            coro_func=jimeng_adapter.receive_token,
        )

        # 2. 注册为每日任务，日期变更时执行
        self.task_manager.start_daily_task(
            name="jimeng_token_receive",
            coro_func=jimeng_adapter.receive_token,
            check_interval_seconds=300,  # 每5分钟检查一次日期变更
            run_immediately=False,  # 启动任务已处理，无需重复执行
        )
        logger.info("[ImageGen] 已配置即梦2API自动领积分任务（启动时+每日）")

    def _adjust_tool_parameters(self, tool: ImageGenerationTool) -> None:
        """根据适配器能力动态调整工具参数。"""
        if not self.generator or not self.generator.adapter:
            return
        capabilities = self.generator.adapter.get_capabilities()
        adjust_tool_parameters(tool, capabilities)

    def create_background_task(self, coro: Coroutine[Any, Any, Any]) -> asyncio.Task:
        """创建后台任务并添加到管理器中。"""
        return self.task_manager.create_task(coro)

    # ---------------------- 核心生图逻辑 ----------------------

    async def _generate_and_send_image_async(
        self,
        prompt: str,
        unified_msg_origin: str,
        images_data: list[tuple[bytes, str]] | None = None,
        aspect_ratio: str = "1:1",
        resolution: str = "1K",
        task_id: str | None = None,
    ) -> None:
        """异步生成图片并发送。"""
        if not self.generator or not self.generator.adapter:
            return

        capabilities = self.generator.adapter.get_capabilities()

        # 检查并清理不支持的参数
        if not (capabilities & ImageCapability.IMAGE_TO_IMAGE) and images_data:
            logger.warning(
                f"[ImageGen] 当前适配器不支持参考图，已忽略 {len(images_data)} 张图片"
            )
            images_data = None

        if not (capabilities & ImageCapability.ASPECT_RATIO) and aspect_ratio != "自动":
            logger.info(
                f"[ImageGen] 当前适配器不支持指定比例，已忽略参数: {aspect_ratio}"
            )
            aspect_ratio = "自动"

        if not (capabilities & ImageCapability.RESOLUTION) and resolution != "1K":
            logger.info(
                f"[ImageGen] 当前适配器不支持指定分辨率，已忽略参数: {resolution}"
            )
            resolution = "1K"

        if not task_id:
            task_id = hashlib.md5(
                f"{time.time()}{unified_msg_origin}".encode()
            ).hexdigest()[:8]

        final_ar = validate_aspect_ratio(aspect_ratio) or None
        if final_ar == "自动":
            final_ar = None
        final_res = validate_resolution(resolution)

        images: list[ImageData] = []
        if images_data:
            for data, mime in images_data:
                images.append(ImageData(data=data, mime_type=mime))

        # 使用信号量控制并发
        if self.semaphore is None:
            await self._do_generate_and_send(
                prompt, unified_msg_origin, images, final_ar, final_res, task_id
            )
            return

        async with self.semaphore:
            await self._do_generate_and_send(
                prompt, unified_msg_origin, images, final_ar, final_res, task_id
            )

    async def _do_generate_and_send(
        self,
        prompt: str,
        unified_msg_origin: str,
        images: list[ImageData],
        aspect_ratio: str | None,
        resolution: str | None,
        task_id: str,
    ) -> None:
        """执行生成逻辑并发送结果。"""
        start_time = time.time()
        if not self.generator:
            logger.warning("[ImageGen] 生成器未初始化，跳过生成请求")
            return
        result = await self.generator.generate(
            GenerationRequest(
                prompt=prompt,
                images=images,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                task_id=task_id,
            )
        )
        end_time = time.time()
        duration = end_time - start_time

        if result.error:
            logger.error(
                f"[ImageGen] 任务 {task_id} 生成失败，耗时: {duration:.2f}s, 错误: {result.error}"
            )
            await self.context.send_message(
                unified_msg_origin,
                MessageChain().message(f"❌ 生成失败: {result.error}"),
            )
            return

        logger.info(
            f"[ImageGen] 任务 {task_id} 生成成功，耗时: {duration:.2f}s, 图片数量: {len(result.images) if result.images else 0}"
        )

        if not result.images:
            return

        # 记录使用次数
        self.usage_manager.record_usage(unified_msg_origin)

        chain = MessageChain()
        for img_bytes in result.images:
            file_path = self.image_processor.save_generated_image(task_id, img_bytes)
            if file_path:
                chain.file_image(file_path)

        info_parts = []
        if self.config_manager.show_generation_info:
            info_parts.append(
                f"✨ 生成成功！\n📊 耗时: {duration:.2f}s\n🖼️ 数量: {len(result.images)}张"
            )

        if self.config_manager.show_model_info and self.config_manager.adapter_config:
            info_parts.append(
                f"🤖 模型: {self.config_manager.adapter_config.name}/{self.config_manager.adapter_config.model}"
            )

        if self.usage_manager.is_daily_limit_enabled():
            count = self.usage_manager.get_usage_count(unified_msg_origin)
            info_parts.append(
                f"📅 今日用量: {count}/{self.usage_manager.get_daily_limit()}"
            )

        if info_parts:
            chain.message("\n" + "\n".join(info_parts))

        await self.context.send_message(unified_msg_origin, chain)


    # ---------------------- 指令处理 ----------------------

    @filter.command("生图")
    async def generate_image_command(self, event: AstrMessageEvent):
        """处理生图指令。"""
        user_id = event.unified_msg_origin

        # 检查频率限制和每日限制
        check_result = self.usage_manager.check_rate_limit(user_id)
        if isinstance(check_result, str):
            yield event.plain_result(check_result)
            return

        masked_uid = mask_sensitive(user_id)

        user_input = (event.message_str or "").strip()
        logger.info(f"[ImageGen] 收到生图指令 - 用户: {masked_uid}, 输入: {user_input}")

        cmd_parts = user_input.split(maxsplit=1)
        if not cmd_parts:
            return

        prompt = cmd_parts[1].strip() if len(cmd_parts) > 1 else ""
        aspect_ratio = self.config_manager.default_aspect_ratio
        resolution = self.config_manager.default_resolution

        # 检查是否命中预设
        matched_preset = None
        extra_content = ""
        if prompt:
            parts = prompt.split(maxsplit=1)
            first_token = parts[0]
            rest = parts[1] if len(parts) > 1 else ""
            if first_token in self.config_manager.presets:
                matched_preset = first_token
                extra_content = rest
            else:
                for name in self.config_manager.presets:
                    if name.lower() == first_token.lower():
                        matched_preset = name
                        extra_content = rest
                        break

        if matched_preset:
            logger.info(f"[ImageGen] 命中预设: {matched_preset}")
            preset_content = self.config_manager.presets[matched_preset]
            try:
                # 预设支持 JSON 格式配置高级参数
                if isinstance(
                    preset_content, str
                ) and preset_content.strip().startswith("{"):
                    preset_data = json.loads(preset_content)
                    if isinstance(preset_data, dict):
                        prompt = preset_data.get("prompt", "")
                        aspect_ratio = preset_data.get("aspect_ratio", aspect_ratio)
                        resolution = preset_data.get("resolution", resolution)
                    else:
                        prompt = preset_content
                else:
                    prompt = preset_content
            except json.JSONDecodeError:
                prompt = preset_content

            if extra_content:
                prompt = f"{prompt} {extra_content}"

        if not prompt:
            yield event.plain_result("❌ 请提供图片生成的提示词或预设名称！")
            return

        # 获取参考图
        images_data = None
        if (
            self.generator
            and self.generator.adapter
            and (
                self.generator.adapter.get_capabilities()
                & ImageCapability.IMAGE_TO_IMAGE
            )
        ):
            images_data = await self.image_processor.fetch_images_from_event(event)

        msg = "已开始生图任务"
        if images_data:
            msg += f"[{len(images_data)}张参考图]"
        if matched_preset:
            msg += f"[预设: {matched_preset}]"
        yield event.plain_result(msg)

        task_id = hashlib.md5(f"{time.time()}{user_id}".encode()).hexdigest()[:8]

        self.create_background_task(
            self._generate_and_send_image_async(
                prompt=prompt,
                images_data=images_data or None,
                unified_msg_origin=event.unified_msg_origin,
                aspect_ratio=aspect_ratio,
                resolution=resolution,
                task_id=task_id,
            )
        )

    @filter.command("生图模型")
    async def model_command(self, event: AstrMessageEvent, model_index: str = ""):
        """切换生图模型。"""
        if not self.config_manager.adapter_config:
            yield event.plain_result("❌ 适配器未初始化")
            return

        models = self.config_manager.adapter_config.available_models or []

        if not model_index:
            lines = ["📋 可用模型列表:"]
            current_model_full = f"{self.config_manager.adapter_config.name}/{self.config_manager.adapter_config.model}"
            for idx, model in enumerate(models, 1):
                marker = " ✓" if model == current_model_full else ""
                lines.append(f"{idx}. {model}{marker}")
            lines.append(f"\n当前使用: {current_model_full}")
            yield event.plain_result("\n".join(lines))
            return

        try:
            index = int(model_index) - 1
            if 0 <= index < len(models):
                raw_model = models[index]  # "供应商名称/模型名称"

                # 更新配置并重新加载
                self.config_manager.save_model_setting(raw_model)
                self.config_manager.reload()

                if self.generator:
                    await self.generator.update_adapter(
                        self.config_manager.adapter_config
                    )

                yield event.plain_result(f"✅ 模型已切换: {raw_model}")
            else:
                yield event.plain_result("❌ 无效的序号")
        except ValueError:
            yield event.plain_result("❌ 请输入有效的数字序号")

    @filter.command("预设")
    async def preset_command(self, event: AstrMessageEvent):
        """管理生图预设。"""
        user_id = event.unified_msg_origin
        masked_uid = mask_sensitive(user_id)
        message_str = (event.message_str or "").strip()
        logger.info(
            f"[ImageGen] 收到预设指令 - 用户: {masked_uid}, 内容: {message_str}"
        )

        parts = message_str.split(maxsplit=1)
        cmd_text = parts[1].strip() if len(parts) > 1 else ""

        if not cmd_text:
            if not self.config_manager.presets:
                yield event.plain_result("📋 当前没有预设")
                return
            preset_list = ["📋 预设列表:"]
            for idx, (name, prompt) in enumerate(
                self.config_manager.presets.items(), 1
            ):
                display = prompt[:20] + "..." if len(prompt) > 20 else prompt
                preset_list.append(f"{idx}. {name}: {display}")
            yield event.plain_result("\n".join(preset_list))
            return

        if cmd_text.startswith("添加 "):
            parts = cmd_text[3:].split(":", 1)
            if len(parts) == 2:
                name, prompt = parts
                self.config_manager.save_preset(name.strip(), prompt.strip())
                yield event.plain_result(f"✅ 预设已添加: {name.strip()}")
            else:
                yield event.plain_result("❌ 格式错误: /预设 添加 名称:内容")
        elif cmd_text.startswith("删除 "):
            name = cmd_text[3:].strip()
            if self.config_manager.delete_preset(name):
                yield event.plain_result(f"✅ 预设已删除: {name}")
            else:
                yield event.plain_result(f"❌ 预设不存在: {name}")
