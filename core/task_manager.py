from __future__ import annotations

import asyncio
import functools
from collections.abc import Callable, Coroutine
from typing import Any

from astrbot.api import logger


class TaskManager:
    """统一的任务管理器，管理插件的后台任务和定时任务。"""

    def __init__(self):
        self.background_tasks: set[asyncio.Task] = set()
        self._loop_tasks: dict[str, asyncio.Task] = {}

    def create_task(self, coro: Coroutine[Any, Any, Any], name: str | None = None) -> asyncio.Task:
        """创建一个普通的后台任务。"""
        task = asyncio.create_task(coro)
        if name:
            task.set_name(name)
        self.background_tasks.add(task)
        task.add_done_callback(self.background_tasks.discard)
        return task

    def start_loop_task(
        self,
        name: str,
        coro_func: Callable[[], Coroutine[Any, Any, Any]],
        interval_seconds: float,
        run_immediately: bool = True,
    ) -> None:
        """启动一个周期性的定时任务。

        Args:
            name: 任务名称，用于唯一标识和日志记录。
            coro_func: 返回协程的函数（任务的主逻辑）。
            interval_seconds: 执行间隔（秒）。
            run_immediately: 是否在启动时立即执行一次。
        """
        if name in self._loop_tasks:
            self.stop_loop_task(name)

        async def _loop():
            if run_immediately:
                try:
                    await coro_func()
                except Exception as e:
                    logger.error(f"[ImageGen] [TaskManager] 定时任务 {name} 初始执行失败: {e}", exc_info=True)

            while True:
                try:
                    await asyncio.sleep(interval_seconds)
                    await coro_func()
                except asyncio.CancelledError:
                    break
                except Exception as e:
                    logger.error(f"[ImageGen] [TaskManager] 定时任务 {name} 执行出错: {e}", exc_info=True)

        task = asyncio.create_task(_loop(), name=f"loop_{name}")
        self._loop_tasks[name] = task
        self.background_tasks.add(task)
        task.add_done_callback(functools.partial(self._on_loop_task_done, name))
        logger.info(f"[ImageGen] [TaskManager] 定时任务 {name} 已启动 (间隔: {interval_seconds}s)")

    def stop_loop_task(self, name: str) -> None:
        """停止指定的定时任务。"""
        if task := self._loop_tasks.pop(name, None):
            if not task.done():
                task.cancel()
            logger.info(f"[ImageGen] [TaskManager] 定时任务 {name} 已停止")

    def _on_loop_task_done(self, name: str, task: asyncio.Task) -> None:
        """定时任务结束时的回调。"""
        self.background_tasks.discard(task)
        self._loop_tasks.pop(name, None)

    async def cancel_all(self):
        """取消所有正在运行的任务。"""
        for task in list(self.background_tasks):
            if not task.done():
                task.cancel()

        if self.background_tasks:
            await asyncio.gather(*self.background_tasks, return_exceptions=True)

        self.background_tasks.clear()
        self._loop_tasks.clear()
        logger.info("[ImageGen] [TaskManager] 所有后台任务已取消")
