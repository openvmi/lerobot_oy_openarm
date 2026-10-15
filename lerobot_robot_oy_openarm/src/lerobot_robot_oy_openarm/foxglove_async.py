#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Keep Foxglove rendering work outside LeRobot's real-time record loop.

LeRobot 0.6 logs visualization data synchronously.  Serializing three raw
640x480 images can therefore consume most of a 30 Hz control interval.  This
module installs a small compatibility patch when the OY OpenArm plugin is
discovered: Foxglove receives the newest pending frame on a worker thread,
while recording and robot control continue without waiting for the UI.

The queue intentionally has latest-frame semantics.  Dropping an obsolete
visualization frame never changes the frame stored in the LeRobot dataset.
"""

from __future__ import annotations

import logging
import os
import sys
import threading
import time
from collections.abc import Callable
from typing import Any

from lerobot.utils import visualization_utils

logger = logging.getLogger(__name__)

_LogVisualization = Callable[..., None]


DEFAULT_MAX_FPS = 30.0


class _LatestFoxgloveDispatcher:
    """Run Foxglove logging asynchronously and retain at most one pending frame."""

    def __init__(
        self,
        log_visualization_data: _LogVisualization,
        max_fps: float = DEFAULT_MAX_FPS,
    ) -> None:
        if max_fps <= 0:
            raise ValueError("max_fps must be positive")
        self._log_visualization_data = log_visualization_data
        self._minimum_interval_s = 1.0 / max_fps
        self._max_fps = max_fps
        self._condition = threading.Condition()
        self._pending: (
            tuple[dict[str, Any] | None, dict[str, Any] | None, bool] | None
        ) = None
        self._thread: threading.Thread | None = None
        self._stopping = False
        self._disabled = False
        self._compress_images: bool | None = None
        self._dropped_frames = 0
        self._last_log_finished_at: float | None = None

    def submit(
        self,
        observation: dict[str, Any] | None,
        action: dict[str, Any] | None,
        compress_images: bool,
    ) -> None:
        # Pin the channel type for the whole Foxglove session.  LeRobot 0.6
        # forgets to forward display_compressed_images to its reset loop; if
        # the type changes, foxglove-sdk rejects RawImage on a channel that was
        # created for CompressedImage (or vice versa).
        with self._condition:
            if self._disabled or self._stopping:
                return
            if self._compress_images is None:
                self._compress_images = compress_images

            # The camera reader replaces its latest NumPy array rather than
            # mutating it.  Shallow copies therefore give this worker stable
            # mappings without copying several megabytes in the record loop.
            item = (
                dict(observation) if observation is not None else None,
                dict(action) if action is not None else None,
                self._compress_images,
            )
            if self._pending is not None:
                self._dropped_frames += 1
            self._pending = item

            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(
                    target=self._run,
                    name="oy-openarm-foxglove",
                    daemon=True,
                )
                self._thread.start()
                logger.info(
                    "Foxglove 已使用后台最新帧模式，显示发送上限为 %.1f Hz；数据集仍按配置 FPS 采集。",
                    self._max_fps,
                )
            self._condition.notify()

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._pending is None and self._stopping:
                    return

                if self._last_log_finished_at is not None and not self._stopping:
                    remaining_s = self._minimum_interval_s - (
                        time.monotonic() - self._last_log_finished_at
                    )
                    if remaining_s > 0:
                        # Waiting on the condition, rather than sleeping, lets
                        # producers replace the pending frame and lets shutdown
                        # wake the worker immediately.
                        self._condition.wait(timeout=remaining_s)
                        continue

                observation, action, compress_images = self._pending
                self._pending = None

            try:
                self._log_visualization_data(
                    "foxglove",
                    observation=observation,
                    action=action,
                    compress_images=compress_images,
                )
                self._last_log_finished_at = time.monotonic()
            except Exception:
                # Visualization must never abort a robot-control or dataset
                # recording session.  Disable it after one failure to avoid a
                # high-rate exception loop, while leaving recording active.
                logger.exception(
                    "Foxglove 后台可视化失败；本次录制将继续，但停止发送可视化数据。"
                )
                with self._condition:
                    self._disabled = True
                    self._pending = None
                return

    def close(self) -> None:
        with self._condition:
            self._stopping = True
            self._condition.notify_all()
            thread = self._thread

        if thread is not None:
            thread.join(timeout=5.0)
            if thread.is_alive():
                logger.warning("Foxglove 后台线程未能在 5 秒内停止。")

        with self._condition:
            dropped_frames = self._dropped_frames
            self._pending = None
            self._thread = None
            self._stopping = False
            self._disabled = False
            self._compress_images = None
            self._dropped_frames = 0
            self._last_log_finished_at = None

        if dropped_frames:
            logger.info(
                "Foxglove 为避免阻塞 30 Hz 数采，共跳过了 %d 个过期的可视化帧；数据集帧未受影响。",
                dropped_frames,
            )


_installed = False


def install_async_foxglove_logging() -> None:
    """Patch an already-loaded ``lerobot-record`` module exactly once."""

    global _installed

    if _installed:
        return

    original_log = visualization_utils.log_visualization_data
    original_shutdown = visualization_utils.shutdown_visualization
    configured_max_fps = os.environ.get("OY_OPENARM_FOXGLOVE_MAX_FPS")
    try:
        max_fps = (
            DEFAULT_MAX_FPS if configured_max_fps is None else float(configured_max_fps)
        )
        if max_fps <= 0:
            raise ValueError
    except ValueError:
        logger.warning(
            "忽略无效的 OY_OPENARM_FOXGLOVE_MAX_FPS=%r，使用默认值 %.1f。",
            configured_max_fps,
            DEFAULT_MAX_FPS,
        )
        max_fps = DEFAULT_MAX_FPS
    dispatcher = _LatestFoxgloveDispatcher(original_log, max_fps=max_fps)

    def log_visualization_data(
        display_mode: str,
        observation: dict[str, Any] | None = None,
        action: dict[str, Any] | None = None,
        compress_images: bool = False,
    ) -> None:
        if display_mode != "foxglove":
            original_log(
                display_mode,
                observation=observation,
                action=action,
                compress_images=compress_images,
            )
            return
        dispatcher.submit(observation, action, compress_images)

    def shutdown_visualization(display_mode: str) -> None:
        if display_mode == "foxglove":
            dispatcher.close()
        original_shutdown(display_mode)

    # Patch the utility module for future imports and the references already
    # imported by the stock lerobot-record entry point.  Plugin registration is
    # invoked from lerobot_record.main(), after that module has been imported.
    visualization_utils.log_visualization_data = log_visualization_data
    visualization_utils.shutdown_visualization = shutdown_visualization

    record_module = sys.modules.get("lerobot.scripts.lerobot_record")
    if record_module is not None:
        record_module.log_visualization_data = log_visualization_data
        record_module.shutdown_visualization = shutdown_visualization

    _installed = True
