#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""CAN transport for the OY OpenArm leader.

Custom protocol V3.08b1 command ``0xA3`` is sent to CAN ID
``0x100 | joint_id``. A response arrives on ``joint_id`` with the payload
``0xA3 + uint16 single-turn count + int32 multi-turn count`` in little-endian
order. There are 16384 counts per revolution.

This module is deliberately read-only: it never enables motors, commands
motion, disables motors, or changes a zero position.
"""

from __future__ import annotations

import importlib
import logging
import math
import os
import shutil
import struct
import subprocess
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType
from typing import Any

from lerobot.utils.import_utils import require_package

logger = logging.getLogger(__name__)

COUNTS_PER_TURN = 16384
READ_ANGLE_COMMAND = 0xA3
SOCKETCAN_INTERFACE_UP = 0x1

ARM_SIDES = ("left", "right")
DEFAULT_JOINT_IDS = tuple(range(1, 9))


@dataclass(frozen=True)
class JointAngle:
    """One decoded joint response.

    ``received_at`` is the CAN backend receive time, not a motor sampling time.
    """

    joint_id: int
    single_count: int
    multi_count: int
    received_at: float

    def angle(self, unit: str = "rad", single_turn: bool = False) -> float:
        if unit not in {"rad", "deg"}:
            raise ValueError(f"unit must be 'rad' or 'deg', got {unit!r}")
        count = self.single_count if single_turn else self.multi_count
        scale = math.tau if unit == "rad" else 360.0
        return count * scale / COUNTS_PER_TURN


def decode_angle_response(message: Any) -> JointAngle | None:
    """Decode an angle response, returning ``None`` for unrelated traffic."""

    data = bytes(getattr(message, "data", b""))
    arbitration_id = getattr(message, "arbitration_id", 0)
    if (
        getattr(message, "is_extended_id", False)
        or getattr(message, "is_remote_frame", False)
        or getattr(message, "is_error_frame", False)
        or getattr(message, "is_fd", False)
        or not 1 <= arbitration_id <= 254
        or getattr(message, "dlc", len(data)) != 7
        or len(data) != 7
        or data[0] != READ_ANGLE_COMMAND
    ):
        return None

    single_count, multi_count = struct.unpack_from("<Hi", data, 1)
    return JointAngle(
        joint_id=arbitration_id,
        single_count=single_count,
        multi_count=multi_count,
        received_at=float(getattr(message, "timestamp", 0.0)),
    )


def _load_python_can() -> ModuleType:
    require_package("python-can", extra="damiao", import_name="can")
    return importlib.import_module("can")


def socketcan_is_up(channel: str) -> bool:
    """Return whether a Linux SocketCAN network interface is up."""

    if not channel or channel != os.path.basename(channel):
        raise ValueError(f"Invalid SocketCAN channel name: {channel!r}")
    flags_path = Path("/sys/class/net") / channel / "flags"
    try:
        flags = int(flags_path.read_text(encoding="ascii").strip(), 0)
    except FileNotFoundError as error:
        raise RuntimeError(f"SocketCAN channel {channel!r} does not exist") from error
    except (OSError, ValueError) as error:
        raise RuntimeError(f"Could not read state for SocketCAN channel {channel!r}: {error}") from error
    return bool(flags & SOCKETCAN_INTERFACE_UP)


def ensure_socketcan_up(
    channel: str,
    bitrate: int,
    *,
    use_can_fd: bool = False,
    data_bitrate: int | None = None,
) -> bool:
    """Bring up a SocketCAN interface when necessary.

    Returns ``True`` when this function configured it and ``False`` when it was
    already up. Configuration requires CAP_NET_ADMIN; for a regular interactive
    user, ``sudo`` may request a password.
    """

    if socketcan_is_up(channel):
        return False

    ip_program = shutil.which("ip")
    if ip_program is None:
        raise RuntimeError(f"Cannot configure {channel!r}: the 'ip' command is not installed")
    prefix: list[str] = []
    if os.geteuid() != 0:
        sudo_program = shutil.which("sudo")
        if sudo_program is None:
            raise RuntimeError(
                f"Cannot configure {channel!r} without root privileges: the 'sudo' command is not installed"
            )
        prefix = [sudo_program]

    type_command = [*prefix, ip_program, "link", "set", channel, "type", "can", "bitrate", str(bitrate)]
    if use_can_fd:
        if data_bitrate is None or data_bitrate <= 0:
            raise ValueError("A positive data_bitrate is required when CAN FD is enabled")
        type_command.extend(["dbitrate", str(data_bitrate), "fd", "on"])
    type_command.extend(["restart-ms", "100"])

    commands = (
        [*prefix, ip_program, "link", "set", channel, "down"],
        type_command,
        [*prefix, ip_program, "link", "set", channel, "up"],
    )
    for command in commands:
        try:
            subprocess.run(command, check=True)  # noqa: S603
        except subprocess.CalledProcessError as error:
            raise RuntimeError(
                f"Could not configure SocketCAN channel {channel!r} "
                f"(exit code {error.returncode}): {' '.join(command)}"
            ) from error

    if not socketcan_is_up(channel):
        raise RuntimeError(f"Configured SocketCAN channel {channel!r}, but it is still down")
    return True


class OyOpenArmBus:
    """Read two OY OpenArm leader arms over reusable CAN connections."""

    def __init__(
        self,
        left_port: str,
        right_port: str,
        *,
        can_interface: str = "socketcan",
        bitrate: int = 1_000_000,
        use_can_fd: bool = False,
        data_bitrate: int | None = None,
        joint_ids: tuple[int, ...] = DEFAULT_JOINT_IDS,
        timeout: float = 0.02,
        unit: str = "rad",
        single_turn: bool = False,
        auto_configure_socketcan: bool = True,
    ) -> None:
        if left_port == right_port:
            raise ValueError("Left and right arms must use different CAN channels")
        if not joint_ids or len(set(joint_ids)) != len(joint_ids):
            raise ValueError("joint_ids must be non-empty and unique")
        if any(not isinstance(joint_id, int) or not 1 <= joint_id <= 254 for joint_id in joint_ids):
            raise ValueError("Every joint ID must be an integer in [1, 254]")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout must be a finite positive number")
        if bitrate <= 0:
            raise ValueError("bitrate must be positive")
        if unit not in {"rad", "deg"}:
            raise ValueError("unit must be 'rad' or 'deg'")
        if can_interface not in {"auto", "socketcan", "slcan", "pcan", "virtual"}:
            raise ValueError("can_interface must be one of: auto, socketcan, slcan, pcan, virtual")

        self.channels = {"left": left_port, "right": right_port}
        self.can_interface = can_interface
        self.bitrate = bitrate
        self.use_can_fd = use_can_fd
        self.data_bitrate = data_bitrate
        self.joint_ids = tuple(joint_ids)
        self.timeout = timeout
        self.unit = unit
        self.single_turn = single_turn
        self.auto_configure_socketcan = auto_configure_socketcan

        self._can: ModuleType | None = None
        self._buses: dict[str, Any] = {}
        self._executor: ThreadPoolExecutor | None = None
        self._lock = threading.RLock()

    @property
    def is_connected(self) -> bool:
        return self._executor is not None and set(self._buses) == set(ARM_SIDES)

    def _resolved_interface(self, channel: str) -> str:
        if self.can_interface != "auto":
            return self.can_interface
        return "slcan" if channel.startswith("/dev/") else "socketcan"

    def _open_bus(self, channel: str) -> Any:
        if self._can is None:
            raise RuntimeError("python-can has not been loaded")
        interface = self._resolved_interface(channel)
        if interface == "socketcan" and self.auto_configure_socketcan:
            started = ensure_socketcan_up(
                channel,
                self.bitrate,
                use_can_fd=self.use_can_fd,
                data_bitrate=self.data_bitrate,
            )
            if started:
                logger.info("Started SocketCAN channel %s at %d bit/s", channel, self.bitrate)

        filters = [{"can_id": joint_id, "can_mask": 0x7FF, "extended": False} for joint_id in self.joint_ids]
        kwargs: dict[str, Any] = {
            "interface": interface,
            "channel": channel,
            "bitrate": self.bitrate,
            "can_filters": filters,
            "receive_own_messages": False,
            "ignore_config": True,
        }
        if interface == "socketcan" and self.use_can_fd:
            kwargs["fd"] = True
            if self.data_bitrate is not None:
                kwargs["data_bitrate"] = self.data_bitrate
        return self._can.Bus(**kwargs)

    def connect(self) -> None:
        with self._lock:
            if self.is_connected:
                raise RuntimeError("OY OpenArm CAN bus is already connected")
            self._can = _load_python_can()
            try:
                for side, channel in self.channels.items():
                    self._buses[side] = self._open_bus(channel)
                self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="oy-openarm")
            except BaseException:
                self._shutdown_unlocked()
                raise
            logger.info(
                "Connected OY OpenArm leader buses: left=%s, right=%s",
                self.channels["left"],
                self.channels["right"],
            )

    def _read_arm(self, side: str) -> tuple[JointAngle | None, ...]:
        bus = self._buses[side]
        if self._can is None:
            raise RuntimeError("python-can has not been loaded")
        try:
            drain_deadline = time.monotonic() + self.timeout
            while bus.recv(timeout=0) is not None:
                if time.monotonic() >= drain_deadline:
                    raise RuntimeError("receive queue remained busy while discarding stale replies")

            deadline = time.monotonic() + self.timeout
            for joint_id in self.joint_ids:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("sending angle requests exceeded the read timeout")
                request = self._can.Message(
                    arbitration_id=0x100 | joint_id,
                    is_extended_id=False,
                    is_fd=False,
                    data=[READ_ANGLE_COMMAND],
                )
                bus.send(request, timeout=remaining)

            samples: dict[int, JointAngle] = {}
            while len(samples) < len(self.joint_ids):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                message = bus.recv(timeout=remaining)
                if message is None:
                    break
                sample = decode_angle_response(message)
                if sample is not None and sample.joint_id in self.joint_ids:
                    samples.setdefault(sample.joint_id, sample)
            return tuple(samples.get(joint_id) for joint_id in self.joint_ids)
        except Exception as error:
            raise RuntimeError(f"Failed reading {side} arm on {self.channels[side]}: {error}") from error

    def read_joint_angles(self) -> dict[str, tuple[JointAngle | None, ...]]:
        """Query both arms concurrently, ordered according to ``joint_ids``."""

        with self._lock:
            if not self.is_connected or self._executor is None:
                raise RuntimeError("OY OpenArm CAN bus is not connected")
            futures = {side: self._executor.submit(self._read_arm, side) for side in ARM_SIDES}
            results: dict[str, tuple[JointAngle | None, ...]] = {}
            errors: list[BaseException] = []
            for side, future in futures.items():
                try:
                    results[side] = future.result()
                except BaseException as error:
                    errors.append(error)
            if errors:
                raise errors[0]
            return results

    def read_positions(self) -> dict[str, float]:
        """Return a flat, complete LeRobot action dictionary for both arms."""

        samples = self.read_joint_angles()
        positions: dict[str, float] = {}
        missing: list[str] = []
        for side in ARM_SIDES:
            for joint_id, sample in zip(self.joint_ids, samples[side], strict=True):
                name = "gripper" if joint_id == 8 else f"joint_{joint_id}"
                feature = f"{side}_{name}.pos"
                if sample is None:
                    missing.append(feature)
                else:
                    positions[feature] = sample.angle(self.unit, self.single_turn)
        if missing:
            raise TimeoutError(f"Timed out waiting for OY OpenArm joints: {', '.join(missing)}")
        return positions

    def _shutdown_unlocked(self) -> None:
        executor, self._executor = self._executor, None
        if executor is not None:
            executor.shutdown(wait=True, cancel_futures=True)
        for side, bus in tuple(self._buses.items()):
            try:
                bus.shutdown()
            except Exception:
                logger.exception("Failed to close %s OY OpenArm CAN bus", side)
        self._buses.clear()
        self._can = None

    def disconnect(self) -> None:
        with self._lock:
            if not self.is_connected and not self._buses:
                return
            self._shutdown_unlocked()
            logger.info("Disconnected OY OpenArm leader buses")

    def __enter__(self) -> OyOpenArmBus:
        self.connect()
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.disconnect()
