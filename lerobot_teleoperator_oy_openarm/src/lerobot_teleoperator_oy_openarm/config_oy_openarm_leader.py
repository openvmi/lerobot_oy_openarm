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

from dataclasses import dataclass

from lerobot.teleoperators.config import TeleoperatorConfig


@dataclass
class OyOpenArmLeaderConfigBase:
    """Configuration shared by OY OpenArm leader teleoperators."""

    # Separate CAN channels are required because both arms use joint IDs 1..8.
    left_port: str
    right_port: str

    # python-can backend. "auto" selects slcan for /dev/* and socketcan otherwise.
    can_interface: str = "socketcan"

    # Custom protocol V3.08b1 normally runs as classic CAN at 1 Mbps.
    can_bitrate: int = 1_000_000
    use_can_fd: bool = False
    can_data_bitrate: int | None = 5_000_000

    # Maximum duration of one complete eight-joint query per arm.
    timeout: float = 0.02

    # Output consumed by OyOpenArmLeader.get_action(). Multi-turn radians are
    # the default and match the follower's radian action convention.
    unit: str = "rad"
    single_turn: bool = False

    # If a SocketCAN interface is down, configure and bring it up. This may
    # require CAP_NET_ADMIN or an interactive sudo password.
    auto_configure_socketcan: bool = True


@TeleoperatorConfig.register_subclass("oy_openarm_leader")
@dataclass
class OyOpenArmLeaderConfig(TeleoperatorConfig, OyOpenArmLeaderConfigBase):
    """LeRobot configuration for the dual-arm OY OpenArm leader."""
