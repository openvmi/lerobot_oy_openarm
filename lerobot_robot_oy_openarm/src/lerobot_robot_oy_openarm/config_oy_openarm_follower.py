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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig
from lerobot.robots.config import RobotConfig

LEFT_DEFAULT_JOINTS_LIMITS: dict[str, tuple[float, float]] = {
    "joint_1": (-75.0, 75.0),
    "joint_2": (-90.0, 9.0),
    "joint_3": (-85.0, 85.0),
    "joint_4": (0.0, 135.0),
    "joint_5": (-85.0, 85.0),
    "joint_6": (-40.0, 40.0),
    "joint_7": (-80.0, 80.0),
    "gripper": (-65.0, 0.0),
}

RIGHT_DEFAULT_JOINTS_LIMITS: dict[str, tuple[float, float]] = {
    "joint_1": (-75.0, 75.0),
    "joint_2": (-9.0, 90.0),
    "joint_3": (-85.0, 85.0),
    "joint_4": (0.0, 135.0),
    "joint_5": (-85.0, 85.0),
    "joint_6": (-40.0, 40.0),
    "joint_7": (-80.0, 80.0),
    "gripper": (-65.0, 0.0),
}


@dataclass
class OyOpenArmFollowerConfigBase:
    """Configuration for the dual-arm OY OpenArm SDK adapter.

    Joint and gripper limits are expressed in degrees because they describe the
    mechanical range configured by the manufacturer. The runtime LeRobot action
    and observation interface uses radians, matching ``OyOpenArmLeader`` and the
    OpenArm SDK joint-servo API.
    """

    # ip地址
    ip: str
    # port端口号
    port: str

    # Kept for backward-compatible config files. This adapter always controls
    # both arms, so only None or "both" are valid.
    side: str | None = None

    # 相机配置
    cameras: dict[str, CameraConfig] = field(default_factory=dict)

    # Mechanical soft limits in degrees. Each config gets an independent copy.
    left_joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: LEFT_DEFAULT_JOINTS_LIMITS.copy()
    )
    right_joint_limits: dict[str, tuple[float, float]] = field(
        default_factory=lambda: RIGHT_DEFAULT_JOINTS_LIMITS.copy()
    )

    # SDK connection and query timeouts, in seconds.
    connection_timeout: float = 5.0
    rpc_timeout: float = 3.0

    # Persistent joint-servo stream settings. A zero publish rate lets the
    # server choose its default. The command watchdog cannot be disabled by the
    # SDK; zero asks the server to use its default value.
    servo_publish_rate_hz: int = 0
    servo_command_timeout_ms: int = 500
    servo_ready_timeout: float = 15.0
    servo_stop_timeout: float = 5.0

    # Opt-in integration for the stock `lerobot-record`/`lerobot-teleoperate`
    # commands. Those commands connect the teleoperator before the robot, so
    # the follower can obtain the active OY leader's current action and align
    # before the first control-loop frame without modifying LeRobot itself.
    align_on_connect: bool = False

    # The SDK advertises min/max positions but not whether increasing metres
    # should correspond to increasing leader angle. Flip either side here when
    # its physical gripper direction is opposite to the leader convention.
    left_gripper_inverted: bool = False
    right_gripper_inverted: bool = False

    # Fail connect() when RCG reports that the robot is not ready.
    require_robot_ready: bool = True


@RobotConfig.register_subclass("oy_openarm_follower")
@dataclass
class OyOpenArmFollowerConfig(RobotConfig, OyOpenArmFollowerConfigBase):
    def __post_init__(self) -> None:
        super().__post_init__()

        if self.side not in (None, "both"):
            raise ValueError("OyOpenArmFollower controls both arms; config.side must be None or 'both'.")
        if not self.ip.strip() or not self.port.strip():
            raise ValueError("config.ip and config.port must be non-empty strings.")
        for name in ("connection_timeout", "rpc_timeout", "servo_ready_timeout", "servo_stop_timeout"):
            if getattr(self, name) <= 0:
                raise ValueError(f"config.{name} must be positive.")
        if self.servo_publish_rate_hz < 0 or self.servo_command_timeout_ms < 0:
            raise ValueError("Servo rate and command timeout must be non-negative.")

        expected = {*(f"joint_{index}" for index in range(1, 8)), "gripper"}
        for side, limits in (("left", self.left_joint_limits), ("right", self.right_joint_limits)):
            if set(limits) != expected:
                missing = sorted(expected - limits.keys())
                unexpected = sorted(limits.keys() - expected)
                raise ValueError(
                    f"config.{side}_joint_limits has invalid keys: "
                    f"missing={missing}, unexpected={unexpected}."
                )
            for joint, bounds in limits.items():
                if len(bounds) != 2 or bounds[0] >= bounds[1]:
                    raise ValueError(
                        f"config.{side}_joint_limits[{joint!r}] must be an increasing (min, max) pair."
                    )
