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

import logging
import time

from lerobot.teleoperators.teleoperator import Teleoperator

try:
    from lerobot.types import RobotAction
except ModuleNotFoundError:
    from lerobot.lerobot_types import RobotAction

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_oy_openarm_leader import OyOpenArmLeaderConfig
from .oy_openarm_bus import OyOpenArmBus
from .pairing import register_active_leader, unregister_active_leader

logger = logging.getLogger(__name__)

OY_OPENARM_ACTION_FEATURES = {
    **{f"left_joint_{joint_id}.pos": float for joint_id in range(1, 8)},
    "left_gripper.pos": float,
    **{f"right_joint_{joint_id}.pos": float for joint_id in range(1, 8)},
    "right_gripper.pos": float,
}


class OyOpenArmLeader(Teleoperator):
    """Read-only dual-arm OY OpenArm leader integrated with LeRobot.

    Hardware communication and custom protocol decoding live in
    :class:`OyOpenArmBus`. This class only adapts the bus lifecycle and its flat
    position dictionary to LeRobot's :class:`Teleoperator` interface.
    """

    config_class = OyOpenArmLeaderConfig
    name = "oy_openarm_leader"

    def __init__(self, config: OyOpenArmLeaderConfig):
        super().__init__(config)
        self.config = config
        self.bus = OyOpenArmBus(
            left_port=config.left_port,
            right_port=config.right_port,
            can_interface=config.can_interface,
            bitrate=config.can_bitrate,
            use_can_fd=config.use_can_fd,
            data_bitrate=config.can_data_bitrate,
            timeout=config.timeout,
            unit=config.unit,
            single_turn=config.single_turn,
            auto_configure_socketcan=config.auto_configure_socketcan,
        )

    @property
    def action_features(self) -> dict[str, type]:
        """Position features returned by :meth:`get_action`."""

        return OY_OPENARM_ACTION_FEATURES.copy()

    @property
    def feedback_features(self) -> dict[str, type]:
        """The current leader protocol does not implement force feedback."""

        return {}

    @property
    def is_connected(self) -> bool:
        return self.bus.is_connected

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """Open and retain both CAN connections.

        ``calibrate`` is accepted for the common LeRobot interface. The custom
        read-only protocol exposes raw driver angles and has no calibration
        command, so no hardware state is changed here.
        """

        del calibrate
        self.bus.connect()
        try:
            register_active_leader(self)
        except Exception:
            self.bus.disconnect()
            raise
        logger.info("%s connected", self)

    @property
    def is_calibrated(self) -> bool:
        # Angles use the motor driver's existing zero point and direction.
        return True

    def calibrate(self) -> None:
        """No-op: calibration/zeroing is external to the read-only protocol."""

    def configure(self) -> None:
        """No-op: connecting must not enable or configure leader motors."""

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        """Read all 16 positions through :meth:`OyOpenArmBus.read_positions`."""

        start = time.perf_counter()
        action = self.bus.read_positions()
        if action.keys() != self.action_features.keys():
            missing = self.action_features.keys() - action.keys()
            unexpected = action.keys() - self.action_features.keys()
            raise RuntimeError(
                f"OY OpenArm action schema mismatch: missing={sorted(missing)}, "
                f"unexpected={sorted(unexpected)}"
            )
        logger.debug("%s read action in %.1f ms", self, (time.perf_counter() - start) * 1e3)
        return action

    @check_if_not_connected
    def send_feedback(self, feedback: dict[str, float]) -> None:
        raise NotImplementedError("Feedback is not implemented for the OY OpenArm leader.")

    @check_if_not_connected
    def disconnect(self) -> None:
        unregister_active_leader(self)
        self.bus.disconnect()
        logger.info("%s disconnected", self)
