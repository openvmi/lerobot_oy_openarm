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

import math
import struct

import pytest
from lerobot.teleoperators.utils import make_teleoperator_from_config
from lerobot_teleoperator_oy_openarm.oy_openarm_bus import (
    COUNTS_PER_TURN,
    READ_ANGLE_COMMAND,
    decode_angle_response,
)

from lerobot_teleoperator_oy_openarm import (
    OY_OPENARM_ACTION_FEATURES,
    OyOpenArmBus,
    OyOpenArmLeader,
    OyOpenArmLeaderConfig,
    get_active_leader,
)

_BUS_MODULE = "lerobot_teleoperator_oy_openarm.oy_openarm_bus"


class FakeMessage:
    def __init__(
        self,
        arbitration_id: int,
        data: bytes | list[int],
        *,
        is_extended_id: bool = False,
        is_remote_frame: bool = False,
        is_error_frame: bool = False,
        is_fd: bool = False,
        timestamp: float = 1.25,
    ) -> None:
        self.arbitration_id = arbitration_id
        self.data = bytes(data)
        self.dlc = len(self.data)
        self.is_extended_id = is_extended_id
        self.is_remote_frame = is_remote_frame
        self.is_error_frame = is_error_frame
        self.is_fd = is_fd
        self.timestamp = timestamp


class FakeCanBus:
    def __init__(self, channel: str, missing_joint: int | None = None) -> None:
        self.channel = channel
        self.missing_joint = missing_joint
        self.pending: list[FakeMessage] = []
        self.requests: list[FakeMessage] = []
        self.shutdown_called = False

    def send(self, request: FakeMessage, timeout: float) -> None:
        del timeout
        self.requests.append(request)
        joint_id = request.arbitration_id & 0xFF
        if joint_id == self.missing_joint:
            return
        direction = 1 if self.channel == "left-test" else -1
        multi_count = direction * joint_id * COUNTS_PER_TURN
        payload = bytes([READ_ANGLE_COMMAND]) + struct.pack("<Hi", joint_id, multi_count)
        self.pending.append(FakeMessage(joint_id, payload))

    def recv(self, timeout: float) -> FakeMessage | None:
        del timeout
        return self.pending.pop(0) if self.pending else None

    def shutdown(self) -> None:
        self.shutdown_called = True


class FakeCanModule:
    Message = FakeMessage

    def __init__(self, missing_joint: int | None = None) -> None:
        self.missing_joint = missing_joint
        self.buses: dict[str, FakeCanBus] = {}

    def Bus(self, **kwargs) -> FakeCanBus:  # noqa: N802
        bus = FakeCanBus(kwargs["channel"], self.missing_joint)
        self.buses[kwargs["channel"]] = bus
        return bus


def make_response(joint_id: int, single_count: int, multi_count: int, **kwargs) -> FakeMessage:
    payload = bytes([READ_ANGLE_COMMAND]) + struct.pack("<Hi", single_count, multi_count)
    return FakeMessage(joint_id, payload, **kwargs)


def test_decode_angle_response() -> None:
    sample = decode_angle_response(make_response(3, COUNTS_PER_TURN // 2, -COUNTS_PER_TURN))
    assert sample is not None
    assert sample.joint_id == 3
    assert sample.angle("deg", single_turn=True) == pytest.approx(180.0)
    assert sample.angle("rad") == pytest.approx(-math.tau)


@pytest.mark.parametrize(
    "message",
    [
        FakeMessage(3, b"\xa2" + b"\x00" * 6),
        FakeMessage(3, b"\xa3" + b"\x00" * 5),
        make_response(3, 0, 0, is_extended_id=True),
        make_response(3, 0, 0, is_fd=True),
        make_response(0x103, 0, 0),
    ],
)
def test_decode_ignores_invalid_frames(message: FakeMessage) -> None:
    assert decode_angle_response(message) is None


def test_bus_reads_both_arms_and_reuses_connections(monkeypatch) -> None:
    fake_can = FakeCanModule()
    monkeypatch.setattr(f"{_BUS_MODULE}._load_python_can", lambda: fake_can)
    bus = OyOpenArmBus(
        "left-test",
        "right-test",
        can_interface="virtual",
        auto_configure_socketcan=False,
    )

    bus.connect()
    first = bus.read_positions()
    second = bus.read_positions()

    assert first == second
    assert first["left_joint_1.pos"] == pytest.approx(math.tau)
    assert first["left_gripper.pos"] == pytest.approx(8 * math.tau)
    assert first["right_joint_1.pos"] == pytest.approx(-math.tau)
    assert first["right_gripper.pos"] == pytest.approx(-8 * math.tau)
    assert len(fake_can.buses) == 2
    assert all(len(fake_bus.requests) == 16 for fake_bus in fake_can.buses.values())

    opened_buses = list(fake_can.buses.values())
    bus.disconnect()
    assert not bus.is_connected
    assert all(fake_bus.shutdown_called for fake_bus in opened_buses)


def test_bus_rejects_incomplete_action(monkeypatch) -> None:
    fake_can = FakeCanModule(missing_joint=8)
    monkeypatch.setattr(f"{_BUS_MODULE}._load_python_can", lambda: fake_can)
    bus = OyOpenArmBus(
        "left-test",
        "right-test",
        can_interface="virtual",
        auto_configure_socketcan=False,
        timeout=0.001,
    )
    bus.connect()
    with pytest.raises(TimeoutError, match="left_gripper.pos"):
        bus.read_positions()
    bus.disconnect()


def test_leader_lifecycle_and_get_action(tmp_path) -> None:
    class FakeLeaderBus:
        def __init__(self) -> None:
            self.is_connected = False

        def connect(self) -> None:
            self.is_connected = True

        def read_positions(self) -> dict[str, float]:
            return {key: float(index) for index, key in enumerate(OY_OPENARM_ACTION_FEATURES)}

        def disconnect(self) -> None:
            self.is_connected = False

    config = OyOpenArmLeaderConfig(
        id="test",
        calibration_dir=tmp_path,
        left_port="can1",
        right_port="can0",
        auto_configure_socketcan=False,
    )
    leader = OyOpenArmLeader(config)
    leader.bus = FakeLeaderBus()

    assert not leader.is_connected
    leader.connect()
    assert leader.is_connected
    assert get_active_leader() is leader
    action = leader.get_action()
    assert action.keys() == leader.action_features.keys()
    assert all(isinstance(value, float) for value in action.values())
    leader.disconnect()
    assert not leader.is_connected
    assert get_active_leader() is None


def test_dynamic_factory_finds_oy_openarm_leader(tmp_path) -> None:
    config = OyOpenArmLeaderConfig(
        id="factory-test",
        calibration_dir=tmp_path,
        left_port="can1",
        right_port="can0",
        auto_configure_socketcan=False,
    )
    teleoperator = make_teleoperator_from_config(config)
    assert isinstance(teleoperator, OyOpenArmLeader)
    assert teleoperator.action_features == OY_OPENARM_ACTION_FEATURES
