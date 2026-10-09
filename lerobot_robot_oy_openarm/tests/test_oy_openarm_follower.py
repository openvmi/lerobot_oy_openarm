#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

import math
from enum import IntEnum
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from lerobot.robots.utils import make_robot_from_config
from lerobot_robot_oy_openarm.config_oy_openarm_follower import OyOpenArmFollowerConfig
from lerobot_robot_oy_openarm.oy_openarm_follower import (
    OY_OPENARM_POSITION_FEATURES,
    OyOpenArmFollower,
)


class Arm(IntEnum):
    BOTH = 0
    LEFT = 1
    RIGHT = 2


class PositionUnit(IntEnum):
    METER = 1
    RADIAN = 2


class FakeServoState:
    def __init__(self, name, is_terminal=False):
        self.name = name
        self.is_terminal = is_terminal


READY = FakeServoState("READY")
RUNNING = FakeServoState("RUNNING")
FAILED = FakeServoState("FAILED", is_terminal=True)
STOPPED = FakeServoState("STOPPED", is_terminal=True)


class FakeSession:
    def __init__(self):
        self.is_closed = False
        self.error = None
        self.sequence = 0
        self.feedback = SimpleNamespace(
            success=True,
            message="joint servo ready",
            state=READY,
            last_received_sequence=0,
            last_published_sequence=0,
            publish_rate_hz=100,
            command_timeout_ms=500,
        )
        self.send = MagicMock(side_effect=self._send)
        self.stop = MagicMock(side_effect=self._stop)
        self.cancel = MagicMock(side_effect=self._cancel)

    def _send(self, *_args, **_kwargs):
        self.sequence += 1
        self.feedback = SimpleNamespace(
            success=True,
            message="target published",
            state=RUNNING,
            last_received_sequence=self.sequence,
            last_published_sequence=self.sequence,
            publish_rate_hz=100,
            command_timeout_ms=500,
        )
        return self.sequence

    def _stop(self, **_kwargs):
        self.feedback = SimpleNamespace(
            success=True,
            message="stopped",
            state=STOPPED,
            last_received_sequence=self.sequence,
            last_published_sequence=self.sequence,
            publish_rate_hz=100,
            command_timeout_ms=500,
        )
        self.is_closed = True

    def _cancel(self):
        self.is_closed = True


class FakeClient:
    instances = []

    def __init__(self, target, default_timeout):
        self.target = target
        self.default_timeout = default_timeout
        self.is_closed = False
        self.session = FakeSession()
        self.wait_until_channel_ready = MagicMock()
        self.close = MagicMock(side_effect=self._close)
        self.system = SimpleNamespace(
            get_status=MagicMock(return_value=SimpleNamespace(server_ready=True, robot_ready=True))
        )
        infos = tuple(
            SimpleNamespace(
                arm=arm,
                position_unit=PositionUnit.METER,
                joint_names=("gripper",),
                min_positions=(minimum,),
                max_positions=(maximum,),
            )
            for arm, minimum, maximum in ((Arm.RIGHT, 0.02, 0.06), (Arm.LEFT, 0.01, 0.05))
        )
        self.end_effectors = SimpleNamespace(
            get_info=MagicMock(return_value=SimpleNamespace(success=True, message="", end_effectors=infos)),
            get_state=MagicMock(),
        )
        self.state = SimpleNamespace(get_joint_states=MagicMock())
        self.joint_servo = SimpleNamespace(start=MagicMock(return_value=self.session))
        self.__class__.instances.append(self)

    def _close(self):
        self.is_closed = True


FAKE_SDK = SimpleNamespace(Arm=Arm, PositionUnit=PositionUnit, OpenArmClient=FakeClient)


@pytest.fixture
def robot(tmp_path):
    FakeClient.instances.clear()
    config = OyOpenArmFollowerConfig(
        id="test",
        calibration_dir=tmp_path,
        ip="127.0.0.1",
        port="50051",
    )
    with patch(
        "lerobot_robot_oy_openarm.oy_openarm_follower._load_openarm_sdk",
        return_value=FAKE_SDK,
    ):
        instance = OyOpenArmFollower(config)
    yield instance
    instance.disconnect()


def _action(value=0.0):
    return dict.fromkeys(OY_OPENARM_POSITION_FEATURES, value)


def test_config_limit_dicts_are_not_shared(tmp_path):
    first = OyOpenArmFollowerConfig(id="a", calibration_dir=tmp_path, ip="localhost", port="1")
    second = OyOpenArmFollowerConfig(id="b", calibration_dir=tmp_path, ip="localhost", port="1")
    first.left_joint_limits["joint_1"] = (-1.0, 1.0)
    assert second.left_joint_limits["joint_1"] == (-75.0, 75.0)


def test_follower_action_schema_matches_oy_openarm_leader():
    expected = {
        **{f"left_joint_{joint_id}.pos": float for joint_id in range(1, 8)},
        "left_gripper.pos": float,
        **{f"right_joint_{joint_id}.pos": float for joint_id in range(1, 8)},
        "right_gripper.pos": float,
    }
    assert expected == OY_OPENARM_POSITION_FEATURES


def test_config_rejects_single_arm_mode(tmp_path):
    with pytest.raises(ValueError, match="controls both arms"):
        OyOpenArmFollowerConfig(id="test", calibration_dir=tmp_path, ip="localhost", port="1", side="left")


def test_dynamic_factory_finds_oy_openarm_follower(tmp_path):
    config = OyOpenArmFollowerConfig(
        id="factory-test",
        calibration_dir=tmp_path,
        ip="127.0.0.1",
        port="50051",
    )
    with patch(
        "lerobot_robot_oy_openarm.oy_openarm_follower._load_openarm_sdk",
        return_value=FAKE_SDK,
    ):
        robot = make_robot_from_config(config)

    assert isinstance(robot, OyOpenArmFollower)
    assert robot.action_features == OY_OPENARM_POSITION_FEATURES


def test_connect_and_disconnect_own_sdk_resources(robot):
    robot.connect()
    client = FakeClient.instances[-1]

    assert robot.is_connected
    assert client.target == "127.0.0.1:50051"
    client.wait_until_channel_ready.assert_called_once_with(timeout=5.0)
    client.system.get_status.assert_called_once_with(timeout=3.0)
    client.end_effectors.get_info.assert_called_once_with(Arm.BOTH, timeout=3.0)

    robot.disconnect()
    assert not robot.is_connected
    client.close.assert_called_once()


def test_connect_can_align_to_registered_leader(tmp_path):
    FakeClient.instances.clear()
    config = OyOpenArmFollowerConfig(
        id="align-test",
        calibration_dir=tmp_path,
        ip="127.0.0.1",
        port="50051",
        align_on_connect=True,
    )
    leader_action = _action(0.1)
    leader = SimpleNamespace(get_action=MagicMock(return_value=leader_action))

    with (
        patch(
            "lerobot_robot_oy_openarm.oy_openarm_follower._load_openarm_sdk",
            return_value=FAKE_SDK,
        ),
        patch(
            "lerobot_robot_oy_openarm.oy_openarm_follower._get_active_oy_openarm_leader",
            return_value=leader,
        ),
    ):
        robot = OyOpenArmFollower(config)
        alignment_session = FakeSession()

        def align_side_effect(_action):
            robot._servo_session = alignment_session

        with patch.object(robot, "calibrate", side_effect=align_side_effect) as align:
            robot.connect()

    leader.get_action.assert_called_once_with()
    align.assert_called_once_with(leader_action)
    alignment_session.stop.assert_called_once_with(timeout=5.0)
    assert robot._servo_session is None
    assert robot.is_connected
    robot.disconnect()


def test_send_action_starts_one_dual_arm_session_and_maps_grippers(robot):
    robot.connect()
    client = FakeClient.instances[-1]
    action = _action()
    action["left_joint_1.pos"] = 10.0
    action["left_gripper.pos"] = math.radians(-65.0)
    action["right_gripper.pos"] = 0.0

    sent = robot.send_action(action)

    # left_joint_1 is clipped at +75 degrees.
    assert sent["left_joint_1.pos"] == pytest.approx(75.0 * 3.141592653589793 / 180.0)
    client.joint_servo.start.assert_called_once_with(
        [Arm.LEFT, Arm.RIGHT],
        publish_rate_hz=0,
        command_timeout_ms=500,
        ready_timeout=15.0,
    )
    args, kwargs = client.session.send.call_args
    assert args[0][Arm.LEFT][0] == sent["left_joint_1.pos"]
    assert len(args[0][Arm.LEFT]) == 7
    assert len(args[0][Arm.RIGHT]) == 7
    assert kwargs["end_effectors"][Arm.LEFT] == pytest.approx([0.01])
    assert kwargs["end_effectors"][Arm.RIGHT] == pytest.approx([0.06])
    assert kwargs["sampled_at_unix_ns"] > 0

    robot.send_action(_action())
    client.joint_servo.start.assert_called_once()


def test_send_action_samples_timestamp_after_session_is_ready(robot):
    robot.connect()
    client = FakeClient.instances[-1]
    events = []

    def start_session(*_args, **_kwargs):
        events.append("start")
        return client.session

    def sample_time():
        events.append("timestamp")
        return 123456789

    client.joint_servo.start.side_effect = start_session
    with patch(
        "lerobot_robot_oy_openarm.oy_openarm_follower.time.time_ns",
        side_effect=sample_time,
    ):
        robot.send_action(_action())

    assert events == ["start", "timestamp"]
    assert client.session.send.call_args.kwargs["sampled_at_unix_ns"] == 123456789


def test_failed_servo_session_is_reported_without_restart_loop(robot):
    robot.connect()
    client = FakeClient.instances[-1]
    robot.send_action(_action())
    client.session.feedback = SimpleNamespace(
        success=False,
        message="target watchdog expired",
        state=FAILED,
        last_received_sequence=1,
        last_published_sequence=0,
        publish_rate_hz=100,
        command_timeout_ms=500,
    )
    client.session.is_closed = True

    with pytest.raises(RuntimeError, match="target watchdog expired.*received=1, published=0"):
        robot.send_action(_action())

    client.joint_servo.start.assert_called_once()


def test_gripper_mapping_can_be_inverted(robot):
    robot.config.left_gripper_inverted = True
    robot.connect()
    action = _action()
    action["left_gripper.pos"] = math.radians(-65.0)

    robot.send_action(action)

    kwargs = FakeClient.instances[-1].session.send.call_args.kwargs
    assert kwargs["end_effectors"][Arm.LEFT] == pytest.approx([0.05])


def test_calibrate_smoothly_aligns_follower_to_leader_action(robot):
    robot.connect()
    client = FakeClient.instances[-1]
    current = _action(0.0)
    current["left_gripper.pos"] = -0.5
    current["right_gripper.pos"] = -0.5
    leader_action = _action(0.1)
    leader_action["left_gripper.pos"] = -0.3
    leader_action["right_gripper.pos"] = -0.3

    with (
        patch.object(robot, "_read_position_observation", side_effect=[current, leader_action]),
        patch("lerobot_robot_oy_openarm.oy_openarm_follower.time.sleep"),
    ):
        robot.calibrate(
            leader_action,
            duration_s=0.02,
            rate_hz=100.0,
            max_velocity_rad_s=100.0,
            position_tolerance_rad=0.001,
            settle_timeout_s=0.01,
        )

    assert client.session.send.call_count == 3
    hold_targets = client.session.send.call_args_list[0].args[0]
    first_targets = client.session.send.call_args_list[1].args[0]
    final_targets = client.session.send.call_args_list[2].args[0]
    assert hold_targets[Arm.LEFT][0] == pytest.approx(0.0)
    assert hold_targets[Arm.RIGHT][0] == pytest.approx(0.0)
    assert client.session.send.call_args_list[0].kwargs["end_effectors"] is None
    assert first_targets[Arm.LEFT][0] == pytest.approx(0.05)
    assert first_targets[Arm.RIGHT][0] == pytest.approx(0.05)
    assert final_targets[Arm.LEFT][0] == pytest.approx(0.1)
    assert final_targets[Arm.RIGHT][0] == pytest.approx(0.1)


def test_calibrate_requires_leader_action(robot):
    robot.connect()
    with pytest.raises(ValueError, match="必须.*主臂"):
        robot.calibrate()


@pytest.mark.parametrize(
    ("parameter", "value"),
    [
        ("duration_s", 0.0),
        ("rate_hz", -1.0),
        ("max_velocity_rad_s", 0.0),
        ("position_tolerance_rad", float("nan")),
        ("settle_timeout_s", 0.0),
    ],
)
def test_calibrate_rejects_invalid_motion_parameters(robot, parameter, value):
    robot.connect()
    kwargs = {parameter: value}
    with pytest.raises(ValueError, match=parameter):
        robot.calibrate(_action(), **kwargs)


def test_get_observation_uses_arm_ids_not_tuple_order(robot):
    robot.connect()
    client = FakeClient.instances[-1]
    client.state.get_joint_states.return_value = (
        SimpleNamespace(arm=Arm.RIGHT, positions_rad=tuple(range(11, 18))),
        SimpleNamespace(arm=Arm.LEFT, positions_rad=tuple(range(1, 8))),
    )
    client.end_effectors.get_state.return_value = SimpleNamespace(
        success=True,
        message="",
        states=(
            SimpleNamespace(
                arm=Arm.RIGHT,
                valid=True,
                position_unit=PositionUnit.METER,
                positions=(0.04,),
                message="",
            ),
            SimpleNamespace(
                arm=Arm.LEFT,
                valid=True,
                position_unit=PositionUnit.METER,
                positions=(0.03,),
                message="",
            ),
        ),
    )

    observation = robot.get_observation()

    assert observation["left_joint_1.pos"] == 1.0
    assert observation["left_joint_7.pos"] == 7.0
    assert observation["right_joint_1.pos"] == 11.0
    assert observation["right_joint_7.pos"] == 17.0
    assert observation["left_gripper.pos"] == pytest.approx(-32.5 * 3.141592653589793 / 180.0)
    assert observation["right_gripper.pos"] == pytest.approx(-32.5 * 3.141592653589793 / 180.0)
    assert observation.keys() == OY_OPENARM_POSITION_FEATURES.keys()


@pytest.mark.parametrize(
    ("mutate", "error"),
    [
        (lambda action: action.pop("left_joint_1.pos"), ValueError),
        (lambda action: action.__setitem__("unexpected.pos", 0.0), ValueError),
        (lambda action: action.__setitem__("left_joint_1.pos", float("nan")), ValueError),
        (lambda action: action.__setitem__("left_joint_1.pos", True), TypeError),
    ],
)
def test_send_action_rejects_invalid_input(robot, mutate, error):
    robot.connect()
    action = _action()
    mutate(action)
    with pytest.raises(error):
        robot.send_action(action)


def test_connect_failure_closes_client(robot):
    class FailingClient(FakeClient):
        def __init__(self, target, default_timeout):
            super().__init__(target, default_timeout)
            self.wait_until_channel_ready = MagicMock(side_effect=TimeoutError("offline"))

    with (
        patch.object(FAKE_SDK, "OpenArmClient", FailingClient),
        pytest.raises(TimeoutError, match="offline"),
    ):
        robot.connect()
    assert not robot.is_connected
    assert FakeClient.instances[-1].is_closed
