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

from __future__ import annotations

import importlib
import logging
import math
import time
from contextlib import suppress
from functools import cached_property
from numbers import Real
from threading import RLock
from types import ModuleType
from typing import Any

from lerobot.cameras import make_cameras_from_configs
from lerobot.robots.robot import Robot

try:
    from lerobot.types import RobotAction, RobotObservation
except ModuleNotFoundError:
    from lerobot.lerobot_types import RobotAction, RobotObservation

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from .config_oy_openarm_follower import OyOpenArmFollowerConfig

logger = logging.getLogger(__name__)

JOINT_NAMES = tuple(f"joint_{index}" for index in range(1, 8))
ARM_POSITION_KEYS = tuple(
    f"{side}_{joint}.pos" for side in ("left", "right") for joint in JOINT_NAMES
)
SERVO_SETTLE_TOLERANCE_RATIO = 0.75
OY_OPENARM_POSITION_FEATURES = {
    **{f"left_{joint}.pos": float for joint in JOINT_NAMES},
    "left_gripper.pos": float,
    **{f"right_{joint}.pos": float for joint in JOINT_NAMES},
    "right_gripper.pos": float,
}


def _load_openarm_sdk() -> ModuleType:
    """按需导入 OpenArm SDK，避免未使用该机械臂时引入硬件依赖。"""

    try:
        return importlib.import_module("openarm_sdk")
    except ImportError as error:
        raise ImportError(
            "OyOpenArmFollower 依赖 'openarm-sdk'。请将 SDK 安装到 LeRobot 所在的同一环境中"
            "（本地源码可执行：`uv pip install -e /path/to/robotic_openarm`）。"
        ) from error


def _get_active_oy_openarm_leader() -> Any:
    """Return the connected leader registered by the companion plugin."""

    try:
        pairing = importlib.import_module("lerobot_teleoperator_oy_openarm.pairing")
    except ModuleNotFoundError as error:
        if error.name == "lerobot_teleoperator_oy_openarm":
            raise RuntimeError(
                "config.align_on_connect=true 需要安装 "
                "'lerobot_teleoperator_oy_openarm' 插件。"
            ) from error
        raise

    leader = pairing.get_active_leader()
    if leader is None:
        raise RuntimeError(
            "config.align_on_connect=true，但当前进程中没有已连接的 "
            "OyOpenArmLeader。请同时传入 --teleop.type=oy_openarm_leader。"
        )
    return leader


def _linear_map(
    value: float, source: tuple[float, float], target: tuple[float, float]
) -> float:
    """将数值从一个线性区间等比例映射到另一个线性区间。"""

    ratio = (value - source[0]) / (source[1] - source[0])
    return target[0] + ratio * (target[1] - target[0])


class OyOpenArmFollower(Robot):
    """完全基于 ``openarm-sdk`` 实现的 OY OpenArm 双臂从臂适配器。

    LeRobot 接口中的关节和夹爪位置统一使用弧度。SDK 的连续伺服接口要求
    手臂关节使用弧度、末端执行器目标使用米，因此这里会在配置的夹爪角度
    范围和 SDK 返回的物理行程范围之间进行双向映射。
    """

    config_class = OyOpenArmFollowerConfig
    name = "oy_openarm_follower"

    def __init__(self, config: OyOpenArmFollowerConfig):
        """根据配置初始化 SDK 引用、运行状态以及相机对象，但不连接硬件。"""

        super().__init__(config)
        self.config = config
        self._sdk = _load_openarm_sdk()
        self._client: Any | None = None
        self._servo_session: Any | None = None
        self._servo_lock = RLock()
        self._arms = (
            ("left", self._sdk.Arm.LEFT),
            ("right", self._sdk.Arm.RIGHT),
        )
        self._gripper_ranges_m: dict[Any, tuple[float, float]] = {}
        self.cameras = make_cameras_from_configs(config.cameras)

    @property
    def _cameras_ft(self) -> dict[str, tuple[int, ...]]:
        """根据相机配置生成 RGB 和深度图像的观测特征描述。"""

        features: dict[str, tuple[int, ...]] = {}
        for name, cfg in self.config.cameras.items():
            if getattr(cfg, "use_rgb", True):
                features[name] = (cfg.height, cfg.width, 3)
            if getattr(cfg, "use_depth", False):
                features[f"{name}_depth"] = (cfg.height, cfg.width, 1)
        return features

    @cached_property
    def observation_features(self) -> dict[str, type | tuple[int, ...]]:
        """返回双臂位置与相机图像组成的完整观测字段定义。"""

        return {**OY_OPENARM_POSITION_FEATURES, **self._cameras_ft}

    @cached_property
    def action_features(self) -> dict[str, type]:
        """返回与 OY OpenArm leader 完全一致的双臂动作字段定义。"""

        return OY_OPENARM_POSITION_FEATURES.copy()

    @property
    def is_connected(self) -> bool:
        """检查 SDK 客户端及所有配置相机是否均处于已连接状态。"""

        return (
            self._client is not None
            and not self._client.is_closed
            and all(camera.is_connected for camera in self.cameras.values())
        )

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        """连接 RCG，检查机器人及夹爪状态，然后连接所有配置相机。"""

        del calibrate  # 标定和电机零点由 RCG 或电机驱动器负责，LeRobot 不重复执行。
        target = f"{self.config.ip}:{self.config.port}"
        logger.info("正在将 %s 连接到 OpenArm RCG：%s", self, target)
        try:
            self._client = self._sdk.OpenArmClient(
                target, default_timeout=self.config.rpc_timeout
            )
            self._client.wait_until_channel_ready(
                timeout=self.config.connection_timeout
            )
            self.configure()
            for camera in self.cameras.values():
                camera.connect()
            if self.config.align_on_connect:
                leader = _get_active_oy_openarm_leader()
                logger.warning("即将在录制前执行 OY OpenArm 主从臂姿态对齐。")
                self.calibrate(leader.get_action())
        except Exception:
            self._cleanup_resources()
            raise
        logger.info("%s 已连接", self)

    @property
    def is_calibrated(self) -> bool:
        """始终返回真，因为 RCG 发布的关节状态已经完成标定并采用国际单位。"""

        # 该适配器在 LeRobot 启动期间绝不能修改电机零点。
        return True

    @check_if_not_connected
    def calibrate(
        self,
        action_pos: RobotAction | None = None,
        *,
        duration_s: float = 3.0,
        rate_hz: float = 50.0,
        max_velocity_rad_s: float = math.radians(20.0),
        position_tolerance_rad: float = math.radians(2.0),
        gripper_tolerance_m: float = 0.002,
        settle_timeout_s: float = 3.0,
    ) -> None:
        """平滑对齐双臂关节和夹爪，不修改电机零点。

        ``action_pos`` 是 leader 返回的 16 维弧度动作。夹爪先通过单次阻塞
        命令对齐，关节随后由临时的 arm-only JointServo 会话对齐。轨迹不做
        碰撞规划，执行前必须清空工作空间。
        """

        if action_pos is None:
            raise ValueError(
                "调用 calibrate() 时必须通过 action_pos 传入主臂的当前关节角度。"
            )

        parameters = {
            "duration_s": duration_s,
            "rate_hz": rate_hz,
            "max_velocity_rad_s": max_velocity_rad_s,
            "position_tolerance_rad": position_tolerance_rad,
            "gripper_tolerance_m": gripper_tolerance_m,
            "settle_timeout_s": settle_timeout_s,
        }
        self._validate_positive_numbers(parameters)
        values = {name: float(value) for name, value in parameters.items()}

        target = self._validated_action(action_pos)
        current = self._read_position_observation()

        # RCG serializes end-effector actions and JointServo operations. Align
        # the grippers first so no slow gripper action can make already-aligned
        # arm joints drift after their temporary servo stream has stopped.
        actual = self._align_grippers(
            target,
            current,
            tolerance_m=values["gripper_tolerance_m"],
        )

        max_joint_error = max(self._joint_errors(target, actual).values())
        if max_joint_error > values["position_tolerance_rad"]:
            self._align_arm_joints(
                current=actual,
                target=target,
                duration_s=values["duration_s"],
                rate_hz=values["rate_hz"],
                max_velocity_rad_s=values["max_velocity_rad_s"],
                position_tolerance_rad=values["position_tolerance_rad"],
                settle_timeout_s=values["settle_timeout_s"],
            )
        else:
            logger.info(
                "手臂关节已在对齐容差内，跳过临时连续伺服会话：最大误差 %.6f 弧度",
                max_joint_error,
            )

        # Controller hand-off can cause a small gravity/holding deviation after
        # STOPPED. Re-read once and perform at most one bounded arm-only
        # correction instead of passively waiting until the final timeout.
        actual = self._read_position_observation()
        post_stop_joint_error = max(self._joint_errors(target, actual).values())
        if post_stop_joint_error > values["position_tolerance_rad"]:
            logger.warning(
                "连续伺服停止后关节最大误差回升至 %.6f 弧度，执行一次关节修正。",
                post_stop_joint_error,
            )
            self._align_arm_joints(
                current=actual,
                target=target,
                duration_s=min(values["duration_s"], 1.0),
                rate_hz=values["rate_hz"],
                max_velocity_rad_s=values["max_velocity_rad_s"],
                position_tolerance_rad=values["position_tolerance_rad"],
                settle_timeout_s=values["settle_timeout_s"],
            )
            actual = self._read_position_observation()

        self._wait_until_aligned(
            target,
            actual,
            rate_hz=values["rate_hz"],
            position_tolerance_rad=values["position_tolerance_rad"],
            gripper_tolerance_m=values["gripper_tolerance_m"],
            timeout_s=values["settle_timeout_s"],
        )

    @staticmethod
    def _validate_positive_numbers(values: dict[str, Real]) -> None:
        """Validate positive, finite motion parameters."""

        for name, value in values.items():
            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(float(value))
            ):
                raise ValueError(f"参数 {name} 必须是有限数值。")
            if value <= 0:
                raise ValueError(f"参数 {name} 必须大于 0。")

    def _align_grippers(
        self,
        target: RobotAction,
        actual: RobotObservation,
        *,
        tolerance_m: float,
    ) -> RobotObservation:
        """Align each gripper after the temporary arm servo has stopped."""

        for side, arm in self._arms:
            key = f"{side}_gripper.pos"
            target_m = self._gripper_to_metres(side, arm, float(target[key]))
            actual_m = self._gripper_to_metres(side, arm, float(actual[key]))
            if abs(target_m - actual_m) <= tolerance_m:
                logger.info(
                    "%s 侧夹爪已在 %.3f 毫米对齐容差内",
                    side,
                    tolerance_m * 1000.0,
                )
                continue

            result = self._client.end_effectors.move(
                arm,
                [target_m],
                self._sdk.PositionUnit.METER,
                timeout=self.config.gripper_action_timeout,
                info_timeout=self.config.rpc_timeout,
            )
            if not bool(getattr(result, "success", False)):
                message = (
                    str(getattr(result, "message", "")).strip()
                    or "服务端未提供错误详情"
                )
                raise RuntimeError(f"OpenArm {side} 侧夹爪对齐失败：{message}")
            actual = self._read_position_observation()
        return actual

    def _wait_until_aligned(
        self,
        target: RobotAction,
        actual: RobotObservation,
        *,
        rate_hz: float,
        position_tolerance_rad: float,
        gripper_tolerance_m: float,
        timeout_s: float,
    ) -> None:
        """Wait until both joints and grippers satisfy their physical tolerances."""

        deadline = time.perf_counter() + timeout_s
        while True:
            joint_errors_rad, gripper_errors_m = self._alignment_errors(target, actual)
            max_joint_error_rad = max(joint_errors_rad.values())
            max_gripper_error_m = max(gripper_errors_m.values())
            if (
                max_joint_error_rad <= position_tolerance_rad
                and max_gripper_error_m <= gripper_tolerance_m
            ):
                logger.info(
                    "主从臂角度对齐完成：最大关节误差 %.6f 弧度，最大夹爪误差 %.3f 毫米",
                    max_joint_error_rad,
                    max_gripper_error_m * 1000.0,
                )
                return

            now = time.perf_counter()
            if now >= deadline:
                self._raise_alignment_timeout(
                    joint_errors_rad,
                    gripper_errors_m,
                    position_tolerance_rad=position_tolerance_rad,
                    gripper_tolerance_m=gripper_tolerance_m,
                )
            time.sleep(min(1.0 / rate_hz, deadline - now))
            actual = self._read_position_observation()

    def _align_arm_joints(
        self,
        *,
        current: RobotObservation,
        target: RobotAction,
        duration_s: float,
        rate_hz: float,
        max_velocity_rad_s: float,
        position_tolerance_rad: float,
        settle_timeout_s: float,
    ) -> None:
        """Align 14 arm joints with an arm-only temporary servo stream."""

        maximum_distance_rad = max(self._joint_errors(target, current).values())
        actual_duration_s = max(duration_s, maximum_distance_rad / max_velocity_rad_s)
        steps = max(1, math.ceil(actual_duration_s * rate_hz))
        period_s = 1.0 / rate_hz
        settle_tolerance_rad = position_tolerance_rad * SERVO_SETTLE_TOLERANCE_RATIO

        # Start before trajectory timing and publish a current-position hold to
        # satisfy RCG's first-target jump protection.
        self._ensure_servo_session()
        try:
            self._submit_action(
                current, include_end_effectors=False, wait_for_publish=True
            )
            started_at = time.perf_counter()
            logger.info(
                "开始主从臂关节对齐：计划时长 %.2f 秒，共 %d 个插值点（不含夹爪）",
                actual_duration_s,
                steps,
            )
            for step in range(1, steps + 1):
                ratio = step / steps
                interpolated = dict(current)
                for key in ARM_POSITION_KEYS:
                    interpolated[key] = float(current[key]) + ratio * (
                        float(target[key]) - float(current[key])
                    )
                self._submit_action(interpolated, include_end_effectors=False)
                next_deadline = started_at + step * period_s
                time.sleep(max(0.0, next_deadline - time.perf_counter()))

            settle_deadline = time.perf_counter() + settle_timeout_s
            while True:
                actual = self._read_position_observation()
                joint_errors = self._joint_errors(target, actual)
                max_joint_error = max(joint_errors.values())
                # Converge more tightly while the servo is active so gravity
                # and controller hand-off still leave the final pose inside
                # the user-facing tolerance after STOPPED.
                if max_joint_error <= settle_tolerance_rad:
                    break
                now = time.perf_counter()
                if now >= settle_deadline:
                    # The tighter stop margin is best-effort, not a new user-
                    # facing acceptance threshold. If the arm is already
                    # inside the configured tolerance, stop normally and let
                    # the post-stop check decide whether correction is needed.
                    if max_joint_error <= position_tolerance_rad:
                        logger.warning(
                            "关节已满足 %.6f 弧度验收容差，但未在等待时间内收敛到 "
                            "%.6f 弧度停止余量；继续执行停止后的最终检查。",
                            position_tolerance_rad,
                            settle_tolerance_rad,
                        )
                        break
                    self._raise_alignment_timeout(
                        joint_errors,
                        {},
                        position_tolerance_rad=position_tolerance_rad,
                        gripper_tolerance_m=1.0,
                    )
                self._submit_action(target, include_end_effectors=False)
                time.sleep(min(period_s, max(0.0, settle_deadline - now)))
        finally:
            # Do not issue any end-effector operation, and do not let connect()
            # return, until RCG confirms that this operation is fully stopped.
            self._stop_servo_session()

    @staticmethod
    def _joint_errors(
        target: RobotAction, actual: RobotObservation
    ) -> dict[str, float]:
        return {
            key: abs(float(target[key]) - float(actual[key]))
            for key in ARM_POSITION_KEYS
        }

    def _alignment_errors(
        self, target: RobotAction, actual: RobotObservation
    ) -> tuple[dict[str, float], dict[str, float]]:
        """Return arm errors in radians and physical gripper errors in metres."""

        joint_errors = self._joint_errors(target, actual)
        gripper_errors = {
            f"{side}_gripper.pos": abs(
                self._gripper_to_metres(side, arm, float(target[f"{side}_gripper.pos"]))
                - self._gripper_to_metres(
                    side, arm, float(actual[f"{side}_gripper.pos"])
                )
            )
            for side, arm in self._arms
        }
        return joint_errors, gripper_errors

    @staticmethod
    def _raise_alignment_timeout(
        joint_errors_rad: dict[str, float],
        gripper_errors_m: dict[str, float],
        *,
        position_tolerance_rad: float,
        gripper_tolerance_m: float,
    ) -> None:
        """Raise a timeout describing the largest normalized alignment error."""

        violations = [
            (
                error / position_tolerance_rad,
                f"{key} 的误差为 {error:.6f} 弧度，"
                f"允许误差为 {position_tolerance_rad:.6f} 弧度。",
            )
            for key, error in joint_errors_rad.items()
            if error > position_tolerance_rad
        ]
        violations.extend(
            (
                error / gripper_tolerance_m,
                f"{key} 的误差为 {error * 1000.0:.3f} 毫米，"
                f"允许误差为 {gripper_tolerance_m * 1000.0:.3f} 毫米。",
            )
            for key, error in gripper_errors_m.items()
            if error > gripper_tolerance_m
        )
        _, worst_error = max(violations, key=lambda item: item[0])
        raise TimeoutError(f"主从臂角度对齐超时：{worst_error}")

    def configure(self) -> None:
        """检查 RCG 与机器人是否就绪，并缓存左右夹爪的米制物理范围。"""

        if self._client is None or self._client.is_closed:
            raise RuntimeError("OpenArm SDK 客户端尚未连接。")

        status = self._client.system.get_status(timeout=self.config.rpc_timeout)
        if self.config.require_robot_ready and (
            not status.server_ready or not status.robot_ready
        ):
            raise RuntimeError(
                "OpenArm RCG 可以访问，但尚未就绪："
                f"server_ready={status.server_ready}, robot_ready={status.robot_ready}。"
            )

        snapshot = self._client.end_effectors.get_info(
            self._sdk.Arm.BOTH, timeout=self.config.rpc_timeout
        )
        if not snapshot.success:
            raise RuntimeError(f"无法查询 OpenArm 末端执行器：{snapshot.message}")

        by_arm = {info.arm: info for info in snapshot.end_effectors}
        ranges: dict[Any, tuple[float, float]] = {}
        for side, arm in self._arms:
            info = by_arm.get(arm)
            if info is None:
                raise RuntimeError(f"OpenArm 未返回 {side} 侧末端执行器信息。")
            if info.position_unit is not self._sdk.PositionUnit.METER:
                raise RuntimeError(
                    f"{side} 侧末端执行器使用 {info.position_unit.name} 单位，"
                    "但连续关节伺服接口要求夹爪目标使用米。"
                )
            if (
                len(info.joint_names) != 1
                or len(info.min_positions) != 1
                or len(info.max_positions) != 1
            ):
                raise RuntimeError(f"{side} 侧末端执行器必须只提供一个位置量。")
            bounds = (float(info.min_positions[0]), float(info.max_positions[0]))
            if (
                not all(math.isfinite(value) for value in bounds)
                or bounds[0] >= bounds[1]
            ):
                raise RuntimeError(f"OpenArm 返回的 {side} 侧夹爪范围无效：{bounds}")
            if getattr(self.config, f"{side}_gripper_inverted"):
                bounds = (bounds[1], bounds[0])
            ranges[arm] = bounds
        self._gripper_ranges_m = ranges

    def _states_by_arm(self, states: tuple[Any, ...], kind: str) -> dict[Any, Any]:
        """按照左右臂枚举整理 SDK 状态，并检查两侧状态是否完整。"""

        by_arm = {state.arm: state for state in states}
        missing = [side for side, arm in self._arms if arm not in by_arm]
        if missing:
            raise RuntimeError(f"OpenArm 的 {kind} 响应缺少以下机械臂：{missing}")
        return by_arm

    def _angular_gripper_bounds(self, side: str) -> tuple[float, float]:
        """读取指定侧夹爪的角度限位，并将配置中的度转换为弧度。"""

        limits = getattr(self.config, f"{side}_joint_limits")["gripper"]
        return math.radians(limits[0]), math.radians(limits[1])

    def _gripper_to_metres(self, side: str, arm: Any, value_rad: float) -> float:
        """将 LeRobot 使用的夹爪弧度值映射为 SDK 连续伺服所需的米制值。"""

        return _linear_map(
            value_rad, self._angular_gripper_bounds(side), self._gripper_ranges_m[arm]
        )

    def _gripper_to_radians(self, side: str, arm: Any, value_m: float) -> float:
        """将 SDK 返回的夹爪米制位置反向映射为 LeRobot 使用的弧度值。"""

        return _linear_map(
            value_m, self._gripper_ranges_m[arm], self._angular_gripper_bounds(side)
        )

    def _read_position_observation(self) -> RobotObservation:
        """只读取双臂关节和夹爪位置，不触发相机采集。"""

        joint_states = self._states_by_arm(
            self._client.state.get_joint_states(
                self._sdk.Arm.BOTH, timeout=self.config.rpc_timeout
            ),
            "关节状态",
        )
        end_snapshot = self._client.end_effectors.get_state(
            self._sdk.Arm.BOTH, timeout=self.config.rpc_timeout
        )
        if not end_snapshot.success:
            raise RuntimeError(
                f"无法读取 OpenArm 末端执行器状态：{end_snapshot.message}"
            )
        end_states = self._states_by_arm(end_snapshot.states, "末端执行器状态")

        position_observation: RobotObservation = {}
        for side, arm in self._arms:
            positions = joint_states[arm].positions_rad
            if len(positions) != len(JOINT_NAMES):
                raise RuntimeError(
                    f"OpenArm {side} 侧关节状态包含 {len(positions)} 个位置值，应为 7 个。"
                )
            for joint, value in zip(JOINT_NAMES, positions, strict=True):
                position_observation[f"{side}_{joint}.pos"] = float(value)

            end_state = end_states[arm]
            if (
                not end_state.valid
                or end_state.position_unit is not self._sdk.PositionUnit.METER
            ):
                raise RuntimeError(
                    f"OpenArm {side} 侧夹爪状态无效或未使用米制单位：{end_state.message}"
                )
            if len(end_state.positions) != 1:
                raise RuntimeError(f"OpenArm {side} 侧夹爪状态必须只包含一个位置值。")
            position_observation[f"{side}_gripper.pos"] = self._gripper_to_radians(
                side, arm, float(end_state.positions[0])
            )
        return position_observation

    @check_if_not_connected
    def get_observation(self) -> RobotObservation:
        """读取左右臂关节、左右夹爪以及所有已配置相机的最新观测。"""

        started_at = time.perf_counter()
        observation = self._read_position_observation()

        for name, camera in self.cameras.items():
            cfg = self.config.cameras[name]
            if getattr(cfg, "use_rgb", True):
                observation[name] = camera.read_latest()
            if getattr(cfg, "use_depth", False):
                observation[f"{name}_depth"] = camera.read_latest_depth()

        logger.debug(
            "%s 读取观测耗时 %.1f 毫秒", self, (time.perf_counter() - started_at) * 1e3
        )
        return observation

    def _validated_action(self, action: RobotAction) -> RobotAction:
        """校验动作字段和值，按左右臂机械限位裁剪，并返回实际待发送动作。"""

        expected = self.action_features.keys()
        if action.keys() != expected:
            missing = sorted(expected - action.keys())
            unexpected = sorted(action.keys() - expected)
            raise ValueError(
                f"OY OpenArm 动作字段不匹配：缺少={missing}，多余={unexpected}"
            )

        result: RobotAction = {}
        for key, raw_value in action.items():
            # LeRobotDataset returns each replayed action value as a scalar
            # torch.Tensor. NumPy scalars/one-element arrays expose the same
            # ``item()`` protocol. Unwrap those values here so the robot
            # boundary consistently works with plain Python numbers without
            # importing either optional numerical package in this plugin.
            if not isinstance(raw_value, Real):
                item = getattr(raw_value, "item", None)
                if callable(item):
                    try:
                        raw_value = item()
                    except (TypeError, ValueError, RuntimeError) as error:
                        raise TypeError(
                            f"动作 action[{key!r}] 必须是单个数值。"
                        ) from error

            if isinstance(raw_value, bool) or not isinstance(raw_value, Real):
                raise TypeError(f"动作 action[{key!r}] 必须是单个数值。")
            value = float(raw_value)
            if not math.isfinite(value):
                raise ValueError(f"动作 action[{key!r}] 必须是有限数值。")
            side, name_with_suffix = key.split("_", maxsplit=1)
            joint = name_with_suffix.removesuffix(".pos")
            low_deg, high_deg = getattr(self.config, f"{side}_joint_limits")[joint]
            low_rad, high_rad = math.radians(low_deg), math.radians(high_deg)
            clipped = min(max(value, low_rad), high_rad)
            if clipped != value:
                logger.debug("将 %s 从 %.6f 弧度裁剪为 %.6f 弧度", key, value, clipped)
            result[key] = clipped
        return result

    @staticmethod
    def _servo_state_name(feedback: Any | None) -> str:
        """返回适合日志显示的伺服状态名称。"""

        if feedback is None:
            return "UNKNOWN"
        state = getattr(feedback, "state", None)
        return str(getattr(state, "name", state if state is not None else "UNKNOWN"))

    def _check_servo_session(self, session: Any) -> Any | None:
        """检查 SDK 传输错误和 RCG 业务终态，不允许静默重启失败会话。"""

        error = session.error
        feedback = session.feedback
        if error is not None:
            raise RuntimeError(f"OpenArm 连续伺服传输失败：{error}") from error

        state = getattr(feedback, "state", None) if feedback is not None else None
        is_terminal = bool(getattr(state, "is_terminal", False))
        success = (
            bool(getattr(feedback, "success", True)) if feedback is not None else True
        )
        if is_terminal or not success:
            message = (
                str(getattr(feedback, "message", "")).strip() or "服务端未提供错误详情"
            )
            received = int(getattr(feedback, "last_received_sequence", 0))
            published = int(getattr(feedback, "last_published_sequence", 0))
            raise RuntimeError(
                "OpenArm 连续伺服会话异常终止："
                f"state={self._servo_state_name(feedback)}, message={message}, "
                f"received={received}, published={published}。"
            )

        if session.is_closed:
            raise RuntimeError(
                "OpenArm 连续伺服流已关闭，但没有收到有效的终态反馈；"
                "为避免反复切换控制器，本次运行不会自动重建会话。"
            )
        return feedback

    def _ensure_servo_session(self) -> Any:
        """创建或返回唯一的健康会话；失败会话必须由上层显式处理。"""

        with self._servo_lock:
            if self._servo_session is not None:
                self._check_servo_session(self._servo_session)
                return self._servo_session

            session = self._client.joint_servo.start(
                [self._sdk.Arm.LEFT, self._sdk.Arm.RIGHT],
                publish_rate_hz=self.config.servo_publish_rate_hz,
                command_timeout_ms=self.config.servo_command_timeout_ms,
                ready_timeout=self.config.servo_ready_timeout,
            )
            try:
                feedback = self._check_servo_session(session)
            except Exception:
                with suppress(Exception):
                    session.cancel()
                raise
            self._servo_session = session
            logger.info(
                "OpenArm 连续伺服会话已就绪：state=%s，发布频率=%s Hz，目标超时=%s ms",
                self._servo_state_name(feedback),
                getattr(feedback, "publish_rate_hz", "unknown"),
                getattr(feedback, "command_timeout_ms", "unknown"),
            )
            return session

    def _wait_until_servo_published(self, session: Any, sequence: int) -> None:
        """等待指定首帧被 RCG 接收并发布，避免在无效会话上启动轨迹。"""

        feedback = self._check_servo_session(session)
        command_timeout_ms = int(
            getattr(
                feedback,
                "command_timeout_ms",
                self.config.servo_command_timeout_ms or 500,
            )
        )
        timeout_s = max(1.0, command_timeout_ms * 2 / 1000.0)
        deadline = time.perf_counter() + timeout_s

        while True:
            feedback = self._check_servo_session(session)
            received = int(getattr(feedback, "last_received_sequence", 0))
            published = int(getattr(feedback, "last_published_sequence", 0))
            if received >= sequence and published >= sequence:
                return

            remaining = deadline - time.perf_counter()
            if remaining <= 0:
                raise TimeoutError(
                    "等待 OpenArm 连续伺服首帧发布超时："
                    f"sequence={sequence}, received={received}, published={published}, "
                    f"state={self._servo_state_name(feedback)}。"
                )
            time.sleep(min(0.005, remaining))

    def _submit_action(
        self,
        action: RobotAction,
        *,
        include_end_effectors: bool = True,
        wait_for_publish: bool = False,
    ) -> RobotAction:
        """提交一帧动作，并按需确认该帧已由 RCG 发布。"""

        sent = self._validated_action(action)
        targets = {
            arm: tuple(sent[f"{side}_{joint}.pos"] for joint in JOINT_NAMES)
            for side, arm in self._arms
        }
        end_effectors = None
        if include_end_effectors:
            end_effectors = {
                arm: [self._gripper_to_metres(side, arm, sent[f"{side}_gripper.pos"])]
                for side, arm in self._arms
            }

        # start() 可能阻塞较长时间，必须先确保会话 READY，再生成目标采样时间戳。
        # send() 的返回值只代表 SDK 本地接受，因此首帧还需等待服务端发布反馈。
        with self._servo_lock:
            session = self._ensure_servo_session()
            sampled_at_unix_ns = time.time_ns()
            sequence = session.send(
                targets,
                end_effectors=end_effectors,
                sampled_at_unix_ns=sampled_at_unix_ns,
            )
            self._check_servo_session(session)
            if wait_for_publish:
                self._wait_until_servo_published(session, sequence)
        return sent

    @check_if_not_connected
    def send_action(self, action: RobotAction) -> RobotAction:
        """校验并裁剪动作、转换夹爪单位，然后提交一帧双臂伺服目标。"""

        return self._submit_action(action)

    def _stop_servo_session(self) -> None:
        """Stop the current stream and detach it only after RCG confirms STOPPED."""

        with self._servo_lock:
            session = self._servo_session
            if session is None:
                return

            # Keep the lock for the complete stop handshake. Otherwise a
            # concurrent send_action() could observe no/current session and
            # race a new JointServo.start() against RCG's controller cleanup.
            try:
                feedback = self._stop_session(session)
            except Exception:
                if not session.is_closed:
                    with suppress(Exception):
                        session.cancel()
                raise
            self._servo_session = None

        logger.info(
            "OpenArm 连续伺服会话已完全停止：state=%s, received=%s, published=%s",
            self._servo_state_name(feedback),
            getattr(feedback, "last_received_sequence", "unknown"),
            getattr(feedback, "last_published_sequence", "unknown"),
        )

    def _stop_session(self, session: Any) -> Any:
        """Stop one SDK session and require a successful STOPPED response."""

        feedback = (
            session.feedback
            if session.is_closed
            else session.stop(timeout=self.config.servo_stop_timeout)
        )
        self._require_servo_stopped(session, feedback)
        return feedback

    def _require_servo_stopped(self, session: Any, feedback: Any | None) -> None:
        """Reject transport errors and every terminal result except successful STOPPED."""

        error = session.error
        if error is not None:
            raise RuntimeError(f"停止 OpenArm 连续伺服时传输失败：{error}") from error

        state_name = self._servo_state_name(feedback)
        success = bool(getattr(feedback, "success", False))
        if state_name == "STOPPED" and success:
            return

        message = (
            str(getattr(feedback, "message", "")).strip()
            if feedback is not None
            else ""
        ) or "服务端未提供错误详情"
        received = int(getattr(feedback, "last_received_sequence", 0))
        published = int(getattr(feedback, "last_published_sequence", 0))
        raise RuntimeError(
            "OpenArm 连续伺服未能正常停止："
            f"state={state_name}, success={success}, message={message}, "
            f"received={received}, published={published}。"
        )

    def _cleanup_resources(self) -> None:
        """尽力停止伺服会话并关闭 SDK 客户端和相机，保证资源完整释放。"""

        with self._servo_lock:
            session = self._servo_session
            if session is not None:
                try:
                    self._stop_session(session)
                except Exception as error:
                    # Stop failure must remain visible, but resource cleanup
                    # still has to close cameras and the SDK channel.
                    logger.warning("未能正常停止 OpenArm 伺服会话：%s", error)
                    if not session.is_closed:
                        with suppress(Exception):
                            session.cancel()
            self._servo_session = None
            client, self._client = self._client, None

        if client is not None:
            try:
                client.close()
            except Exception as error:
                logger.warning("未能正常关闭 OpenArm SDK 客户端：%s", error)

        for name, camera in self.cameras.items():
            try:
                if camera.is_connected:
                    camera.disconnect()
            except Exception as error:
                logger.warning("未能正常断开相机 %s：%s", name, error)
        self._gripper_ranges_m = {}

    def disconnect(self) -> None:
        """停止本对象持有的伺服会话，并释放 SDK 客户端与相机资源。"""

        self._cleanup_resources()
        logger.info("%s 已断开连接", self)
