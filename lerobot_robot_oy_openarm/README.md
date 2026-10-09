# lerobot_robot_oy_openarm

LeRobot third-party robot plugin for the dual-arm OY OpenArm follower. It talks
to OpenArm RCG through `openarm-sdk` and exposes both arms, both grippers, and
optional LeRobot cameras through the standard `Robot` interface.

## Installation

```bash
pip install -e ./lerobot_robot_oy_openarm
```

The distribution and import package are both named
`lerobot_robot_oy_openarm`, so LeRobot's `register_third_party_plugins()` can
discover and import the plugin automatically.

## Usage

```bash
lerobot-teleoperate \
  --robot.type=oy_openarm_follower \
  --robot.ip=127.0.0.1 \
  --robot.port=50051 \
  --robot.align_on_connect=true \
  --teleop.type=oy_openarm_leader \
  --teleop.left_port=can1 \
  --teleop.right_port=can0
```

The follower controls both arms. Joint and gripper actions use radians at the
LeRobot boundary; gripper values are mapped to the physical metre ranges
reported by OpenArm RCG.

`align_on_connect` is disabled by default because it moves the follower. When
explicitly enabled, the follower reads the connected OY OpenArm leader through
the companion teleoperator plugin, aligns once, and stops the temporary servo
session before LeRobot starts its control loop. The stock `lerobot-record` and
`lerobot-teleoperate` commands both connect the teleoperator before the robot,
so no modification to LeRobot is required.
