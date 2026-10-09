# lerobot_teleoperator_oy_openarm

LeRobot third-party teleoperator plugin for the dual-arm OY OpenArm leader. It
reads the eight joints on each arm through two persistent CAN connections and
publishes a 16-position LeRobot action dictionary.

## Installation

```bash
pip install -e ./lerobot_teleoperator_oy_openarm
```

The distribution and import package are both named
`lerobot_teleoperator_oy_openarm`, so LeRobot's
`register_third_party_plugins()` can discover and import the plugin
automatically.

## Usage

```bash
lerobot-teleoperate \
  --teleop.type=oy_openarm_leader \
  --teleop.left_port=can1 \
  --teleop.right_port=can0 \
  --robot.type=oy_openarm_follower \
  --robot.ip=127.0.0.1 \
  --robot.port=50051 \
  --robot.align_on_connect=true
```

The default backend is Linux SocketCAN at 1 Mbps. When automatic interface
configuration is enabled, bringing a CAN interface up may require
`CAP_NET_ADMIN` or `sudo`.
