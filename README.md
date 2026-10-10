# LeRobot OY OpenArm Plugins

本项目为 OY OpenArm 提供两个独立的 LeRobot 第三方插件：

- `lerobot_robot_oy_openarm`：双臂从臂（Follower）插件，通过 OpenArm SDK/RCG 控制机械臂。
- `lerobot_teleoperator_oy_openarm`：双臂主臂（Leader）插件，通过两路 CAN 读取主臂关节位置。

安装后，LeRobot 的 `register_third_party_plugins()` 会自动发现这两个插件，因此无需修改
LeRobot 源码即可使用 `lerobot-teleoperate`、`lerobot-record` 和 `lerobot-rollout`。

> **安全提示：** 主从对齐和遥操作会真实移动机械臂。运行前必须确认急停可用，并清空机械臂工作空间。

## 1. 安装

建议先进入安装 LeRobot 的 Python 环境：

```bash
conda activate lerobot_py312
```

从 PyPI 安装两个插件：

```bash
pip install lerobot_robot_oy_openarm
pip install lerobot_teleoperator_oy_openarm
```

如果正在本仓库中开发，可改用 editable 安装：

```bash
cd ~/projects/lerobot_oy_openarm

pip install -e ./lerobot_robot_oy_openarm
pip install -e ./lerobot_teleoperator_oy_openarm
```

检查插件是否能被 LeRobot 发现：

```bash
python - <<'PY'
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.robots.config import RobotConfig
from lerobot.teleoperators.config import TeleoperatorConfig

register_third_party_plugins()

print(RobotConfig.get_known_choices()["oy_openarm_follower"])
print(TeleoperatorConfig.get_known_choices()["oy_openarm_leader"])
PY
```

正常情况下会显示来自以下 Python 包的配置类：

```text
lerobot_robot_oy_openarm
lerobot_teleoperator_oy_openarm
```

## 2. 硬件准备

### 2.1 启动 OpenArm RCG

先启动机器人控制网关，并确认 gRPC 端口可访问。例如：

```bash
ros2 launch oym_rcg_py rcg.launch.py
ss -lnt | grep 50051
```

以下示例假设 RCG 地址为 `127.0.0.1:50051`。

### 2.2 配置主臂 CAN

以下示例假设左主臂使用 `can1`，右主臂使用 `can0`，波特率为 1 Mbps：

```bash
sudo ip link set can0 down
sudo ip link set can0 type can bitrate 1000000 restart-ms 100
sudo ip link set can0 up

sudo ip link set can1 down
sudo ip link set can1 type can bitrate 1000000 restart-ms 100
sudo ip link set can1 up
```

检查接口状态：

```bash
ip -details link show can0
ip -details link show can1
```

如果实际接线相反，请交换后续命令中的 `left_port` 和 `right_port`。

### 2.3 查找相机

```bash
lerobot-find-cameras opencv
```

本文后续示例使用三台相机，请根据实际设备修改 `index_or_path`：

```bash
export OY_CAMERAS='{chest: {type: opencv, index_or_path: 16, width: 640, height: 480, fps: 30}, left_wrist: {type: opencv, index_or_path: 4, width: 640, height: 480, fps: 30}, right_wrist: {type: opencv, index_or_path: 10, width: 640, height: 480, fps: 30}}'
```

## 3. 主从遥操作

使用标准 `lerobot-teleoperate` 命令：

```bash
lerobot-teleoperate \
  --robot.type=oy_openarm_follower \
  --robot.id=oy_follower \
  --robot.ip=127.0.0.1 \
  --robot.port=50051 \
  --robot.side=both \
  --robot.align_on_connect=true \
  --robot.servo_command_timeout_ms=2000 \
  --robot.left_gripper_inverted=true \
  --robot.right_gripper_inverted=true \
  --robot.cameras="$OY_CAMERAS" \
  --teleop.type=oy_openarm_leader \
  --teleop.id=oy_leader \
  --teleop.left_port=can1 \
  --teleop.right_port=can0 \
  --teleop.timeout=0.1 \
  --teleop.auto_configure_socketcan=false \
  --fps=30 \
  --display_data=false
```

`--robot.align_on_connect=true` 会在控制循环开始前读取主臂姿态，并让从臂平滑对齐。
该选项默认关闭，只有明确启用时才会在连接阶段移动机器人。

## 4. 使用 `lerobot-record` 采集数据

如需上传 Hugging Face Hub，先登录并设置用户名：

```bash
hf auth login
export HF_USER=<你的 Hugging Face 用户名>
```

定义数据集名称：

```bash
export DATASET_REPO_ID=${HF_USER}/oy_openarm_pick_cube
```

开始采集：

```bash
lerobot-record \
  --robot.type=oy_openarm_follower \
  --robot.id=oy_follower \
  --robot.ip=127.0.0.1 \
  --robot.port=50051 \
  --robot.side=both \
  --robot.align_on_connect=true \
  --robot.servo_command_timeout_ms=2000 \
  --robot.left_gripper_inverted=true \
  --robot.right_gripper_inverted=true \
  --robot.cameras="$OY_CAMERAS" \
  --teleop.type=oy_openarm_leader \
  --teleop.id=oy_leader \
  --teleop.left_port=can1 \
  --teleop.right_port=can0 \
  --teleop.timeout=0.1 \
  --teleop.auto_configure_socketcan=false \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --dataset.no_stamp=true \
  --dataset.num_episodes=50 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=15 \
  --dataset.fps=30 \
  --dataset.single_task="Pick up the cube and place it in the target area" \
  --dataset.streaming_encoding=true \
  --dataset.encoder_threads=2 \
  --dataset.push_to_hub=false \
  --display_data=false
```

如果只保存到本机：

```bash
--dataset.push_to_hub=false
```

数据默认保存在 Hugging Face/LeRobot 本地缓存中。`--dataset.no_stamp=true` 用于保持
`repo_id` 不变，后续训练时可以直接使用相同的 `DATASET_REPO_ID`。

录制过程常用按键：

- `→` 或 `n`：提前结束当前 episode。
- `←` 或 `r`：丢弃当前 episode 并重新录制。
- `Esc` 或 `q`：停止录制。

### 4.1 采集时使用 Foxglove

OY OpenArm robot 插件会自动把 Foxglove 序列化和网络发送移到后台线程。后台默认以
30 FPS 为目标，并且只保留最新的待显示帧：如果 Foxglove 处理不及，插件会跳过过期的
**可视化帧**，但不会跳过数据集帧，也不会阻塞动作发送和机器人控制循环。

建议使用 JPEG 压缩图像，避免三路 `640x480` RawImage 产生约 79 MiB/s 的数据：

```bash
lerobot-record \
  ... \
  --dataset.fps=30 \
  --dataset.streaming_encoding=true \
  --dataset.rgb_encoder.vcodec=h264 \
  --dataset.rgb_encoder.preset=ultrafast \
  --dataset.rgb_encoder.crf=23 \
  --dataset.rgb_encoder.g=30 \
  --dataset.encoder_threads=2 \
  --display_data=true \
  --display_mode=foxglove \
  --display_compressed_images=true
```

Foxglove 默认连接地址为 `ws://127.0.0.1:8765`。插件会在整个录制和 episode 间的 reset
阶段保持相同的图像通道类型，因此支持多 episode 录制。Foxglove 页面为了跟上实时数据
可能以低于 30 Hz 的频率刷新；这不会改变数据集的目标 FPS。若终端仍出现
`Record loop is running slower ... than the target FPS`，则说明相机、机器人状态读取或视频编码
本身已经超过 33.3 ms，需要继续降低分辨率或检查对应耗时。

如需调整 Foxglove 后台发送上限，可在命令前设置环境变量；同时实现 30 FPS 可视化和
30 FPS 采集时保持默认值：

```bash
OY_OPENARM_FOXGLOVE_MAX_FPS=30 lerobot-record ...
```

本项目默认运行环境若没有真实的 NVIDIA/Intel/AMD 硬件编码设备，不要使用
`--dataset.rgb_encoder.vcodec=auto`。仅在 FFmpeg 列出编码器名称并不表示对应硬件真实可用；
此时应显式使用上述 `h264` 软件编码。默认的 `libsvtav1` 虽然压缩率更高，但三路实时 AV1
编码会产生大量工作线程，容易与 30 Hz 控制循环发生调度竞争。

## 5. 使用 ACT 训练模型

确保训练相关依赖已经安装：

```bash
pip install "lerobot[training]"
```

使用刚才采集的数据训练 ACT：

```bash
lerobot-train \
  --dataset.repo_id="$DATASET_REPO_ID" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --output_dir=outputs/train/act_oy_openarm_pick_cube \
  --job_name=act_oy_openarm_pick_cube \
  --steps=100000 \
  --wandb.enable=false
```

训练检查点默认位于：

```text
outputs/train/act_oy_openarm_pick_cube/checkpoints/
```

最新模型通常可以通过以下路径访问：

```text
outputs/train/act_oy_openarm_pick_cube/checkpoints/last/pretrained_model
```

如果显存不足，可以减少 batch size：

```bash
--batch_size=4
```

如需启用 Weights & Biases：

```bash
wandb login
```

并将训练参数改为：

```bash
--wandb.enable=true
```

从最近的本地检查点继续训练：

```bash
lerobot-train \
  --config_path=outputs/train/act_oy_openarm_pick_cube/checkpoints/last/pretrained_model/train_config.json \
  --resume=true
```

## 6. 使用 ACT 模型推理

LeRobot 0.6 使用 `lerobot-rollout` 在真实机器人上运行训练好的策略。

推理时的相机名称、分辨率、顺序和关节特征必须与采集训练数据时保持一致：

```bash
export ACT_POLICY=outputs/train/act_oy_openarm_pick_cube/checkpoints/last/pretrained_model

lerobot-rollout \
  --strategy.type=base \
  --policy.path="$ACT_POLICY" \
  --device=cuda \
  --robot.type=oy_openarm_follower \
  --robot.id=oy_follower \
  --robot.ip=127.0.0.1 \
  --robot.port=50051 \
  --robot.side=both \
  --robot.servo_command_timeout_ms=2000 \
  --robot.left_gripper_inverted=true \
  --robot.right_gripper_inverted=true \
  --robot.cameras="$OY_CAMERAS" \
  --task="Pick up the cube and place it in the target area" \
  --duration=60 \
  --display_data=false
```

ACT 本身不依赖自然语言任务描述，但建议让 `--task` 与采集时的
`--dataset.single_task` 保持一致。

### 推理前的初始姿态

自主推理通常不连接主臂，因此不要在上面的命令中设置
`--robot.align_on_connect=true`。运行模型前，应手动或通过一次遥操作将机械臂移动到训练数据中
使用的安全初始姿态，然后退出遥操作并启动 `lerobot-rollout`。

## 7. 常见问题

### 插件无法发现

确认两个插件安装在运行 LeRobot 命令的同一个 Python 环境：

```bash
which python
which lerobot-record
python -m pip show lerobot_robot_oy_openarm
python -m pip show lerobot_teleoperator_oy_openarm
```

### 不要在 import 路径中加入 `src`

正确：

```python
import lerobot_robot_oy_openarm
import lerobot_teleoperator_oy_openarm
```

错误：

```python
import lerobot_robot_oy_openarm.src.lerobot_robot_oy_openarm
```

### 连续伺服会话 `EXPIRED`

如果出现 `target watchdog expired`，说明控制循环在看门狗期限内没有发送新动作。建议：

- 设置 `--robot.servo_command_timeout_ms=2000`。
- 使用 `--dataset.streaming_encoding=true`。
- 根据 CPU 性能调整 `--dataset.encoder_threads`。
- 检查相机读取是否阻塞或实际帧率是否低于 `--dataset.fps`。

### `align_on_connect` 找不到主臂

自动对齐只适用于同时配置以下参数的遥操作或数采命令：

```text
--robot.type=oy_openarm_follower
--teleop.type=oy_openarm_leader
--robot.align_on_connect=true
```

标准 `lerobot-record` 和 `lerobot-teleoperate` 会先连接主臂，再连接从臂，因此可以直接使用该功能。
