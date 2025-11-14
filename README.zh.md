# WujiHand 重定向系统

[English](README.md) | 中文

基于 DexRetargeting 算法的高精度手部姿态重定向系统，支持 Vision Pro 手部跟踪输入，实现实时遥操作。

## 演示视频

https://github.com/user-attachments/assets/232eec4a-4b04-43cb-bc4b-a64d8fe1d18b

或直接访问: https://vimeo.com/1136862746

## 快速开始

```bash
# 克隆仓库（包含子模块）
git clone --recurse-submodules <repository-url>
cd wuji_retargeting

# 如果已经克隆但未包含子模块，运行以下命令初始化子模块：
# git submodule update --init --recursive

# 安装依赖
pip install -r requirements.txt

# 以可编辑模式安装包（用于开发）
pip install -e .

# 运行仿真（无需硬件）
cd example
python teleop_sim.py
```

### 子模块说明

本项目包含以下子模块：
- `example/utils/mujoco-sim`: MuJoCo 仿真模型

如果克隆时未使用 `--recurse-submodules`，请运行：
```bash
git submodule update --init --recursive
```

## 项目结构

```
wuji_retargeting/
├── example/
│   ├── teleop_real.py      # 真实硬件控制
│   ├── teleop_sim.py        # MuJoCo 仿真
│   ├── input_devices/       # Vision Pro 输入层
│   │   ├── base.py          # 输入设备基础接口
│   │   ├── visionpro.py     # Vision Pro 实时输入
│   │   └── visionpro_replay.py  # Vision Pro 回放输入
│   ├── data/                # 录制数据目录
│   └── utils/               # 工具函数
│       ├── avp_utils.py      # Apple Vision Pro 工具函数
│       └── mujoco-sim/       # MuJoCo 仿真模型
├── wuji_retargeting/        # 核心重定向包
│   ├── __init__.py          # 包导出
│   ├── retarget.py          # 高级重定向接口
│   ├── opt.py               # DexPilot 优化器
│   ├── robot.py             # 机器人运动学封装
│   ├── mediapipe.py         # MediaPipe 格式转换
│   ├── urdf/                # URDF 模型（left.urdf, right.urdf）
│   └── meshes/              # URDF 模型使用的 3D 网格文件
├── requirements.txt         # Python 依赖
├── pyproject.toml          # 包配置
└── README.md               # 本文件
```

## 使用方法

### 仿真模式（无需硬件）

```bash
cd example
python teleop_sim.py
```

编辑 `teleop_sim.py` 配置：
- `hand_side`: "right" | "left"
- `input_device_type`: "visionpro_real" | "visionpro_replay"
- `visionpro_record_path`: 录制文件路径

### 真实硬件

```bash
cd example
python teleop_real.py
```

首次在 Linux 上运行前，需要授予 WujiHand 控制器的 USB 访问权限：

```bash
# 临时（重启后失效）
sudo chmod a+rw /dev/ttyUSB0

# 可选：持久访问
sudo usermod -a -G dialout $USER
# 然后需要注销/重新登录

# 可选：udev 规则（替代手动 chmod）
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="0483", MODE="0666"' |
sudo tee /etc/udev/rules.d/95-wujihand.rules &&
sudo udevadm control --reload-rules &&
sudo udevadm trigger
```

编辑 `teleop_real.py` 配置：
- `hand_side`: "right" | "left"
- `input_device_type`: "visionpro_real" | "visionpro_replay"
- 主机只接入一个真实手时不需要序列号（示例只支持单手）

## 输入设备

所有输入设备共享一套轻量 API，`teleop_*` 脚本可以在不同数据源之间切换而无需调整代码。

### 接口

- `get_fingers_data() -> dict` 返回 `{"left_fingers": np.ndarray, "right_fingers": np.ndarray}`
- `cleanup()` 释放资源并完成任何录制（如果有）

要接入新设备，请继承 `InputDeviceBase`，实现以上方法，并在 `example/input_devices/__init__.py` 中导出类。

### VisionPro（实时跟踪）

```python
from example.input_devices import VisionPro

device = VisionPro(
    ip="192.168.50.127"    # Vision Pro 设备 IP
)

data = device.get_fingers_data()
device.cleanup()
```

- Vision Pro 与主机需在同一局域网
- 启动连接前，请在 Vision Pro 上运行 `tracking_streamer`
- 详细部署流程见 [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop)

### VisionProReplay（录制回放）

```python
from example.input_devices import VisionProReplay

device = VisionProReplay(
    record_path="record_example.pkl",  # 默认在 example/data/ 中查找
    playback_speed=0.7  # 可选：回放速度倍数
)

data = device.get_fingers_data()
device.cleanup()
```

- 默认录制文件：`example/data/record_example.pkl`
- 播放到末尾时自动循环
- 适用于仿真演示或离线调试

## 依赖

- `wujihandpy`: WujiHand 控制库
- `avp_stream`: Vision Pro 数据流
- `mujoco`: 物理仿真（仿真模式需要）
- 完整列表见 `requirements.txt`

## 注意事项

- Vision Pro 需在同一网络并运行 `tracking_streamer`
- MuJoCo 模型位于 `example/utils/mujoco-sim/model/`
- Vision Pro 设置请参考 [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop)

## 致谢

本项目基于以下研究和开源项目构建：

- **DexPilot**: 重定向算法基于 T. Chen 等人在论文 ["DexPilot: Vision Based Teleoperation of Dexterous Robotic Hand-Arm System"](https://arxiv.org/abs/1910.03135) 中提出的 DexPilot 方法。优化器实现将原始的四指手算法适配为支持五指手。

- **DexRetargeting**: 本项目遵循 DexRetargeting 框架进行灵巧手姿态重定向，提供灵活的关节顺序处理和模块化的优化器设计。

- **VisionProTeleop**: Vision Pro 手部跟踪集成基于 Improbable AI 的 [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop) 项目。

## 许可证

本项目采用 MIT 许可证 - 详见 [LICENSE](LICENSE) 文件。
