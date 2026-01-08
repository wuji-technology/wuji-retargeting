中文 | [English](README.md)

# WujiHand Retargeting

基于 Wuji Retargeting 算法的 WujiHand 手部姿态重定向系统。

## 演示

https://github.com/user-attachments/assets/4e58e677-421d-40a0-9860-cc80b4a4b17c

## 环境要求

- Python >= 3.10

## 安装

```bash
git clone --recurse-submodules <repository-url>
cd wuji_retargeting
git lfs pull
pip install -r requirements.txt
pip install -e .
```

### 故障排除

**pinocchio 安装问题**：如果从 PyPI 镜像源安装遇到问题，请使用官方源：

```bash
pip install pin==3.8.0 -i https://pypi.org/simple
```

**macOS MuJoCo 窗口**：仿真脚本需使用 `mjpython` 代替 `python`：

```bash
mjpython teleop_sim.py --play data/avp1.pkl --hand left
```

## 快速开始

### 仿真

```bash
cd example

# 回放 MediaPipe 录制（默认）
python teleop_sim.py --play data/avp1.pkl --hand left

# Vision Pro 实时遥操作
python teleop_sim.py --input visionpro --ip <your-vision-pro-ip> --hand left
```

### 真机控制

```bash
cd example

# 简单运行（默认回放 data/avp1.pkl，右手）
python teleop_real.py

# 回放 MediaPipe 录制
python teleop_real.py --play data/avp1.pkl --hand right

# Vision Pro 实时输入（推荐）
python teleop_real.py --input visionpro --ip <your-vision-pro-ip> --hand right

# 录制输入数据
python teleop_real.py --input visionpro --record
```

Linux USB 权限设置:
```bash
sudo chmod a+rw /dev/ttyUSB0
```

## 命令参考

### teleop_sim.py

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `config/adaptive_analytical_avp.yaml` | YAML 配置文件 |
| `--hand` | `left` | 手的方向 (`left`/`right`) |
| `--input` | - | 输入类型 (`visionpro`/`mediapipe_replay`) |
| `--play FILE` | - | 播放 MediaPipe 录制 |
| `--ip` | `192.168.50.127` | Vision Pro IP 地址 |
| `--speed` | `1.0` | 播放速度 |
| `--no-loop` | - | 禁用循环播放 |
| `--record` | - | 录制输入数据 |
| `--output FILE` | - | 录制输出路径 |

### teleop_real.py

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `config/adaptive_analytical_avp.yaml` | YAML 配置文件 |
| `--hand` | `right` | 手的方向 (`left`/`right`) |
| `--input` | - | 输入类型 (`visionpro`/`mediapipe_replay`) |
| `--play FILE` | - | 播放 MediaPipe 录制 |
| `--ip` | `192.168.50.127` | Vision Pro IP 地址 |
| `--speed` | `1.0` | 播放速度 |
| `--no-loop` | - | 禁用循环播放 |
| `--record` | - | 录制输入数据 |
| `--output FILE` | - | 录制输出路径 |

## API

```python
from wuji_retargeting import Retargeter

retargeter = Retargeter.from_yaml("config/adaptive_analytical_avp.yaml", hand_side="right")
qpos = retargeter.retarget(raw_keypoints)  # (21, 3) -> (20,)
```

## 优化器

系统使用 `AdaptiveOptimizerAnalytical` - Huber 损失 + 手写解析梯度 + NLopt SLSQP。

### 优化公式

```
min_q  L(q) + λ||q - q_prev||²
s.t.   q_min ≤ q ≤ q_max
```

其中 `λ` 是 `norm_delta`（速度正则化权重）。

### 自适应混合

```
L = Σ_i [α_i * L_tip_dir_vec_i + (1-α_i) * L_full_hand_i]

α_i = 1           如果 d_i < d1  (捏合 → 使用 TipDirVec)
α_i = 0           如果 d_i > d2  (张开 → 使用 FullHandVec)
α_i = 插值        其他情况
```

- `d_i`: 拇指到第 i 个手指的指尖距离
- `d1`, `d2`: 捏合阈值（默认: 2.0cm, 4.0cm）

## 配置

**注意**：默认配置针对 Apple Vision Pro 调优。使用其他输入设备时，可能需要根据手型调整 `scaling` 和 `segment_scaling` 等参数。

### 配置文件结构

```yaml
optimizer:
  type: "AdaptiveOptimizerAnalytical"
  hand_side: "left"  # 或 "right"

retarget:
  # Huber 损失阈值
  huber_delta: 2.0             # 位置 Huber 阈值 (cm)
  huber_delta_dir: 0.5         # 方向 Huber 阈值

  # 损失权重
  w_pos: 1.0           # 指尖位置权重
  w_dir: 10.0          # 指尖方向权重
  w_full_hand: 1.0     # 全手权重

  # 正则化
  norm_delta: 0.04     # 速度正则化权重

  # 缩放
  scaling: 1.0         # 全局 MediaPipe 缩放

  # 每个手指的分段缩放 [PIP, DIP, TIP]
  segment_scaling:
    thumb:  [1.0, 1.0, 1.0]
    index:  [1.0, 1.03, 1.05]
    middle: [1.0, 1.0, 1.0]
    ring:   [1.0, 1.0, 1.0]
    pinky:  [1.05, 1.15, 1.15]

  # 捏合阈值 (cm)
  pinch_thresholds:
    index:  { d1: 2.0, d2: 4.0 }
    middle: { d1: 2.0, d2: 4.0 }
    ring:   { d1: 2.0, d2: 4.0 }
    pinky:  { d1: 2.0, d2: 4.0 }

  # MediaPipe 旋转 (度)
  mediapipe_rotation:
    x: 0.0   # 横滚
    y: 0.0   # 俯仰
    z: 0.0   # 偏航

  # 低通滤波器 (0~1, 越小越平滑)
  lp_alpha: 0.4
```

### 参数参考

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `huber_delta` | `2.0` | 位置 Huber 阈值 (cm) |
| `huber_delta_dir` | `0.5` | 方向 Huber 阈值 |
| `w_pos` | `1.0` | 指尖位置损失权重 |
| `w_dir` | `10.0` | 指尖方向损失权重 |
| `w_full_hand` | `1.0` | 全手损失权重 |
| `norm_delta` | `0.04` | 速度正则化权重 |
| `scaling` | `1.0` | 全局 MediaPipe 缩放 |
| `segment_scaling` | - | 每个手指缩放 `{thumb: [a,b,c], ...}` |
| `pinch_thresholds` | - | `{finger: {d1: val, d2: val}, ...}` (cm) |
| `mediapipe_rotation` | `{x:0, y:0, z:0}` | MediaPipe 旋转 (度) |
| `lp_alpha` | `0.4` | 低通滤波器系数 |

## 项目结构

```
wuji_retargeting/
├── wuji_retargeting/       # 核心包
│   ├── retarget.py         # Retargeter 接口
│   ├── opt/                # 优化器包
│   │   ├── base.py         # BaseOptimizer, TimingStats, LPFilter
│   │   └── adaptive_analytical.py  # AdaptiveOptimizerAnalytical
│   ├── robot.py            # Pinocchio 运动学
│   ├── mediapipe.py        # 坐标变换
│   └── wuji_hand_description/  # URDF/mesh 子模块
├── example/
│   ├── teleop_sim.py       # MuJoCo 仿真
│   ├── teleop_real.py      # 硬件控制
│   ├── input_devices/      # 输入设备模块
│   │   ├── visionpro.py    # VisionPro 输入
│   │   └── mediapipe_replay.py  # MediaPipe 录制回放
│   ├── config/             # YAML 配置文件
│   ├── data/               # 录制数据
│   └── utils/mujoco-sim/   # MuJoCo 模型子模块
└── requirements.txt
```

## 引用

如果您觉得本项目有用，请考虑引用：

```bibtex
@software{wuji2025retargeting,
  title={WujiHand Retargeting},
  author={Guanqi He and Wentao Zhang},
  year={2025},
  url={https://github.com/wuji-technology/wuji_retargeting},
  note={* Equal contribution}
}
```

## 许可证

MIT
