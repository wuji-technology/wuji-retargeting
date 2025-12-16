中文 | [English](README.md)

# WujiHand Retargeting

基于 DexPilot 算法的 WujiHand 手部姿态重定向系统。

## 演示

https://vimeo.com/1136862746

## 环境要求

- Python >= 3.10

## 安装

```bash
git clone --recurse-submodules <repository-url>
cd wuji_retargeting
pip install -r requirements.txt
pip install -e .
```

**注意**: 需要通过 conda 安装 `pinocchio`（不要用 pip）:
```bash
conda install -c conda-forge pinocchio
```

### 故障排除: pinocchio 安装问题

如果从 PyPI 镜像源安装 `pinocchio` 遇到问题，请固定以下版本并使用官方源：

```bash
pip install pinocchio==3.8.0 cmeel==0.89.0 -i https://pypi.org/simple
```

## 快速开始

### 仿真

```bash
cd example

# Vision Pro 实时遥操作
mjpython teleop_sim.py --input visionpro_real --hand left --ip <your-vision-pro-ip>

# 回放 Vision Pro 录制
mjpython teleop_sim.py --input input_data_replay --play data/avp1.pkl --hand right --config config/adaptive_analytical_avp.yaml

# 回放输入数据
mjpython teleop_sim.py --input input_data_replay --play data/manus1.pkl --hand left
```

### 真机控制

```bash
cd example

# Vision Pro 实时输入（推荐）
python teleop_real.py --input visionpro_real --ip <your-vision-pro-ip> --hand right

# 简单运行（默认回放 data/avp1.pkl，右手）
python teleop_real.py

# 回放输入数据文件
python teleop_real.py --play data/avp1.pkl --hand right

# Manus 手套实时输入
python teleop_real.py --input manus_glove --glove-id 0 --hand left --config config/adaptive_analytical_manus.yaml

# 录制输入数据
python teleop_real.py --input visionpro_real --record
```

Linux USB 权限设置:
```bash
sudo chmod a+rw /dev/ttyUSB0
```

## 优化器

| 优化器 | 损失函数 | 梯度 | 求解器 | 速度 |
|--------|---------|------|--------|------|
| `AdaptiveOptimizerAnalytical` | Huber | 解析梯度 | NLopt SLSQP | ~650 Hz (manus), ~430 Hz (avp) |
| `AdaptiveOptimizerQP` | L2 | QP | quadprog | ~700 Hz (manus), ~460 Hz (avp) |

**推荐**: `AdaptiveOptimizerAnalytical` - Huber 损失对异常值更鲁棒。

## 命令参考

### teleop_sim.py

| 选项 | 默认值 | 说明 |
|------|--------|------|
| `--config` | `config/adaptive_analytical_manus.yaml` | YAML 配置文件 |
| `--hand` | `left` | 手的方向 (`left`/`right`) |
| `--input` | - | 输入类型 (`visionpro_real`/`visionpro_replay`/`input_data_replay`) |
| `--replay FILE` | - | 回放 Vision Pro 录制（配合 `visionpro_replay` 使用） |
| `--play FILE` | - | 播放输入数据（配合 `input_data_replay` 使用） |
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
| `--input` | - | 输入类型 (`visionpro_real`/`manus_glove`/`input_data_replay`) |
| `--play FILE` | - | 播放输入数据文件（MediaPipe 格式） |
| `--ip` | `192.168.50.127` | Vision Pro IP 地址 |
| `--glove-id` | `0` | Manus 手套 ID |
| `--speed` | `1.0` | 播放速度 |
| `--no-loop` | - | 禁用循环播放 |
| `--record` | - | 录制输入数据 |
| `--output FILE` | - | 录制输出路径 |

## API

```python
from wuji_retargeting import Retargeter

retargeter = Retargeter.from_yaml("config/adaptive_analytical_manus.yaml", hand_side="right")
qpos = retargeter.retarget(raw_keypoints)  # (21, 3) -> (20,)
```

## 优化公式

两种优化器求解相同的问题，但使用不同的损失函数:

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

### AdaptiveOptimizerAnalytical

使用 **Huber 损失** + 手写解析梯度 + NLopt SLSQP:

```
L_tip_dir_vec = w_pos * Huber(||v_tip - v_tip_ref||) + w_dir * Huber(||d_tip - d_tip_ref||)
L_full_hand = w_full * Huber(||v_full - v_full_ref||)
```

### AdaptiveOptimizerQP

使用 **L2 损失** + Gauss-Newton QP 求解器:

```
L_tip_dir_vec = w_pos * ||v_tip - v_tip_ref||² + w_dir * ||d_tip - d_tip_ref||²
L_full_hand = w_full * ||v_full - v_full_ref||²
```

每次迭代线性化并求解 QP:
```
min  0.5 * ||J*Δq + r||² + 0.5 * λ||q + Δq - q_prev||²
s.t. q_min ≤ q + Δq ≤ q_max
```

## 配置

### 配置文件结构

```yaml
optimizer:
  type: "AdaptiveOptimizerAnalytical"  # 或 "AdaptiveOptimizerQP"
  hand_side: "left"  # 或 "right"

retarget:
  # Huber 损失阈值（仅 Analytical）
  huber_delta: 2.0             # 位置 Huber 阈值 (cm)
  huber_delta_dir: 0.5         # 方向 Huber 阈值

  # 损失权重
  w_pos: 1.0           # 指尖位置权重
  w_dir: 10.0          # 指尖方向权重
  w_full_hand: 1.0     # 全手权重

  # 正则化
  norm_delta: 0.04     # 速度正则化权重

  # QP 求解器设置（仅 QP）
  qp_max_iters: 10     # 最大 Gauss-Newton 迭代次数
  qp_tol: 1e-4         # 收敛容差

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
| `qp_max_iters` | `10` | 最大 Gauss-Newton 迭代次数（仅 QP） |
| `qp_tol` | `1e-4` | 收敛容差（仅 QP） |
| `scaling` | `1.0` | 全局 MediaPipe 缩放 |
| `segment_scaling` | - | 每个手指缩放 `{thumb: [a,b,c], ...}` |
| `pinch_thresholds` | - | `{finger: {d1: val, d2: val}, ...}` (cm) |
| `mediapipe_rotation` | `{x:0, y:0, z:0}` | MediaPipe 旋转 (度) |
| `lp_alpha` | `0.4` | 低通滤波器系数 |

### 配置文件

| 配置文件 | 优化器 | 数据源 |
|---------|--------|--------|
| `adaptive_analytical_manus.yaml` | Analytical (Huber) | Manus 手套 |
| `adaptive_analytical_avp.yaml` | Analytical (Huber) | Vision Pro |
| `adaptive_qp_manus.yaml` | QP (L2) | Manus 手套 |
| `adaptive_qp_avp.yaml` | QP (L2) | Vision Pro |

## 项目结构

```
wuji_retargeting/
├── wuji_retargeting/       # 核心包
│   ├── retarget.py         # Retargeter 接口
│   ├── opt/                # 优化器包
│   │   ├── base.py         # BaseOptimizer, TimingStats, LPFilter
│   │   ├── adaptive_analytical.py  # AdaptiveOptimizerAnalytical
│   │   └── adaptive_qp.py  # AdaptiveOptimizerQP
│   ├── robot.py            # Pinocchio 运动学
│   └── mediapipe.py        # 坐标变换
├── example/
│   ├── teleop_sim.py       # MuJoCo 仿真
│   ├── teleop_real.py      # 硬件控制
│   ├── input_devices/      # 输入设备驱动
│   │   ├── base.py         # InputDeviceBase 基类
│   │   ├── visionpro.py    # VisionPro 实时输入
│   │   ├── visionpro_replay.py  # VisionPro 录制回放
│   │   ├── input_data_replay.py # 输入数据回放
│   │   └── manus_glove.py  # Manus 手套输入
│   ├── config/             # YAML 配置文件
│   └── data/               # 录制数据
└── requirements.txt
```

## 许可证

MIT
