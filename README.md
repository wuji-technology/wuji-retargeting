[中文](README.zh.md) | English

# WujiHand Retargeting

Hand pose retargeting system for WujiHand using Wuji Retargeting algorithm.

## Demo

https://github.com/user-attachments/assets/4e58e677-421d-40a0-9860-cc80b4a4b17c

## Requirements

- Python >= 3.10

## Install

```bash
git clone --recurse-submodules <repository-url>
cd wuji_retargeting
pip install -r requirements.txt
pip install -e .
```

### Troubleshooting: pinocchio Installation

If you encounter issues installing `pinocchio` from PyPI (e.g., from mirrors or alternative sources), pin the following versions and use the official PyPI source:

```bash
pip install pin==3.8.0 cmeel==0.89.0 -i https://pypi.org/simple
```

## Quick Start

### Simulation

```bash
cd example

# Replay MediaPipe recording (default)
mjpython teleop_sim.py --play data/avp1.pkl --hand left

# Real-time teleoperation with Vision Pro
mjpython teleop_sim.py --input visionpro --ip <your-vision-pro-ip> --hand left
```

### Real Hardware

```bash
cd example

# Simple run with default (replay data/avp1.pkl, right hand)
python teleop_real.py

# Replay MediaPipe recording
python teleop_real.py --play data/avp1.pkl --hand right

# Live Vision Pro input (recommended)
python teleop_real.py --input visionpro --ip <your-vision-pro-ip> --hand right

# Record input data while using Vision Pro
python teleop_real.py --input visionpro --record
```

Linux USB permission:
```bash
sudo chmod a+rw /dev/ttyUSB0
```

## Command Reference

### teleop_sim.py

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `config/adaptive_analytical_avp.yaml` | YAML config file |
| `--hand` | `left` | Hand side (`left`/`right`) |
| `--input` | - | Input type (`visionpro`/`mediapipe_replay`) |
| `--play FILE` | - | Play MediaPipe recording (shortcut for `--input mediapipe_replay`) |
| `--ip` | `192.168.50.127` | Vision Pro IP address |
| `--speed` | `1.0` | Playback speed |
| `--no-loop` | - | Disable looping |
| `--record` | - | Record input data |
| `--output FILE` | - | Recording output path |

### teleop_real.py

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `config/adaptive_analytical_avp.yaml` | YAML config file |
| `--hand` | `right` | Hand side (`left`/`right`) |
| `--input` | - | Input type (`visionpro`/`mediapipe_replay`) |
| `--play FILE` | - | Play MediaPipe recording (shortcut for `--input mediapipe_replay`) |
| `--ip` | `192.168.50.127` | Vision Pro IP address |
| `--speed` | `1.0` | Playback speed |
| `--no-loop` | - | Disable looping |
| `--record` | - | Record input data |
| `--output FILE` | - | Recording output path |

## API

```python
from wuji_retargeting import Retargeter

retargeter = Retargeter.from_yaml("config/adaptive_analytical_avp.yaml", hand_side="right")
qpos = retargeter.retarget(raw_keypoints)  # (21, 3) -> (20,)
```

## Optimizer

The system uses `AdaptiveOptimizerAnalytical` - Huber loss with hand-written analytical gradients + NLopt SLSQP.

### Optimization Formulation

```
min_q  L(q) + λ||q - q_prev||²
s.t.   q_min ≤ q ≤ q_max
```

Where `λ` is `norm_delta` (velocity regularization).

### Adaptive Blending

```
L = Σ_i [α_i * L_tip_dir_vec_i + (1-α_i) * L_full_hand_i]

α_i = 1           if d_i < d1  (pinching → use TipDirVec)
α_i = 0           if d_i > d2  (open → use FullHandVec)
α_i = interp      otherwise
```

- `d_i`: thumb-to-finger_i tip distance
- `d1`, `d2`: pinch thresholds (default: 2.0cm, 4.0cm)

## Configuration

### Config File Structure

```yaml
optimizer:
  type: "AdaptiveOptimizerAnalytical"
  hand_side: "left"  # or "right"

retarget:
  # Huber loss thresholds
  huber_delta: 2.0             # Position Huber threshold (cm)
  huber_delta_dir: 0.5         # Direction Huber threshold

  # Loss weights
  w_pos: 1.0           # Tip position weight
  w_dir: 10.0          # Tip direction weight
  w_full_hand: 1.0     # Full hand weight

  # Regularization
  norm_delta: 0.04     # Velocity regularization weight

  # Scaling
  scaling: 1.0         # Global MediaPipe scaling

  # Per-finger segment scaling [PIP, DIP, TIP]
  segment_scaling:
    thumb:  [1.0, 1.0, 1.0]
    index:  [1.0, 1.03, 1.05]
    middle: [1.0, 1.0, 1.0]
    ring:   [1.0, 1.0, 1.0]
    pinky:  [1.05, 1.15, 1.15]

  # Pinch thresholds (cm)
  pinch_thresholds:
    index:  { d1: 2.0, d2: 4.0 }
    middle: { d1: 2.0, d2: 4.0 }
    ring:   { d1: 2.0, d2: 4.0 }
    pinky:  { d1: 2.0, d2: 4.0 }

  # MediaPipe rotation (degrees)
  mediapipe_rotation:
    x: 0.0   # Roll
    y: 0.0   # Pitch
    z: 0.0   # Yaw

  # Low-pass filter (0~1, smaller=smoother)
  lp_alpha: 0.4
```

### Parameter Reference

| Parameter | Default | Description |
|-----------|---------|-------------|
| `huber_delta` | `2.0` | Position Huber threshold (cm) |
| `huber_delta_dir` | `0.5` | Direction Huber threshold |
| `w_pos` | `1.0` | Tip position cost weight |
| `w_dir` | `10.0` | Tip direction cost weight |
| `w_full_hand` | `1.0` | Full hand cost weight |
| `norm_delta` | `0.04` | Velocity regularization weight |
| `scaling` | `1.0` | Global MediaPipe scaling |
| `segment_scaling` | - | Per-finger scaling `{thumb: [a,b,c], ...}` |
| `pinch_thresholds` | - | `{finger: {d1: val, d2: val}, ...}` (cm) |
| `mediapipe_rotation` | `{x:0, y:0, z:0}` | MediaPipe rotation (degrees) |
| `lp_alpha` | `0.4` | Low-pass filter coefficient |

## Project Structure

```
wuji_retargeting/
├── wuji_retargeting/       # Core package
│   ├── retarget.py         # Retargeter interface
│   ├── opt/                # Optimizer package
│   │   ├── base.py         # BaseOptimizer, TimingStats, LPFilter
│   │   └── adaptive_analytical.py  # AdaptiveOptimizerAnalytical
│   ├── robot.py            # Pinocchio kinematics
│   ├── mediapipe.py        # Coordinate transform
│   └── wuji_hand_description/  # URDF/mesh submodule
├── example/
│   ├── teleop_sim.py       # MuJoCo simulation
│   ├── teleop_real.py      # Hardware control
│   ├── input_devices/      # Input device modules
│   │   ├── visionpro.py    # VisionPro input
│   │   └── mediapipe_replay.py  # MediaPipe recording replay
│   ├── config/             # YAML configs
│   ├── data/               # Recording data
│   └── utils/mujoco-sim/   # MuJoCo model submodule
└── requirements.txt
```

## License

MIT
