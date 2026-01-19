[中文](README.zh.md) | English

# wuji-retargeting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Hand pose retargeting system for Wuji Hand. High-precision retargeting based on adaptive analytical optimization, supporting Apple Vision Pro hand tracking for real-time teleoperation.

https://github.com/user-attachments/assets/72116289-7a33-4a6b-83ca-fb4d9aaece0d

## Table of Contents

- [Repository Structure](#repository-structure)
- [Usage](#usage)
  - [Prerequisites](#prerequisites)
    - [Apple Vision Pro Setup](#apple-vision-pro-setup)
  - [Installation](#installation)
  - [Running](#running)
    - [Quick Start](#quick-start)
    - [Command Reference](#command-reference)
    - [API](#api)
    - [Optimizer](#optimizer)
    - [Configuration](#configuration)
- [Citation](#citation)
- [Acknowledgement](#acknowledgement)
- [Contact](#contact)

## Repository Structure

```text
├── wuji_retargeting/
│   ├── opt/
│   │   └── ...
│   └── wuji_hand_description/
│       └── ...
├── example/
│   ├── input_devices/
│   │   └── ...
│   ├── config/
│   │   └── ...
│   ├── data/
│   │   └── ...
│   └── utils/
│       └── ...
├── requirements.txt
└── README.md
```

### Directory Description

| Directory | Description |
|-----------|-------------|
| `wuji_retargeting/` | Core package containing retargeter interface, optimizer modules, kinematics, and coordinate transforms |
| `wuji_retargeting/opt/` | Optimizer implementations including adaptive analytical optimizer |
| `wuji_retargeting/wuji_hand_description/` | URDF and mesh submodule for Wuji Hand |
| `example/` | Demonstration scripts for simulation and hardware control |
| `example/input_devices/` | Input device modules (Vision Pro, MediaPipe replay) |
| `example/config/` | YAML configuration files |
| `example/data/` | Sample recording data |

## Usage

### Prerequisites

- Python >= 3.10

#### Apple Vision Pro Setup

To use Vision Pro for real-time hand tracking, install the streaming app from [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop):

##### 1. Install VisionOS App

Install **Tracking Streamer** from the [App Store](https://apps.apple.com/us/app/tracking-streamer/id6478969032) on your Apple Vision Pro.

##### 2. Install Python Library

```bash
pip install --upgrade avp_stream
```

> **Note**: Keep the library updated. The latest App Store version requires `avp_stream >= 2.50.0`. The VisionOS app will display a warning if your Python library is outdated.

##### 3. Optional: iOS Companion App

Install **Tracking Manager** from the App Store on your iPhone/iPad for managing recordings, settings, and camera calibration.

No additional network configuration is required. Everything works out of the box after installation.

### Installation

```bash
git clone --recurse-submodules <repository-url>
cd wuji_retargeting
git lfs pull
pip install -r requirements.txt
pip install -e .
```

#### Troubleshooting

**pinocchio Installation**: If you encounter issues installing `pinocchio` from PyPI mirrors, use the official source:

```bash
pip install pin==3.8.0 -i https://pypi.org/simple
```

**macOS MuJoCo Viewer**: Use `mjpython` instead of `python` for simulation scripts:

```bash
mjpython teleop_sim.py --play data/avp1.pkl --hand left
```

### Running

#### Quick Start

##### Simulation

```bash
cd example

# Replay MediaPipe recording (default)
python teleop_sim.py --play data/avp1.pkl --hand left

# Real-time teleoperation with Vision Pro
python teleop_sim.py --input visionpro --ip <your-vision-pro-ip> --hand left
```

##### Real Hardware

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

#### Command Reference

##### teleop_sim.py

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

##### teleop_real.py

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

#### API

```python
from wuji_retargeting import Retargeter

retargeter = Retargeter.from_yaml("config/adaptive_analytical_avp.yaml", hand_side="right")
qpos = retargeter.retarget(raw_keypoints)  # (21, 3) -> (20,)
```

#### Optimizer

The system uses `AdaptiveOptimizerAnalytical` - Huber loss with hand-written analytical gradients + NLopt SLSQP.

##### Optimization Formulation

```
min_q  L(q) + λ||q - q_prev||²
s.t.   q_min ≤ q ≤ q_max
```

Where `λ` is `norm_delta` (velocity regularization).

##### Adaptive Blending

```
L = Σ_i [α_i * L_tip_dir_vec_i + (1-α_i) * L_full_hand_i]

α_i = 1           if d_i < d1  (pinching → use TipDirVec)
α_i = 0           if d_i > d2  (open → use FullHandVec)
α_i = interp      otherwise
```

- `d_i`: thumb-to-finger_i tip distance
- `d1`, `d2`: pinch thresholds (default: 2.0cm, 4.0cm)

#### Configuration

**Note**: The default configuration is tuned for Apple Vision Pro. When using other input devices, you may need to adjust `scaling` and `segment_scaling` parameters based on your hand shape.

##### Config File Structure

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

##### Parameter Reference

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

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{wuji2025retargeting,
  title={WujiHand Retargeting},
  author={Guanqi He and Wentao Zhang},
  year={2025},
  url={https://github.com/wuji-technology/wuji_retargeting},
  note={* Equal contribution}
}
```

## Acknowledgement

This project is built upon several excellent open-source projects:

- [MuJoCo](https://mujoco.org/) for physics simulation
- [dex-retargeting](https://github.com/dexsuite/dex-retargeting) for hand retargeting algorithms
- [DexPilot](https://arxiv.org/abs/1910.03135) for vision-based teleoperation insights
- [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop) for Apple Vision Pro streaming

## Contact

For any questions, please contact support@wuji.tech.
