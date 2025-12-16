[中文](README.zh.md) | English

# WujiHand Retargeting

Hand pose retargeting system for WujiHand using DexPilot algorithm.

## Demo

https://vimeo.com/1136862746

## Requirements

- Python >= 3.10

## Install

```bash
git clone --recurse-submodules <repository-url>
cd wuji_retargeting
pip install -r requirements.txt
pip install -e .
```

**Note**: Install `pinocchio` via conda (not pip):
```bash
conda install -c conda-forge pinocchio
```

### Troubleshooting: pinocchio Installation

If you encounter issues installing `pinocchio` from PyPI (e.g., from mirrors or alternative sources), pin the following versions and use the official PyPI source:

```bash
pip install pinocchio==3.8.0 cmeel==0.89.0 -i https://pypi.org/simple
```

## Quick Start

### Simulation

```bash
cd example

# Real-time teleoperation with Vision Pro
mjpython teleop_sim.py --input visionpro_real --hand left --ip <your-vision-pro-ip>

# Replay Vision Pro recording
mjpython teleop_sim.py --input input_data_replay --play data/avp1.pkl --hand right --config config/adaptive_analytical_avp.yaml

# Replay input data log
mjpython teleop_sim.py --input input_data_replay --play data/manus1.pkl --hand left
```

### Real Hardware

```bash
cd example

# Live Vision Pro input (recommended)
python teleop_real.py --input visionpro_real --ip <your-vision-pro-ip> --hand right

# Simple run with default (replay data/avp1.pkl, right hand)
python teleop_real.py

# Replay input data file
python teleop_real.py --play data/avp1.pkl --hand right

# Live Manus glove input
python teleop_real.py --input manus_glove --glove-id 0 --hand left --config config/adaptive_analytical_manus.yaml

# Record input data while using Vision Pro
python teleop_real.py --input visionpro_real --record
```

Linux USB permission:
```bash
sudo chmod a+rw /dev/ttyUSB0
```

## Optimizers

| Optimizer | Loss | Gradient | Solver | Speed |
|-----------|------|----------|--------|-------|
| `AdaptiveOptimizerAnalytical` | Huber | Analytical | NLopt SLSQP | ~650 Hz (manus), ~430 Hz (avp) |
| `AdaptiveOptimizerQP` | L2 | QP | quadprog | ~700 Hz (manus), ~460 Hz (avp) |

**Recommended**: `AdaptiveOptimizerAnalytical` - Huber loss is more robust to outliers.

## Command Reference

### teleop_sim.py

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `config/adaptive_analytical_manus.yaml` | YAML config file |
| `--hand` | `left` | Hand side (`left`/`right`) |
| `--input` | - | Input type (`visionpro_real`/`visionpro_replay`/`input_data_replay`) |
| `--replay FILE` | - | Replay Vision Pro recording (use with `visionpro_replay`) |
| `--play FILE` | - | Play input data log (use with `input_data_replay`) |
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
| `--input` | - | Input type (`visionpro_real`/`manus_glove`/`input_data_replay`) |
| `--play FILE` | - | Play input data file (MediaPipe format) |
| `--ip` | `192.168.50.127` | Vision Pro IP address |
| `--glove-id` | `0` | Manus glove ID |
| `--speed` | `1.0` | Playback speed |
| `--no-loop` | - | Disable looping |
| `--record` | - | Record input data |
| `--output FILE` | - | Recording output path |

## API

```python
from wuji_retargeting import Retargeter

retargeter = Retargeter.from_yaml("config/adaptive_analytical_manus.yaml", hand_side="right")
qpos = retargeter.retarget(raw_keypoints)  # (21, 3) -> (20,)
```

## Optimization Formulation

Both optimizers solve the same problem with different loss functions:

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

### AdaptiveOptimizerAnalytical

Uses **Huber loss** with hand-written analytical gradients + NLopt SLSQP:

```
L_tip_dir_vec = w_pos * Huber(||v_tip - v_tip_ref||) + w_dir * Huber(||d_tip - d_tip_ref||)
L_full_hand = w_full * Huber(||v_full - v_full_ref||)
```

### AdaptiveOptimizerQP

Uses **L2 loss** with Gauss-Newton QP solver:

```
L_tip_dir_vec = w_pos * ||v_tip - v_tip_ref||² + w_dir * ||d_tip - d_tip_ref||²
L_full_hand = w_full * ||v_full - v_full_ref||²
```

At each iteration, linearize and solve QP:
```
min  0.5 * ||J*Δq + r||² + 0.5 * λ||q + Δq - q_prev||²
s.t. q_min ≤ q + Δq ≤ q_max
```

## Configuration

### Config File Structure

```yaml
optimizer:
  type: "AdaptiveOptimizerAnalytical"  # or "AdaptiveOptimizerQP"
  hand_side: "left"  # or "right"

retarget:
  # Huber loss thresholds (Analytical only)
  huber_delta: 2.0             # Position Huber threshold (cm)
  huber_delta_dir: 0.5         # Direction Huber threshold

  # Loss weights
  w_pos: 1.0           # Tip position weight
  w_dir: 10.0          # Tip direction weight
  w_full_hand: 1.0     # Full hand weight

  # Regularization
  norm_delta: 0.04     # Velocity regularization weight

  # QP solver settings (QP only)
  qp_max_iters: 10     # Maximum Gauss-Newton iterations
  qp_tol: 1e-4         # Convergence tolerance

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
| `qp_max_iters` | `10` | Maximum Gauss-Newton iterations (QP only) |
| `qp_tol` | `1e-4` | Convergence tolerance (QP only) |
| `scaling` | `1.0` | Global MediaPipe scaling |
| `segment_scaling` | - | Per-finger scaling `{thumb: [a,b,c], ...}` |
| `pinch_thresholds` | - | `{finger: {d1: val, d2: val}, ...}` (cm) |
| `mediapipe_rotation` | `{x:0, y:0, z:0}` | MediaPipe rotation (degrees) |
| `lp_alpha` | `0.4` | Low-pass filter coefficient |

### Config Files

| Config | Optimizer | Data Source |
|--------|-----------|-------------|
| `adaptive_analytical_manus.yaml` | Analytical (Huber) | Manus glove |
| `adaptive_analytical_avp.yaml` | Analytical (Huber) | Vision Pro |
| `adaptive_qp_manus.yaml` | QP (L2) | Manus glove |
| `adaptive_qp_avp.yaml` | QP (L2) | Vision Pro |

## Project Structure

```
wuji_retargeting/
├── wuji_retargeting/       # Core package
│   ├── retarget.py         # Retargeter interface
│   ├── opt/                # Optimizer package
│   │   ├── base.py         # BaseOptimizer, TimingStats, LPFilter
│   │   ├── adaptive_analytical.py  # AdaptiveOptimizerAnalytical
│   │   └── adaptive_qp.py  # AdaptiveOptimizerQP
│   ├── robot.py            # Pinocchio kinematics
│   └── mediapipe.py        # Coordinate transform
├── example/
│   ├── teleop_sim.py       # MuJoCo simulation
│   ├── teleop_real.py      # Hardware control
│   ├── input_devices/      # Input device drivers
│   │   ├── base.py         # InputDeviceBase
│   │   ├── visionpro.py    # VisionPro live input
│   │   ├── visionpro_replay.py  # VisionPro recording replay
│   │   ├── input_data_replay.py # Input data replay
│   │   └── manus_glove.py  # Manus glove input
│   ├── config/             # YAML configs
│   └── data/               # Recording data
└── requirements.txt
```

## License

MIT
