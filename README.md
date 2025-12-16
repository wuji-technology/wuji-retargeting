# WujiHand Retargeting

Hand pose retargeting system for WujiHand using DexPilot algorithm.

## Demo

https://vimeo.com/1136862746

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

## Usage

### Simulation

```bash
cd example
mjpython teleop_sim.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | `config/adaptive_manus.yaml` | YAML config file |
| `--hand` | `left` | Hand side (`left`/`right`) |
| `--input` | - | Input type (`visionpro_real`/`visionpro_replay`/`input_data_replay`) |
| `--replay FILE` | - | Replay VisionPro recording |
| `--play FILE` | - | Play input data log |
| `--ip` | `192.168.50.127` | VisionPro IP address |
| `--speed` | `1.0` | Playback speed |
| `--no-loop` | - | Disable looping |
| `--record` | - | Record input data |
| `--output FILE` | - | Recording output path |

### Real Hardware

```bash
cd example
python teleop_real.py
```

Linux USB permission:
```bash
sudo chmod a+rw /dev/ttyUSB0
```

### Evaluation

```bash
cd example
python eval.py [OPTIONS]
```

| Option | Default | Description |
|--------|---------|-------------|
| `--config` | (required) | YAML config file |
| `--name` | - | Experiment name |
| `--eval-dir` | `data/eval` | Evaluation data directory |
| `--filter` | - | Filter files by prefix (`manus`/`avp`) |
| `--view` | - | Open MuJoCo viewer for each frame |
| `--play FILE` | - | Play recording continuously |
| `--enable-dynamics` | - | Enable MuJoCo dynamics simulation |
| `--output-dir` | - | Append to existing output |

## API

```python
from wuji_retargeting import Retargeter

retargeter = Retargeter.from_yaml("config/adaptive_manus.yaml", hand_side="right")
qpos = retargeter.retarget(raw_keypoints)  # (21, 3) -> (20,)
```

## Optimization Formulation

All optimizers solve the following problem:

```
min_q  L(q) + λ||q - q_prev||²
s.t.   q_min ≤ q ≤ q_max
```

Where `λ` is `norm_delta` (velocity regularization).

### TipDirVecOptimizer

Matches fingertip positions and directions:

```
L = w_pos * L_pos + w_dir * L_dir + w_full * L_full

L_pos = mean(Huber(||v_tip^robot - v_tip^ref||, δ))
L_dir = mean(Huber(||d_tip^robot - d_tip^ref||, δ_dir))
L_full = mean(||v_full^robot - v_full^ref||²)
```

- `v_tip`: wrist → fingertip vector (5 vectors)
- `d_tip`: link4 → fingertip direction (normalized)
- `v_full`: wrist → {PIP, DIP, TIP} vectors (15 vectors)

### FullHandVecOptimizer

Matches full hand segment vectors:

```
L = w_vec * mean(Huber(||v_i^robot - s_i * v_i^ref||, δ))
```

- `v_i`: wrist → {PIP, DIP, TIP} for each finger (15 vectors)
- `s_i`: per-segment scaling factor

### AdaptiveOptimizer

Blends TipDirVec and FullHandVec per-finger based on pinch distance:

```
L = Σ_i [α_i * L_tip_dir_vec_i + (1-α_i) * L_full_hand_i]

α_i = 1           if d_i < d1  (pinching)
α_i = 0           if d_i > d2  (open)
α_i = interp      otherwise
```

- `d_i`: thumb-to-finger_i tip distance
- `d1`, `d2`: pinch thresholds

## Configuration

### Optimizer Selection

```yaml
optimizer:
  type: "AdaptiveOptimizer"  # or "TipDirVecOptimizer", "FullHandVecOptimizer"
```

### Retarget Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `huber_delta` | `2.0` | Position Huber loss threshold (cm) |
| `huber_delta_dir` | `0.5` | Direction Huber loss threshold |
| `norm_delta` | `0.04` | Velocity regularization weight |
| `lp_alpha` | `0.4` | Low-pass filter coefficient (0~1, smaller=smoother) |

### TipDirVecOptimizer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w_pos` | `1.0` | Tip position cost weight |
| `w_dir` | `10.0` | Tip direction cost weight |
| `w_full_hand_vec` | `0.001` | Full hand regularization weight |
| `scaling` | `1.0` | Global MediaPipe scaling |
| `project_tip_dir` | `false` | Project tip direction onto finger plane |

### FullHandVecOptimizer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w_vec` | `1.0` | Full hand vector cost weight |
| `segment_scaling` | - | Per-finger scaling `{thumb: [a,b,c], ...}` |

### AdaptiveOptimizer

| Parameter | Default | Description |
|-----------|---------|-------------|
| `w_pos` | `1.0` | Tip position weight |
| `w_dir` | `10.0` | Tip direction weight |
| `w_full_hand` | `1.0` | Full hand weight |
| `scaling` | `1.0` | Global scaling |
| `segment_scaling` | - | Per-finger scaling |
| `pinch_thresholds` | - | `{index: {d1: 2.0, d2: 4.0}, ...}` (cm) |

### MediaPipe Rotation

```yaml
retarget:
  mediapipe_rotation:
    x: 0.0   # Roll (degrees)
    y: 0.0   # Pitch (degrees)
    z: 15.0  # Yaw (degrees)
```

### Example Config

```yaml
optimizer:
  type: "AdaptiveOptimizer"

retarget:
  huber_delta: 2.0
  huber_delta_dir: 0.5
  norm_delta: 0.04

  w_pos: 1.0
  w_dir: 10.0
  w_full_hand: 1.0
  scaling: 1.0

  segment_scaling:
    thumb:  [1.0, 1.0, 1.0]
    index:  [1.0, 1.03, 1.05]
    middle: [1.0, 1.0, 1.0]
    ring:   [1.0, 1.0, 1.0]
    pinky:  [1.05, 1.15, 1.15]

  pinch_thresholds:
    index:  { d1: 2.0, d2: 4.0 }
    middle: { d1: 2.0, d2: 4.0 }
    ring:   { d1: 2.0, d2: 4.0 }
    pinky:  { d1: 2.0, d2: 4.0 }

  mediapipe_rotation:
    x: 0.0
    y: 0.0
    z: 0.0
```

## Project Structure

```
wuji_retargeting/
├── wuji_retargeting/       # Core package
│   ├── retarget.py         # Retargeter interface
│   ├── opt.py              # Optimizers (TipDirVec/FullHandVec/Adaptive)
│   ├── robot.py            # Pinocchio kinematics
│   └── mediapipe.py        # Coordinate transform
├── example/
│   ├── teleop_sim.py       # MuJoCo simulation
│   ├── teleop_real.py      # Hardware control
│   ├── eval.py             # Evaluation script
│   └── config/             # YAML configs
└── requirements.txt
```

## License

MIT
