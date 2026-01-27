# wuji-retargeting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Release](https://img.shields.io/github/v/release/wuji-technology/wuji-retargeting)](https://github.com/wuji-technology/wuji-retargeting/releases)

Hand pose retargeting system for Wuji Hand. High-precision retargeting based on adaptive analytical optimization, supporting Apple Vision Pro hand tracking for real-time teleoperation.

https://github.com/user-attachments/assets/72116289-7a33-4a6b-83ca-fb4d9aaece0d

**Get started with [Quick Start](#quick-start). For detailed documentation, please refer to [Retargeting Tutorial](https://docs.wuji.tech/docs/en/wuji-hand/latest/retargeting-user-guide/introduction) on Wuji Docs Center.**

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

## Quick Start

### Installation

```bash
git clone --recurse-submodules https://github.com/wuji-technology/wuji-retargeting.git
cd wuji-retargeting
pip install -r requirements.txt
pip install -e .
```

### Running

```bash
cd example

# Simulation - replay recording
python teleop_sim.py --play data/avp1.pkl --hand left

# Real hardware
python teleop_real.py --play data/avp1.pkl --hand right
```

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

For any questions, please contact [support@wuji.tech](mailto:support@wuji.tech).
