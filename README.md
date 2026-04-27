# wuji-retargeting

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE) [![Release](https://img.shields.io/github/v/release/wuji-technology/wuji-retargeting)](https://github.com/wuji-technology/wuji-retargeting/releases)

Hand pose retargeting system for Wuji Hand. High-precision retargeting based on adaptive analytical and key-vector optimization, supporting Apple Vision Pro, video files, Intel RealSense, and ZED cameras for real-time teleoperation.

https://github.com/user-attachments/assets/72116289-7a33-4a6b-83ca-fb4d9aaece0d

**Get started with [Quick Start](#quick-start). For detailed documentation, please refer to [Retargeting Tutorial](https://docs.wuji.tech/docs/en/wuji-hand/latest/retargeting-user-guide/introduction) on Wuji Docs Center.**

## Repository Structure

```text
├── wuji_retargeting/
│   ├── opt/
│   │   └── ...
│   ├── viz/
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
| `wuji_retargeting/viz/` | Visualization tools for parameter tuning (TuningViewer, SkeletonDrawer) |
| `wuji_retargeting/opt/` | Optimizer implementations: adaptive analytical optimizer and configurable key-vector optimizer |
| `wuji_retargeting/wuji_hand_description/` | URDF and mesh submodule for Wuji Hand |
| `example/` | Demonstration scripts for simulation and hardware control |
| `example/input_devices/` | Input device modules (Vision Pro, MediaPipe replay, video, RealSense, ZED) |
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

# Simulation - replay recording (default: adaptive analytical optimizer)
python teleop_sim.py --play data/avp1.pkl --hand left

# Simulation - key-vector optimizer
python teleop_sim.py --play data/avp1.pkl --hand right --config config/vector/vector_avp.yaml

# Real hardware
python teleop_real.py --play data/avp1.pkl --hand right
```

### Video / RealSense / ZED Input

In addition to Apple Vision Pro, you can use MP4 video files, Intel RealSense cameras, or STEREOLABS ZED cameras as input sources with MediaPipe hand tracking.

```bash
# MP4 video input
pip install -e ".[video]"
mjpython teleop_sim.py --video path/to/hand_video.mp4 --hand right

# Intel RealSense camera
pip install -e ".[realsense]"
mjpython teleop_sim.py --realsense --hand right

# ZED camera
pip install -e ".[zed]"
mjpython teleop_sim.py --zed --hand right
```

Use `--show-video` to display the camera/video feed with MediaPipe landmarks overlay for input verification.

### Parameter Tuning Visualization Tool

An interactive tuning tool that displays **three skeleton layers** simultaneously to help you understand and adjust retargeting parameters:

| Color | Layer | Description |
|-------|-------|-------------|
| Orange | Input | Raw MediaPipe keypoints from hand tracking |
| Cyan | Scaled Target | Keypoints after `segment_scaling` applied |
| White | Robot FK | Actual robot joint positions from forward kinematics |

**Quick start:**

```bash
cd example

# Launch tuning viewer with recording data
mjpython tuning_tool.py --play data/avp1.pkl --hand left

# Or add --tuning to existing teleop commands
mjpython teleop_sim.py --play data/avp1.pkl --hand left --tuning
```

**Workflow:**
1. Start the viewer with a recording, video, or live input
2. Open the retarget config YAML in your editor (e.g., `config/adaptive_analytical_avp.yaml`)
3. Modify parameters (e.g., `segment_scaling`, `lp_alpha`, `norm_delta`)
4. Save the file — the viewer **auto-reloads** and shows parameter changes in real-time
5. Compare the three skeleton layers to evaluate your tuning

The viewer highlights affected fingers in red when parameters change, and prints tuning guidance in the terminal.

## Manus Glove Input

This retargeting library is designed as a **pure Python package** and does not introduce ROS2 or other middleware dependencies. Since Manus gloves require the Manus SDK (C++) and ROS2 for data acquisition, Manus input is not directly supported in this repository.

To use Manus gloves as the input device for retargeting, please use the [wuji-hand-teleop](https://github.com/wuji-technology/wuji-hand-teleop) repository, which integrates the full Manus ROS2 driver and retargeting pipeline:

```bash
ros2 launch wuji_teleop_bringup wuji_teleop_hand.launch.py hand_input:=manus
```

For more details, refer to the [wuji-hand-teleop documentation](https://github.com/wuji-technology/wuji-hand-teleop).

## Citation

If you find this project useful, please consider citing:

```bibtex
@software{wuji2026retargeting,
  title={WujiHand Retargeting},
  author={Guanqi He and Wentao Zhang},
  year={2026},
  url={https://github.com/wuji-technology/wuji-retargeting},
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
