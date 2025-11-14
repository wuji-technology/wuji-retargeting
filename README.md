# WujiHand Retargeting

[中文](README.zh.md) | English

High-precision hand pose retargeting system for WujiHand using DexRetargeting algorithm. Supports Vision Pro hand tracking input for real-time teleoperation.

## Demo

https://github.com/user-attachments/assets/232eec4a-4b04-43cb-bc4b-a64d8fe1d18b

Or visit directly: https://vimeo.com/1136862746

## Quick Start

```bash
# Clone with submodules
git clone --recurse-submodules <repository-url>
cd wuji_retargeting

# If you already cloned without submodules, initialize them with:
# git submodule update --init --recursive

# Install dependencies
pip install -r requirements.txt

# Install package in editable mode (for development)
pip install -e .

# Run simulation (no hardware needed)
cd example
python teleop_sim.py
```

### Submodules

This project includes the following submodules:
- `example/utils/mujoco-sim`: MuJoCo simulation models

If you cloned without `--recurse-submodules`, run:
```bash
git submodule update --init --recursive
```

## Project Structure

```
wuji_retargeting/
├── example/
│   ├── teleop_real.py      # Real hardware control
│   ├── teleop_sim.py        # MuJoCo simulation
│   ├── input_devices/       # Vision Pro input layer
│   │   ├── base.py          # Base input device interface
│   │   ├── visionpro.py     # Vision Pro real-time input
│   │   └── visionpro_replay.py  # Vision Pro replay input
│   ├── data/                # Recorded data directory
│   └── utils/               # Utilities
│       ├── avp_utils.py      # Apple Vision Pro utilities
│       └── mujoco-sim/       # MuJoCo simulation models
├── wuji_retargeting/        # Core retargeting package
│   ├── __init__.py          # Package exports
│   ├── retarget.py          # High-level retargeting interface
│   ├── opt.py               # DexPilot optimizer
│   ├── robot.py             # Robot kinematics wrapper
│   ├── mediapipe.py         # MediaPipe format conversion
│   ├── urdf/                # URDF models (left.urdf, right.urdf)
│   └── meshes/              # 3D mesh files for URDF models
├── requirements.txt         # Python dependencies
├── pyproject.toml          # Package configuration
└── README.md               # This file
```

## Usage

### Simulation (No Hardware)

```bash
cd example
python teleop_sim.py
```

Edit `teleop_sim.py` to configure:
- `hand_side`: "right" | "left"
- `input_device_type`: "visionpro_real" | "visionpro_replay"
- `visionpro_record_path`: Recording file path

### Real Hardware

```bash
cd example
python teleop_real.py
```

Before first run on Linux, grant USB access to the WujiHand controller:

```bash
# Temporary (until next reboot)
sudo chmod a+rw /dev/ttyUSB0

# Optional: persistent access
sudo usermod -a -G dialout $USER
# then log out / log back in

# Optional: udev rule (replaces manual chmod)
echo 'SUBSYSTEM=="usb", ATTR{idVendor}=="0483", MODE="0666"' |
sudo tee /etc/udev/rules.d/95-wujihand.rules &&
sudo udevadm control --reload-rules &&
sudo udevadm trigger
```

Edit `teleop_real.py` to configure:
- `hand_side`: "right" | "left"
- `input_device_type`: "visionpro_real" | "visionpro_replay"
- No serial number needed when only one hand is connected (example supports single-hand only)

## Input Devices

All input devices expose a small, consistent API so that `teleop_*` scripts can switch sources without code changes.

### Interface

- `get_fingers_data() -> dict` returns `{"left_fingers": np.ndarray, "right_fingers": np.ndarray}`
- `cleanup()` releases resources and finalises any recordings

Add a new device by inheriting from `InputDeviceBase`, implementing the two methods above, and exporting the class in `example/input_devices/__init__.py`.

### VisionPro (real-time tracking)

```python
from example.input_devices import VisionPro

device = VisionPro(
    ip="192.168.50.127"   # Vision Pro device IP
)

data = device.get_fingers_data()
device.cleanup()
```

- Vision Pro must share a network with the host running the teleop script.
- Launch the `tracking_streamer` app on Vision Pro before connecting.
- See [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop) for the streaming setup.

### VisionProReplay (recorded data)

```python
from example.input_devices import VisionProReplay

device = VisionProReplay(
    record_path="record_example.pkl",  # Looked up under example/data/
    playback_speed=0.7  # Optional: playback speed multiplier
)

data = device.get_fingers_data()
device.cleanup()
```

- Default recording: `example/data/record_example.pkl`
- Automatically loops when reaching the end
- Handy for simulation runs or offline debugging without hardware

## Requirements

- `wujihandpy`: WujiHand control library
- `avp_stream`: Vision Pro data stream
- `mujoco`: Physics simulation (for simulation mode)
- See `requirements.txt` for complete list

## Notes

- Vision Pro must be on the same network and running `tracking_streamer`
- MuJoCo models are in `example/utils/mujoco-sim/model/`
- See [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop) for Vision Pro setup

## Acknowledgments

This project is built upon the following research and open-source projects:

- **DexPilot**: The retargeting algorithm is based on the DexPilot method proposed in ["DexPilot: Vision Based Teleoperation of Dexterous Robotic Hand-Arm System"](https://arxiv.org/abs/1910.03135) by T. Chen et al. The optimizer implementation adapts the original four-fingered hand algorithm to support five-fingered hands.

- **DexRetargeting**: This project follows the DexRetargeting framework for dexterous hand pose retargeting, providing flexible joint ordering and modular optimizer design.

- **VisionProTeleop**: Vision Pro hand tracking integration is based on [VisionProTeleop](https://github.com/Improbable-AI/VisionProTeleop) by Improbable AI.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
