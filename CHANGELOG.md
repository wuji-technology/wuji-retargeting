# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2026.05.23]

### Changed

- Tuned the Wuji Glove left/right example configs for better fit: refined per-finger segment scaling, adjusted MediaPipe rotation calibration, and disabled the wrist/thumb offsets in favor of the SDK's built-in per-device calibration

## [2026.05.18]

### Added

- Added Custom Input Device Integration Guide describing how to integrate a custom hand input device with a small wrapper class
- Added `Custom Input Devices` section to README linking to the new integration guide
- Added Wuji Glove input device (`wuji_glove_device.py`) with SDK-driven 21 MediaPipe keypoint output
- Added Wuji Glove example configs for left and right hands
- Added offset calibration tool (`calibrate_offset.py`) for neutral-pose wrist/thumb offsets
- Added `--input wuji_glove` support in `example/teleop_sim.py` and `example/teleop_real.py`
- Added Wuji Glove live mode to `example/tuning_tool.py`
- Added optimizer extensions for thumb PIP skipping, anti-hyperextension, DIP/PIP coupling, optional URDF override, and retarget keypoint offsets
- Added `wuji-sdk>=0.10.0` dependency and optional `tuning` extra

### Changed

- Updated docs to recommend Wuji Glove as the live input path while keeping other input sources available

## [2026.04.27] - 2026-04-27

### Added

- Added a key-vector retargeting optimizer with example configs for Apple Vision Pro and video
- Added ZED camera support as a real-time hand input
- Added interactive parameter tuning visualizer with three-layer skeleton comparison, HUD, and fingertip highlighting; supports hot-reload and playback

## [0.2.0] - 2026-04-07

### Added

- Added MP4 video input device with MediaPipe hand tracking
- Added Intel RealSense camera input device with MediaPipe hand tracking
- Added an abstract base class for input devices
- Added a video/RealSense configuration preset
- Added `--video`, `--realsense`, `--show-video` CLI flags to teleop scripts
- Added optional `realsense` and `video` install extras

## [0.1.1] - 2026-02-02

### Changed

- Upgraded to v0.2.1 hand description

### Fixed

- Removed invalid MDX syntax in appendix documentation

## [0.1.0] - 2026-01-21

Initial public release.

### Added

- Added core retargeting system with a unified retargeter interface
- Added adaptive analytical optimizer with Huber loss and hand-written analytical gradients
- Added low-pass filtering for smooth motion output
- Added Apple Vision Pro hand tracking support
- Added MediaPipe recording playback support
- Added input data recording functionality
- Added MuJoCo simulation example
- Added real hardware control example
- Added YAML-based configuration system with per-finger scaling and pinch thresholds

[Unreleased]: https://github.com/wuji-technology/wuji-retargeting/compare/v2026.05.23...HEAD
[2026.05.23]: https://github.com/wuji-technology/wuji-retargeting/compare/v2026.05.18...v2026.05.23
[2026.05.18]: https://github.com/wuji-technology/wuji-retargeting/compare/v2026.04.27...v2026.05.18
[2026.04.27]: https://github.com/wuji-technology/wuji-retargeting/compare/v0.2.0...v2026.04.27
[0.2.0]: https://github.com/wuji-technology/wuji-retargeting/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/wuji-technology/wuji-retargeting/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/wuji-technology/wuji-retargeting/releases/tag/v0.1.0
