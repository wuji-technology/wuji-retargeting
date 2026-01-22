# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] - 2026-01-21

Initial public release.

### Added

- Core retargeting system with `Retargeter` interface
- Adaptive analytical optimizer (`AdaptiveOptimizerAnalytical`) with Huber loss and hand-written analytical gradients
- Apple Vision Pro hand tracking support via `avp_stream`
- MediaPipe recording playback support
- MuJoCo simulation examples (`teleop_sim.py`)
- Real hardware control examples (`teleop_real.py`)
- YAML-based configuration system with per-finger scaling and pinch thresholds
- Low-pass filtering for smooth motion output
- Input data recording functionality

[Unreleased]: https://github.com/wuji-technology/wuji_retargeting/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/wuji-technology/wuji_retargeting/releases/tag/v0.1.0
