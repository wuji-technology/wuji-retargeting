# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [2026.04.27] - 2026-04-27

### Added

- Added `VectorOptimizer`, a key-vector retargeting optimizer with example configs for AVP and video
- Added ZED camera support as a real-time hand input
- Added interactive parameter tuning visualizer with three-layer skeleton comparison, HUD, and fingertip highlighting; supports hot-reload and playback

## [0.2.0] - 2026-04-07

### Added

- Added MP4 video input device with MediaPipe hand tracking (`video_mediapipe.py`)
- Added Intel RealSense camera input device with MediaPipe hand tracking (`realsense_mediapipe.py`)
- Added `InputDeviceBase` abstract base class for input devices (`base.py`)
- Added video/RealSense config (`adaptive_analytical_video.yaml`)
- Added `--video`, `--realsense`, `--show-video` CLI flags to teleop scripts
- Added optional `[realsense]` and `[video]` extras in `pyproject.toml`

## [0.1.1] - 2026-02-02

### Changed

- Upgrade to v0.2.1 hand description (#63)

### Fixed

- Remove invalid MDX syntax in appendix.mdx (#61)

## [0.1.0] - 2026-01-21                                                                                       
                                                                                                                
Initial public release.                                                                                       
                                                                                                                
### Added                                                                                                     
                                                                                                                
- Added core retargeting system with `Retargeter` interface                                                   
- Added adaptive analytical optimizer (`AdaptiveOptimizerAnalytical`) with Huber loss and hand-written analytical gradients                                                                                          
- Added low-pass filtering for smooth motion output                                                           
- Added Apple Vision Pro hand tracking support via `avp_stream`                                               
- Added MediaPipe recording playback support                                                                  
- Added input data recording functionality                                                                    
- Added MuJoCo simulation example (`teleop_sim.py`)                                                           
- Added real hardware control example (`teleop_real.py`)                                                      
- Added YAML-based configuration system with per-finger scaling and pinch thresholds                          
                                                                                                                
[Unreleased]: https://github.com/wuji-technology/wuji-retargeting/compare/v2026.04.27...HEAD
[2026.04.27]: https://github.com/wuji-technology/wuji-retargeting/compare/v0.2.0...v2026.04.27
[0.2.0]: https://github.com/wuji-technology/wuji-retargeting/compare/v0.1.1...v0.2.0
[0.1.1]: https://github.com/wuji-technology/wuji_retargeting/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/wuji-technology/wuji_retargeting/releases/tag/v0.1.0
