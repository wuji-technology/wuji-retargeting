# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

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
                                                                                                                
[Unreleased]: https://github.com/wuji-technology/wuji_retargeting/compare/v0.1.1...HEAD
[0.1.1]: https://github.com/wuji-technology/wuji_retargeting/compare/v0.1.0...v0.1.1
[0.1.0]: https://github.com/wuji-technology/wuji_retargeting/releases/tag/v0.1.0
