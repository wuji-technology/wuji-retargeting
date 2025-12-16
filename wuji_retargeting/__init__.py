"""Wuji Hand Retargeting Module.

Provides hand pose retargeting from MediaPipe format to Wuji Hand joint angles.

Main classes:
- Retargeter: High-level unified interface (recommended)
- BaseOptimizer: Low-level optimizer access

Example:
    from wuji_retargeting import Retargeter

    retargeter = Retargeter.from_yaml("config/adaptive_manus.yaml", hand_side="right")
    qpos = retargeter.retarget(raw_keypoints)  # (21, 3) -> (20,)
"""

from .retarget import Retargeter
from .opt import BaseOptimizer, LPFilter
from .mediapipe import apply_mediapipe_transformations

__all__ = [
    "Retargeter",
    "BaseOptimizer",
    "LPFilter",
    "apply_mediapipe_transformations",
]
