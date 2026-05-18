"""Optimizers for hand retargeting.

AdaptiveOptimizerAnalytical - Recommended optimizer using Huber loss + analytical gradients + NLopt SLSQP.
Uses adaptive blending between TipDirVec and FullHandVec based on pinch distance.

VectorOptimizer - Generic key-vector matching optimizer. Minimizes distances between
configurable (origin_link -> task_link) robot vectors and corresponding MediaPipe
keypoint vectors. Configurable via `retarget.key_vectors` in YAML.

All parameters are read from YAML configuration files.
"""

from .base import (
    BaseOptimizer,
    LPFilter,
    TimingStats,
    M_TO_CM,
    CM_TO_M,
)
from .adaptive_analytical import AdaptiveOptimizerAnalytical
from .vector import VectorOptimizer


__all__ = [
    "BaseOptimizer",
    "AdaptiveOptimizerAnalytical",
    "VectorOptimizer",
    "LPFilter",
    "TimingStats",
    "M_TO_CM",
    "CM_TO_M",
]
