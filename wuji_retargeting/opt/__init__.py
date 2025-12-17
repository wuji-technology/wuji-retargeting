"""Optimizers for hand retargeting.

AdaptiveOptimizerAnalytical - Recommended optimizer using Huber loss + analytical gradients + NLopt SLSQP.
Uses adaptive blending between TipDirVec and FullHandVec based on pinch distance.

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


__all__ = [
    "BaseOptimizer",
    "AdaptiveOptimizerAnalytical",
    "LPFilter",
    "TimingStats",
    "M_TO_CM",
    "CM_TO_M",
]
