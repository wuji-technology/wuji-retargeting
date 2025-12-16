"""Optimizers for hand retargeting.

AdaptiveOptimizerQP - Recommended optimizer using QP solver with Gauss-Newton method.
Uses L2 loss with adaptive blending between TipDirVec and FullHandVec based on pinch distance.

All parameters are read from YAML configuration files.
"""

from .base import (
    BaseOptimizer,
    LPFilter,
    TimingStats,
    M_TO_CM,
    CM_TO_M,
)
from .adaptive_qp import AdaptiveOptimizerQP
from .adaptive_analytical import AdaptiveOptimizerAnalytical
from .tip_dir_vec import TipDirVecOptimizer
from .tip_dir_vec_qp import TipDirVecOptimizerQP
from .tip_dir_vec_analytical import TipDirVecOptimizerAnalytical


__all__ = [
    "BaseOptimizer",
    "AdaptiveOptimizerQP",
    "AdaptiveOptimizerAnalytical",
    "TipDirVecOptimizer",
    "TipDirVecOptimizerQP",
    "TipDirVecOptimizerAnalytical",
    "LPFilter",
    "TimingStats",
    "M_TO_CM",
    "CM_TO_M",
]
