"""Unified retargeting interface for Wuji Hand.

Provides a high-level interface that handles:
- MediaPipe coordinate transformation
- Optional rotation adjustment
- Optimizer selection (TipDirVec/FullHandVec/Adaptive)
- Low-pass filtering

Usage:
    retargeter = Retargeter.from_yaml("config/adaptive_manus.yaml", hand_side="right")
    qpos = retargeter.retarget(raw_keypoints)  # (21, 3) -> (20,)
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import yaml

from .opt import BaseOptimizer, LPFilter
from .mediapipe import apply_mediapipe_transformations


class Retargeter:
    """Unified retargeting interface for Wuji Hand.

    Encapsulates the complete retargeting pipeline:
    1. MediaPipe coordinate transformation (raw -> wrist frame)
    2. Optional rotation adjustment (configured via mediapipe_rotation)
    3. IK optimization (TipDirVec/FullHandVec/Adaptive)
    4. Low-pass filtering for smooth output

    Attributes:
        optimizer: The underlying optimizer instance
        lp_filter: Low-pass filter for smoothing
        hand_side: 'left' or 'right'
    """

    def __init__(self, config: dict, hand_side: str = "right"):
        """Initialize retargeter.

        Args:
            config: Configuration dict (from YAML)
            hand_side: 'left' or 'right'
        """
        self.config = config
        self.hand_side = hand_side.lower()

        if self.hand_side not in ['left', 'right']:
            raise ValueError(f"hand_side must be 'left' or 'right', got {hand_side}")

        # Ensure hand_side in config
        if 'optimizer' not in config:
            config['optimizer'] = {}
        config['optimizer']['hand_side'] = self.hand_side

        # Create optimizer
        self.optimizer = BaseOptimizer.from_config(config)

        # Create low-pass filter
        retarget_config = config.get('retarget', {})
        lp_alpha = retarget_config.get('lp_alpha', 0.2)
        self.lp_filter = LPFilter(lp_alpha)

        # Rotation adjustment
        self.rotation_xyz = retarget_config.get('mediapipe_rotation', {})

    @classmethod
    def from_yaml(cls, yaml_path: str, hand_side: str = "right") -> "Retargeter":
        """Create retargeter from YAML configuration file.

        Args:
            yaml_path: Path to YAML configuration file
            hand_side: 'left' or 'right'

        Returns:
            Retargeter instance
        """
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        return cls(config, hand_side)

    @classmethod
    def from_config(cls, config: dict, hand_side: str = "right") -> "Retargeter":
        """Create retargeter from configuration dict.

        Args:
            config: Configuration dict
            hand_side: 'left' or 'right'

        Returns:
            Retargeter instance
        """
        return cls(config, hand_side)

    def retarget(
        self,
        raw_keypoints: np.ndarray,
        apply_filter: bool = True,
    ) -> np.ndarray:
        """Retarget raw MediaPipe keypoints to joint angles.

        Args:
            raw_keypoints: (21, 3) raw MediaPipe keypoints
            apply_filter: Whether to apply low-pass filter

        Returns:
            qpos: (20,) joint angles
        """
        # Apply coordinate transformation
        mediapipe_kp = apply_mediapipe_transformations(raw_keypoints, self.hand_side)

        # Apply rotation adjustment if configured
        if self.rotation_xyz:
            mediapipe_kp = self._apply_rotation(mediapipe_kp)

        # Solve IK
        qpos = self.optimizer.solve(mediapipe_kp)

        # Apply filter
        if apply_filter:
            qpos = self.lp_filter.next(qpos)

        return qpos

    def retarget_verbose(
        self,
        raw_keypoints: np.ndarray,
        apply_filter: bool = True,
    ) -> Tuple[np.ndarray, dict]:
        """Retarget with verbose output for visualization.

        Args:
            raw_keypoints: (21, 3) raw MediaPipe keypoints
            apply_filter: Whether to apply low-pass filter

        Returns:
            Tuple of (qpos, verbose_dict) where verbose_dict contains:
                - mediapipe_kp: Transformed keypoints
                - qpos_unfiltered: Joint angles before filtering
                - qpos: Final joint angles
                - cost: Optimization cost
                - pinch_alphas: (AdaptiveOptimizer only) alpha values
        """
        # Apply coordinate transformation
        mediapipe_kp = apply_mediapipe_transformations(raw_keypoints, self.hand_side)

        # Apply rotation adjustment if configured
        if self.rotation_xyz:
            mediapipe_kp = self._apply_rotation(mediapipe_kp)

        # Solve IK
        qpos = self.optimizer.solve(mediapipe_kp)

        # Apply filter
        if apply_filter:
            filtered_qpos = self.lp_filter.next(qpos)
        else:
            filtered_qpos = qpos

        # Build verbose dict
        verbose_dict = {
            'mediapipe_kp': mediapipe_kp.copy(),
            'qpos_unfiltered': qpos.copy(),
            'qpos': filtered_qpos.copy(),
            'cost': self.optimizer.compute_cost(qpos, mediapipe_kp),
        }

        # Add optimizer-specific data
        if hasattr(self.optimizer, '_compute_pinch_alpha'):
            verbose_dict['pinch_alphas'] = self.optimizer._compute_pinch_alpha(mediapipe_kp)

        return filtered_qpos, verbose_dict

    def _apply_rotation(self, keypoints: np.ndarray) -> np.ndarray:
        """Apply rotation adjustment to keypoints.

        Uses extrinsic XYZ Euler angles (rotations around fixed axes).

        Args:
            keypoints: (21, 3) keypoints in wrist frame

        Returns:
            Rotated keypoints
        """
        from scipy.spatial.transform import Rotation

        x_deg = self.rotation_xyz.get('x', 0.0)
        y_deg = self.rotation_xyz.get('y', 0.0)
        z_deg = self.rotation_xyz.get('z', 0.0)

        if x_deg == 0 and y_deg == 0 and z_deg == 0:
            return keypoints

        rot = Rotation.from_euler('xyz', [x_deg, y_deg, z_deg], degrees=True)
        return keypoints @ rot.as_matrix().T

    def reset_filter(self):
        """Reset low-pass filter state."""
        self.lp_filter.reset()

    def reset(self):
        """Reset all state (filter and optimizer)."""
        self.lp_filter.reset()
        self.optimizer.last_qpos = None

    @property
    def num_joints(self) -> int:
        """Number of joint angles."""
        return self.optimizer.num_joints


__all__ = ["Retargeter"]
