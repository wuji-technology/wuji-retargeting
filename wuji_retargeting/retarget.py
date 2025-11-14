"""High-level retargeting interface for Wuji Hand using DexPilot algorithm."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .robot import RobotWrapper
from .opt import DexPilotOptimizer, LPFilter


# Package root for URDF path resolution
_THIS_FILE = Path(__file__).resolve()
_PACKAGE_ROOT = _THIS_FILE.parent

# Wuji Hand fixed configuration
WUJI_WRIST_LINK_NAME = "palm_link"
WUJI_FINGER_TIP_LINK_NAMES = [
    "finger1_tip_link",
    "finger2_tip_link",
    "finger3_tip_link",
    "finger4_tip_link",
    "finger5_tip_link",
]
WUJI_FINGER_TIP_SCALING = [1.0, 1.1, 1.0, 1.2, 1.3]  # Default scaling for thumb to pinky
LOW_PASS_ALPHA = 0.2  # Low-pass filter alpha (smaller = smoother but more latency)


@dataclass
class RetargetingResult:
    """Retargeting result containing robot joint positions and intermediate data."""
    robot_qpos: np.ndarray      # Robot joint positions (20,)
    mediapipe_pose: np.ndarray  # MediaPipe format hand pose (21, 3)
    reference: np.ndarray       # Reference vectors used in optimization


class WujiHandRetargeter:
    """Retargeter for Wuji Hand using DexPilot algorithm."""
    
    def __init__(self, hand_side: str = "right"):
        """
        Initialize retargeter for specified hand.
        
        Args:
            hand_side: "right" or "left"
        """
        self.hand_side = hand_side.lower()
        if self.hand_side not in ["right", "left"]:
            raise ValueError(f"hand_side must be 'right' or 'left', got {hand_side}")
        
        # Build URDF path (from package directory)
        urdf_path = (_PACKAGE_ROOT / f"urdf/{self.hand_side}.urdf").resolve()
        if not urdf_path.exists():
            raise ValueError(f"URDF path {urdf_path} does not exist")
        
        # Load robot model
        robot = RobotWrapper(str(urdf_path))
        
        # Build optimizer with Wuji Hand hardcoded configuration
        self.optimizer = DexPilotOptimizer(
            robot,
            robot.dof_joint_names,
            finger_tip_link_names=WUJI_FINGER_TIP_LINK_NAMES,
            wrist_link_name=WUJI_WRIST_LINK_NAME,
            finger_scaling=WUJI_FINGER_TIP_SCALING,
        )
        
        # Joint limits (always enabled for Wuji Hand)
        joint_limits = robot.joint_limits[self.optimizer.idx_pin2target]
        self.optimizer.set_joint_limit(joint_limits)
        self.joint_limits = joint_limits
        
        # Store optimizer and filter
        self.filter = LPFilter(LOW_PASS_ALPHA)
        
        # Initialize last joint positions for warm start
        self.last_qpos = joint_limits.mean(1).astype(np.float32)
    
    def retarget(self, mediapipe_pose: np.ndarray) -> RetargetingResult:
        """
        Retarget MediaPipe format hand pose to Wuji Hand joint positions.
        
        Args:
            mediapipe_pose: MediaPipe format hand pose (21, 3) - hand landmarks in 3D
            
        Returns:
            RetargetingResult with robot_qpos, mediapipe_pose, and reference
        """
        mediapipe_pose = np.asarray(mediapipe_pose, dtype=np.float64)
        if mediapipe_pose.shape != (21, 3):
            raise ValueError(f"Expected mediapipe_pose shape (21, 3), got {mediapipe_pose.shape}")
        
        # Compute reference vectors (task - origin)
        indices = self.optimizer.target_link_human_indices
        reference = mediapipe_pose[indices[1], :] - mediapipe_pose[indices[0], :]
        
        # Run retargeting optimization
        robot_qpos = self._retarget_optimization(ref_value=reference)
        
        return RetargetingResult(
            robot_qpos=robot_qpos,
            mediapipe_pose=mediapipe_pose,
            reference=reference,
        )
    
    def _retarget_optimization(self, ref_value: np.ndarray) -> np.ndarray:
        """Internal method to run optimization and filtering."""
        qpos = self.optimizer.retarget(
            ref_value=ref_value.astype(np.float32),
            last_qpos=np.clip(
                self.last_qpos, self.joint_limits[:, 0], self.joint_limits[:, 1]
            ),
        )
        self.last_qpos = qpos
        
        # Apply low-pass filter
        robot_qpos = self.filter.next(qpos)
        return robot_qpos


__all__ = [
    "WujiHandRetargeter",
    "RetargetingResult",
]

