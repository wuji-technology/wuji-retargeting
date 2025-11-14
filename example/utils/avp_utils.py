"""Apple Vision Pro (AVP) specific utilities for MediaPipe format conversion."""

from typing import Optional
import numpy as np

from wuji_retargeting.mediapipe import (
    apply_mediapipe_transformations,
)
from wuji_retargeting import WujiHandRetargeter, RetargetingResult


def convert_hand_matrices_to_mediapipe(hand_matrices):
    """
    Convert Vision Pro hand matrices (25x4x4) to MediaPipe format (21x3).
    
    Args:
        hand_matrices: numpy array of shape (25, 4, 4) - transformation matrices
        
    Returns:
        mediapipe_pose: numpy array of shape (21, 3) - MediaPipe landmarks
    """
    mediapipe_pose = np.zeros((21, 3))
    
    joint_mapping = [
        0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24
    ]
    
    for mp_idx, vp_joint_idx in enumerate(joint_mapping):
        position = hand_matrices[vp_joint_idx][:3, 3]
        mediapipe_pose[mp_idx] = position
    
    return mediapipe_pose


def convert_vision_pro_to_mediapipe_format(hand_matrices, hand_type: str = "right"):
    """
    Convert Vision Pro hand matrices to MediaPipe format with proper coordinate transformations.
    
    Args:
        hand_matrices: numpy array of shape (25, 4, 4) - Vision Pro transformation matrices
        hand_type: "right" or "left" - determines coordinate system
        
    Returns:
        joint_pos: numpy array of shape (21, 3) - transformed landmarks in MediaPipe format
    """
    hand_type = hand_type.lower()
    
    keypoint_3d_array = convert_hand_matrices_to_mediapipe(hand_matrices)
    joint_pos = apply_mediapipe_transformations(keypoint_3d_array, hand_type)
    
    return joint_pos


# Cache for retargeter instances
_retargeter_cache = {}


def retarget_vision_pro(hand_matrices: np.ndarray, hand_side: str = "right", retargeter: Optional[WujiHandRetargeter] = None) -> RetargetingResult:
    """
    Retarget Vision Pro hand matrices to Wuji Hand joint positions.
    This is a convenience wrapper that combines AVP conversion and retargeting.
    
    Args:
        hand_matrices: numpy array of shape (25, 4, 4) - Vision Pro transformation matrices
        hand_side: "right" or "left" - determines coordinate system
        retargeter: Optional WujiHandRetargeter instance to reuse. If None, creates or reuses cached instance.
        
    Returns:
        RetargetingResult with robot_qpos, mediapipe_pose, and reference
    """
    # Convert Vision Pro matrices to MediaPipe format
    mediapipe_pose = convert_vision_pro_to_mediapipe_format(hand_matrices, hand_type=hand_side)
    
    # Use provided retargeter or get/create cached one
    if retargeter is None:
        if hand_side not in _retargeter_cache:
            _retargeter_cache[hand_side] = WujiHandRetargeter(hand_side=hand_side)
        retargeter = _retargeter_cache[hand_side]
    
    return retargeter.retarget(mediapipe_pose)


__all__ = [
    'convert_hand_matrices_to_mediapipe',
    'convert_vision_pro_to_mediapipe_format',
    'retarget_vision_pro',
]

