"""Parameter-to-joint/finger mapping for tuning guidance.

Provides human-readable descriptions of what each retargeting parameter
controls, which fingers/joints it affects, and how adjusting it changes behavior.
"""

# Maps parameter path -> description dict
PARAM_FINGER_MAP = {
    # === Per-finger segment scaling (primary tuning parameters) ===
    "segment_scaling.thumb": {
        "fingers": ["thumb"],
        "joints": ["finger1_link2", "finger1_link3", "finger1_link4"],
        "desc": "Thumb segment length scaling [PIP, DIP, TIP]",
        "effect": "Increase -> thumb target skeleton extends further from wrist",
    },
    "segment_scaling.index": {
        "fingers": ["index"],
        "joints": ["finger2_link2", "finger2_link3", "finger2_link4"],
        "desc": "Index finger segment length scaling [PIP, DIP, TIP]",
        "effect": "Compensates for length difference between human and robot index finger",
    },
    "segment_scaling.middle": {
        "fingers": ["middle"],
        "joints": ["finger3_link2", "finger3_link3", "finger3_link4"],
        "desc": "Middle finger segment length scaling [PIP, DIP, TIP]",
        "effect": "Compensates for length difference between human and robot middle finger",
    },
    "segment_scaling.ring": {
        "fingers": ["ring"],
        "joints": ["finger4_link2", "finger4_link3", "finger4_link4"],
        "desc": "Ring finger segment length scaling [PIP, DIP, TIP]",
        "effect": "Compensates for length difference between human and robot ring finger",
    },
    "segment_scaling.pinky": {
        "fingers": ["pinky"],
        "joints": ["finger5_link2", "finger5_link3", "finger5_link4"],
        "desc": "Pinky finger segment length scaling [PIP, DIP, TIP]",
        "effect": "Compensates for length difference between human and robot pinky finger",
    },

    # === Global parameters ===
    "scaling": {
        "fingers": ["all"],
        "desc": "Global hand scaling factor",
        "effect": "Increase -> all target points move further from wrist (for larger hands)",
    },
    "lp_alpha": {
        "fingers": ["all"],
        "desc": "Low-pass filter coefficient (0.0-1.0)",
        "effect": "Decrease -> smoother but more latency; Increase -> more responsive but may jitter",
    },
    "norm_delta": {
        "fingers": ["all"],
        "desc": "Velocity regularization weight",
        "effect": "Increase -> motion is smoother/more stable; Decrease -> tracks hand more closely",
    },

    # === Pinch thresholds ===
    "pinch_thresholds.index": {
        "fingers": ["thumb", "index"],
        "desc": "Index finger pinch activation thresholds {d1, d2} (cm)",
        "effect": "d1: below this distance, full pinch mode; d2: above this, full open mode",
    },
    "pinch_thresholds.middle": {
        "fingers": ["thumb", "middle"],
        "desc": "Middle finger pinch activation thresholds {d1, d2} (cm)",
        "effect": "Same as index pinch thresholds but for middle finger",
    },
    "pinch_thresholds.ring": {
        "fingers": ["thumb", "ring"],
        "desc": "Ring finger pinch activation thresholds {d1, d2} (cm)",
        "effect": "Same as index pinch thresholds but for ring finger",
    },
    "pinch_thresholds.pinky": {
        "fingers": ["thumb", "pinky"],
        "desc": "Pinky finger pinch activation thresholds {d1, d2} (cm)",
        "effect": "Same as index pinch thresholds but for pinky finger",
    },

    # === Rotation ===
    "mediapipe_rotation.x": {
        "fingers": ["all"],
        "desc": "Input coordinate X-axis rotation compensation (degrees)",
        "effect": "Rotates all input keypoints around X-axis to correct systematic offset",
    },
    "mediapipe_rotation.y": {
        "fingers": ["all"],
        "desc": "Input coordinate Y-axis rotation compensation (degrees)",
        "effect": "Rotates all input keypoints around Y-axis to correct systematic offset",
    },
    "mediapipe_rotation.z": {
        "fingers": ["all"],
        "desc": "Input coordinate Z-axis rotation compensation (degrees)",
        "effect": "Rotates all input keypoints around Z-axis to correct systematic offset",
    },

    # === Video-specific ===
    "video_input.z_scale": {
        "fingers": ["all"],
        "desc": "Monocular depth amplification factor (video mode only)",
        "effect": "Increase -> finger forward/backward motion is amplified",
    },
}

# Finger name -> MediaPipe keypoint indices
FINGER_KEYPOINT_INDICES = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
    "all": list(range(0, 21)),
}


def get_param_description(param_path: str, lang: str = "en") -> str:
    """Get human-readable description for a parameter.

    Args:
        param_path: Dot-separated parameter path (e.g., "segment_scaling.thumb")
        lang: Language code. English is the only supported language.

    Returns:
        Description string, or empty string if not found
    """
    info = PARAM_FINGER_MAP.get(param_path)
    if info is None:
        return ""
    _ = lang
    desc = info.get("desc", "")
    effect = info.get("effect", "")
    fingers = ", ".join(info.get("fingers", []))
    return f"{desc} | Fingers: {fingers} | {effect}"


def get_affected_fingers(param_path: str) -> list:
    """Get list of finger names affected by a parameter.

    Args:
        param_path: Dot-separated parameter path

    Returns:
        List of finger names, or empty list if not found
    """
    info = PARAM_FINGER_MAP.get(param_path)
    if info is None:
        return []
    return info.get("fingers", [])


def get_affected_keypoint_indices(param_path: str) -> list:
    """Get MediaPipe keypoint indices affected by a parameter.

    Args:
        param_path: Dot-separated parameter path

    Returns:
        List of keypoint indices
    """
    fingers = get_affected_fingers(param_path)
    indices = []
    for finger in fingers:
        indices.extend(FINGER_KEYPOINT_INDICES.get(finger, []))
    return sorted(set(indices))
