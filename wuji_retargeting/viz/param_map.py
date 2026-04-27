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
        "desc_zh": "拇指各段长度缩放 [PIP, DIP, TIP]",
        "effect": "Increase -> thumb target skeleton extends further from wrist",
        "effect_zh": "调大 -> 拇指目标骨架变长，机器人拇指会更伸展",
    },
    "segment_scaling.index": {
        "fingers": ["index"],
        "joints": ["finger2_link2", "finger2_link3", "finger2_link4"],
        "desc": "Index finger segment length scaling [PIP, DIP, TIP]",
        "desc_zh": "食指各段长度缩放 [PIP, DIP, TIP]",
        "effect": "Compensates for length difference between human and robot index finger",
        "effect_zh": "补偿人手与机器手食指的长度差异",
    },
    "segment_scaling.middle": {
        "fingers": ["middle"],
        "joints": ["finger3_link2", "finger3_link3", "finger3_link4"],
        "desc": "Middle finger segment length scaling [PIP, DIP, TIP]",
        "desc_zh": "中指各段长度缩放 [PIP, DIP, TIP]",
        "effect": "Compensates for length difference between human and robot middle finger",
        "effect_zh": "补偿人手与机器手中指的长度差异",
    },
    "segment_scaling.ring": {
        "fingers": ["ring"],
        "joints": ["finger4_link2", "finger4_link3", "finger4_link4"],
        "desc": "Ring finger segment length scaling [PIP, DIP, TIP]",
        "desc_zh": "无名指各段长度缩放 [PIP, DIP, TIP]",
        "effect": "Compensates for length difference between human and robot ring finger",
        "effect_zh": "补偿人手与机器手无名指的长度差异",
    },
    "segment_scaling.pinky": {
        "fingers": ["pinky"],
        "joints": ["finger5_link2", "finger5_link3", "finger5_link4"],
        "desc": "Pinky finger segment length scaling [PIP, DIP, TIP]",
        "desc_zh": "小指各段长度缩放 [PIP, DIP, TIP]",
        "effect": "Compensates for length difference between human and robot pinky finger",
        "effect_zh": "补偿人手与机器手小指的长度差异",
    },

    # === Global parameters ===
    "scaling": {
        "fingers": ["all"],
        "desc": "Global hand scaling factor",
        "desc_zh": "全局手部缩放比例",
        "effect": "Increase -> all target points move further from wrist (for larger hands)",
        "effect_zh": "调大 -> 所有目标点离手腕更远，适合手大的用户",
    },
    "lp_alpha": {
        "fingers": ["all"],
        "desc": "Low-pass filter coefficient (0.0-1.0)",
        "desc_zh": "低通滤波系数 (0.0-1.0)",
        "effect": "Decrease -> smoother but more latency; Increase -> more responsive but may jitter",
        "effect_zh": "调小 -> 更平滑但延迟大；调大 -> 更灵敏但可能抖动",
    },
    "norm_delta": {
        "fingers": ["all"],
        "desc": "Velocity regularization weight",
        "desc_zh": "速度正则化权重",
        "effect": "Increase -> motion is smoother/more stable; Decrease -> tracks hand more closely",
        "effect_zh": "调大 -> 运动更平稳；调小 -> 跟手更紧",
    },

    # === Pinch thresholds ===
    "pinch_thresholds.index": {
        "fingers": ["thumb", "index"],
        "desc": "Index finger pinch activation thresholds {d1, d2} (cm)",
        "desc_zh": "食指捏取激活距离阈值 {d1, d2} (cm)",
        "effect": "d1: below this distance, full pinch mode; d2: above this, full open mode",
        "effect_zh": "d1: 低于此距离进入捏取模式；d2: 高于此距离完全退出捏取模式",
    },
    "pinch_thresholds.middle": {
        "fingers": ["thumb", "middle"],
        "desc": "Middle finger pinch activation thresholds {d1, d2} (cm)",
        "desc_zh": "中指捏取激活距离阈值 {d1, d2} (cm)",
        "effect": "Same as index pinch thresholds but for middle finger",
        "effect_zh": "与食指捏取阈值相同，但作用于中指",
    },
    "pinch_thresholds.ring": {
        "fingers": ["thumb", "ring"],
        "desc": "Ring finger pinch activation thresholds {d1, d2} (cm)",
        "desc_zh": "无名指捏取激活距离阈值 {d1, d2} (cm)",
        "effect": "Same as index pinch thresholds but for ring finger",
        "effect_zh": "与食指捏取阈值相同，但作用于无名指",
    },
    "pinch_thresholds.pinky": {
        "fingers": ["thumb", "pinky"],
        "desc": "Pinky finger pinch activation thresholds {d1, d2} (cm)",
        "desc_zh": "小指捏取激活距离阈值 {d1, d2} (cm)",
        "effect": "Same as index pinch thresholds but for pinky finger",
        "effect_zh": "与食指捏取阈值相同，但作用于小指",
    },

    # === Rotation ===
    "mediapipe_rotation.x": {
        "fingers": ["all"],
        "desc": "Input coordinate X-axis rotation compensation (degrees)",
        "desc_zh": "输入坐标系 X 轴旋转补偿（度）",
        "effect": "Rotates all input keypoints around X-axis to correct systematic offset",
        "effect_zh": "绕 X 轴旋转所有输入关键点以校正系统偏移",
    },
    "mediapipe_rotation.y": {
        "fingers": ["all"],
        "desc": "Input coordinate Y-axis rotation compensation (degrees)",
        "desc_zh": "输入坐标系 Y 轴旋转补偿（度）",
        "effect": "Rotates all input keypoints around Y-axis to correct systematic offset",
        "effect_zh": "绕 Y 轴旋转所有输入关键点以校正系统偏移",
    },
    "mediapipe_rotation.z": {
        "fingers": ["all"],
        "desc": "Input coordinate Z-axis rotation compensation (degrees)",
        "desc_zh": "输入坐标系 Z 轴旋转补偿（度）",
        "effect": "Rotates all input keypoints around Z-axis to correct systematic offset",
        "effect_zh": "绕 Z 轴旋转所有输入关键点以校正系统偏移",
    },

    # === Video-specific ===
    "video_input.z_scale": {
        "fingers": ["all"],
        "desc": "Monocular depth amplification factor (video mode only)",
        "desc_zh": "单目深度放大倍数（仅视频模式）",
        "effect": "Increase -> finger forward/backward motion is amplified",
        "effect_zh": "调大 -> 手指前后运动更明显",
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


def get_param_description(param_path: str, lang: str = "zh") -> str:
    """Get human-readable description for a parameter.

    Args:
        param_path: Dot-separated parameter path (e.g., "segment_scaling.thumb")
        lang: Language code ("zh" for Chinese, "en" for English)

    Returns:
        Description string, or empty string if not found
    """
    info = PARAM_FINGER_MAP.get(param_path)
    if info is None:
        return ""
    desc_key = "desc_zh" if lang == "zh" else "desc"
    effect_key = "effect_zh" if lang == "zh" else "effect"
    desc = info.get(desc_key, info.get("desc", ""))
    effect = info.get(effect_key, info.get("effect", ""))
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
