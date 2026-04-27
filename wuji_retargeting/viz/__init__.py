"""Visualization tools for hand retargeting parameter tuning.

Provides an interactive MuJoCo-based viewer that displays three skeleton layers
(input, scaled target, robot FK) simultaneously, with YAML-driven configuration
and hot-reload support for real-time parameter tuning.

Usage:
    from wuji_retargeting.viz import TuningViewer

    viewer = TuningViewer(hand_side="left", retarget_config_path="config.yaml")
    viewer.play_recording(data)
"""

from .tuning_viewer import TuningViewer
from .skeleton_drawer import SkeletonDrawer
from .config_watcher import ConfigWatcher
from .param_map import PARAM_FINGER_MAP, get_param_description

__all__ = [
    "TuningViewer",
    "SkeletonDrawer",
    "ConfigWatcher",
    "PARAM_FINGER_MAP",
    "get_param_description",
]
