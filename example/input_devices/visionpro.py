import numpy as np
from avp_stream import VisionProStreamer
from wuji_retargeting.mediapipe import MediaPipeSmoother

from .base import InputDeviceBase

# VisionPro 25 joints to MediaPipe 21 landmarks mapping
VP_TO_MEDIAPIPE = (
    0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 19, 21, 22, 23, 24
)


def convert_vp_to_mediapipe(fingers_mat: np.ndarray) -> np.ndarray:
    """Convert VisionPro (25, 4, 4) to MediaPipe (21, 3) format."""
    mediapipe_pose = np.zeros((21, 3), dtype=np.float32)
    for mp_idx, vp_idx in enumerate(VP_TO_MEDIAPIPE):
        mediapipe_pose[mp_idx] = fingers_mat[vp_idx][:3, 3]
    return mediapipe_pose


class VisionPro(InputDeviceBase):
    def __init__(self, ip: str = "192.168.50.127"):
        self.streamer = VisionProStreamer(ip=ip)
        self._mediapipe_smoother = MediaPipeSmoother(buffer_size=5)

    def get_fingers_data(self) -> dict:
        data = self.streamer.latest
        return {
            "left_fingers": convert_vp_to_mediapipe(data["left_fingers"]),
            "right_fingers": convert_vp_to_mediapipe(data["right_fingers"]),
        }