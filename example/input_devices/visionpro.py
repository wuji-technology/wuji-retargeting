from avp_stream import VisionProStreamer
from wuji_retargeting.mediapipe import MediaPipeSmoother

from .base import InputDeviceBase

class VisionPro(InputDeviceBase):
    def __init__(self, ip: str = "192.168.50.127"):
        self.streamer = VisionProStreamer(ip=ip)
        self._mediapipe_smoother = MediaPipeSmoother(buffer_size=5)

    def get_fingers_data(self) -> dict:
        data = self.streamer.latest
        return {
            "left_fingers": data["left_fingers"],
            "right_fingers": data["right_fingers"],
        }