import pickle
import time
from pathlib import Path
from .base import InputDeviceBase

def _get_data_dir() -> Path:
    """Get the data directory path."""
    example_dir = Path(__file__).parent.parent
    return example_dir / "data"

class VisionProReplay(InputDeviceBase):
    def __init__(self, record_path: str, playback_speed: float = 0.7):
        record_path = Path(record_path)
        if not record_path.is_absolute():
            record_path = _get_data_dir() / record_path

        with open(record_path, "rb") as f:
            self.recording = pickle.load(f)

        self.playback_speed = playback_speed
        self.current_frame = 0
        self.total_frames = len(self.recording)
        self.start_time = None
        self.frame_interval = 0.005  # 200Hz (same as recording)

    def get_fingers_data(self) -> dict:
        if self.start_time is None:
            self.start_time = time.time()

        elapsed = (time.time() - self.start_time) * self.playback_speed
        target_frame = int(elapsed / self.frame_interval)

        target_frame = target_frame % self.total_frames
        self.current_frame = target_frame
        data = self.recording[self.current_frame]
        return {
            "left_fingers": data["left_fingers"],
            "right_fingers": data["right_fingers"],
        }
