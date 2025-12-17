"""
MediaPipe Replay device for teleoperation.
Replays recorded MediaPipe hand tracking data.

The .pkl file format:
[
    {
        "t": float,  # timestamp
        "left_fingers": np.ndarray (21, 3),  # MediaPipe format
        "right_fingers": np.ndarray (21, 3),  # MediaPipe format
    },
    ...
]

This device replays recorded hand tracking data for demo and testing.
"""

import pickle
import time
from pathlib import Path
import numpy as np


def _get_example_dir() -> Path:
    """Get the example directory (parent of input_devices)."""
    return Path(__file__).resolve().parents[1]


class MediaPipeReplay:
    """Replay recorded MediaPipe hand tracking data."""

    def __init__(
        self,
        record_path: str,
        playback_speed: float = 1.0,
        loop: bool = True,
    ):
        """
        Initialize input data replay device.

        Args:
            record_path: Path to the .pkl file (absolute or relative to example dir)
            playback_speed: Speed multiplier for playback (1.0 = original speed)
            loop: Whether to loop the recording
        """
        record_path = Path(record_path)
        if not record_path.is_absolute():
            record_path = _get_example_dir() / record_path

        if not record_path.exists():
            raise FileNotFoundError(f"Recording file not found: {record_path}")

        with open(record_path, "rb") as f:
            self.recording = pickle.load(f)

        self.playback_speed = playback_speed
        self.loop = loop
        self.current_frame = 0
        self.total_frames = len(self.recording)
        self.start_time = None
        self._finished = False

        # Get the duration from timestamps
        if self.total_frames > 1:
            self.duration = self.recording[-1]["t"] - self.recording[0]["t"]
        else:
            self.duration = 0.0

        # Check if timestamps are valid (not all zeros)
        self.use_frame_counter = self.duration < 0.01
        self.default_fps = 30.0  # Default FPS when timestamps are invalid

        print(f"Loaded input data recording: {record_path}")
        print(f"  Total frames: {self.total_frames}")
        if self.use_frame_counter:
            print(f"  Mode: frame counter (timestamps invalid)")
        else:
            print(f"  Duration: {self.duration:.2f}s")

    def get_fingers_data(self) -> dict:
        """
        Return finger data in MediaPipe (21, 3) format.

        Returns:
            dict with "left_fingers" and "right_fingers" keys
        """
        if self._finished:
            # Return zeros when finished (non-loop mode)
            empty = np.zeros((21, 3), dtype=np.float32)
            return {"left_fingers": empty, "right_fingers": empty}

        if self.start_time is None:
            self.start_time = time.time()

        # Find the frame based on elapsed time or frame counter
        if self.use_frame_counter:
            # Use frame counter mode for invalid timestamps
            elapsed = (time.time() - self.start_time) * self.playback_speed
            target_frame = int(elapsed * self.default_fps)
        else:
            elapsed = (time.time() - self.start_time) * self.playback_speed
            target_frame = self._find_frame_by_time(elapsed)

        if target_frame >= self.total_frames:
            if self.loop:
                # Reset for looping
                self.start_time = time.time()
                target_frame = 0
            else:
                self._finished = True
                empty = np.zeros((21, 3), dtype=np.float32)
                return {"left_fingers": empty, "right_fingers": empty}

        self.current_frame = target_frame
        data = self.recording[self.current_frame]

        return {
            "left_fingers": data["left_fingers"],
            "right_fingers": data["right_fingers"],
        }

    def _find_frame_by_time(self, elapsed: float) -> int:
        """Find the frame index for the given elapsed time."""
        if self.total_frames == 0:
            return 0

        base_time = self.recording[0]["t"]

        # Binary search for efficiency
        left, right = 0, self.total_frames - 1
        while left < right:
            mid = (left + right + 1) // 2
            if self.recording[mid]["t"] - base_time <= elapsed:
                left = mid
            else:
                right = mid - 1

        return left

    def is_finished(self) -> bool:
        """Check if playback is finished (only for non-loop mode)."""
        return self._finished

    def reset(self):
        """Reset playback to the beginning."""
        self.start_time = None
        self.current_frame = 0
        self._finished = False

    def get_progress(self) -> tuple[int, int]:
        """Get current progress as (current_frame, total_frames)."""
        return self.current_frame, self.total_frames
