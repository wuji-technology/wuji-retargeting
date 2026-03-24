"""
MediaPipe Video input device for teleoperation.
Reads an MP4 video file, runs MediaPipe hand detection, and outputs
hand landmarks in the standard (21, 3) format.

Usage:
    mjpython teleop_sim.py --video data/right.mp4 --hand right
"""

import time
from pathlib import Path

import cv2
import numpy as np
import mediapipe as mp

from .base import InputDeviceBase


def _get_example_dir() -> Path:
    return Path(__file__).resolve().parents[1]


# Reference finger segment lengths (meters) from AVP stereo tracking data.
# These match the actual hand proportions that the retargeting pipeline is tuned for.
# [MCP-PIP, PIP-DIP, DIP-TIP] for each finger
_REFERENCE_SEGMENT_LENGTHS = {
    'thumb':  [0.0505, 0.0318, 0.0302],
    'index':  [0.0418, 0.0243, 0.0223],
    'middle': [0.0489, 0.0289, 0.0227],
    'ring':   [0.0422, 0.0274, 0.0227],
    'pinky':  [0.0343, 0.0195, 0.0201],
}

# MediaPipe landmark index groups per finger: [MCP, PIP, DIP, TIP]
_FINGER_INDICES = {
    'thumb':  [1, 2, 3, 4],
    'index':  [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring':   [13, 14, 15, 16],
    'pinky':  [17, 18, 19, 20],
}


class VideoMediaPipe(InputDeviceBase):
    """Read MP4 video and extract hand landmarks via MediaPipe."""

    # MediaPipe hand skeleton connections for drawing
    _HAND_CONNECTIONS = [
        (0,1),(1,2),(2,3),(3,4),
        (0,5),(5,6),(6,7),(7,8),
        (0,9),(9,10),(10,11),(11,12),
        (0,13),(13,14),(14,15),(15,16),
        (0,17),(17,18),(18,19),(19,20),
        (5,9),(9,13),(13,17),
    ]

    def __init__(
        self,
        video_path: str,
        hand_side: str = "right",
        playback_speed: float = 1.0,
        loop: bool = True,
        video_config: dict = None,
        show_video: bool = False,
    ):
        """
        Args:
            video_path: Path to MP4 video file
            hand_side: Expected hand side in the video ('left' or 'right')
            playback_speed: Playback speed multiplier
            loop: Whether to loop the video
            video_config: Video input config dict (from yaml 'video_input' section)
            show_video: Whether to display video with MediaPipe landmarks
        """
        self.show_video = show_video
        video_path = Path(video_path)
        if not video_path.is_absolute():
            video_path = _get_example_dir() / video_path

        if not video_path.exists():
            raise FileNotFoundError(f"Video file not found: {video_path}")

        self.video_path = str(video_path)
        self.hand_side = hand_side.lower()
        self.playback_speed = playback_speed
        self.loop = loop

        # Read parameters from config (with defaults)
        cfg = video_config or {}
        self.z_scale = cfg.get('z_scale', 2.5)
        self.correct_segments = cfg.get('correct_segments', True)
        self._reference_wrist_to_mid_mcp = cfg.get('reference_wrist_to_mid_mcp', 0.09)

        # Open video
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # MediaPipe hands detector
        self.mp_hands = mp.solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        # Pre-process all frames to avoid real-time detection latency
        self._landmarks = []
        self._raw_landmarks = []  # normalized (0-1) coords for drawing on video
        self._frames = [] if self.show_video else None  # cache video frames
        self._preprocess_video()

        # Playback state
        self.start_time = None
        self.current_frame = 0
        self._finished = False

        print(f"Loaded video: {video_path}")
        print(f"  Resolution: {self.frame_width}x{self.frame_height}")
        print(f"  FPS: {self.fps:.1f}, Total frames: {self.total_frames}")
        print(f"  Detected hand in {sum(1 for lm in self._landmarks if lm is not None)}/{self.total_frames} frames")
        print(f"  Hand side: {self.hand_side}, z_scale: {self.z_scale}, correct_segments: {self.correct_segments}")

    def _preprocess_video(self):
        """Run MediaPipe on all frames and cache landmarks."""
        print("Pre-processing video frames with MediaPipe...")
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        # MediaPipe reports handedness from camera view (mirrored).
        # A right hand in real life appears as "Left" in MediaPipe output.
        expected_mp_label = "Left" if self.hand_side == "right" else "Right"

        last_valid = None

        last_valid_raw = None

        for i in range(self.total_frames):
            ret, frame = self.cap.read()
            if not ret:
                self._landmarks.append(last_valid)
                self._raw_landmarks.append(last_valid_raw)
                if self._frames is not None:
                    self._frames.append(None)
                continue

            if self._frames is not None:
                self._frames.append(frame.copy())

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.mp_hands.process(rgb)

            kp = None
            raw_lm = None
            if results.multi_hand_landmarks and results.multi_handedness:
                for hand_lm, hand_cls in zip(
                    results.multi_hand_landmarks, results.multi_handedness
                ):
                    label = hand_cls.classification[0].label
                    if label == expected_mp_label:
                        kp = self._landmarks_to_array(hand_lm)
                        raw_lm = [(lm.x, lm.y) for lm in hand_lm.landmark]
                        break

                # If expected hand not found, use the first detected hand
                if kp is None and results.multi_hand_landmarks:
                    kp = self._landmarks_to_array(results.multi_hand_landmarks[0])
                    raw_lm = [(lm.x, lm.y) for lm in results.multi_hand_landmarks[0].landmark]

            if kp is not None:
                kp = self._process_landmarks(kp)
                last_valid = kp
                last_valid_raw = raw_lm

            self._landmarks.append(kp if kp is not None else last_valid)
            self._raw_landmarks.append(raw_lm if raw_lm is not None else last_valid_raw)

        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    def _landmarks_to_array(self, hand_landmarks) -> np.ndarray:
        """Convert MediaPipe hand landmarks to (21, 3) numpy array.

        MediaPipe gives:
          x, y: normalized image coordinates [0, 1]
          z: relative depth (roughly same scale as x, negative = closer)

        We convert to pixel-scale 3D coordinates with z amplified
        to compensate for monocular depth compression.
        """
        kp = np.array(
            [[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark],
            dtype=np.float32,
        )
        # Scale to pixel space. z uses x-scale per MediaPipe convention,
        # amplified by z_scale to compensate for monocular depth compression.
        kp[:, 0] *= self.frame_width
        kp[:, 1] *= self.frame_height
        kp[:, 2] *= self.frame_width * self.z_scale
        return kp

    def _process_landmarks(self, kp: np.ndarray) -> np.ndarray:
        """Center at wrist and scale to approximate real-world meters.

        Uses wrist-to-middle-finger-MCP distance as reference (~0.09m).
        Optionally corrects per-finger segment lengths using anthropometric data.
        """
        # Center at wrist
        kp = kp - kp[0:1, :]

        # Global scale: match wrist-to-middle-MCP distance
        dist = np.linalg.norm(kp[9])
        if dist < 1e-6:
            return kp
        scale = self._reference_wrist_to_mid_mcp / dist
        kp = kp * scale

        # Per-segment scaling to correct finger proportions
        if self.correct_segments:
            kp = self._correct_segment_lengths(kp)

        return kp

    def _correct_segment_lengths(self, kp: np.ndarray) -> np.ndarray:
        """Correct individual finger segment lengths to match anthropometric data.

        Scales each segment (MCP->PIP, PIP->DIP, DIP->TIP) to match
        expected lengths while preserving directions.
        """
        kp_corrected = kp.copy()

        for finger_name, indices in _FINGER_INDICES.items():
            ref_lengths = _REFERENCE_SEGMENT_LENGTHS[finger_name]
            mcp_i, pip_i, dip_i, tip_i = indices

            # Base point for this finger chain
            base = kp_corrected[mcp_i].copy()

            # Segment 1: MCP -> PIP
            seg1 = kp[pip_i] - kp[mcp_i]
            seg1_len = np.linalg.norm(seg1)
            if seg1_len > 1e-6:
                seg1_dir = seg1 / seg1_len
                kp_corrected[pip_i] = base + seg1_dir * ref_lengths[0]

            # Segment 2: PIP -> DIP
            seg2 = kp[dip_i] - kp[pip_i]
            seg2_len = np.linalg.norm(seg2)
            if seg2_len > 1e-6:
                seg2_dir = seg2 / seg2_len
                kp_corrected[dip_i] = kp_corrected[pip_i] + seg2_dir * ref_lengths[1]

            # Segment 3: DIP -> TIP
            seg3 = kp[tip_i] - kp[dip_i]
            seg3_len = np.linalg.norm(seg3)
            if seg3_len > 1e-6:
                seg3_dir = seg3 / seg3_len
                kp_corrected[tip_i] = kp_corrected[dip_i] + seg3_dir * ref_lengths[2]

        return kp_corrected

    def get_fingers_data(self) -> dict:
        if self._finished:
            empty = np.zeros((21, 3), dtype=np.float32)
            return {"left_fingers": empty, "right_fingers": empty}

        if self.start_time is None:
            self.start_time = time.time()

        # Determine target frame from elapsed time
        elapsed = (time.time() - self.start_time) * self.playback_speed
        target_frame = int(elapsed * self.fps)

        if target_frame >= self.total_frames:
            if self.loop:
                self.start_time = time.time()
                target_frame = 0
            else:
                self._finished = True
                empty = np.zeros((21, 3), dtype=np.float32)
                return {"left_fingers": empty, "right_fingers": empty}

        self.current_frame = target_frame
        kp = self._landmarks[self.current_frame]

        # Show video with landmarks overlay
        if self.show_video:
            self._show_video_frame(self.current_frame)

        empty = np.zeros((21, 3), dtype=np.float32)
        if kp is None:
            return {"left_fingers": empty, "right_fingers": empty}

        result = {"left_fingers": empty.copy(), "right_fingers": empty.copy()}
        result[f"{self.hand_side}_fingers"] = kp
        return result

    def _show_video_frame(self, frame_idx: int):
        """Display video frame with MediaPipe landmarks overlay."""
        if self._frames is None or self._frames[frame_idx] is None:
            return

        frame = self._frames[frame_idx].copy()
        raw_lm = self._raw_landmarks[frame_idx]

        if raw_lm is not None:
            h, w = frame.shape[:2]
            pts = [(int(x * w), int(y * h)) for x, y in raw_lm]

            # Draw skeleton lines
            for s, e in self._HAND_CONNECTIONS:
                cv2.line(frame, pts[s], pts[e], (0, 255, 0), 2)

            # Draw keypoints
            for i, pt in enumerate(pts):
                cv2.circle(frame, pt, 4, (0, 0, 255), -1)
                cv2.putText(frame, str(i), (pt[0]+5, pt[1]-5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1)

        # Resize for display
        scale = 480 / frame.shape[0]
        display = cv2.resize(frame, None, fx=scale, fy=scale)
        cv2.putText(display, f"Frame {frame_idx}/{self.total_frames}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.imshow("MediaPipe Video", display)
        cv2.waitKey(1)

    def is_finished(self) -> bool:
        return self._finished

    def reset(self):
        self.start_time = None
        self.current_frame = 0
        self._finished = False

    def __del__(self):
        if hasattr(self, "cap") and self.cap is not None:
            self.cap.release()
        if hasattr(self, "mp_hands"):
            self.mp_hands.close()
        if hasattr(self, "show_video") and self.show_video:
            cv2.destroyAllWindows()
