"""Input device for Wuji Glove via wuji_sdk.

Connects to a Wuji Glove via wuji_sdk's SdkManager and subscribes to
HandSkeleton data (21 MediaPipe joints). Returns (21, 3) position arrays
compatible with the existing retargeting pipeline.

Usage:
    device = WujiGloveDevice(hand_side="right", device_name="glove")
    data = device.get_fingers_data()
    # data["right_fingers"] -> np.ndarray (21, 3)
"""

from typing import Dict, Optional

import numpy as np

from .base import InputDeviceBase

try:
    from wuji_sdk import SdkManager
    WUJI_SDK_AVAILABLE = True
except ImportError:
    WUJI_SDK_AVAILABLE = False


class WujiGloveDevice(InputDeviceBase):
    """Input device that reads Wuji Glove MediaPipe data via ``wuji_sdk``."""

    def __init__(
        self,
        hand_side: Optional[str] = "right",
        device_name: str = "glove",
        sn: Optional[str] = None,
    ):
        """Initialize the Wuji Glove input device.

        Args:
            hand_side: Hand side, ``"left"`` or ``"right"``. If ``None``,
                infer the side from the SDK ``frame_id``.
            device_name: SDK device name used for routing and handle management.
            sn: Device serial number. Required in multi-device setups; may be
                ``None`` when auto-connecting a single glove.
        """
        if not WUJI_SDK_AVAILABLE:
            raise ImportError(
                "wuji_sdk is not installed. "
                "Please install wuji_sdk to use WujiGloveDevice."
            )

        if hand_side is None:
            normalized_hand_side = None
        else:
            normalized_hand_side = hand_side.lower()
            if normalized_hand_side not in {"left", "right"}:
                raise ValueError(
                    f"hand_side must be 'left', 'right', or None, got {hand_side!r}"
                )

        self._hand_side = normalized_hand_side
        self._device_name = device_name
        self._last_data: Dict[str, Optional[np.ndarray]] = {
            "left_fingers": None,
            "right_fingers": None,
        }

        manager = SdkManager.instance()
        if sn:
            self._device = manager.connect(sn=sn, device_name=device_name)
        else:
            self._device = manager.auto_connect(device_name=device_name)
        self._sub = self._device.hand_skeleton().subscribe()

    def get_fingers_data(self) -> Dict[str, Optional[np.ndarray]]:
        """Return the latest non-blocking ``(21, 3)`` skeleton frame.

        Returns:
            {"left_fingers": np.ndarray | None, "right_fingers": np.ndarray | None}
            Returns the cached previous frame when no new data is available.
        """
        skeleton = self._sub.recv()
        if skeleton is None:
            return self._last_data

        # Drain queue to keep only the latest frame,
        # preventing lag buildup when SDK pushes faster than we consume.
        while True:
            newer = self._sub.recv()
            if newer is None:
                break
            skeleton = newer

        # Extract 21 joint positions as (x, y, z) coordinates.
        keypoints = np.array(
            [j.pose.position for j in skeleton.joints],
            dtype=np.float32,
        )
        if keypoints.shape != (21, 3):
            print(f"Warning: unexpected skeleton shape {keypoints.shape}, skipping frame")
            return self._last_data

        # Infer the active hand side when it is not fixed by the caller.
        hand_side = self._hand_side
        if hand_side is None:
            hand_side = self._detect_hand_side(skeleton)

        result: Dict[str, Optional[np.ndarray]] = {
            "left_fingers": None,
            "right_fingers": None,
        }
        result[f"{hand_side}_fingers"] = keypoints
        self._last_data = result
        return result

    def cleanup(self):
        """Release SDK resources."""
        self._sub = None
        self._device = None

    @staticmethod
    def _detect_hand_side(skeleton) -> str:
        """Infer the hand side from ``skeleton.header.frame_id``.

        Returns ``"left"`` for ``"l_wrist"`` and ``"right"`` for
        ``"r_wrist"``. Falls back to ``"right"`` when the frame cannot be
        identified.
        """
        frame_id = skeleton.header.frame_id
        if frame_id.startswith("l"):
            return "left"
        return "right"
