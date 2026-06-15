"""Multi-layer skeleton drawing for MuJoCo viewer.

Draws three skeleton layers with configurable colors:
- Input (MediaPipe): raw input keypoints
- Scaled Target: keypoints after segment_scaling applied
- Robot FK: actual robot joint positions from forward kinematics
"""

import numpy as np
import mujoco


def _compute_arrow_rotation_matrix(direction: np.ndarray) -> np.ndarray:
    """Compute rotation matrix to align Z-axis with given direction.

    MuJoCo capsule is aligned along Z-axis by default.
    """
    direction = direction / (np.linalg.norm(direction) + 1e-8)
    z_axis = np.array([0, 0, 1])

    if np.abs(np.dot(direction, z_axis)) > 0.999:
        if direction[2] > 0:
            return np.eye(3)
        else:
            return np.diag([1, -1, -1])

    axis = np.cross(z_axis, direction)
    axis = axis / (np.linalg.norm(axis) + 1e-8)
    angle = np.arccos(np.clip(np.dot(z_axis, direction), -1, 1))

    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    return R


# MediaPipe hand skeleton connections
SKELETON_CONNECTIONS = [
    # Thumb
    (0, 1), (1, 2), (2, 3), (3, 4),
    # Index
    (0, 5), (5, 6), (6, 7), (7, 8),
    # Middle
    (0, 9), (9, 10), (10, 11), (11, 12),
    # Ring
    (0, 13), (13, 14), (14, 15), (15, 16),
    # Pinky
    (0, 17), (17, 18), (18, 19), (19, 20),
    # Palm cross-connections
    (5, 9), (9, 13), (13, 17),
]

# Finger index ranges for highlight
FINGER_KEYPOINT_RANGES = {
    "thumb": [1, 2, 3, 4],
    "index": [5, 6, 7, 8],
    "middle": [9, 10, 11, 12],
    "ring": [13, 14, 15, 16],
    "pinky": [17, 18, 19, 20],
}

# MediaPipe index -> robot link name
MEDIAPIPE_TO_ROBOT_LINK = {
    0: "palm_link",
    1: "finger1_link1", 2: "finger1_link2", 3: "finger1_link3", 4: "finger1_link4",
    5: "finger2_link1", 6: "finger2_link2", 7: "finger2_link3", 8: "finger2_link4",
    9: "finger3_link1", 10: "finger3_link2", 11: "finger3_link3", 12: "finger3_link4",
    13: "finger4_link1", 14: "finger4_link2", 15: "finger4_link3", 16: "finger4_link4",
    17: "finger5_link1", 18: "finger5_link2", 19: "finger5_link3", 20: "finger5_link4",
}

# Default layer configuration
DEFAULT_LAYER_CONFIG = {
    "mediapipe_input": {
        "enabled": True,
        "color": [1.0, 0.5, 0.0, 0.6],
        "line_color": [1.0, 0.6, 0.2, 0.5],
        "point_size": 0.004,
        "line_width": 0.002,
    },
    "scaled_target": {
        "enabled": True,
        "color": [0.0, 0.8, 0.8, 0.8],
        "line_color": [0.2, 0.8, 1.0, 0.7],
        "point_size": 0.004,
        "line_width": 0.002,
    },
    "robot_fk": {
        "enabled": True,
        "color": [1.0, 1.0, 1.0, 0.95],
        "line_color": [1.0, 1.0, 1.0, 0.8],
        "point_size": 0.005,
        "line_width": 0.003,
    },
    "highlight_color": [1.0, 0.0, 0.0, 1.0],
}


class SkeletonDrawer:
    """Draws multi-layer hand skeletons in a MuJoCo scene.

    Supports three layers (input/scaled/FK), each with independent
    color and visibility settings driven by YAML configuration.
    """

    TIP_INDICES = [4, 8, 12, 16, 20]

    def __init__(self, model, data, hand_side: str, viz_config: dict = None,
                 link_naming: dict = None):
        """Initialize skeleton drawer.

        Args:
            model: MuJoCo model
            data: MuJoCo data
            hand_side: 'left' or 'right'
            viz_config: Visualization config dict (skeleton section from tuning_viz.yaml)
            link_naming: Optional optimizer.link_naming block. When set (e.g. WH120's
                r_wrist / r_index_finger_* scheme), the MediaPipe->body map and
                the wrist-body lookup are built from it so the overlay matches the
                optimizer's link resolution. Omitting it uses the default WH110 map.
        """
        self.model = model
        self.data = data
        self.hand_side = hand_side.lower()
        self._mp_to_robot, self._wrist_candidates = self._resolve_body_names(link_naming)
        self._build_robot_link_ids()
        self.update_config(viz_config or {})

    def update_config(self, viz_config: dict):
        """Update visualization config (called on hot-reload).

        Args:
            viz_config: The 'skeleton' section from tuning_viz.yaml
        """
        self.layer_configs = {}
        for layer_name in ("mediapipe_input", "scaled_target", "robot_fk"):
            default = DEFAULT_LAYER_CONFIG[layer_name]
            user = viz_config.get(layer_name, {})
            self.layer_configs[layer_name] = {
                "enabled": user.get("enabled", default["enabled"]),
                "color": np.array(
                    user.get("color", default["color"]), dtype=np.float32
                ),
                "line_color": np.array(
                    user.get("line_color", default["line_color"]), dtype=np.float32
                ),
                "point_size": user.get("point_size", default["point_size"]),
                "line_width": user.get("line_width", default["line_width"]),
            }

        hl = viz_config.get("highlight_color", DEFAULT_LAYER_CONFIG["highlight_color"])
        self.highlight_color = np.array(hl, dtype=np.float32)
        self.highlight_indices = set()
        self.draw_skeleton_lines = viz_config.get("draw_lines", True)

    def _resolve_body_names(self, link_naming: dict):
        """Resolve (MediaPipe-index -> body-name map, wrist-body candidates).

        With ``optimizer.link_naming`` (e.g. WH120's anatomical scheme) both are
        built from the templates: MP wrist->palm, and per finger
        MCP->link1, PIP->pip, DIP->dip, TIP->tip. Without it, fall back to the
        default WH110 map + base/palm candidate list.
        """
        if not link_naming:
            wrist_candidates = [
                f"{self.hand_side}_base_link",
                "base_link",
                "hand_base",
                "palm_link",
                f"{self.hand_side}_palm_link",
            ]
            return dict(MEDIAPIPE_TO_ROBOT_LINK), wrist_candidates

        prefix = link_naming.get("prefix", "")
        fingers = link_naming["fingers"]
        palm = prefix + link_naming["palm"]

        def nm(role, finger):
            return prefix + link_naming[role].format(finger=finger)

        mp_to_robot = {0: palm}
        for k, finger in enumerate(fingers):
            mp_to_robot[1 + 4 * k] = nm("link1", finger)  # MCP
            mp_to_robot[2 + 4 * k] = nm("pip", finger)    # PIP
            mp_to_robot[3 + 4 * k] = nm("dip", finger)    # DIP
            mp_to_robot[4 + 4 * k] = nm("tip", finger)    # TIP
        # The MJCF often fuses the wrist into the root body and drops *_tip bodies,
        # so the MANO-frame body may be the configured palm (URDF) OR the root
        # "{prefix}base" (MJCF). Tip MP indices that have no body are skipped.
        return mp_to_robot, [palm, prefix + "base"]

    def _build_robot_link_ids(self):
        """Build mapping from MediaPipe index to MuJoCo body ID."""
        self.robot_body_ids = {}
        for mp_idx, link_name in self._mp_to_robot.items():
            body_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, link_name
            )
            if body_id < 0:
                prefixed_name = f"{self.hand_side}_{link_name}"
                body_id = mujoco.mj_name2id(
                    self.model, mujoco.mjtObj.mjOBJ_BODY, prefixed_name
                )
            if body_id >= 0:
                self.robot_body_ids[mp_idx] = body_id

        # Find MANO-frame body. With link_naming this is the configured palm
        # (e.g. r_wrist); otherwise try base_link variants then palm_link.
        for name in self._wrist_candidates:
            wrist_id = mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, name
            )
            if wrist_id >= 0:
                self.wrist_body_id = wrist_id
                break
        else:
            raise ValueError(
                "Failed to find wrist body for MANO-frame visualization. "
                f"Tried {self._wrist_candidates}"
            )

    def get_robot_keypoints(self) -> np.ndarray:
        """Get robot joint positions from FK (21, 3)."""
        positions = np.full((21, 3), np.nan, dtype=np.float64)
        for mp_idx, body_id in self.robot_body_ids.items():
            positions[mp_idx] = self.data.xpos[body_id].copy()
        return positions

    def get_wrist_transform(self) -> tuple:
        """Get wrist (position, rotation_matrix)."""
        wrist_pos = self.data.xpos[self.wrist_body_id].copy()
        wrist_rot = self.data.xmat[self.wrist_body_id].reshape(3, 3).copy()
        return wrist_pos, wrist_rot

    def set_highlight_fingers(self, finger_names: list):
        """Set which fingers to highlight (e.g., after parameter change).

        Args:
            finger_names: List of finger names ("thumb", "index", etc.)
        """
        self.highlight_indices = set()
        for name in finger_names:
            if name == "all":
                self.highlight_indices = set(range(21))
                return
            indices = FINGER_KEYPOINT_RANGES.get(name, [])
            self.highlight_indices.update(indices)

    def clear_highlight(self):
        """Clear finger highlight."""
        self.highlight_indices = set()

    def draw(
        self,
        scene,
        mediapipe_kp: np.ndarray = None,
        scaled_kp: np.ndarray = None,
        pinch_alphas: np.ndarray = None,
    ):
        """Draw all enabled skeleton layers.

        All keypoints should be in wrist-relative frame.

        Args:
            scene: MuJoCo scene (viewer.user_scn)
            mediapipe_kp: (21, 3) raw MediaPipe keypoints in wrist frame
            scaled_kp: (21, 3) scaled keypoints in wrist frame
            pinch_alphas: (5,) alpha values per finger
        """
        scene.ngeom = 0
        wrist_pos, wrist_rot = self.get_wrist_transform()

        # Layer 1: MediaPipe input (orange)
        if mediapipe_kp is not None and self.layer_configs["mediapipe_input"]["enabled"]:
            world_kp = (mediapipe_kp @ wrist_rot.T) + wrist_pos
            cfg = self.layer_configs["mediapipe_input"]
            if self.draw_skeleton_lines:
                self._draw_lines(scene, world_kp, cfg["line_color"], cfg["line_width"])
            self._draw_points(scene, world_kp, cfg["color"], cfg["point_size"])

        # Layer 2: Scaled target (cyan)
        if scaled_kp is not None and self.layer_configs["scaled_target"]["enabled"]:
            world_kp = (scaled_kp @ wrist_rot.T) + wrist_pos
            cfg = self.layer_configs["scaled_target"]
            if self.draw_skeleton_lines:
                self._draw_lines(scene, world_kp, cfg["line_color"], cfg["line_width"])
            self._draw_points(scene, world_kp, cfg["color"], cfg["point_size"])

        # Layer 3: Robot FK (white)
        if self.layer_configs["robot_fk"]["enabled"]:
            robot_kp = self.get_robot_keypoints()
            cfg = self.layer_configs["robot_fk"]
            if self.draw_skeleton_lines:
                self._draw_lines(scene, robot_kp, cfg["line_color"], cfg["line_width"])
            self._draw_points(scene, robot_kp, cfg["color"], cfg["point_size"])

        # Pinch indicators
        if pinch_alphas is not None and mediapipe_kp is not None:
            world_kp = (mediapipe_kp @ wrist_rot.T) + wrist_pos
            self._draw_pinch_indicators(scene, world_kp, pinch_alphas)

        # MANO coordinate axes at wrist (RGB = XYZ)
        self._draw_mano_axes(scene, wrist_pos, wrist_rot)

    def _draw_points(self, scene, positions, color, size):
        """Draw spheres at keypoint positions."""
        for i in range(len(positions)):
            if scene.ngeom >= scene.maxgeom:
                break
            if not np.all(np.isfinite(positions[i])):
                continue
            # Use highlight color if this keypoint is highlighted
            draw_color = color
            if i in self.highlight_indices:
                draw_color = self.highlight_color

            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([size, 0, 0]),
                positions[i].astype(np.float64),
                np.eye(3).flatten().astype(np.float64),
                draw_color,
            )
            scene.ngeom += 1

    def _draw_lines(self, scene, keypoints, color, width):
        """Draw skeleton connection lines."""
        for start_idx, end_idx in SKELETON_CONNECTIONS:
            if scene.ngeom >= scene.maxgeom:
                break
            start = keypoints[start_idx]
            end = keypoints[end_idx]
            if not (np.all(np.isfinite(start)) and np.all(np.isfinite(end))):
                continue
            mid = (start + end) / 2
            direction = end - start
            length = np.linalg.norm(direction)
            if not np.isfinite(length) or length < 1e-6:
                continue
            direction = direction / length
            rot_matrix = _compute_arrow_rotation_matrix(direction)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.array([width, length / 2, 0]),
                mid.astype(np.float64),
                rot_matrix.flatten().astype(np.float64),
                color,
            )
            scene.ngeom += 1

    def _draw_mano_axes(self, scene, wrist_pos, wrist_rot, length=0.04, width=0.0015):
        """Draw RGB XYZ axes at wrist for MANO frame reference.

        Right hand: +x into palm, +y thumb side, +z finger direction.
        Left hand:  +x out of palm, +y small-finger side, +z finger direction.
        """
        colors = [
            np.array([1.0, 0.0, 0.0, 0.9], dtype=np.float32),  # X: red
            np.array([0.0, 1.0, 0.0, 0.9], dtype=np.float32),  # Y: green
            np.array([0.0, 0.0, 1.0, 0.9], dtype=np.float32),  # Z: blue
        ]
        for axis_idx in range(3):
            if scene.ngeom >= scene.maxgeom:
                break
            direction_world = wrist_rot[:, axis_idx]
            end = wrist_pos + direction_world * length
            mid = (wrist_pos + end) / 2
            rot_matrix = _compute_arrow_rotation_matrix(direction_world)
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                np.array([width, length / 2, 0]),
                mid.astype(np.float64),
                rot_matrix.flatten().astype(np.float64),
                colors[axis_idx],
            )
            scene.ngeom += 1

    def _draw_pinch_indicators(self, scene, mediapipe_world, pinch_alphas):
        """Draw red spheres at fingertips proportional to pinch alpha."""
        for i, tip_idx in enumerate(self.TIP_INDICES):
            if scene.ngeom >= scene.maxgeom:
                break
            alpha = pinch_alphas[i]
            if alpha > 0.01 and np.all(np.isfinite(mediapipe_world[tip_idx])):
                red_color = np.array([1.0, 0.0, 0.0, alpha * 0.8], dtype=np.float32)
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom],
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.004, 0, 0]),
                    mediapipe_world[tip_idx].astype(np.float64),
                    np.eye(3).flatten().astype(np.float64),
                    red_color,
                )
                scene.ngeom += 1
