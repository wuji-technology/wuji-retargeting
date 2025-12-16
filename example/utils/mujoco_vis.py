"""MuJoCo visualization utilities for teleoperation."""

import numpy as np
import mujoco


class HandVisualization:
    """Persistent visualization with pre-allocated geometries for debug.

    Visualizes 4 types of keypoints:
    - MediaPipe input keypoints (21 points): light transparent green
    - Scaled MediaPipe positions (21 points): solid green
    - Robot FK positions (6 points): solid red
    - URDF tip points (5 points): purple
    """

    def __init__(self, scene):
        """Create visualization geometries once (59 spheres total)."""
        # Colors for each keypoint type
        self.mediapipe_color = np.array([0.0, 1.0, 0.0, 0.2])       # light transparent green
        self.scaled_color = np.array([0.0, 1.0, 0.0, 1.0])          # solid green
        self.robot_color = np.array([1.0, 0.0, 0.0, 1.0])           # solid red
        self.urdf_tip_color = np.array([0.5, 0.0, 0.5, 1.0])        # purple

        self.mediapipe_geoms = []        # 21 spheres
        self.scaled_geoms = []           # 21 spheres
        self.robot_geoms = []            # 6 spheres
        self.urdf_tip_geoms = []         # 5 spheres

        # Create 21 sphere geometries for MediaPipe keypoints
        for _ in range(21):
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.002, 0, 0]),
                np.zeros(3),
                np.eye(3).flatten(),
                self.mediapipe_color
            )
            self.mediapipe_geoms.append(scene.ngeom)
            scene.ngeom += 1

        # Create 21 sphere geometries for scaled MediaPipe positions
        for _ in range(21):
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.002, 0, 0]),
                np.zeros(3),
                np.eye(3).flatten(),
                self.scaled_color
            )
            self.scaled_geoms.append(scene.ngeom)
            scene.ngeom += 1

        # Create 6 sphere geometries for robot FK positions (solid red)
        for _ in range(6):
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.002, 0, 0]),
                np.zeros(3),
                np.eye(3).flatten(),
                self.robot_color
            )
            self.robot_geoms.append(scene.ngeom)
            scene.ngeom += 1

        # Create 5 sphere geometries for URDF tip points (purple)
        for _ in range(5):
            if scene.ngeom >= scene.maxgeom:
                break
            mujoco.mjv_initGeom(
                scene.geoms[scene.ngeom],
                mujoco.mjtGeom.mjGEOM_SPHERE,
                np.array([0.002, 0, 0]),
                np.zeros(3),
                np.eye(3).flatten(),
                self.urdf_tip_color
            )
            self.urdf_tip_geoms.append(scene.ngeom)
            scene.ngeom += 1

        self.final_ngeom = scene.ngeom

    def update(self, scene, mediapipe_pos, scaled_pos, robot_pos, urdf_tip_pos=None):
        """Update geometry positions with new keypoints.

        Args:
            scene: MuJoCo scene
            mediapipe_pos: (21, 3) Full MediaPipe keypoints
            scaled_pos: (21, 3) Scaled MediaPipe positions
            robot_pos: (6, 3) Robot FK body positions
            urdf_tip_pos: (5, 3) Optional URDF tip positions (purple spheres)
        """
        mediapipe_pos = np.asarray(mediapipe_pos, dtype=np.float32)
        scaled_pos = np.asarray(scaled_pos, dtype=np.float32)
        robot_pos = np.asarray(robot_pos, dtype=np.float32)

        # Update MediaPipe sphere positions
        for i, geom_idx in enumerate(self.mediapipe_geoms):
            if i < len(mediapipe_pos):
                scene.geoms[geom_idx].pos[:] = mediapipe_pos[i]

        # Update scaled MediaPipe sphere positions
        for i, geom_idx in enumerate(self.scaled_geoms):
            if i < len(scaled_pos):
                scene.geoms[geom_idx].pos[:] = scaled_pos[i]

        # Update robot FK sphere positions
        for i, geom_idx in enumerate(self.robot_geoms):
            if i < len(robot_pos):
                scene.geoms[geom_idx].pos[:] = robot_pos[i]

        # Update URDF tip sphere positions
        if urdf_tip_pos is not None:
            urdf_tip_pos = np.asarray(urdf_tip_pos, dtype=np.float32)
            for i, geom_idx in enumerate(self.urdf_tip_geoms):
                if i < len(urdf_tip_pos):
                    scene.geoms[geom_idx].pos[:] = urdf_tip_pos[i]

        # Restore scene.ngeom (update_scene() resets it)
        scene.ngeom = self.final_ngeom
