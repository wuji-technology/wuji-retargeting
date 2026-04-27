"""Key-vector-based retargeting optimizer."""

from __future__ import annotations

from typing import List, Optional

import numpy as np

from .base import BaseOptimizer, M_TO_CM, huber_loss_np, huber_loss_grad_np


class VectorOptimizer(BaseOptimizer):
    """Key-vector-based retargeting optimizer.

    Minimizes distances between corresponding human-robot vector pairs.
    Each vector is defined by an (origin_link, task_link) pair on the robot,
    matched to a corresponding (origin_kp, task_kp) pair of MediaPipe keypoints.

    Reference:
        arxiv 2506.11916, dex-retargeting VectorOptimizer

    Loss:
        L(q) = (1/N) * sum_i Huber( ||robot_vec[i] - scale[i] * target_vec[i]|| )
             + norm_delta * ||q - q_prev||^2

        robot_vec[i]  = FK(task_link[i]) - FK(origin_link[i])
        target_vec[i] = mp_kp[task_kp[i]] - mp_kp[origin_kp[i]]
    """

    def __init__(self, config: dict):
        super().__init__(config)

        retarget_config = config.get('retarget', {})

        kv_config = retarget_config.get('key_vectors', self._default_key_vectors())
        if not kv_config:
            raise ValueError("key_vectors must be a non-empty list")

        MP_KP_MAX = 20  # MediaPipe hand: 21 keypoints indexed 0..20
        for i, kv in enumerate(kv_config):
            for k in ('origin', 'task', 'origin_kp', 'task_kp'):
                if k not in kv:
                    raise ValueError(f"key_vectors[{i}] missing required field '{k}'")
            if not isinstance(kv['origin'], str) or not isinstance(kv['task'], str):
                raise ValueError(f"key_vectors[{i}]: 'origin' and 'task' must be strings")
            for k in ('origin_kp', 'task_kp'):
                if not isinstance(kv[k], int) or not (0 <= kv[k] <= MP_KP_MAX):
                    raise ValueError(
                        f"key_vectors[{i}]: '{k}' must be int in [0, {MP_KP_MAX}], got {kv[k]!r}"
                    )
            if 'scale' in kv and not isinstance(kv['scale'], (int, float)):
                raise ValueError(
                    f"key_vectors[{i}]: 'scale' must be numeric, got {type(kv['scale']).__name__}"
                )

        self._origin_link_names_kv: List[str] = [kv['origin'] for kv in kv_config]
        self._task_link_names_kv: List[str]   = [kv['task']   for kv in kv_config]
        self._origin_kp_indices = np.array([kv['origin_kp'] for kv in kv_config], dtype=int)
        self._task_kp_indices   = np.array([kv['task_kp']   for kv in kv_config], dtype=int)
        self._vector_scalings   = np.array([kv.get('scale', 1.0) for kv in kv_config], dtype=np.float64)
        self.num_vectors        = len(kv_config)

        # Deduplicate links to avoid redundant FK/Jacobian computations.
        # palm_link appears as origin for all 15 vectors — compute it only once.
        all_link_names = list(dict.fromkeys(
            self._origin_link_names_kv + self._task_link_names_kv
        ))
        self._kv_computed_link_names   = all_link_names
        self._kv_computed_link_indices = [
            self.robot.get_link_index(name) for name in all_link_names
        ]
        self._kv_origin_indices = np.array(
            [all_link_names.index(name) for name in self._origin_link_names_kv], dtype=int
        )
        self._kv_task_indices = np.array(
            [all_link_names.index(name) for name in self._task_link_names_kv], dtype=int
        )

    # ------------------------------------------------------------------
    # Default 15-vector configuration
    # ------------------------------------------------------------------

    @staticmethod
    def _default_key_vectors() -> list:
        """15 vectors: wrist -> {link3, link4, tip_link} for each finger.

        MediaPipe keypoint indices:
            0: wrist
            Thumb:  2=mcp, 3=ip,  4=tip
            Index:  6=pip, 7=dip,  8=tip
            Middle: 10=pip,11=dip, 12=tip
            Ring:   14=pip,15=dip, 16=tip
            Pinky:  18=pip,19=dip, 20=tip
        """
        return [
            # thumb (finger1)
            {'origin': 'palm_link', 'task': 'finger1_link3',    'origin_kp': 0, 'task_kp':  2, 'scale': 1.0},
            {'origin': 'palm_link', 'task': 'finger1_link4',    'origin_kp': 0, 'task_kp':  3, 'scale': 1.0},
            {'origin': 'palm_link', 'task': 'finger1_tip_link', 'origin_kp': 0, 'task_kp':  4, 'scale': 1.0},
            # index (finger2)
            {'origin': 'palm_link', 'task': 'finger2_link3',    'origin_kp': 0, 'task_kp':  6, 'scale': 1.0},
            {'origin': 'palm_link', 'task': 'finger2_link4',    'origin_kp': 0, 'task_kp':  7, 'scale': 1.0},
            {'origin': 'palm_link', 'task': 'finger2_tip_link', 'origin_kp': 0, 'task_kp':  8, 'scale': 1.0},
            # middle (finger3)
            {'origin': 'palm_link', 'task': 'finger3_link3',    'origin_kp': 0, 'task_kp': 10, 'scale': 1.0},
            {'origin': 'palm_link', 'task': 'finger3_link4',    'origin_kp': 0, 'task_kp': 11, 'scale': 1.0},
            {'origin': 'palm_link', 'task': 'finger3_tip_link', 'origin_kp': 0, 'task_kp': 12, 'scale': 1.0},
            # ring (finger4)
            {'origin': 'palm_link', 'task': 'finger4_link3',    'origin_kp': 0, 'task_kp': 14, 'scale': 1.0},
            {'origin': 'palm_link', 'task': 'finger4_link4',    'origin_kp': 0, 'task_kp': 15, 'scale': 1.0},
            {'origin': 'palm_link', 'task': 'finger4_tip_link', 'origin_kp': 0, 'task_kp': 16, 'scale': 1.0},
            # pinky (finger5)
            {'origin': 'palm_link', 'task': 'finger5_link3',    'origin_kp': 0, 'task_kp': 18, 'scale': 1.0},
            {'origin': 'palm_link', 'task': 'finger5_link4',    'origin_kp': 0, 'task_kp': 19, 'scale': 1.0},
            {'origin': 'palm_link', 'task': 'finger5_tip_link', 'origin_kp': 0, 'task_kp': 20, 'scale': 1.0},
        ]

    # ------------------------------------------------------------------
    # Target vector computation
    # ------------------------------------------------------------------

    def _compute_target_vectors(self, keypoints: np.ndarray) -> np.ndarray:
        """Compute scaled target vectors from MediaPipe keypoints.

        Args:
            keypoints: (21, 3) MediaPipe keypoints in wrist frame (meters)

        Returns:
            (N, 3) target vectors in cm
        """
        origin_kp = keypoints[self._origin_kp_indices]  # (N, 3)
        task_kp   = keypoints[self._task_kp_indices]    # (N, 3)
        vecs = (task_kp - origin_kp) * self._vector_scalings[:, None] * M_TO_CM
        return vecs.astype(np.float64)

    # ------------------------------------------------------------------
    # Loss and analytical gradient
    # ------------------------------------------------------------------

    def _loss_and_grad(
        self,
        qpos: np.ndarray,
        target_vectors: np.ndarray,
        last_qpos: Optional[np.ndarray],
    ) -> tuple[float, np.ndarray]:
        """Compute loss and analytical gradient."""
        qpos = np.asarray(qpos, dtype=np.float64)

        # --- Forward kinematics ---
        self.robot.compute_forward_kinematics(qpos)
        positions = np.array([
            self.robot.get_link_pose(idx)[:3, 3]
            for idx in self._kv_computed_link_indices
        ], dtype=np.float64) * M_TO_CM

        # --- Batch Jacobians: (num_unique_links, 3, nq) ---
        Js = self.robot.compute_all_jacobians_batch(
            qpos, self._kv_computed_link_indices
        ) * M_TO_CM

        # --- Per-vector positions and Jacobians ---
        origin_pos = positions[self._kv_origin_indices]  # (N, 3)
        task_pos   = positions[self._kv_task_indices]    # (N, 3)
        J_origin   = Js[self._kv_origin_indices]         # (N, 3, nq)
        J_task     = Js[self._kv_task_indices]           # (N, 3, nq)

        # --- Vector residuals ---
        robot_vec = task_pos - origin_pos
        diff      = robot_vec - target_vectors
        dist      = np.linalg.norm(diff, axis=1)

        # --- Loss: mean Huber ---
        total_loss = np.mean(huber_loss_np(dist, self.huber_delta))

        # --- Gradient ---
        huber_grad  = huber_loss_grad_np(dist, self.huber_delta)
        diff_normed = diff / (dist[:, None] + 1e-8)

        total_grad = np.zeros(self.num_joints, dtype=np.float64)
        N = self.num_vectors
        for i in range(N):
            J_diff = J_task[i] - J_origin[i]
            total_grad += (huber_grad[i] / N) * (diff_normed[i] @ J_diff)

        # --- Regularization ---
        if last_qpos is not None:
            total_loss += self.norm_delta * np.sum((qpos - last_qpos) ** 2)
            total_grad += 2.0 * self.norm_delta * (qpos - last_qpos)

        return total_loss, total_grad

    # ------------------------------------------------------------------
    # NLopt objective factory
    # ------------------------------------------------------------------

    def _get_objective(self, target_vectors: np.ndarray, last_qpos: Optional[np.ndarray]):
        target_vectors = np.asarray(target_vectors, dtype=np.float64)
        if last_qpos is not None:
            last_qpos = np.asarray(last_qpos, dtype=np.float64)

        def objective(x: np.ndarray, grad_out: np.ndarray) -> float:
            loss, grad = self._loss_and_grad(
                np.asarray(x, dtype=np.float64), target_vectors, last_qpos
            )
            if grad_out.size > 0:
                grad_out[:] = grad
            return float(loss)

        return objective

    # ------------------------------------------------------------------
    # BaseOptimizer interface
    # ------------------------------------------------------------------

    def solve(
        self,
        mediapipe_keypoints: np.ndarray,
        last_qpos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        mediapipe_keypoints = np.asarray(mediapipe_keypoints, dtype=np.float64)
        if mediapipe_keypoints.shape != (21, 3):
            raise ValueError(f"Expected shape (21, 3), got {mediapipe_keypoints.shape}")

        target_vectors = self._compute_target_vectors(mediapipe_keypoints)
        reg_qpos  = self._get_reg_qpos(last_qpos)
        init_qpos = self._get_init_qpos(last_qpos)

        objective_fn = self._get_objective(target_vectors, reg_qpos)
        return self._run_optimization(objective_fn, init_qpos)

    def compute_cost(self, qpos: np.ndarray, mediapipe_keypoints: np.ndarray) -> float:
        target_vectors = self._compute_target_vectors(
            np.asarray(mediapipe_keypoints, dtype=np.float64)
        )
        loss, _ = self._loss_and_grad(
            np.asarray(qpos, dtype=np.float64), target_vectors, None
        )
        return float(loss)
