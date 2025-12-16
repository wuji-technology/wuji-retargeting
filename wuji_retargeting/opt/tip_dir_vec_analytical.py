"""TipDirVec optimizer with analytical gradients for hand retargeting."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np

from .base import BaseOptimizer, M_TO_CM, TimingStats, huber_loss_np, huber_loss_grad_np


class TipDirVecOptimizerAnalytical(BaseOptimizer):
    """TipDirVec optimizer with analytical (hand-written) gradients.

    Loss function (Huber):
        L = w_pos * L_tip_pos + w_dir * L_tip_dir + w_full_hand_vec * L_full_hand + λ||Δq||²

    Where:
        - L_tip_pos: Huber loss on wrist->tip vectors (5 vectors) - unscaled mediapipe
        - L_tip_dir: Huber loss on tip directions (link4->tip, 5 vectors) - unscaled mediapipe
        - L_full_hand_vec: Huber loss on full hand vectors (15 vectors) - scaled mediapipe

    Uses hand-written analytical gradients + NLopt SLSQP for optimization.
    """

    def __init__(self, config: dict):
        """Initialize TipDirVecOptimizerAnalytical."""
        super().__init__(config)

        # Initialize timing stats
        self._timing = TimingStats()
        self._enable_timing = True

        retarget_config = config.get('retarget', {})

        # TipDirVec parameters
        self.huber_delta_dir = retarget_config.get('huber_delta_dir', 0.5)
        self.w_pos = retarget_config.get('w_pos', 1.0)
        self.w_dir = retarget_config.get('w_dir', 10.0)
        self.w_full_hand_vec = retarget_config.get('w_full_hand_vec', 0.1)

        # Segment scaling for full hand regularization
        segment_scaling_config = retarget_config.get('segment_scaling', {})
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        self.segment_scaling = np.ones((5, 3), dtype=np.float64)
        for i, finger_name in enumerate(finger_names):
            if finger_name in segment_scaling_config:
                self.segment_scaling[i] = np.array(segment_scaling_config[finger_name])

        # Add link1 for finger plane computation
        self.link1_names = [f"finger{i}_link1" for i in range(1, 6)]
        all_link_names = (
            [self.origin_link_name] +
            self.task_link_names +
            self.link3_names +
            self.link4_names +
            self.link1_names
        )
        self.computed_link_names = list(dict.fromkeys(all_link_names))
        self.computed_link_indices = [
            self.robot.get_link_index(name) for name in self.computed_link_names
        ]
        self.link1_indices = [
            self.computed_link_names.index(name) for name in self.link1_names
        ]

    def solve(
        self,
        mediapipe_keypoints: np.ndarray,
        last_qpos: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Solve for joint angles."""
        if self._enable_timing:
            t_total_start = time.perf_counter()
            t_preprocess_start = time.perf_counter()
            self._timing.start_frame()

        mediapipe_keypoints = np.asarray(mediapipe_keypoints, dtype=np.float64)
        if mediapipe_keypoints.shape != (21, 3):
            raise ValueError(f"Expected shape (21, 3), got {mediapipe_keypoints.shape}")

        reg_qpos = self._get_reg_qpos(last_qpos)
        init_qpos = self._get_init_qpos(last_qpos)

        # Compute targets
        # tip_pos and tip_dir use unscaled mediapipe
        target_tip_vectors = self._compute_tip_vectors(mediapipe_keypoints, 1.0)
        target_tip_dirs = self._compute_tip_dirs(mediapipe_keypoints)
        # full_hand uses segment_scaling
        target_full_hand_vectors = self._compute_full_hand_vectors(
            mediapipe_keypoints, self.segment_scaling
        )

        if self._enable_timing:
            self._timing.preprocess_ms += (time.perf_counter() - t_preprocess_start) * 1000
            t_nlopt_start = time.perf_counter()

        objective_fn = self._get_objective_analytical(
            target_tip_vectors, target_tip_dirs, target_full_hand_vectors, reg_qpos
        )
        result = self._run_optimization(objective_fn, init_qpos)

        if self._enable_timing:
            self._timing.nlopt_ms += (time.perf_counter() - t_nlopt_start) * 1000
            self._timing.total_ms += (time.perf_counter() - t_total_start) * 1000
            self._timing.call_count += 1
            self._timing.end_frame(self.opt.get_numevals())

        return result

    def compute_cost(
        self,
        qpos: np.ndarray,
        mediapipe_keypoints: np.ndarray,
    ) -> float:
        """Compute cost for given joint angles."""
        target_tip_vectors = self._compute_tip_vectors(mediapipe_keypoints, 1.0)
        target_tip_dirs = self._compute_tip_dirs(mediapipe_keypoints)
        target_full_hand_vectors = self._compute_full_hand_vectors(
            mediapipe_keypoints, self.segment_scaling
        )
        loss, _ = self._loss_and_grad_analytical(
            qpos, target_tip_vectors, target_tip_dirs, target_full_hand_vectors, None
        )
        return float(loss)

    def _loss_and_grad_analytical(
        self,
        qpos: np.ndarray,
        target_tip_vectors: np.ndarray,
        target_tip_dirs: np.ndarray,
        target_full_hand_vectors: np.ndarray,
        last_qpos: Optional[np.ndarray],
    ) -> tuple[float, np.ndarray]:
        """Compute loss and gradient analytically."""
        qpos = np.asarray(qpos, dtype=np.float64)

        # Forward kinematics
        if self._enable_timing:
            t_fk_start = time.perf_counter()

        self.robot.compute_forward_kinematics(qpos)
        positions = np.array([
            self.robot.get_link_pose(idx)[:3, 3] for idx in self.computed_link_indices
        ], dtype=np.float64) * M_TO_CM

        if self._enable_timing:
            self._timing.fk_ms += (time.perf_counter() - t_fk_start) * 1000
            t_jac_start = time.perf_counter()

        # Get Jacobians (num_links, 3, nq)
        Js = self.robot.compute_all_jacobians_batch(qpos, self.computed_link_indices) * M_TO_CM

        if self._enable_timing:
            self._timing.jacobian_ms += (time.perf_counter() - t_jac_start) * 1000
            t_grad_start = time.perf_counter()

        # Extract positions
        origin_pos = positions[self.origin_indices]  # (5, 3)
        task_pos = positions[self.task_indices]  # (5, 3)
        link3_pos = positions[self.link3_indices]  # (5, 3)
        link4_pos = positions[self.link4_indices]  # (5, 3)
        wrist_pos = positions[self.origin_indices[0]]  # (3,)

        # Get Jacobians
        J_origin = Js[self.origin_indices]  # (5, 3, nq)
        J_task = Js[self.task_indices]  # (5, 3, nq)
        J_link3 = Js[self.link3_indices]  # (5, 3, nq)
        J_link4 = Js[self.link4_indices]  # (5, 3, nq)
        J_wrist = Js[self.origin_indices[0]]  # (3, nq)

        total_loss = 0.0
        total_grad = np.zeros(self.num_joints, dtype=np.float64)

        # === Tip Position Loss (Huber) ===
        robot_tip_vec = task_pos - origin_pos  # (5, 3)
        diff_pos = robot_tip_vec - target_tip_vectors  # (5, 3)
        dist_pos = np.linalg.norm(diff_pos, axis=1)  # (5,)
        loss_tip_pos = huber_loss_np(dist_pos, self.huber_delta)  # (5,)

        # Gradient
        huber_grad_pos = huber_loss_grad_np(dist_pos, self.huber_delta)  # (5,)
        diff_normed_pos = diff_pos / (dist_pos[:, None] + 1e-8)  # (5, 3)
        for i in range(5):
            grad_coeff = self.w_pos * huber_grad_pos[i] / 5.0  # mean over 5 fingers
            J_diff = J_task[i] - J_origin[i]  # (3, nq)
            total_grad += grad_coeff * (diff_normed_pos[i] @ J_diff)

        # === Tip Direction Loss (Huber) ===
        robot_tip_dir_vec = task_pos - link4_pos  # (5, 3)
        robot_tip_dir_norm = np.linalg.norm(robot_tip_dir_vec, axis=1, keepdims=True)  # (5, 1)
        robot_tip_dirs = robot_tip_dir_vec / (robot_tip_dir_norm + 1e-8)  # (5, 3)

        diff_dir = robot_tip_dirs - target_tip_dirs  # (5, 3)
        dist_dir = np.linalg.norm(diff_dir, axis=1)  # (5,)
        loss_tip_dir = huber_loss_np(dist_dir, self.huber_delta_dir)  # (5,)

        # Gradient for normalized direction
        huber_grad_dir = huber_loss_grad_np(dist_dir, self.huber_delta_dir)  # (5,)
        diff_normed_dir = diff_dir / (dist_dir[:, None] + 1e-8)  # (5, 3)
        for i in range(5):
            grad_coeff = self.w_dir * huber_grad_dir[i] / 5.0
            u = robot_tip_dirs[i]  # (3,)
            n = robot_tip_dir_norm[i, 0]  # scalar
            # Jacobian of normalization: (I - u*u^T) / n
            J_norm = (np.eye(3) - np.outer(u, u)) / (n + 1e-8)  # (3, 3)
            J_diff = J_task[i] - J_link4[i]  # (3, nq)
            total_grad += grad_coeff * (diff_normed_dir[i] @ J_norm @ J_diff)

        # === Full Hand Vec Loss (Huber) - regularization ===
        robot_pip_vec = link3_pos - wrist_pos  # (5, 3)
        robot_dip_vec = link4_pos - wrist_pos  # (5, 3)
        robot_tip_vec_full = task_pos - wrist_pos  # (5, 3)

        target_pip = target_full_hand_vectors[:5]
        target_dip = target_full_hand_vectors[5:10]
        target_tip = target_full_hand_vectors[10:15]

        diff_pip = robot_pip_vec - target_pip
        diff_dip = robot_dip_vec - target_dip
        diff_tip = robot_tip_vec_full - target_tip

        dist_pip = np.linalg.norm(diff_pip, axis=1)
        dist_dip = np.linalg.norm(diff_dip, axis=1)
        dist_tip = np.linalg.norm(diff_tip, axis=1)

        loss_pip = huber_loss_np(dist_pip, self.huber_delta)
        loss_dip = huber_loss_np(dist_dip, self.huber_delta)
        loss_tip_full = huber_loss_np(dist_tip, self.huber_delta)
        loss_full_hand = (np.mean(loss_pip) + np.mean(loss_dip) + np.mean(loss_tip_full)) / 3.0

        # Gradients for full hand vectors
        huber_grad_pip = huber_loss_grad_np(dist_pip, self.huber_delta)
        huber_grad_dip = huber_loss_grad_np(dist_dip, self.huber_delta)
        huber_grad_tip = huber_loss_grad_np(dist_tip, self.huber_delta)

        diff_normed_pip = diff_pip / (dist_pip[:, None] + 1e-8)
        diff_normed_dip = diff_dip / (dist_dip[:, None] + 1e-8)
        diff_normed_tip = diff_tip / (dist_tip[:, None] + 1e-8)

        for i in range(5):
            grad_coeff = self.w_full_hand_vec / 3.0 / 5.0  # mean over 3 segments and 5 fingers
            # PIP gradient
            total_grad += grad_coeff * huber_grad_pip[i] * (diff_normed_pip[i] @ (J_link3[i] - J_wrist))
            # DIP gradient
            total_grad += grad_coeff * huber_grad_dip[i] * (diff_normed_dip[i] @ (J_link4[i] - J_wrist))
            # TIP gradient
            total_grad += grad_coeff * huber_grad_tip[i] * (diff_normed_tip[i] @ (J_task[i] - J_wrist))

        # === Total Loss ===
        total_loss = (
            self.w_pos * np.mean(loss_tip_pos) +
            self.w_dir * np.mean(loss_tip_dir) +
            self.w_full_hand_vec * loss_full_hand
        )

        # === Regularization ===
        if last_qpos is not None:
            total_loss += self.norm_delta * np.sum((qpos - last_qpos) ** 2)
            total_grad += 2.0 * self.norm_delta * (qpos - last_qpos)

        if self._enable_timing:
            self._timing.gradient_ms += (time.perf_counter() - t_grad_start) * 1000

        return total_loss, total_grad

    def get_timing_stats(self) -> TimingStats:
        """Get timing statistics."""
        return self._timing

    def reset_timing_stats(self):
        """Reset timing statistics."""
        self._timing.reset()

    def set_timing_enabled(self, enabled: bool):
        """Enable or disable timing instrumentation."""
        self._enable_timing = enabled

    def _get_objective_analytical(
        self,
        target_tip_vectors: np.ndarray,
        target_tip_dirs: np.ndarray,
        target_full_hand_vectors: np.ndarray,
        last_qpos: Optional[np.ndarray],
    ):
        """Create NLopt objective function with analytical gradient."""
        target_tip_vectors = np.asarray(target_tip_vectors, dtype=np.float64)
        target_tip_dirs = np.asarray(target_tip_dirs, dtype=np.float64)
        target_full_hand_vectors = np.asarray(target_full_hand_vectors, dtype=np.float64)
        if last_qpos is not None:
            last_qpos = np.asarray(last_qpos, dtype=np.float64)

        def objective(x, grad_out):
            qpos = np.asarray(x, dtype=np.float64)
            loss, grad = self._loss_and_grad_analytical(
                qpos, target_tip_vectors, target_tip_dirs, target_full_hand_vectors, last_qpos
            )
            if grad_out.size > 0:
                grad_out[:] = grad
            # Record iteration loss for plotting
            if self._enable_timing:
                self._timing.record_iter_loss(float(loss))
            return float(loss)

        return objective
