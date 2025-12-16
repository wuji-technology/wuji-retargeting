"""TipDirVec optimizer using QP solver for hand retargeting."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import quadprog

from .base import BaseOptimizer, M_TO_CM, TimingStats


class TipDirVecOptimizerQP(BaseOptimizer):
    """TipDirVec optimizer using QP solver with Gauss-Newton method.

    Loss function (L2):
        L = w_pos * L_tip_pos + w_dir * L_tip_dir + w_full_hand_vec * L_full_hand + λ||Δq||²

    Where:
        - L_tip_pos: L2 loss on wrist->tip vectors (5 vectors)
        - L_tip_dir: L2 loss on tip directions (link4->tip, 5 vectors)
        - L_full_hand_vec: L2 loss on full hand vectors (15 vectors) as regularization
    """

    def __init__(self, config: dict):
        """Initialize TipDirVecOptimizerQP."""
        super().__init__(config)

        # Initialize timing stats
        self._timing = TimingStats()
        self._enable_timing = True

        retarget_config = config.get('retarget', {})

        # TipDirVec parameters
        self.w_pos = retarget_config.get('w_pos', 1.0)
        self.w_dir = retarget_config.get('w_dir', 10.0)
        self.w_full_hand_vec = retarget_config.get('w_full_hand_vec', 0.1)
        self.scaling = retarget_config.get('scaling', 1.0)

        # Segment scaling for full hand regularization
        segment_scaling_config = retarget_config.get('segment_scaling', {})
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        self.segment_scaling = np.ones((5, 3), dtype=np.float64)
        for i, finger_name in enumerate(finger_names):
            if finger_name in segment_scaling_config:
                self.segment_scaling[i] = np.array(segment_scaling_config[finger_name])

        # QP solver settings
        self.max_iters = int(retarget_config.get('qp_max_iters', 10))
        self.tol = float(retarget_config.get('qp_tol', 1e-4))

        # Add link1 for finger plane computation (kept for compatibility)
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
        """Solve for joint angles using QP."""
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
        # tip_vectors and tip_dirs use unscaled mediapipe (scaling=1.0)
        target_tip_vectors = self._compute_tip_vectors(mediapipe_keypoints, 1.0)
        target_tip_dirs = self._compute_tip_dirs(mediapipe_keypoints)
        # full_hand_vectors use segment_scaling for regularization
        target_full_hand_vectors = self._compute_full_hand_vectors(
            mediapipe_keypoints, self.segment_scaling
        )

        if self._enable_timing:
            self._timing.preprocess_ms += (time.perf_counter() - t_preprocess_start) * 1000
            t_qp_start = time.perf_counter()

        # Run iterative QP solve (Gauss-Newton)
        qpos = init_qpos.copy()
        num_iters = 0

        for _ in range(self.max_iters):
            num_iters += 1
            qpos_new = self._qp_step(
                qpos, target_tip_vectors, target_tip_dirs,
                target_full_hand_vectors, reg_qpos
            )

            # Check convergence
            delta = np.linalg.norm(qpos_new - qpos)
            qpos = qpos_new

            if self._enable_timing:
                loss = self._compute_loss(
                    qpos, target_tip_vectors, target_tip_dirs,
                    target_full_hand_vectors, reg_qpos
                )
                self._timing.record_iter_loss(float(loss))

            if delta < self.tol:
                break

        if self._enable_timing:
            self._timing.nlopt_ms += (time.perf_counter() - t_qp_start) * 1000
            self._timing.total_ms += (time.perf_counter() - t_total_start) * 1000
            self._timing.call_count += 1
            self._timing.end_frame(num_iters)

        self.last_qpos = qpos.astype(np.float64)
        return qpos.astype(np.float32)

    def _qp_step(
        self,
        qpos: np.ndarray,
        target_tip_vectors: np.ndarray,
        target_tip_dirs: np.ndarray,
        target_full_hand_vectors: np.ndarray,
        reg_qpos: Optional[np.ndarray],
    ) -> np.ndarray:
        """Perform one QP step using Gauss-Newton linearization."""
        if self._enable_timing:
            t_fk_start = time.perf_counter()

        # Forward kinematics
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
            t_qp_start = time.perf_counter()

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

        nq = self.num_joints

        # Build stacked Jacobian and residual for QP
        J_stack = []
        r_stack = []

        for i in range(5):
            # === Tip Position ===
            w_tip_pos = np.sqrt(self.w_pos)
            J_tip_pos = (J_task[i] - J_origin[i]) * w_tip_pos  # (3, nq)
            r_tip_pos = ((task_pos[i] - origin_pos[i]) - target_tip_vectors[i]) * w_tip_pos  # (3,)
            J_stack.append(J_tip_pos)
            r_stack.append(r_tip_pos)

            # === Tip Direction ===
            w_tip_dir = np.sqrt(self.w_dir)
            robot_tip_dir_vec = task_pos[i] - link4_pos[i]  # (3,)
            robot_tip_dir_norm = np.linalg.norm(robot_tip_dir_vec) + 1e-8
            robot_tip_dir = robot_tip_dir_vec / robot_tip_dir_norm  # (3,)

            # Jacobian of normalization: (I - u*u^T) / ||v|| @ (J_task - J_link4)
            J_norm = (np.eye(3) - np.outer(robot_tip_dir, robot_tip_dir)) / robot_tip_dir_norm
            J_tip_dir = (J_norm @ (J_task[i] - J_link4[i])) * w_tip_dir  # (3, nq)
            r_tip_dir = (robot_tip_dir - target_tip_dirs[i]) * w_tip_dir  # (3,)
            J_stack.append(J_tip_dir)
            r_stack.append(r_tip_dir)

            # === Full Hand Vectors (regularization) ===
            w_full = np.sqrt(self.w_full_hand_vec / 3.0)

            # PIP: link3 - wrist
            J_pip = (J_link3[i] - J_wrist) * w_full
            r_pip = ((link3_pos[i] - wrist_pos) - target_full_hand_vectors[i]) * w_full
            J_stack.append(J_pip)
            r_stack.append(r_pip)

            # DIP: link4 - wrist
            J_dip = (J_link4[i] - J_wrist) * w_full
            r_dip = ((link4_pos[i] - wrist_pos) - target_full_hand_vectors[5 + i]) * w_full
            J_stack.append(J_dip)
            r_stack.append(r_dip)

            # TIP: task - wrist
            J_tip = (J_task[i] - J_wrist) * w_full
            r_tip = ((task_pos[i] - wrist_pos) - target_full_hand_vectors[10 + i]) * w_full
            J_stack.append(J_tip)
            r_stack.append(r_tip)

        # Stack all
        J_all = np.vstack(J_stack)  # (75, nq)
        r_all = np.concatenate(r_stack)  # (75,)

        # QP formulation
        JtJ = J_all.T @ J_all  # (nq, nq)
        Jtr = J_all.T @ r_all  # (nq,)

        # P = J^T*J + λI
        P = JtJ + self.norm_delta * np.eye(nq)

        # c = J^T*r - λ*(q_reg - q_current)
        if reg_qpos is not None:
            c = Jtr - self.norm_delta * (reg_qpos - qpos)
        else:
            c = Jtr

        # Solve QP with box constraints
        lb = self.robot.joint_limits[:, 0]
        ub = self.robot.joint_limits[:, 1]

        C = np.vstack([np.eye(nq), -np.eye(nq)]).T  # (nq, 2*nq)
        b_ineq = np.concatenate([lb, -ub])  # (2*nq,)

        try:
            P_reg = P + 1e-8 * np.eye(nq)
            dq, _, _, _, _, _ = quadprog.solve_qp(P_reg, -c, C, b_ineq)
            qpos_new = qpos + dq
        except ValueError:
            qpos_new = qpos - 0.1 * c
            qpos_new = np.clip(qpos_new, lb, ub)

        if self._enable_timing:
            self._timing.gradient_ms += (time.perf_counter() - t_qp_start) * 1000

        return qpos_new

    def _compute_loss(
        self,
        qpos: np.ndarray,
        target_tip_vectors: np.ndarray,
        target_tip_dirs: np.ndarray,
        target_full_hand_vectors: np.ndarray,
        reg_qpos: Optional[np.ndarray],
    ) -> float:
        """Compute L2 loss for monitoring."""
        # Forward kinematics
        self.robot.compute_forward_kinematics(qpos)
        positions = np.array([
            self.robot.get_link_pose(idx)[:3, 3] for idx in self.computed_link_indices
        ], dtype=np.float64) * M_TO_CM

        # Extract positions
        origin_pos = positions[self.origin_indices]
        task_pos = positions[self.task_indices]
        link3_pos = positions[self.link3_indices]
        link4_pos = positions[self.link4_indices]
        wrist_pos = positions[self.origin_indices[0]]

        total_loss = 0.0

        for i in range(5):
            # Tip position loss
            robot_tip_vec = task_pos[i] - origin_pos[i]
            diff_pos = robot_tip_vec - target_tip_vectors[i]
            loss_tip_pos = 0.5 * np.sum(diff_pos ** 2)

            # Tip direction loss
            robot_tip_dir_vec = task_pos[i] - link4_pos[i]
            robot_tip_dir = robot_tip_dir_vec / (np.linalg.norm(robot_tip_dir_vec) + 1e-8)
            diff_dir = robot_tip_dir - target_tip_dirs[i]
            loss_tip_dir = 0.5 * np.sum(diff_dir ** 2)

            # Full hand loss
            pip_diff = (link3_pos[i] - wrist_pos) - target_full_hand_vectors[i]
            dip_diff = (link4_pos[i] - wrist_pos) - target_full_hand_vectors[5 + i]
            tip_diff = (task_pos[i] - wrist_pos) - target_full_hand_vectors[10 + i]
            loss_full = (0.5 * np.sum(pip_diff ** 2) +
                         0.5 * np.sum(dip_diff ** 2) +
                         0.5 * np.sum(tip_diff ** 2)) / 3.0

            total_loss += self.w_pos * loss_tip_pos + self.w_dir * loss_tip_dir + self.w_full_hand_vec * loss_full

        # Regularization
        if reg_qpos is not None:
            total_loss += self.norm_delta * np.sum((qpos - reg_qpos) ** 2)

        return total_loss

    def compute_cost(
        self,
        qpos: np.ndarray,
        mediapipe_keypoints: np.ndarray,
    ) -> float:
        """Compute cost for given joint angles."""
        # tip_vectors and tip_dirs use unscaled mediapipe
        target_tip_vectors = self._compute_tip_vectors(mediapipe_keypoints, 1.0)
        target_tip_dirs = self._compute_tip_dirs(mediapipe_keypoints)
        # full_hand_vectors use segment_scaling
        target_full_hand_vectors = self._compute_full_hand_vectors(
            mediapipe_keypoints, self.segment_scaling
        )
        return self._compute_loss(
            qpos, target_tip_vectors, target_tip_dirs,
            target_full_hand_vectors, None
        )

    def get_timing_stats(self) -> TimingStats:
        """Get timing statistics."""
        return self._timing

    def reset_timing_stats(self):
        """Reset timing statistics."""
        self._timing.reset()

    def set_timing_enabled(self, enabled: bool):
        """Enable or disable timing instrumentation."""
        self._enable_timing = enabled
