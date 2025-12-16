"""Adaptive optimizer using QP solver for hand retargeting."""

from __future__ import annotations

import time
from typing import Optional

import numpy as np
import quadprog

from .base import BaseOptimizer, M_TO_CM, TimingStats


class AdaptiveOptimizerQP(BaseOptimizer):
    """Adaptive optimizer using QP solver.

    Same loss function as AdaptiveOptimizerL2 but solved using QP formulation.
    The L2 loss is purely quadratic, making it suitable for QP:

        min 0.5 * ||residual(q)||^2 + 0.5 * λ||q - q_prev||^2

    This is reformulated as standard QP:
        min 0.5 * q^T * P * q + c^T * q
        s.t. lb <= q <= ub

    Using Gauss-Newton linearization:
        residual(q) ≈ residual(q0) + J * (q - q0)
        ||residual(q)||^2 ≈ ||J*q - (J*q0 - r0)||^2 = ||J*q - b||^2

    Where:
        P = J^T @ J + λI
        c = -J^T @ b - λ * q_prev
    """

    def __init__(self, config: dict):
        """Initialize AdaptiveOptimizerQP."""
        super().__init__(config)

        # Initialize timing stats
        self._timing = TimingStats()
        self._enable_timing = True

        retarget_config = config.get('retarget', {})

        # TipDirVec parameters
        self.w_pos = retarget_config.get('w_pos', 1.0)
        self.w_dir = retarget_config.get('w_dir', 10.0)
        self.w_full_hand_reg = retarget_config.get('w_full_hand_reg', 0.1)  # Full hand regularization for TipDirVec mode
        self.scaling = retarget_config.get('scaling', 1.0)
        self.project_tip_dir = retarget_config.get('project_tip_dir', False)

        # FullHandVec parameters
        self.w_full_hand = retarget_config.get('w_full_hand', 1.0)
        segment_scaling_config = retarget_config.get('segment_scaling', {})
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        self.segment_scaling = np.ones((5, 3), dtype=np.float64)
        for i, finger_name in enumerate(finger_names):
            if finger_name in segment_scaling_config:
                self.segment_scaling[i] = np.array(segment_scaling_config[finger_name])

        # Pinch thresholds
        pinch_config = retarget_config.get('pinch_thresholds', {})
        self.d1 = np.array([
            pinch_config.get('index', {}).get('d1', 2.0),
            pinch_config.get('middle', {}).get('d1', 2.0),
            pinch_config.get('ring', {}).get('d1', 2.0),
            pinch_config.get('pinky', {}).get('d1', 2.0),
        ], dtype=np.float64)
        self.d2 = np.array([
            pinch_config.get('index', {}).get('d2', 4.0),
            pinch_config.get('middle', {}).get('d2', 4.0),
            pinch_config.get('ring', {}).get('d2', 4.0),
            pinch_config.get('pinky', {}).get('d2', 4.0),
        ], dtype=np.float64)

        # QP solver settings
        self.max_iters = int(retarget_config.get('qp_max_iters', 10))
        self.tol = float(retarget_config.get('qp_tol', 1e-4))

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

    def _compute_pinch_alpha(self, mediapipe_keypoints: np.ndarray) -> np.ndarray:
        """Compute alpha weights for each finger."""
        thumb_tip = mediapipe_keypoints[self.MP_TIP_INDICES[0]]
        finger_tips = mediapipe_keypoints[self.MP_TIP_INDICES[1:]]
        distances = np.linalg.norm(finger_tips - thumb_tip, axis=1) * M_TO_CM
        alphas_4 = np.clip((self.d2 - distances) / (self.d2 - self.d1 + 1e-8), 0.0, 1.0)
        alpha_thumb = np.max(alphas_4)
        return np.concatenate([[alpha_thumb], alphas_4])

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

        alphas = self._compute_pinch_alpha(mediapipe_keypoints)
        target_tip_vectors = self._compute_tip_vectors(mediapipe_keypoints, self.scaling)
        target_tip_dirs = self._compute_tip_dirs(mediapipe_keypoints)
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
                target_full_hand_vectors, alphas, reg_qpos
            )

            # Check convergence
            delta = np.linalg.norm(qpos_new - qpos)
            qpos = qpos_new

            if self._enable_timing:
                loss = self._compute_loss(
                    qpos, target_tip_vectors, target_tip_dirs,
                    target_full_hand_vectors, alphas, reg_qpos
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
        alphas: np.ndarray,
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
        # Total residual dimension: 5*(3 + 3 + 3*3) = 5*(3 + 3 + 9) = 75
        # - tip_pos: 5*3 = 15 (weighted by sqrt(alpha * w_pos))
        # - tip_dir: 5*3 = 15 (weighted by sqrt(alpha * w_dir))
        # - full_hand: 5*9 = 45 (weighted by sqrt((1-alpha) * w_full_hand / 3))

        J_stack = []
        r_stack = []

        for i in range(5):
            # === Tip Position (weighted) ===
            w_tip_pos = np.sqrt(alphas[i] * self.w_pos)
            # residual = robot_tip_vec - target_tip_vec
            # robot_tip_vec = task_pos - origin_pos
            # d(robot_tip_vec)/dq = J_task - J_origin
            J_tip_pos = (J_task[i] - J_origin[i]) * w_tip_pos  # (3, nq)
            r_tip_pos = ((task_pos[i] - origin_pos[i]) - target_tip_vectors[i]) * w_tip_pos  # (3,)
            J_stack.append(J_tip_pos)
            r_stack.append(r_tip_pos)

            # === Tip Direction (weighted) ===
            w_tip_dir = np.sqrt(alphas[i] * self.w_dir)
            # robot_tip_dir = normalized(task_pos - link4_pos)
            # For direction, we linearize around current robot_tip_dir
            robot_tip_dir_vec = task_pos[i] - link4_pos[i]  # (3,)
            robot_tip_dir_norm = np.linalg.norm(robot_tip_dir_vec) + 1e-8
            robot_tip_dir = robot_tip_dir_vec / robot_tip_dir_norm  # (3,)

            # Jacobian of normalization: (I - u*u^T) / ||v|| @ (J_task - J_link4)
            J_norm = (np.eye(3) - np.outer(robot_tip_dir, robot_tip_dir)) / robot_tip_dir_norm
            J_tip_dir = (J_norm @ (J_task[i] - J_link4[i])) * w_tip_dir  # (3, nq)
            r_tip_dir = (robot_tip_dir - target_tip_dirs[i]) * w_tip_dir  # (3,)
            J_stack.append(J_tip_dir)
            r_stack.append(r_tip_dir)

            # === Full Hand Vectors (weighted) ===
            # Combine: (1-alpha)*w_full_hand for FullHandVec mode + alpha*w_full_hand_reg for TipDirVec regularization
            w_full_combined = (1.0 - alphas[i]) * self.w_full_hand + alphas[i] * self.w_full_hand_reg
            w_full = np.sqrt(w_full_combined / 3.0)

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

        # QP formulation: min 0.5 * ||J*q - b||^2 + 0.5 * λ||q - q_reg||^2
        # where b = J*q0 - r0 (Gauss-Newton target)
        # Expanding: 0.5 * q^T * (J^T*J + λI) * q - q^T * (J^T*b + λ*q_reg) + const
        # Standard QP: min 0.5 * q^T * P * q + c^T * q
        # P = J^T*J + λI
        # c = -J^T*b - λ*q_reg = -J^T*(J*q0 - r0) - λ*q_reg
        #   = -J^T*J*q0 + J^T*r0 - λ*q_reg

        JtJ = J_all.T @ J_all  # (nq, nq)
        Jtr = J_all.T @ r_all  # (nq,)

        # P = J^T*J + λI (regularization for qpos smoothness)
        P = JtJ + self.norm_delta * np.eye(nq)

        # c = J^T*r - λ*q_reg  (since r = current residual, we want to minimize it)
        # Actually for Gauss-Newton: we want to minimize ||J*dq + r||^2
        # where dq = q_new - q_current
        # Expanding: min 0.5 * dq^T * J^T*J * dq + dq^T * J^T*r + const
        # Plus regularization: 0.5 * λ * ||q_new - q_reg||^2
        #   = 0.5 * λ * ||(q_current + dq) - q_reg||^2
        #   = 0.5 * λ * ||dq - (q_reg - q_current)||^2
        # Combined: 0.5 * dq^T * (J^T*J + λI) * dq + dq^T * (J^T*r - λ*(q_reg - q_current))
        # So c = J^T*r - λ*(q_reg - q_current) for dq

        if reg_qpos is not None:
            c = Jtr - self.norm_delta * (reg_qpos - qpos)
        else:
            c = Jtr

        # Solve QP with box constraints
        # quadprog format: min 0.5 * x^T * G * x - a^T * x
        # s.t. C^T * x >= b
        # So G = P, a = -c

        # Box constraints: lb <= q <= ub
        # For quadprog: C^T * x >= b format
        # q >= lb  =>  I * q >= lb
        # q <= ub  =>  -I * q >= -ub
        lb = self.robot.joint_limits[:, 0]
        ub = self.robot.joint_limits[:, 1]

        C = np.vstack([np.eye(nq), -np.eye(nq)]).T  # (nq, 2*nq)
        b_ineq = np.concatenate([lb, -ub])  # (2*nq,)

        try:
            # quadprog.solve_qp expects G to be positive definite
            # Add small regularization to ensure positive definiteness
            P_reg = P + 1e-8 * np.eye(nq)
            dq, _, _, _, _, _ = quadprog.solve_qp(P_reg, -c, C, b_ineq)
            qpos_new = qpos + dq
        except ValueError:
            # Fallback: just take gradient step with clipping
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
        alphas: np.ndarray,
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

            # Combine with alpha weights
            # TipDirVec: w_pos * pos + w_dir * dir + w_full_hand_reg * full (regularization)
            # FullHandVec: w_full_hand * full
            loss_tip_dir_vec = self.w_pos * loss_tip_pos + self.w_dir * loss_tip_dir + self.w_full_hand_reg * loss_full
            loss_full_hand = self.w_full_hand * loss_full
            total_loss += alphas[i] * loss_tip_dir_vec + (1.0 - alphas[i]) * loss_full_hand

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
        alphas = self._compute_pinch_alpha(mediapipe_keypoints)
        target_tip_vectors = self._compute_tip_vectors(mediapipe_keypoints, self.scaling)
        target_tip_dirs = self._compute_tip_dirs(mediapipe_keypoints)
        target_full_hand_vectors = self._compute_full_hand_vectors(
            mediapipe_keypoints, self.segment_scaling
        )
        return self._compute_loss(
            qpos, target_tip_vectors, target_tip_dirs,
            target_full_hand_vectors, alphas, None
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
