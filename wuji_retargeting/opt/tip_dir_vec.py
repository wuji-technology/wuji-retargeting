"""TipDirVec optimizer for hand retargeting."""

from __future__ import annotations

from typing import Optional

import autograd.numpy as anp
from autograd import grad
import numpy as np

from .base import BaseOptimizer, M_TO_CM, huber_loss


class TipDirVecOptimizer(BaseOptimizer):
    """Optimizer using tip position + tip direction with full hand vec regularization.

    Loss function:
        L = w_pos * L_tip_pos + w_dir * L_tip_dir + w_full_hand_vec * L_full_hand_vec + λ||Δq||²

    Where:
        - L_tip_pos: Huber loss on wrist->tip vectors (5 vectors)
        - L_tip_dir: Huber loss on tip directions (link4->tip, 5 vectors)
        - L_full_hand_vec: L2 loss on full hand vectors (15 vectors) as regularization
    """

    def __init__(self, config: dict):
        """Initialize TipDirVecOptimizer.

        Args:
            config: Configuration dict with 'optimizer' and 'retarget' sections
        """
        super().__init__(config)

        retarget_config = config.get('retarget', {})
        self.huber_delta_dir = retarget_config.get('huber_delta_dir', 0.5)
        self.w_pos = retarget_config.get('w_pos', 1.0)
        self.w_dir = retarget_config.get('w_dir', 1.0)
        self.w_full_hand_vec = retarget_config.get('w_full_hand_vec', 0.01)
        self.scaling = retarget_config.get('scaling', 1.0)
        self.project_tip_dir = retarget_config.get('project_tip_dir', True)

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
        mediapipe_keypoints = np.asarray(mediapipe_keypoints, dtype=np.float64)
        if mediapipe_keypoints.shape != (21, 3):
            raise ValueError(f"Expected shape (21, 3), got {mediapipe_keypoints.shape}")

        reg_qpos = self._get_reg_qpos(last_qpos)
        init_qpos = self._get_init_qpos(last_qpos)

        # Compute targets
        target_tip_vectors = self._compute_tip_vectors(mediapipe_keypoints, self.scaling)
        target_tip_dirs = self._compute_tip_dirs(mediapipe_keypoints)
        target_full_hand_vectors = self._compute_full_hand_vectors(
            mediapipe_keypoints, np.ones((5, 3))
        )

        # Create objective and run optimization
        objective_fn = self._get_objective(
            target_tip_vectors, target_tip_dirs, target_full_hand_vectors, reg_qpos
        )
        return self._run_optimization(objective_fn, init_qpos)

    def compute_cost(
        self,
        qpos: np.ndarray,
        mediapipe_keypoints: np.ndarray,
    ) -> float:
        """Compute cost for given joint angles."""
        target_tip_vectors = self._compute_tip_vectors(mediapipe_keypoints, self.scaling)
        target_tip_dirs = self._compute_tip_dirs(mediapipe_keypoints)
        target_full_hand_vectors = self._compute_full_hand_vectors(
            mediapipe_keypoints, np.ones((5, 3))
        )
        return float(self._loss_fn(
            qpos, target_tip_vectors, target_tip_dirs, target_full_hand_vectors, None
        ))

    def _loss_fn(
        self,
        qpos: np.ndarray,
        target_tip_vectors: np.ndarray,
        target_tip_dirs: np.ndarray,
        target_full_hand_vectors: np.ndarray,
        last_qpos: Optional[np.ndarray],
    ) -> float:
        """Compute loss."""
        # FK
        positions_flat = self.robot.differentiable_fk(qpos, self.computed_link_indices)
        positions = positions_flat.reshape(-1, 3) * M_TO_CM

        # Extract positions
        origin_positions = positions[self.origin_indices]
        task_positions = positions[self.task_indices]
        link1_positions = positions[self.link1_indices]
        link3_positions = positions[self.link3_indices]
        link4_positions = positions[self.link4_indices]

        # === Tip position cost ===
        robot_tip_vec = task_positions - origin_positions
        vec_diff = robot_tip_vec - target_tip_vectors
        vec_dist = anp.sqrt(anp.sum(vec_diff ** 2, axis=1))
        loss_pos = anp.mean(huber_loss(vec_dist, self.huber_delta))

        # === Tip direction cost ===
        robot_tip_dir_vec = task_positions - link4_positions
        robot_tip_dir_norm = anp.sqrt(anp.sum(robot_tip_dir_vec ** 2, axis=1, keepdims=True))
        robot_tip_dirs = robot_tip_dir_vec / (robot_tip_dir_norm + 1e-8)

        # Optional projection onto finger plane
        if self.project_tip_dir:
            v1 = link4_positions - link1_positions
            v2 = task_positions - link1_positions
            plane_normal = anp.stack([
                v1[:, 1] * v2[:, 2] - v1[:, 2] * v2[:, 1],
                v1[:, 2] * v2[:, 0] - v1[:, 0] * v2[:, 2],
                v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0],
            ], axis=1)
            plane_normal_norm = anp.sqrt(anp.sum(plane_normal ** 2, axis=1, keepdims=True))
            plane_normal = plane_normal / (plane_normal_norm + 1e-8)

            dot_product = anp.sum(target_tip_dirs * plane_normal, axis=1, keepdims=True)
            target_tip_dirs_proj = target_tip_dirs - dot_product * plane_normal
            proj_norm = anp.sqrt(anp.sum(target_tip_dirs_proj ** 2, axis=1, keepdims=True))
            target_tip_dirs_proj = target_tip_dirs_proj / (proj_norm + 1e-8)
        else:
            target_tip_dirs_proj = target_tip_dirs

        dir_diff = robot_tip_dirs - target_tip_dirs_proj
        dir_dist = anp.sqrt(anp.sum(dir_diff ** 2, axis=1))
        loss_dir = anp.mean(huber_loss(dir_dist, self.huber_delta_dir))

        # === Full hand vec regularization ===
        # Robot vectors: wrist->link3 (PIP), wrist->link4 (DIP), wrist->tip (TIP)
        wrist_pos = positions[self.origin_indices[0]]
        robot_pip_vec = link3_positions - wrist_pos
        robot_dip_vec = link4_positions - wrist_pos
        robot_tip_vec_full = task_positions - wrist_pos
        robot_full_hand_vec = anp.vstack([robot_pip_vec, robot_dip_vec, robot_tip_vec_full])

        full_hand_diff = robot_full_hand_vec - target_full_hand_vectors
        loss_full_hand = anp.mean(anp.sum(full_hand_diff ** 2, axis=1))

        # === Total loss ===
        loss = (
            self.w_pos * loss_pos +
            self.w_dir * loss_dir +
            self.w_full_hand_vec * loss_full_hand
        )

        if last_qpos is not None:
            loss = loss + self.norm_delta * anp.sum((qpos - last_qpos) ** 2)

        return loss

    def _get_objective(
        self,
        target_tip_vectors: np.ndarray,
        target_tip_dirs: np.ndarray,
        target_full_hand_vectors: np.ndarray,
        last_qpos: Optional[np.ndarray],
    ):
        """Create NLopt objective function."""
        target_tip_vectors = np.asarray(target_tip_vectors, dtype=np.float64)
        target_tip_dirs = np.asarray(target_tip_dirs, dtype=np.float64)
        target_full_hand_vectors = np.asarray(target_full_hand_vectors, dtype=np.float64)
        if last_qpos is not None:
            last_qpos = np.asarray(last_qpos, dtype=np.float64)

        def loss_fn(qpos):
            return self._loss_fn(
                qpos, target_tip_vectors, target_tip_dirs, target_full_hand_vectors, last_qpos
            )

        loss_grad_fn = grad(loss_fn)

        def objective(x, grad_out):
            qpos = np.asarray(x, dtype=np.float64)
            loss = float(loss_fn(qpos))
            if grad_out.size > 0:
                grad_out[:] = np.array(loss_grad_fn(qpos), dtype=np.float64)
            return loss

        return objective
