from typing import List

import autograd.numpy as anp
from autograd import grad
import nlopt
import numpy as np

from .robot import RobotWrapper


def _smooth_l1_loss_autograd(diff, beta: float):
    """Huber (Smooth L1) loss implemented with autograd-compatible ops."""
    beta = anp.maximum(beta, 1e-12)
    abs_diff = anp.abs(diff)
    quadratic = 0.5 * diff**2 / beta
    linear = abs_diff - 0.5 * beta
    return anp.where(abs_diff < beta, quadratic, linear)

class DexPilotOptimizer:
    """Retargeting optimizer using the method proposed in DexPilot

    This is a broader adaptation of the original optimizer delineated in the DexPilot paper.
    While the initial DexPilot study focused solely on the four-fingered Allegro Hand, this version of the optimizer
    embraces the same principles for both four-fingered and five-fingered hands. It projects the distance between the
    thumb and the other fingers to facilitate more stable grasping.
    Reference: https://arxiv.org/abs/1910.03135

    Args:
        robot:
        target_joint_names:
        finger_tip_link_names:
        wrist_link_name:
        gamma:
    """

    retargeting_type = "DEXPILOT"

    def __init__(
        self,
        robot: RobotWrapper,
        target_joint_names: List[str],
        finger_tip_link_names: List[str],
        wrist_link_name: str,
        finger_scaling: List[float],
        huber_delta=0.03,
        norm_delta=4e-3,
        project_dist=0.02,
        escape_dist=0.03,
        max_iter=100,
        eta1=1e-4,
        eta2=3e-2,
    ):
        # Wuji Hand has 5 fingers
        self.num_fingers = 5
        self.finger_scaling = np.array(finger_scaling, dtype=np.float32)

        # Generate link indices inline (finger-finger pairs + wrist-to-finger)
        origin_link_index = []
        task_link_index = []
        for i in range(1, self.num_fingers):
            for j in range(i + 1, self.num_fingers + 1):
                origin_link_index.append(j)
                task_link_index.append(i)
        for i in range(1, self.num_fingers + 1):
            origin_link_index.append(0)
            task_link_index.append(i)
        
        self.origin_finger_indices = np.array(origin_link_index, dtype=int)
        self.task_finger_indices = np.array(task_link_index, dtype=int)

        # Build target link human indices and names
        target_link_human_indices = (
            np.stack([origin_link_index, task_link_index], axis=0) * 4
        ).astype(int)
        link_names = [wrist_link_name] + finger_tip_link_names
        target_origin_link_names = [link_names[i] for i in origin_link_index]
        target_task_link_names = [link_names[i] for i in task_link_index]

        # Initialize robot and joint mappings
        self.robot = robot
        joint_names = robot.dof_joint_names
        idx_pin2target = []
        for target_joint_name in target_joint_names:
            if target_joint_name not in joint_names:
                raise ValueError(
                    f"Joint {target_joint_name} given does not appear to be in robot XML."
                )
            idx_pin2target.append(joint_names.index(target_joint_name))
        self.target_joint_names = target_joint_names
        self.idx_pin2target = np.array(idx_pin2target)
        self.opt = nlopt.opt(nlopt.LD_SLSQP, len(idx_pin2target))
        
        # Target
        self.target_link_human_indices = target_link_human_indices
        
        # DexPilot specific
        self.origin_link_names = target_origin_link_names
        self.task_link_names = target_task_link_names
        self.huber_delta = float(huber_delta)
        self.norm_delta = norm_delta

        # Computation cache for better performance
        # For one link used in multiple vectors, e.g. hand palm, we do not want to compute it multiple times
        self.computed_link_names = list(
            set(target_origin_link_names).union(set(target_task_link_names))
        )
        self.origin_link_indices = np.array(
            [self.computed_link_names.index(name) for name in target_origin_link_names],
            dtype=int,
        )
        self.task_link_indices = np.array(
            [self.computed_link_names.index(name) for name in target_task_link_names],
            dtype=int,
        )

        # Sanity check and cache link indices
        self.computed_link_indices = [self.robot.get_link_index(name) for name in self.computed_link_names]

        self.opt.set_ftol_abs(1e-6)
        self.opt.set_maxeval(max_iter)  # Set maximum number of function evaluations
        
        # DexPilot projection parameters
        self.project_dist = project_dist
        self.escape_dist = escape_dist
        self.eta1 = eta1
        self.eta2 = eta2
        
        # DexPilot cache for projection
        (
            self.projected,
            self.s2_project_index_origin,
            self.s2_project_index_task,
            self.projected_dist,
        ) = self._set_dexpilot_cache(self.num_fingers, eta1, eta2)
        
        self.vector_scaling = self._build_vector_scaling()

    def set_joint_limit(self, joint_limits: np.ndarray, epsilon=1e-3):
        """Set joint limits for optimization."""
        expected_shape = (len(self.idx_pin2target), 2)
        if joint_limits.shape != expected_shape:
            raise ValueError(
                f"Expect joint limits have shape: {expected_shape}, but get {joint_limits.shape}"
            )
        self.opt.set_lower_bounds((joint_limits[:, 0] - epsilon).tolist())
        self.opt.set_upper_bounds((joint_limits[:, 1] + epsilon).tolist())
    
    def set_max_iter(self, max_iter: int):
        """Set maximum number of iterations for optimization.
        
        Args:
            max_iter: Maximum number of function evaluations (iterations)
        """
        if max_iter <= 0:
            raise ValueError("max_iter must be positive")
        self.opt.set_maxeval(max_iter)

    def retarget(self, ref_value, last_qpos):
        """
        Compute the retargeting results using non-linear optimization.
        
        Args:
            ref_value: the reference value in cartesian space as input
            last_qpos: the last retargeting results or initial value

        Returns:
            Joint position of robot, the joint order and dim is consistent with self.target_joint_names
        """
        objective_fn = self.get_objective_function(
            ref_value, np.array(last_qpos).astype(np.float32)
        )

        self.opt.set_min_objective(objective_fn)
        try:
            qpos = self.opt.optimize(last_qpos)
            return np.array(qpos, dtype=np.float32)
        except RuntimeError as e:
            print(e)
            return np.array(last_qpos, dtype=np.float32)


    @staticmethod
    def _set_dexpilot_cache(num_fingers, eta1, eta2):
        """
        Initialize DexPilot projection cache.
        
        Returns:
            projected: boolean array indicating projection state (initialized to False)
            s2_project_index_origin: indices for secondary projection origin
            s2_project_index_task: indices for secondary projection task
            projected_dist: distance values for projection (eta1 for S1, eta2 for S2)
        """
        projected = np.zeros(num_fingers * (num_fingers - 1) // 2, dtype=bool)
        
        s2_project_index_origin = []
        s2_project_index_task = []
        for i in range(0, num_fingers - 2):
            for j in range(i + 1, num_fingers - 1):
                s2_project_index_origin.append(j)
                s2_project_index_task.append(i)
        
        projected_dist = np.array(
            [eta1] * (num_fingers - 1)
            + [eta2] * ((num_fingers - 1) * (num_fingers - 2) // 2)
        )
        
        return projected, s2_project_index_origin, s2_project_index_task, projected_dist

    def _build_vector_scaling(self) -> np.ndarray:
        """Build per-vector scaling factors based on provided per-finger multipliers."""
        factors = np.ones(len(self.origin_finger_indices), dtype=np.float32)
        for idx in range(len(self.origin_finger_indices)):
            indices = []
            if self.origin_finger_indices[idx] > 0:
                indices.append(self.origin_finger_indices[idx] - 1)
            if self.task_finger_indices[idx] > 0:
                indices.append(self.task_finger_indices[idx] - 1)
            if indices:
                factors[idx] = float(np.mean(self.finger_scaling[indices]))
        return factors

    def get_objective_function(
        self, target_vector: np.ndarray, last_qpos: np.ndarray
    ):
        """Create fully differentiable objective function for NLopt optimization."""
        target_vector = np.asarray(target_vector, dtype=np.float64)
        
        len_proj = self.num_fingers * (self.num_fingers - 1) // 2
        len_s2 = len(self.s2_project_index_task)
        len_s1 = len_proj - len_s2
        
        # Update projection indicator based on target vector distances
        target_vec_dist = np.linalg.norm(target_vector[:len_proj], axis=1)
        self.projected[:len_s1][target_vec_dist[0:len_s1] < self.project_dist] = True
        self.projected[:len_s1][target_vec_dist[0:len_s1] > self.escape_dist] = False
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[:len_s1][self.s2_project_index_origin],
            self.projected[:len_s1][self.s2_project_index_task],
        )
        self.projected[len_s1:len_proj] = np.logical_and(
            self.projected[len_s1:len_proj], target_vec_dist[len_s1:len_proj] <= 0.02
        )
        
        # Update weight vector based on projection state
        normal_weight = np.ones(len_proj, dtype=np.float64)
        high_weight = np.array([200] * len_s1 + [400] * len_s2, dtype=np.float64)
        weight_proj = np.where(self.projected, high_weight, normal_weight)
        
        # Weight for vectors from wrist to fingertips
        # Higher weight ensures better intuitive mapping due to pose detection errors
        weight = np.concatenate([
            weight_proj,
            np.ones(self.num_fingers, dtype=np.float64) * (len_proj + self.num_fingers),
        ])
        
        # Compute reference distance vector
        normal_vec = target_vector * self.vector_scaling[:, None]  # (len_vec, 3)
        dir_vec = target_vector[:len_proj] / (target_vec_dist[:, None] + 1e-6)  # (len_proj, 3)
        projected_vec = dir_vec * self.projected_dist[:, None]  # (len_proj, 3)
        
        # Compute final reference vector (use projected vector if projected, else normal vector)
        reference_vec_proj = np.where(
            self.projected[:, None], projected_vec, normal_vec[:len_proj]
        )  # (len_proj, 3)
        reference_vec = np.concatenate(
            [reference_vec_proj, normal_vec[len_proj:]], axis=0
        ).astype(np.float64)  # (total_vec, 3)
        
        # Prepare constants for Autograd
        body_shape = (len(self.computed_link_indices), 3)
        num_vec = reference_vec.shape[0]
        
        # Define fully differentiable loss: qpos -> scalar
        def loss_fn(qpos):
            # FK (differentiable via VJP)
            body = self.robot.differentiable_fk(qpos, self.computed_link_indices).reshape(body_shape)
            
            # Position error
            robot_vec = body[self.task_link_indices, :] - body[self.origin_link_indices, :]
            diff_vec = robot_vec - anp.array(reference_vec)
            vec_dist = anp.linalg.norm(diff_vec, axis=1)
            huber = _smooth_l1_loss_autograd(vec_dist, self.huber_delta)
            pos_loss = anp.sum(huber * anp.array(weight)) / num_vec
            
            # Regularization
            reg_loss = self.norm_delta * anp.sum((qpos - last_qpos) ** 2)
            
            return pos_loss + reg_loss
        
        loss_grad_fn = grad(loss_fn)
        
        def objective(x: np.ndarray, grad: np.ndarray) -> float:
            qpos = np.asarray(x, dtype=np.float64)
            loss = float(loss_fn(qpos))
            if grad.size > 0:
                grad[:] = np.array(loss_grad_fn(qpos), dtype=np.float64)
            return loss
        
        return objective

class LPFilter:
    """Low-pass filter for smoothing joint positions."""
    def __init__(self, alpha):
        self.alpha = alpha
        self.y = None
        self.is_init = False

    def next(self, x):
        if not self.is_init:
            self.y = x
            self.is_init = True
            return self.y.copy()
        self.y = self.y + self.alpha * (x - self.y)
        return self.y.copy()

    def reset(self):
        self.y = None
        self.is_init = False



