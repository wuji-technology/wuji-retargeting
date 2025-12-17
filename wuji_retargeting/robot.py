from typing import List
import time

import numpy as np
import numpy.typing as npt
import pinocchio as pin
from autograd.extend import primitive, defvjp


def skew_to_axial(S: np.ndarray) -> np.ndarray:
    """Extract axial vector from skew-symmetric matrix.

    For skew-symmetric S = [[0, -z, y], [z, 0, -x], [-y, x, 0]],
    returns [x, y, z].
    """
    return np.array([S[2, 1] - S[1, 2],
                     S[0, 2] - S[2, 0],
                     S[1, 0] - S[0, 1]])


class RobotWrapper:
    """Pinocchio robot wrapper with Autograd-compatible forward kinematics."""

    def __init__(self, urdf_path: str):
        # Create robot model and data
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.data: pin.Data = self.model.createData()

        if self.model.nv != self.model.nq:
            raise NotImplementedError("Cannot handle robot with special joint.")

        # Cache for differentiable FK primitives
        self._fk_primitives = {}
        self._fk_poses_primitives = {}  # For poses (position + rotation)

        # Timing statistics for FK and Jacobian
        self._timing_enabled = False
        self._fk_time_sum = 0.0
        self._jacobian_time_sum = 0.0
        self._fk_call_count = 0
        self._jacobian_call_count = 0

    def enable_timing(self, enabled: bool = True):
        """Enable or disable timing statistics."""
        self._timing_enabled = enabled
        if enabled:
            self.reset_timing()

    def reset_timing(self):
        """Reset timing statistics."""
        self._fk_time_sum = 0.0
        self._jacobian_time_sum = 0.0
        self._fk_call_count = 0
        self._jacobian_call_count = 0

    def get_timing_stats(self):
        """Get timing statistics.

        Returns:
            dict with keys:
                - fk_avg_us: average FK time in microseconds
                - jacobian_avg_us: average Jacobian time in microseconds
                - fk_call_count: number of FK calls
                - jacobian_call_count: number of Jacobian calls
        """
        fk_avg = (self._fk_time_sum / self._fk_call_count * 1e6) if self._fk_call_count > 0 else 0
        jac_avg = (self._jacobian_time_sum / self._jacobian_call_count * 1e6) if self._jacobian_call_count > 0 else 0
        return {
            'fk_avg_us': fk_avg,
            'jacobian_avg_us': jac_avg,
            'fk_call_count': self._fk_call_count,
            'jacobian_call_count': self._jacobian_call_count,
        }

    @property
    def dof_joint_names(self) -> List[str]:
        """Return names of joints with DOF > 0."""
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def joint_limits(self):
        """Return joint limits as (lower, upper) pairs."""
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    def get_link_index(self, name: str) -> int:
        """Get frame index by name."""
        return self.model.getFrameId(name, pin.BODY)

    # -------------------------------------------------------------------------- #
    # Standard Kinematics Methods
    # -------------------------------------------------------------------------- #
    def compute_forward_kinematics(self, qpos: npt.NDArray):
        """Compute forward kinematics for all links."""
        pin.forwardKinematics(self.model, self.data, qpos)

    def get_link_pose(self, link_id: int) -> npt.NDArray:
        """Get link pose as 4x4 homogeneous matrix."""
        pose: pin.SE3 = pin.updateFramePlacement(self.model, self.data, link_id)
        return pose.homogeneous

    def compute_single_link_local_jacobian(self, qpos, link_id: int) -> npt.NDArray:
        """Compute Jacobian for a single link."""
        J = pin.computeFrameJacobian(self.model, self.data, qpos, link_id)
        return J

    def compute_all_jacobians_batch(self, qpos: npt.NDArray, link_indices: List[int]) -> npt.NDArray:
        """Batch compute position Jacobians for multiple links.

        This is more efficient than calling compute_single_link_local_jacobian
        multiple times because it uses computeJointJacobians once.

        Args:
            qpos: Joint positions
            link_indices: List of frame indices

        Returns:
            jacobians: (num_links, 3, nq) position Jacobians in world frame
        """
        qpos = np.asarray(qpos, dtype=np.float64)

        # Compute all joint Jacobians at once (updates data.J internally)
        pin.computeJointJacobians(self.model, self.data, qpos)
        # Update all frame placements
        pin.updateFramePlacements(self.model, self.data)

        jacobians = []
        for idx in link_indices:
            # getFrameJacobian reuses computed joint Jacobians (faster than computeFrameJacobian)
            J_local = pin.getFrameJacobian(self.model, self.data, idx, pin.LOCAL)
            # Get rotation to transform to world frame
            R = self.data.oMf[idx].rotation
            # Only take position part (3, nq) and transform to world frame
            J_world_pos = R @ J_local[:3, :]
            jacobians.append(J_world_pos)

        return np.stack(jacobians, axis=0)

    # -------------------------------------------------------------------------- #
    # Differentiable Kinematics Methods (Autograd-compatible)
    # -------------------------------------------------------------------------- #
    def differentiable_fk(self, qpos, link_indices: List[int]):
        """Autograd-compatible FK. Returns flattened link positions (num_links * 3,)."""
        # Create primitive for this link combination if not cached
        key = tuple(sorted(link_indices))
        if key not in self._fk_primitives:
            self._fk_primitives[key] = self._make_fk_primitive(link_indices)
        
        return self._fk_primitives[key](qpos)
    
    def _make_fk_primitive(self, link_indices: List[int]):
        """Create Autograd primitive for specific link indices."""
        robot = self

        @primitive
        def fk(qpos):
            qpos = np.asarray(qpos, dtype=np.float64)
            if robot._timing_enabled:
                t0 = time.perf_counter()
            robot.compute_forward_kinematics(qpos)
            positions = np.array([
                robot.get_link_pose(idx)[:3, 3] for idx in link_indices
            ], dtype=np.float64)
            if robot._timing_enabled:
                robot._fk_time_sum += time.perf_counter() - t0
                robot._fk_call_count += 1
            return positions.flatten()

        def fk_vjp(ans, qpos):
            qpos = np.asarray(qpos, dtype=np.float64)

            def vjp_fn(g):
                # g: upstream gradient (num_links * 3,)
                g = g.reshape(-1, 3)

                if robot._timing_enabled:
                    t0 = time.perf_counter()

                # Use batch Jacobian computation (more efficient)
                J = robot.compute_all_jacobians_batch(qpos, link_indices)  # (num_links, 3, nq)

                if robot._timing_enabled:
                    robot._jacobian_time_sum += time.perf_counter() - t0
                    robot._jacobian_call_count += 1

                # Chain rule: grad_qpos = sum over links of (g @ J)
                return np.einsum('li,lij->j', g, J)

            return vjp_fn

        defvjp(fk, fk_vjp)
        return fk

    # -------------------------------------------------------------------------- #
    # Differentiable FK with Poses (Position + Rotation)
    # -------------------------------------------------------------------------- #
    def differentiable_fk_poses(self, qpos, link_indices: List[int]):
        """Autograd-compatible FK returning positions and rotations.

        Returns:
            poses: (num_links * 12,) - flattened [pos(3), rot(9)] per link
                   First num_links*3 elements are positions,
                   next num_links*9 elements are flattened rotation matrices.
        """
        key = ('poses', tuple(sorted(link_indices)))
        if key not in self._fk_poses_primitives:
            self._fk_poses_primitives[key] = self._make_fk_poses_primitive(link_indices)

        return self._fk_poses_primitives[key](qpos)

    def _make_fk_poses_primitive(self, link_indices: List[int]):
        """Create Autograd primitive for FK returning positions and rotations."""
        robot = self
        num_links = len(link_indices)

        @primitive
        def fk_poses(qpos):
            qpos = np.asarray(qpos, dtype=np.float64)
            if robot._timing_enabled:
                t0 = time.perf_counter()

            robot.compute_forward_kinematics(qpos)
            positions = []
            rotations = []
            for idx in link_indices:
                pose = robot.get_link_pose(idx)
                positions.append(pose[:3, 3])
                rotations.append(pose[:3, :3].flatten())

            if robot._timing_enabled:
                robot._fk_time_sum += time.perf_counter() - t0
                robot._fk_call_count += 1

            return np.concatenate([
                np.array(positions, dtype=np.float64).flatten(),
                np.array(rotations, dtype=np.float64).flatten()
            ])

        def fk_poses_vjp(ans, qpos):
            qpos = np.asarray(qpos, dtype=np.float64)

            def vjp_fn(g):
                g_pos = g[:num_links * 3].reshape(-1, 3)
                g_rot = g[num_links * 3:].reshape(-1, 9)

                if robot._timing_enabled:
                    t0 = time.perf_counter()

                robot.compute_forward_kinematics(qpos)
                grad_qpos = np.zeros(robot.model.nq)

                for i, idx in enumerate(link_indices):
                    # Get full Jacobian (6, nq): [linear; angular]
                    J_local = robot.compute_single_link_local_jacobian(qpos, idx)
                    R = robot.get_link_pose(idx)[:3, :3]

                    # Position gradient
                    J_pos = R @ J_local[:3, :]  # (3, nq)
                    grad_qpos += g_pos[i] @ J_pos

                    # Rotation gradient
                    J_omega = R @ J_local[3:, :]  # (3, nq) world frame
                    G = g_rot[i].reshape(3, 3)
                    GR = G.T @ R
                    axial = skew_to_axial(GR - GR.T)
                    grad_qpos += J_omega.T @ axial

                if robot._timing_enabled:
                    robot._jacobian_time_sum += time.perf_counter() - t0
                    robot._jacobian_call_count += 1

                return grad_qpos

            return vjp_fn

        defvjp(fk_poses, fk_poses_vjp)
        return fk_poses

