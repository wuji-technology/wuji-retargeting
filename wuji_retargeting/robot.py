from typing import List
import time

import numpy as np
import numpy.typing as npt
import pinocchio as pin


class RobotWrapper:
    """Pinocchio robot wrapper for forward kinematics."""

    def __init__(self, urdf_path: str):
        # Create robot model and data
        self.model: pin.Model = pin.buildModelFromUrdf(urdf_path)
        self.data: pin.Data = self.model.createData()

        if self.model.nv != self.model.nq:
            raise NotImplementedError("Cannot handle robot with special joint.")

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

    def compute_fk_batch(self, qpos: npt.NDArray, link_indices: List[int]) -> npt.NDArray:
        """Batch compute FK positions for multiple links.

        Args:
            qpos: Joint positions
            link_indices: List of frame indices

        Returns:
            positions: (num_links * 3,) flattened positions
        """
        qpos = np.asarray(qpos, dtype=np.float64)
        pin.forwardKinematics(self.model, self.data, qpos)
        pin.updateFramePlacements(self.model, self.data)

        positions = []
        for idx in link_indices:
            pos = self.data.oMf[idx].translation
            positions.append(pos)

        return np.concatenate(positions)
