from isaaclab.assets.rigid_object import RigidObjectData

import omni.physics.tensors.impl.api as physx
import torch
import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import TimestampedBuffer


class RigidObjectModelData(RigidObjectData):

    def __init__(self, root_physx_view: physx.RigidBodyView, device: str):
        super().__init__(root_physx_view, device)
        self.surface_pts_b: torch.Tensor = None
        self._surface_pts_w = TimestampedBuffer()

    @property
    def surface_pts_w(self):

        if self._surface_pts_w.timestamp < self._sim_timestamp:
            body_pose = self.root_state_w[:, :7]
            pos, qwxyz = body_pose[:, :3], body_pose[:, 3:]
            self._surface_pts_w.data = math_utils.transform_points(self.surface_pts_b, pos, qwxyz)
        return self._surface_pts_w.data
