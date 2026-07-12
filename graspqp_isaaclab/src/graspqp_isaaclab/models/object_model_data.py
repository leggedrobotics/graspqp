# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

import isaaclab.utils.math as math_utils
import omni.physics.tensors.impl.api as physx
import torch
from isaaclab.assets.rigid_object import RigidObjectData
from isaaclab.utils.buffers import TimestampedBuffer


class RigidObjectModelData(RigidObjectData):
    def __init__(self, root_physx_view: physx.RigidBodyView, device: str):
        super().__init__(root_physx_view, device)
        self.surface_pts_b: torch.Tensor = None
        self._surface_pts_w = TimestampedBuffer()
        self._cog_b: torch.Tensor = None

    @property
    def surface_pts_w(self):

        if self._surface_pts_w.timestamp < self._sim_timestamp:
            body_pose = self.root_state_w[:, :7]
            pos, qwxyz = body_pose[:, :3], body_pose[:, 3:]
            self._surface_pts_w.data = math_utils.transform_points(self.surface_pts_b, pos, qwxyz)
        return self._surface_pts_w.data

    @property
    def cog_b(self) -> torch.Tensor:
        if self._cog_b is None:
            self._cog_b = torch.mean(self.surface_pts_b, dim=-2)
            
        return self._cog_b