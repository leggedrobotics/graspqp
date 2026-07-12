# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

from isaaclab.assets.articulation import ArticulationData

import omni.physics.tensors.impl.api as physx
import torch
import isaaclab.utils.math as math_utils
from isaaclab.utils.buffers import TimestampedBuffer


class ArticulationModelData(ArticulationData):

    surface_pts_b: list[torch.Tensor] | None = None
    tracked_body_mask: torch.Tensor | None = None

    def __init__(self, root_physx_view: physx.ArticulationView, device: str):
        super().__init__(root_physx_view, device)
        self._surface_pts_w = TimestampedBuffer()
        self._surface_normals_w = TimestampedBuffer()

    def _empty_surface_tensor(self) -> torch.Tensor:
        return torch.empty((self.root_state_w.shape[0], 0, 3), device=self.device)

    def _surface_list(self, name: str) -> list[torch.Tensor]:
        return getattr(self, name, None) or []

    @property
    def surface_pts_b(self):
        surface_pts_b = self._surface_list("_surface_pts_b")
        if len(surface_pts_b) == 0:
            return self._empty_surface_tensor()
        return torch.cat(surface_pts_b, dim=1)

    @property
    def surface_normals_b(self):
        surface_normals_b = self._surface_list("_surface_normals_b")
        if len(surface_normals_b) == 0:
            return self._empty_surface_tensor()
        return torch.cat(surface_normals_b, dim=1)

    @property
    def surface_pts_w(self):

        if self._surface_pts_w.timestamp < self._sim_timestamp:
            surface_pts_b = self._surface_list("_surface_pts_b")
            if len(surface_pts_b) == 0:
                self._surface_pts_w.data = self._empty_surface_tensor()
                self._surface_pts_w.timestamp = self._sim_timestamp
                return self._surface_pts_w.data
            body_poses = self.body_state_w[:, self.tracked_body_mask, :7]
            pos, qwxyz = body_poses[..., :3], body_poses[..., 3:]
            points = []

            # TODO, could be parallelized
            for obj_idx, pts_b in enumerate(surface_pts_b):
                points.append(math_utils.transform_points(pts_b, pos[:, obj_idx], qwxyz[:, obj_idx]))

            self._surface_pts_w.data = torch.cat(points, dim=1)
            self._surface_pts_w.timestamp = self._sim_timestamp
        return self._surface_pts_w.data

    @property
    def surface_normals_w(self):

        if self._surface_normals_w.timestamp < self._sim_timestamp:
            surface_normals_b = self._surface_list("_surface_normals_b")
            if len(surface_normals_b) == 0:
                self._surface_normals_w.data = self._empty_surface_tensor()
                self._surface_normals_w.timestamp = self._sim_timestamp
                return self._surface_normals_w.data
            body_poses = self.body_state_w[:, self.tracked_body_mask, :7]
            pos, qwxyz = body_poses[..., :3], body_poses[..., 3:]
            normals = []

            # TODO, could be parallelized
            for obj_idx, normals_b in enumerate(surface_normals_b):
                normals.append(math_utils.transform_points(normals_b, pos[:, obj_idx], qwxyz[:, obj_idx]))

            self._surface_normals_w.data = torch.cat(normals, dim=1)
            self._surface_normals_w.timestamp = self._sim_timestamp

        return self._surface_normals_w.data

    @property
    def cog_b(self):
        return self.surface_pts_b.mean(dim=-2)
