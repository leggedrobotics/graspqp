# Copyright (c) 2025 ETH Zurich, René Zurbrügg
#
# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from .articulation_model import ArticulationModel


@configclass
class ArticulationModelCfg(ArticulationCfg):
    """Configuration parameters for a rigid object."""

    @configclass
    class MeshTargetCfg:
        """Configuration for different ray-cast targets."""

        target_prim_expr: str = "MISSING"
        """The regex to specify the target prim to ray cast against."""

        n_surface_pts: int = 2048

        offset_compensation: bool = True
        """Whether to apply offset compensation."""

    @configclass
    class SdfTargetCfg:
        """Configuration for different SDF targets."""

        target_prim_expr: str = "MISSING"
        """The regex to specify the target prim to ray cast against."""
        max_sdf_pts: int = 2048

    class_type: type = ArticulationModel

    mesh_targets_cfg: dict[str, MeshTargetCfg] | None = None

    sdf_targets_cfg: dict[str, SdfTargetCfg] | None = None

    def __post_init__(self):
        """Post init method."""
        super().__post_init__()

        # Set the default mesh target if not provided
        if self.mesh_targets_cfg is None:
            self.mesh_targets_cfg = {}
        if self.sdf_targets_cfg is None:
            self.sdf_targets_cfg = {}

    joint_pos_limits_min: dict[str, tuple[float, float] | float] | None = None
    """Min joint limits for each robot joint. If None, the default limits from the usd are used."""

    joint_pos_limits_max: dict[str, tuple[float, float] | float] | None = None
    """Max joint limits for each robot joint. If None, the default limits from the usd are used."""