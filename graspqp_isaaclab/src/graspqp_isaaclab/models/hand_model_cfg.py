# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

from typing import Callable

from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.ui.components import ListComponentCfg
from isaaclab.utils import configclass

from .hand_model import HandModel


@configclass
class HandModelCfg(ArticulationCfg, ListComponentCfg):
    """Configuration parameters for a rigid object."""

    @configclass
    class MeshTargetCfg:
        """Configuration for different ray-cast targets."""

        target_prim_expr: str = "MISSING"
        """The regex to specify the target prim to ray cast against."""

    class_type: type = HandModel

    hand_model_name: str = "MISSING"
    
    grasp_type: str = "default" 

    root_body: str | None = None

    contact_mode: str = "all"  # or random

    forward_axis: tuple[float, float, float] = (0.0, 0.0, 1.0)
    """Approach ("forward") axis of the gripper in the ee frame. Used by reg_gravity to keep
    the forward axis perpendicular to gravity. Override per gripper config if needed."""

    surface_pts: int | None = None

    entries: list[ListComponentCfg.ListEntryCfg] | None = [
        ListComponentCfg.ListEntryCfg(name="contact_points", label="Contact Points", enabled=False),
        ListComponentCfg.ListEntryCfg(name="contact_normals", label="Contact Normals", enabled=False),
        ListComponentCfg.ListEntryCfg(name="surface_points", label="Surface Points", enabled=False),
        ListComponentCfg.ListEntryCfg(name="collision_spheres", label="Collision Spheres", enabled=False),
        ListComponentCfg.ListEntryCfg(name="projected_gravity", label="Projected Gravity", enabled=False),
    ]

    @classmethod
    def from_articulation_cfg(cls, articulation_cfg: ArticulationCfg, **kwargs):
        data = articulation_cfg.to_dict()
        params = {}
        for key in data.keys():
            params[key] = getattr(articulation_cfg, key)

        # update with kwargs
        for key in kwargs.keys():
            params[key] = kwargs[key]

        params["class_type"] = HandModel
        return cls(**params)

    actuated_joints_expr: list[str] | str | None = None
    """Regular expression to specify the actuated joints. Defaults to None which means all joints are actuated."""

    mimic_joints: dict[str, dict[str, float | str]] | None = None

    init_fnc: Callable | None = None
