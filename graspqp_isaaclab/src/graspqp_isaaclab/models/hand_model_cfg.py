from typing import Callable
from isaaclab.assets.articulation import ArticulationCfg

from .hand_model import HandModel

from isaaclab.utils import configclass


@configclass
class HandModelCfg(ArticulationCfg):
    """Configuration parameters for a rigid object."""

    @configclass
    class MeshTargetCfg:
        """Configuration for different ray-cast targets."""

        target_prim_expr: str = "MISSING"
        """The regex to specify the target prim to ray cast against."""

    class_type: type = HandModel

    hand_model_name: str = "MISSING"

    root_body: str | None = None

    contact_mode: str = "all"  # or random

    surface_pts: int | None = None

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
