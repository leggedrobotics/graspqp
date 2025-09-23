from isaaclab.assets.rigid_object import RigidObjectCfg
from isaaclab.utils import configclass

from .object_model import RigidObjectModel


@configclass
class RigidObjectModelCfg(RigidObjectCfg):
    """Configuration parameters for a rigid object."""

    @configclass
    class MeshTargetCfg:
        """Configuration for different ray-cast targets."""

        target_prim_expr: str = "MISSING"
        """The regex to specify the target prim to ray cast against."""

    class_type: type = RigidObjectModel

    n_surface_pts: int = 2048

    mesh_target_cfg: MeshTargetCfg | None = None
