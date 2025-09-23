import omni.physics.tensors.impl.api as physx
import torch
from isaaclab.assets.articulation import ArticulationData


class HandModelData(ArticulationData):
    def __init__(self, root_physx_view: physx.RigidBodyView, device: str):
        super().__init__(root_physx_view, device)

    def __repr__(self):
        # print all tensors in the data class
        attrs = [f"{k}={v if isinstance(v, torch.Tensor) else v}" for k, v in self.__dict__.items() if not k.startswith("_")]
        return "ArticulationData(" + "\n ".join(attrs) + ")"
