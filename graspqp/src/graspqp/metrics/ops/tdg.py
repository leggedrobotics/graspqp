from abc import abstractmethod
from dataclasses import dataclass
from typing import List

# Third Party
import numpy as np
import torch


@dataclass(frozen=True)
class TensorDeviceType:
    device: torch.device = torch.device("cuda", 0)
    dtype: torch.dtype = torch.float32
    collision_geometry_dtype: torch.dtype = torch.float32
    collision_gradient_dtype: torch.dtype = torch.float32
    collision_distance_dtype: torch.dtype = torch.float32

    @staticmethod
    def from_basic(device: str, dev_id: int):
        return TensorDeviceType(torch.device(device, dev_id))

    def to_device(self, data_tensor):
        if isinstance(data_tensor, torch.Tensor):
            return data_tensor.to(device=self.device, dtype=self.dtype)
        else:
            return torch.as_tensor(np.array(data_tensor), device=self.device, dtype=self.dtype)

    def to_int8_device(self, data_tensor):
        return data_tensor.to(device=self.device, dtype=torch.int8)

    def cpu(self):
        return TensorDeviceType(device=torch.device("cpu"), dtype=self.dtype)

    def as_torch_dict(self):
        return {"device": self.device, "dtype": self.dtype}


@torch.jit.script
def normalize_vector(v: torch.Tensor):
    return v / torch.clamp(v.norm(dim=-1, p=2, keepdim=True), min=1e-12)


def np_normalize(x_vec):
    return x_vec / (np.linalg.norm(x_vec, axis=-1)[:, None] + 1e-8)


def random_sample_points_on_sphere(dim_num, point_num):
    points = np.random.randn(point_num, dim_num)
    points = np_normalize(points)
    # print(f"Finish generating! Got {points.shape[0]} points (without duplication) on S^{dim_num-1}!")
    return points


class GraspEnergyBase:
    # friction coef: [miu_1, miu_2]. If miu_2 != 0, use soft finger contact model.
    miu_coef: List

    tensor_args: TensorDeviceType

    def __init__(self, miu_coef, obj_gravity_center, obj_obb_length, tensor_args, enable_density):
        self.miu_coef = miu_coef
        # friction cone parameters
        assert self.miu_coef[0] <= 1 and self.miu_coef[0] > 0

        self.tensor_args = tensor_args
        self.obj_gravity_center = None
        self.obj_obb_length = None
        self.reset(obj_gravity_center, obj_obb_length)

        # for rotation matrix construction
        self.rot_base1 = self.tensor_args.to_device([0, 1, 0])
        self.rot_base2 = self.tensor_args.to_device([0, 0, 1])
        self.enable_density = enable_density
        self.grasp_batch = self.force_batch = 0
        return

    def utils_1axis_to_3axes(self, axis_0):
        """
        One 3D direction to three 3D axes for constructing 3x3 rotation matrix.

        Parameters
        ----------
        axis_0: [..., 3]

        Returns
        ----------
        axis_0: [..., 3]
        axis_1: [..., 3]
        axis_2: [..., 3]

        """

        tmp_rot_base1 = self.rot_base1.view([1] * (len(axis_0.shape) - 1) + [3])
        tmp_rot_base2 = self.rot_base2.view([1] * (len(axis_0.shape) - 1) + [3])

        proj_xy = (axis_0 * tmp_rot_base1).sum(dim=-1, keepdim=True).abs()
        axis_1 = torch.where(proj_xy > 0.99, tmp_rot_base2, tmp_rot_base1)  # avoid normal prependicular to axis_y1
        # NOTE the next line is necessary for gradient descent! Otherwise the differentiability is bad if normal is similar to axis_1!
        axis_1 = normalize_vector(axis_1 - (axis_1 * axis_0).sum(dim=-1, keepdim=True) * axis_0).detach()
        axis_1 = normalize_vector(axis_1 - (axis_1 * axis_0).sum(dim=-1, keepdim=True) * axis_0)
        axis_2 = torch.cross(axis_0, axis_1, dim=-1)
        return axis_0, axis_1, axis_2

    def construct_grasp_matrix(self, pos, normal):
        axis_0, axis_1, axis_2 = self.utils_1axis_to_3axes(normal)
        env_num = self.obj_gravity_center.shape[0]
        batch_num = pos.shape[0] // env_num
        # NOTE: obj_gravity_center and obj_obb_length are used to normalized the position to roughly align the magnitude of torque with force.
        relative_pos = (
            (pos.view(env_num, batch_num, -1, 3) - self.obj_gravity_center.view(-1, 1, 1, 3))
            / self.obj_obb_length.view(-1, 1, 1, 1)
        ).view(env_num * batch_num, -1, 3)
        w0 = torch.cat([axis_0, torch.cross(relative_pos, axis_0, dim=-1)], dim=-1)
        w1 = torch.cat([axis_1, torch.cross(relative_pos, axis_1, dim=-1)], dim=-1)
        w2 = torch.cat([axis_2, torch.cross(relative_pos, axis_2, dim=-1)], dim=-1)
        if self.miu_coef[1] > 0:
            w3 = torch.cat([axis_0 * 0.0, axis_0 * self.miu_coef[1]], dim=-1)
            grasp_matrix = torch.stack([w0, w1, w2, w3], dim=-1)  # [b, n, 6, 4]
        else:
            grasp_matrix = torch.stack([w0, w1, w2], dim=-1)  # [b, n, 6, 3]
        contact_frame = torch.stack([axis_0, axis_1, axis_2], dim=-1)
        return grasp_matrix, contact_frame

    def estimate_density(self, normal):
        cos_theta = (normal.unsqueeze(-2) * normal.unsqueeze(-3)).sum(dim=-1)  # [b, n, n]
        density = 1 / torch.clamp(torch.clamp(cos_theta, min=0).sum(dim=-1), min=1e-4)
        return density.detach()

    @abstractmethod
    def forward(self, pos, normal, test_wrenches):
        raise NotImplementedError

    def reset(self, gravity_center, obb_length):
        if gravity_center is not None:
            if self.obj_gravity_center is not None and gravity_center.shape == self.obj_gravity_center.shape:
                self.obj_gravity_center[:] = torch.tensor(gravity_center)
            else:
                self.obj_gravity_center = self.tensor_args.to_device(gravity_center)
        if obb_length is not None:
            if self.obj_obb_length is not None and obb_length.shape == self.obj_obb_length.shape:
                self.obj_obb_length[:] = torch.tensor(obb_length)
            else:
                self.obj_obb_length = self.tensor_args.to_device(obb_length)
        return


class TDGEnergy(GraspEnergyBase):
    def __init__(self, miu_coef, obj_gravity_center, obj_obb_length, tensor_args, enable_density, **args):
        super().__init__(miu_coef, obj_gravity_center, obj_obb_length, tensor_args, enable_density)
        self.miu_tensor = self.tensor_args.to_device(self.miu_coef[0])
        self.direct_num = 1000

        # NOTE: This TWS is for force closure. You can modify it to some limited direction to suit your task.
        direction_3D = random_sample_points_on_sphere(3, self.direct_num)
        direction_6D = np.concatenate([direction_3D, direction_3D * 0.0], axis=-1)
        self.target_direction_6D = self.tensor_args.to_device(direction_6D).unsqueeze(0)  # [1, P, 6]

        # self.target_direction_6D = torch.eye(6, device=self.tensor_args.device, dtype=self.tensor_args.dtype).unsqueeze(0)
        self.F_center_direction = self.tensor_args.to_device([1, 0, 0])[None, None, None]

        # NOTE: We find that the soft contact version doesn't make great difference, so it is not included.
        if self.miu_coef[1] > 0:
            raise NotImplementedError
        return

    def GWS(self, G_matrix, normal):
        """Approximate the GWS boundary by dense samples.

        Returns
        ----------
        w: [b, P, 6]
        """
        # solve q_W(u): q_W(u) equals to G * q_F(G^T @ u), so first solve q_F(u')
        direction_F = normalize_vector(
            (self.target_direction_6D.unsqueeze(1) @ G_matrix).transpose(2, 1)
        )  # G^T @ u: [b, P, n, 3] or [b, P, n, 4]
        proj_on_cn = (direction_F * self.F_center_direction).sum(dim=-1, keepdim=True)  # [b, P, n, 1]
        perp_to_cn = direction_F - proj_on_cn * self.F_center_direction  # [b, P, n, 3]  or [b, P, n, 4]

        angles = torch.acos(torch.clamp(proj_on_cn, min=-1, max=1))  # [b, P, n, 1]
        bottom_length = self.miu_tensor
        bottom_angle = torch.atan(bottom_length)

        region1 = angles <= bottom_angle
        region2 = (angles > bottom_angle) & (angles <= np.pi / 2)
        region3 = angles > np.pi / 2
        perp_norm = perp_to_cn.norm(dim=-1, keepdim=True)

        # a more continuous approximation
        help3 = perp_norm / (perp_norm - 2 * bottom_length * torch.clamp(proj_on_cn, max=0))
        help2 = self.F_center_direction + bottom_length * normalize_vector(perp_to_cn)
        argmin_3D_on_normalized_cone = (
            region1 * (self.F_center_direction + perp_to_cn / torch.clamp(proj_on_cn, min=torch.cos(bottom_angle) / 2))
            + region2 * help2
            + region3 * help3 * help2
        )  # [b, P, n, 3]

        # get q_W(u) = G * q_F(G^T @ u)
        w = (G_matrix.unsqueeze(1) @ argmin_3D_on_normalized_cone.unsqueeze(-1)).squeeze(-1)  # [b, P, n, 6]

        # NOTE: use density to change the force bound. It can help to synthesize more human-like pose, i.e. four fingers on one side and the thumb finger on another.
        if self.enable_density:
            density = self.estimate_density(normal)
            final_w = (w * density.unsqueeze(1).unsqueeze(-1)).sum(dim=2)  # [b, P, 6]
        else:
            final_w = w.sum(dim=2)
        return final_w

    def forward(self, pos, normal):
        # G: F \in R^3 (or R^4) -> W \in R^6
        G_matrix, contact_frame = self.construct_grasp_matrix(pos, normal)
        w = self.GWS(G_matrix, normal)
        cos_wt = (normalize_vector(w) * self.target_direction_6D).sum(dim=-1)
        gras_energy = (1 - cos_wt).mean(dim=-1, keepdim=True)
        return gras_energy


class TDGSpanMetric(torch.nn.Module):
    def __init__(self, device="cpu"):
        super().__init__()

        tensor_args = TensorDeviceType(device=device)
        config = {
            "miu_coef": [0.2, 0.0],
            "obj_gravity_center": torch.tensor([[0.0, 0.0, 0.0]]),
            "obj_obb_length": torch.tensor([0.2]),
            "enable_density": True,
            "tensor_args": tensor_args,
        }
        self.tdg_energy = TDGEnergy(**config)

    def forward(
        self, contact_pts: torch.Tensor, contact_normals: torch.Tensor, cog, torque_weight=0.0, with_solution=False, **kwargs
    ):
        contact_pts = contact_pts - cog.unsqueeze(1)
        energy = 100 * self.tdg_energy.forward(contact_pts, contact_normals).squeeze(
            -1
        )  # multiply with 100 to make it similar scale
        return energy, None
