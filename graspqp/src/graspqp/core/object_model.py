# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT
#
# Portions derived from DexGraspNet (https://github.com/PKU-EPIC/DexGraspNet),
# MIT License, Copyright (c) 2023 Jialiang Zhang, Ruicheng Wang.

"""Object representation used during grasp optimization.

Defines :class:`ObjectModel`, which loads one or more object meshes, samples a
farthest-point surface point cloud for each, and builds a signed distance field
(SDF) with a selectable backend (WARP / TorchSDF / Kaolin, chosen via the
``SDF_BACKEND`` environment variable). The SDF query :meth:`ObjectModel.cal_distance`
is differentiable w.r.t. the query points, letting the contact/penetration
energies pull the hand onto the object surface.

Based on Dexgraspnet: https://pku-epic.github.io/DexGraspNet/
"""

import contextlib
import glob
import os

import numpy as np
import torch

from .pytorch3d_compat import Meshes, sample_farthest_points, sample_points_from_meshes
import trimesh as tm

# Try to import pytorch3d, but make it optional
try:
    import pytorch3d.ops
    import pytorch3d.structures

    PYTORCH3D_AVAILABLE = True
except ImportError:
    PYTORCH3D_AVAILABLE = False

SDF_BACKEND = os.environ.get("SDF_BACKEND", "WARP").upper()

if SDF_BACKEND == "TORCHSDF":
    from torchsdf import compute_sdf, index_vertices_by_faces
elif SDF_BACKEND == "WARP":
    import warp as wp

    from graspqp.utils import warp as wp_utils
elif SDF_BACKEND == "KAOLIN":
    import kaolin

with contextlib.suppress(ImportError):
    import plotly.graph_objects as go

import time


class ObjectModel:
    """Batched object meshes with a differentiable signed distance field.

    Holds a list of object meshes (one entry per object code), each replicated
    ``batch_size_each`` times so a batch of grasps can be optimized against the
    same object in parallel. After :meth:`initialize`, exposes the sampled
    surface point cloud (``surface_points_tensor``), per-object scales
    (``object_scale_tensor``) and the SDF query :meth:`cal_distance`.

    Attributes:
        device: Torch device holding the tensors.
        batch_size_each (int): Number of grasps optimized per object.
        num_samples (int): Number of surface points sampled per object.
        sdf_library (str): Active SDF backend (``"WARP"``, ``"TORCHSDF"`` or
            ``"KAOLIN"``).
        object_scale_tensor (torch.Tensor): ``(n_objects, batch_size_each)``
            per-grasp object scales.
        surface_points_tensor (torch.Tensor): ``(n_objects * batch_size_each,
            num_samples, 3)`` sampled surface points.
    """

    def __init__(self, data_root_path, batch_size_each, scale=1.0, num_samples=2000, device="cuda"):
        """Create an object model (meshes are loaded later in :meth:`initialize`).

        Args:
            data_root_path (str): Root directory containing per-object mesh
                folders.
            batch_size_each (int): Batch size (number of grasps) per object.
            scale (float): Global scale factor applied to every loaded mesh's
                vertices (in meters).
            num_samples (int): Number of object surface points to sample with
                farthest-point sampling. If 0, surface sampling is skipped.
            device (str | torch.device): Device for the torch tensors.
        """

        self.device = device
        self.batch_size_each = batch_size_each
        self.data_root_path = data_root_path
        self.num_samples = num_samples

        self.object_code_list = None
        self.object_scale_tensor = None
        self.scale = scale
        self.object_mesh_list = None
        self.object_face_verts_list = None
        # self.scale_choice = torch.tensor([0.06, 0.08, 0.1, 0.12, 0.15], dtype=torch.float, device=self.device)u
        self.scale_choice = torch.tensor([1.0], dtype=torch.float, device=self.device)
        self.sdf_library = SDF_BACKEND
        # Build WARP meshes with generalized-winding-number support and resolve inside/outside with
        # it (robust for non-watertight / inconsistently-wound meshes). Off by default (faster BVH,
        # pseudo-normal sign). Set via ``initialize(use_winding_number=True)``.
        self.use_winding_number = False
        self._cog = None

    @property
    def cog(self):
        """torch.Tensor: ``(n_objects * batch_size_each, 3)`` centroid of each
        object's sampled surface points, computed lazily and cached."""
        if self._cog is None:
            self._cog = self.surface_points_tensor.mean(dim=1)
        return self._cog

    def initialize(
        self,
        object_code_list,
        sdf_library=SDF_BACKEND,
        resample_with_fps=True,
        extension=".obj",
        convention=None,
        use_winding_number=False,
    ):
        """Load meshes, choose per-grasp scales and sample surface points.

        Populates ``object_mesh_list``, ``object_scale_tensor``,
        ``object_face_verts_list`` (the backend-specific SDF acceleration
        structure) and ``surface_points_tensor``.

        Args:
            object_code_list (list[str] | str): Object code(s) naming
                subdirectories under ``data_root_path``. A bare string is
                promoted to a single-element list.
            sdf_library (str): SDF backend to build, one of ``"WARP"``,
                ``"TORCHSDF"`` or ``"KAOLIN"``. Defaults to the module-level
                ``SDF_BACKEND``.
            resample_with_fps (bool): Kept for API compatibility (surface
                sampling always uses farthest-point sampling).
            extension (str): Mesh file extension to search for when the default
                ``coacd`` mesh paths are absent.
            convention (str | None): Up-axis convention of the source meshes.
                ``"y-up"`` remaps axes to z-up; ``"z-up"`` (or None) leaves them
                unchanged.
            use_winding_number (bool): WARP backend only. Build the meshes with
                generalized-winding-number support and use it to resolve
                inside/outside in the SDF query -- robust for meshes that are
                not perfectly watertight / consistently wound. Off by default
                (faster BVH build, pseudo-normal sign).

        Raises:
            ValueError: If an object mesh cannot be found, has too few vertices,
                or ``convention`` is unrecognized.
        """
        self.sdf_library = sdf_library.upper()
        self.use_winding_number = use_winding_number
        if not isinstance(object_code_list, list):
            object_code_list = [object_code_list]
        self.object_code_list = object_code_list
        self.object_scale_tensor = []
        self.object_mesh_list = []
        self.object_face_verts_list = []
        self.surface_points_tensor = []
        for object_code in object_code_list:
            self.object_scale_tensor.append(
                self.scale_choice[
                    torch.randint(0, self.scale_choice.shape[0], (self.batch_size_each,), device=self.device)
                ]
            )
            mesh_path = os.path.join(self.data_root_path, object_code, "coacd", "remeshed.obj")
            if not os.path.exists(mesh_path):
                mesh_path = os.path.join(self.data_root_path, object_code, "coacd", "decomposed.obj")
            if not os.path.exists(mesh_path):
                print(f"Mesh not found, trying to find {extension} files in the directory")
                meshes = glob.glob(os.path.join(self.data_root_path, object_code, f"*{extension}"))
                # assume .usd files
                self.object_scale_tensor[-1] = 0 * self.object_scale_tensor[-1] + 1.0
                # check if remshed mesh is available
                remeshed_meshes = [mesh for mesh in meshes if "remeshed.obj" in mesh]
                if len(remeshed_meshes) == 1:
                    mesh_path = remeshed_meshes[0]
                else:
                    if len(meshes) == 0:
                        raise ValueError(f"Object {object_code} not found")
                    if len(meshes) > 1:
                        print("Warning: multiple meshes found, using the first one. Please check the data.")
                    mesh_path = meshes[0]

            print(f"Loading object {object_code} from {mesh_path}")
            mesh = tm.load(mesh_path, force="mesh", process=True)
            if len(mesh.vertices) < 100:
                raise ValueError(f"Object {object_code} has too few vertices, please check the data.")

            if convention is not None:
                if convention == "y-up":
                    # need to flip y and z
                    x = mesh.vertices[:, 0].copy()
                    y = mesh.vertices[:, 1].copy()
                    z = mesh.vertices[:, 2].copy()

                    mesh.vertices[:, 1] = -z
                    mesh.vertices[:, 2] = y
                    mesh.vertices[:, 0] = x
                elif convention == "z-up":
                    pass
                else:
                    raise ValueError(f"Unknown convention {convention}")
            mesh.vertices = mesh.vertices * self.scale
            self.object_mesh_list.append(mesh)
            # self.object_mesh_list.append(tm.load(os.path.join(self.data_root_path, object_code, "coacd", "decomposed.obj"), force="mesh", process=False))

            object_verts = torch.Tensor(self.object_mesh_list[-1].vertices).to(self.device)
            object_faces = torch.Tensor(self.object_mesh_list[-1].faces).long().to(self.device)
            # drop vertices below zero
            # mask = object_verts[..., -1] > 0.005 # 5mm
            # invalid_idx = torch.where(mask == False)[0]
            # invalid_faces = torch.isin(object_faces.view(-1), invalid_idx).view(-1, 3).any(dim=-1)
            # object_faces = object_faces[~invalid_faces]
            if self.sdf_library == "TORCHSDF":
                self.object_face_verts_list.append(index_vertices_by_faces(object_verts, object_faces))
            elif self.sdf_library == "WARP":

                # create warp mesh from vertices and faces
                link_vertices, link_faces = object_verts.cpu().numpy(), object_faces.cpu().numpy()
                verts_wp = wp.from_numpy(np.ascontiguousarray(link_vertices), device=str(self.device), dtype=wp.vec3)
                faces_wp = wp.from_numpy(
                    np.ascontiguousarray(link_faces.flatten()), device=str(self.device), dtype=wp.int32
                )
                wp_mesh = wp.Mesh(points=verts_wp, indices=faces_wp, support_winding_number=self.use_winding_number)

                self.object_face_verts_list.append(wp_mesh)
            elif self.sdf_library == "KAOLIN":
                link_face_verts = kaolin.ops.mesh.index_vertices_by_faces(object_verts.unsqueeze(0), object_faces)
                self.object_face_verts_list.append((link_face_verts, object_faces, object_verts))

            if self.num_samples != 0:
                vertices = torch.tensor(self.object_mesh_list[-1].vertices, dtype=torch.float, device=self.device)
                faces = torch.tensor(self.object_mesh_list[-1].faces, dtype=torch.float, device=self.device)
                mesh = Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))

                dense_point_cloud = sample_points_from_meshes(mesh, num_samples=100 * self.num_samples)

                surface_points = sample_farthest_points(dense_point_cloud, K=self.num_samples)[0][0]

                surface_points = surface_points.to(dtype=torch.float, device=self.device)
                self.surface_points_tensor.append(surface_points)
        self.object_scale_tensor = torch.stack(self.object_scale_tensor, dim=0)

        if self.num_samples != 0:
            self.surface_points_tensor = torch.stack(self.surface_points_tensor, dim=0).repeat_interleave(
                self.batch_size_each, dim=0
            )  # (n_objects * batch_size_each, num_samples, 3)

    def cal_distance(self, x, with_closest_points=False):
        """Query the object SDF at a batch of points and return contact normals.

        The distance sign convention is: interior points are positive, exterior
        points are negative. The query is differentiable w.r.t. ``x`` so grasp
        energies can backprop contact gradients through it. Points are divided
        by the per-object scale before the query and the returned distances are
        rescaled back to meters.

        Args:
            x (torch.Tensor): ``(B, n_contact, 3)`` hand contact points in the
                object frame (meters). ``B`` must be ``n_objects * batch_size_each``.
            with_closest_points (bool): If True, also return the closest points
                on the object meshes.

        Returns:
            tuple: ``(distance, normals)`` with shapes ``(B, n_contact)`` and
            ``(B, n_contact, 3)`` -- signed distances (inside positive) and the
            outward contact normal at each point. When ``with_closest_points``
            is True, a third tensor ``closest_points`` of shape
            ``(B, n_contact, 3)`` is appended.
        """
        _, n_points, _ = x.shape
        x = x.reshape(-1, self.batch_size_each * n_points, 3)
        distance = []
        normals = []
        closest_points = []
        scale = self.object_scale_tensor.repeat_interleave(n_points, dim=1)
        x = x / scale.unsqueeze(2)
        for i in range(len(self.object_mesh_list)):
            if self.sdf_library == "TORCHSDF":
                face_verts = self.object_face_verts_list[i]
                dis, dis_signs, normal, _ = compute_sdf(x[i], face_verts)
            elif self.sdf_library == "WARP":
                mesh = self.object_face_verts_list[i]
                # Differentiable w.r.t. the query points x[i]. The bare calc_sdf_field crossed the
                # torch<->warp boundary with no autograd hook, so the hand contact points received
                # NO gradient from the object SDF (force-closure / contact energies couldn't pull
                # the hand onto the surface). CalcSdfField supplies d(sdf)/dx = outward normal.
                dis_local, normal = wp_utils.CalcSdfField.apply(x[i], mesh.id, 1e6, self.use_winding_number)
                dis_signs = torch.where(dis_local > 0, 1, -1)
                dis = dis_local**2
            else:
                # dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                (face_verts, face_indexes, verts) = self.object_face_verts_list[i]
                # capture the closest-face index (was discarded) to build the contact normal.
                # kaolin returns a leading batch dim (1, M); the downstream (and the WARP/TORCHSDF
                # branches) work on (M, ...), so squeeze it off to keep shapes consistent -- the
                # original KAOLIN branch was shape-broken too, not just missing `normal`.
                dis, closest_face_idx, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
                    x[i].unsqueeze(0), face_verts
                )
                dis = dis.squeeze(0)  # (M,)
                closest_face_idx = closest_face_idx.squeeze(0).long()  # (M,)
                dis_signs = kaolin.ops.mesh.check_sign(verts.unsqueeze(0), face_indexes, x[i].unsqueeze(0)).squeeze(0)
                dis_signs = torch.where(
                    dis_signs,
                    -1 * torch.ones_like(dis_signs, dtype=torch.int32),
                    torch.ones_like(dis_signs, dtype=torch.int32),
                )
                # Outward face normal of the closest face -- the KAOLIN branch previously never set
                # `normal`, so cal_distance raised UnboundLocalError. Matches the WARP branch's
                # wp.mesh_eval_face_normal (same mesh winding); constant w.r.t. the query point.
                cf = face_verts[0, closest_face_idx]  # (M, 3, 3)
                normal = torch.nn.functional.normalize(
                    torch.linalg.cross(cf[:, 1] - cf[:, 0], cf[:, 2] - cf[:, 0], dim=-1), dim=-1
                )  # (M, 3)

            if with_closest_points:
                closest_points.append(x[i] - dis.sqrt().unsqueeze(1) * normal)

            dis = torch.sqrt(dis + 1e-8)
            dis = dis * (-dis_signs)
            distance.append(dis)
            normals.append(normal * dis_signs.unsqueeze(1))
        distance = torch.stack(distance)
        normals = torch.stack(normals)
        distance = distance * scale
        distance = distance.reshape(-1, n_points)
        normals = normals.reshape(-1, n_points, 3)

        if with_closest_points:
            closest_points = (torch.stack(closest_points) * scale.unsqueeze(2)).reshape(-1, n_points, 3)
            return distance, normals, closest_points
        return distance, normals

    def get_plotly_data(self, i, color="lightgreen", opacity=1.0, pose=None, simplify=True, offset=[0, 0, 0]):
        """
        Get visualization data for plotly.graph_objects

        Parameters
        ----------
        i: int
            index of data
        color: str
            color of mesh
        opacity: float
            opacity
        pose: (4, 4) matrix
            homogeneous transformation matrix

        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        model_index = i // self.batch_size_each
        model_scale = self.object_scale_tensor[model_index, i % self.batch_size_each].detach().cpu().numpy()
        mesh = self.object_mesh_list[model_index]
        vertices = mesh.vertices * model_scale
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
            vertices = vertices @ pose[:3, :3].T + pose[:3, 3]

        if len(vertices) > 2000 and simplify:
            import open3d as o3d

            o3d_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(vertices), triangles=o3d.utility.Vector3iVector(mesh.faces)
            )
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(2000)
            mesh = tm.Trimesh(vertices=np.array(o3d_mesh.vertices), faces=np.array(o3d_mesh.triangles))
        if offset is not None:
            vertices += np.array(offset)
        data = go.Mesh3d(
            x=vertices[:, 0],
            y=vertices[:, 1],
            z=vertices[:, 2],
            i=mesh.faces[:, 0],
            j=mesh.faces[:, 1],
            k=mesh.faces[:, 2],
            color=color,
            opacity=opacity,
            legendgroup="Object",
            showlegend=True,
        )
        all_data = [data]

        return all_data

    def get_open3d_data(self, i, pose=None):
        """
        Get visualization data for open3d.geometry

        Parameters
        ----------
        i: int
            index of data
        pose: (4, 4) matrix
            homogeneous transformation matrix

        Returns
        -------
        data: open3d.geometry.TriangleMesh
        """
        import open3d as o3d

        model_index = i // self.batch_size_each
        model_scale = self.object_scale_tensor[model_index, i % self.batch_size_each].detach().cpu().numpy()
        mesh = self.object_mesh_list[model_index]
        vertices = mesh.vertices * model_scale
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
            vertices = vertices @ pose[:3, :3].T + pose[:3, 3]
        data = tm.Trimesh(vertices, mesh.faces)

        vertices, faces = data.vertices, data.faces
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        return mesh
