"""
Based on Dexgraspnet: https://pku-epic.github.io/DexGraspNet/
"""

import contextlib
import glob
import os

import numpy as np
import pytorch3d.ops
import pytorch3d.structures
import torch
import trimesh as tm

SDF_BACKEND = os.environ.get("SDF_BACKEND", "TORCHSDF").upper()

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
    def __init__(self, data_root_path, batch_size_each, scale=1.0, num_samples=2000, device="cuda"):
        """
        Create a Object Model

        Parameters
        ----------
        data_root_path: str
            directory to object meshes
        batch_size_each: int
            batch size for each objects
        num_samples: int
            numbers of object surface points, sampled with fps
        device: str | torch.Device
            device for torch tensors
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
        self._cog = None

    @property
    def cog(self):
        if self._cog is None:
            self._cog = self.surface_points_tensor.mean(dim=1)
        return self._cog

    def initialize(
        self, object_code_list, sdf_library=SDF_BACKEND, resample_with_fps=True, extension=".obj", convention=None
    ):
        """
        Initialize Object Model with list of objects

        Choose scales, load meshes, sample surface points

        Parameters
        ----------
        object_code_list: list | str
            list of object codes
        """
        self.sdf_library = sdf_library.upper()
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
                    # if len(meshes) == 0:
                    #     raise ValueError(f"Object {object_code} not found")
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
                wp_mesh = wp.Mesh(points=verts_wp, indices=faces_wp)

                self.object_face_verts_list.append(wp_mesh)
            elif self.sdf_library == "KAOLIN":
                link_face_verts = kaolin.ops.mesh.index_vertices_by_faces(object_verts.unsqueeze(0), object_faces)
                self.object_face_verts_list.append((link_face_verts, object_faces, object_verts))

            if self.num_samples != 0:
                vertices = torch.tensor(self.object_mesh_list[-1].vertices, dtype=torch.float, device=self.device)
                faces = torch.tensor(self.object_mesh_list[-1].faces, dtype=torch.float, device=self.device)
                mesh = pytorch3d.structures.Meshes(vertices.unsqueeze(0), faces.unsqueeze(0))

                dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * self.num_samples)

                if resample_with_fps:
                    surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=self.num_samples)[0][0]
                else:
                    # sample random points
                    surface_points = dense_point_cloud[0].clone()
                    surface_points = surface_points[torch.randint(0, surface_points.shape[0], (self.num_samples,))]

                surface_points.to(dtype=float, device=self.device)
                self.surface_points_tensor.append(surface_points)
        self.object_scale_tensor = torch.stack(self.object_scale_tensor, dim=0)

        if self.num_samples != 0:
            self.surface_points_tensor = torch.stack(self.surface_points_tensor, dim=0).repeat_interleave(
                self.batch_size_each, dim=0
            )  # (n_objects * batch_size_each, num_samples, 3)

    def cal_distance(self, x, with_closest_points=False):
        """
        Calculate signed distances from hand contact points to object meshes and return contact normals

        Interiors are positive, exteriors are negative

        Use our modified Kaolin package

        Parameters
        ----------
        x: (B, `n_contact`, 3) torch.Tensor
            hand contact points
        with_closest_points: bool
            whether to return closest points on object meshes

        Returns
        -------
        distance: (B, `n_contact`) torch.Tensor
            signed distances from hand contact points to object meshes, inside is positive
        normals: (B, `n_contact`, 3) torch.Tensor
            contact normal vectors defined by gradient
        closest_points: (B, `n_contact`, 3) torch.Tensor
            contact points on object meshes, returned only when `with_closest_points is True`
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
                pts = wp.from_torch(x[i], dtype=wp.vec3)
                distance_wp, normal_wp = wp_utils.calc_sdf_field(points_wp=pts, mesh_id=mesh.id)
                dis_local = wp.to_torch(distance_wp)
                dis_signs = torch.where(dis_local > 0, 1, -1)
                dis = dis_local**2
                normal = wp.to_torch(normal_wp)
            else:
                # dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                (face_verts, face_indexes, verts) = self.object_face_verts_list[i]
                dis, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(x[i].unsqueeze(0), face_verts)
                dis_signs = kaolin.ops.mesh.check_sign(verts.unsqueeze(0), face_indexes, x[i].unsqueeze(0))
                dis_signs = torch.where(
                    dis_signs,
                    -1 * torch.ones_like(dis_signs, dtype=torch.int32),
                    torch.ones_like(dis_signs, dtype=torch.int32),
                )

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
