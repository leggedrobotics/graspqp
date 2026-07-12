# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, ClassVar
import weakref

import carb
import isaaclab.sim as sim_utils
from graspqp_isaaclab.utils.appcompat import resolve_prim_pose, resolve_prim_scale
import numpy as np
import torch
import trimesh
import warp as wp
from isaaclab.assets.rigid_object import RigidObject
from isaaclab.utils.math import matrix_from_quat
from pxr import UsdPhysics

from .object_model_data import RigidObjectModelData
from graspqp_isaaclab.utils import warp as wp_mesh
from graspqp_isaaclab.utils.warp import convert_to_warp_mesh

if TYPE_CHECKING:
    from .object_model_cfg import RigidObjectModelCfg


# Define primitive mesh types for collision detection
PRIMITIVE_MESH_TYPES = ("Sphere", "Cube", "Cylinder", "Cone", "Capsule", "Plane")

def convert_faces_to_triangles(faces: np.ndarray, point_counts: np.ndarray) -> np.ndarray:
    """Converts quad mesh face indices into triangle face indices.

    This function expects an array of faces (indices) and the number of points per face. It then converts potential
    quads into triangles and returns the new triangle face indices as a numpy array of shape (n_faces_new, 3).

    Args:
        faces: The faces of the quad mesh as a one-dimensional array. Shape is (N,).
        point_counts: The number of points per face. Shape is (N,).

    Returns:
        The new face ids with triangles. Shape is (n_faces_new, 3).
    """
    # check if the mesh is already triangulated
    if (point_counts == 3).all():
        return faces.reshape(-1, 3)  # already triangulated
    all_faces = []

    vertex_counter = 0
    # Iterates over all faces of the mesh to triangulate them.
    # could be very slow for large meshes
    for num_points in point_counts:
        # Triangulate n-gons (n>4) using fan triangulation
        for i in range(num_points - 2):
            triangle = np.array([faces[vertex_counter], faces[vertex_counter + 1 + i], faces[vertex_counter + 2 + i]])
            all_faces.append(triangle)

        vertex_counter += num_points
    return np.asarray(all_faces)


def create_trimesh_from_geom_mesh(mesh_prim):
    """Reads the vertices and faces of a mesh prim.

    The function reads the vertices and faces of a mesh prim and returns it. If the underlying mesh is a quad mesh,
    it converts it to a triangle mesh.

    Args:
        mesh_prim: The mesh prim to read the vertices and faces from.

    Returns:
        A trimesh.Trimesh object containing the mesh geometry.
    """
    from pxr import UsdGeom
    
    if mesh_prim.GetTypeName() != "Mesh":
        raise ValueError(f"Prim at path '{mesh_prim.GetPath()}' is not a mesh.")
    # cast into UsdGeomMesh
    mesh = UsdGeom.Mesh(mesh_prim)

    # read the vertices and faces
    points = np.asarray(mesh.GetPointsAttr().Get()).copy()

    # Load faces and convert to triangle if needed. (Default is quads)
    num_vertex_per_face = np.asarray(mesh.GetFaceVertexCountsAttr().Get())
    indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())
    return points, convert_faces_to_triangles(indices, num_vertex_per_face)


def create_mesh_from_geom_shape(shape_prim):
    """Create a mesh from a USD geometric shape prim."""
    from pxr import UsdGeom
    
    prim_type = shape_prim.GetTypeName()
    
    if prim_type == "Sphere":
        sphere = UsdGeom.Sphere(shape_prim)
        radius = float(sphere.GetRadiusAttr().Get() or 1.0)
        mesh = trimesh.creation.icosphere(radius=radius, subdivisions=4)
    elif prim_type == "Cube":
        cube = UsdGeom.Cube(shape_prim)
        size = float(cube.GetSizeAttr().Get() or 2.0)
        mesh = trimesh.creation.box(extents=[size, size, size])
    elif prim_type == "Cylinder":
        cylinder = UsdGeom.Cylinder(shape_prim)
        radius = float(cylinder.GetRadiusAttr().Get() or 1.0)
        height = float(cylinder.GetHeightAttr().Get() or 2.0)
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    elif prim_type == "Capsule":
        capsule = UsdGeom.Capsule(shape_prim)
        radius = float(capsule.GetRadiusAttr().Get() or 1.0)
        height = float(capsule.GetHeightAttr().Get() or 2.0)
        mesh = trimesh.creation.capsule(radius=radius, height=height)
    elif prim_type == "Cone":
        cone = UsdGeom.Cone(shape_prim)
        radius = float(cone.GetRadiusAttr().Get() or 1.0)
        height = float(cone.GetHeightAttr().Get() or 2.0)
        mesh = trimesh.creation.cylinder(radius=radius, height=height, sections=32)
    else:
        raise ValueError(f"Unsupported shape type: {prim_type}")
    
    return mesh


def _get_prim_view(prim_path_expr: str, physics_sim_view, max_depth: int = 2) -> str:
    if max_depth < 0:
        raise RuntimeError(f"Failed to find a non-xform parent prim for path expression: {prim_path_expr}")

    api_prim = sim_utils.find_first_matching_prim(prim_path_expr)
    if api_prim is None:
        raise RuntimeError(f"Failed to find a prim at path expression: {prim_path_expr}")

    if api_prim.HasAPI(UsdPhysics.ArticulationRootAPI):
        view = physics_sim_view.create_articulation_view(prim_path_expr.replace(".*", "*"))
    elif api_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        view = physics_sim_view.create_rigid_body_view(prim_path_expr.replace(".*", "*"))
    else:
        # Check if the parent prim is a physics prim
        parent_path_expr = "/".join(prim_path_expr.split("/")[:-1])
        view = _get_prim_view(parent_path_expr, physics_sim_view, max_depth - 1)
    return view


def _registered_points_idx(points: np.ndarray, registered_points: list[np.ndarray | None]) -> int:
    """Check if the points are already registered in the list of registered points.

    Args:
        points: The points to check.
        registered_points: The list of registered points.

    Returns:
        The index of the registered points if found, otherwise -1.
    """
    for idx, reg_points in enumerate(registered_points):
        if reg_points is None:
            continue
        if reg_points.shape == points.shape and (reg_points == points).all():
            return idx
    return -1


class RigidObjectModel(RigidObject):
    cfg: RigidObjectModelCfg

    meshes: ClassVar[dict[str, list[list[wp.Mesh]]]] = {}
    """The warp meshes available for raycasting. Stored as a dictionary.

    For each target_prim_cfg in the mesh_tracker_cfg.mesh_prim_paths, the dictionary stores the warp meshes
    for each environment instance. The list has shape (num_envs, num_meshes_per_env).
    Note that wp.Mesh are references to the warp mesh objects, so they are not duplicated for each environment if
    not necessary.

    The keys correspond to the prim path for the meshes, and values are the corresponding warp Mesh objects.

    .. note::
           We store a global dictionary of all warp meshes to prevent re-loading the mesh for different ray-cast sensor instances.
    """

    mesh_views: ClassVar[dict[str, object]] = {}

    local_tfs: ClassVar[dict[str, torch.Tensor]] = {}
    """The views of the meshes available for raycasting.

    The keys correspond to the prim path for the meshes, and values are the corresponding views of the prims.

    .. note::
           We store a global dictionary of all views to prevent re-loading for different ray-cast sensor instances.
    """

    def __init__(self, cfg: RigidObjectModelCfg):
        # Ugly visualizer solution.
        self._debug_vis_cb_fnc = None
        self._debug_vis_toggle_fnc = None

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        self._setup_vis_terms(terms=["Surface Pointcloud"])
        self._vis_frame = None
        self._num_meshes_per_env = {}

        super(RigidObjectModel, self).__init__(cfg)

        self._n_surface_pts = cfg.n_surface_pts
        self._mesh_target = cfg.mesh_target_cfg

    def _create_rigid_object_data(self):
        return RigidObjectModelData(self._root_physx_view, self.device)

    # def _create_rigid_object_data(self) -> RigidObjectData:
    #     """Create rigid object data instance.

    #     Returns:
    #         Rigid object data instance.
    #     """
    #     return RigidObjectData(self._root_physx_view, self.device)

    def _initialize_impl(self):
        super()._initialize_impl()
        # load the meshes by parsing the stage
        self._initialize_warp_meshes()

    def calc_contact_normals(
        self,
        contact_pts_w: torch.Tensor,
        env_ids: Sequence[int] | None = None,
        pose: torch.Tensor | None = None,
        collision=True,
    ):
        body_pose = self._data.root_state_w[:, :7]

        if env_ids is None:
            env_ids = torch.arange(len(body_pose), device=body_pose.device)

        if pose is None:
            mesh_pos = body_pose[env_ids, :3].unsqueeze(1)
            mesh_rot = body_pose[env_ids, 3:7].unsqueeze(1)
        else:
            mesh_pos = pose[:, :3].unsqueeze(1)
            mesh_rot = pose[:, 3:7].unsqueeze(1)

        sdf, triangle_normals = wp_mesh.calc_obj_distances(
            wp.array2d(self._object_mesh_ids, dtype=wp.uint64, device=self.device),
            mesh_pos,
            mesh_rot,
            contact_pts_w,
            max_dist=1e6,
            env_ids=env_ids,
        )
        sdf = sdf.squeeze(1)
        triangle_normals = triangle_normals.squeeze(1)
        return triangle_normals, sdf

    def _initialize_warp_meshes(self):

        target_cfg = self._mesh_target
        prim_path = target_cfg.target_prim_expr

        paths = sim_utils.find_matching_prim_paths(prim_path)
        if len(paths) == 0:
            raise RuntimeError(f"Failed to find a prim at path expression: {prim_path}")

        RigidObjectModel.mesh_views[prim_path] = _get_prim_view(prim_path, self._physics_sim_view)
        loaded_vertices: list[np.ndarray | None] = []
        wp_meshes = []
        pytorch3d_meshes = []

        for path_idx, path in enumerate(paths):

            # check if the prim is a primitive object - handle these as special types
            mesh_prim = sim_utils.get_first_matching_child_prim(
                path, lambda prim: prim.GetTypeName() in PRIMITIVE_MESH_TYPES
            )

            if mesh_prim is None:
                # obtain the mesh prim
                mesh_prim = sim_utils.get_first_matching_child_prim(path, lambda prim: prim.GetTypeName() == "Mesh")
                main_prim = sim_utils.find_first_matching_prim(path)
                if mesh_prim is None or not mesh_prim.IsValid():
                    raise RuntimeError(f"Invalid mesh prim path: {paths}")

                points, faces = create_trimesh_from_geom_mesh(mesh_prim)
                mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
                scale = resolve_prim_scale(mesh_prim)
                mesh.apply_scale(scale)

                relative_pos, relative_quat = resolve_prim_pose(mesh_prim, main_prim)
                relative_pos = torch.tensor(relative_pos, dtype=torch.float32)
                relative_quat = torch.tensor(relative_quat, dtype=torch.float32)
                rotation = matrix_from_quat(relative_quat)
                transform = np.eye(4)
                transform[:3, :3] = rotation.numpy()
                transform[:3, 3] = relative_pos.numpy()
                mesh.apply_transform(transform)
                points, faces = mesh.vertices, mesh.faces
                if (
                    str(RigidObjectModel.mesh_views[prim_path].prim_paths[path_idx]) != str(mesh_prim.GetPath())
                    and False
                ):
                    raise RuntimeError(
                        f"Mesh prim path mismatch: {RigidObjectModel.mesh_views[prim_path].prim_paths[path_idx]} vs {mesh_prim.GetPath()}"
                    )
                    # find relative path
                    parent_prim = sim_utils.find_matching_prims(
                        RigidObjectModel.mesh_views[prim_path].prim_paths[path_idx]
                    )[0]
                    # pos, orientation = sim_utils.get_relative_chain_pose_from_usd(mesh_prim, parent_prim)
                    # RigidObjectModel.local_tfs[prim_path] = torch.cat([pos, orientation], dim=-1)
                    # points = (
                    #     transform_points(
                    #         torch.from_numpy(points).to(pos.device, dtype=pos.dtype),
                    #         pos,
                    #         orientation,
                    #     )
                    #     .cpu()
                    #     .numpy()
                    # )
                    wp_mesh = convert_to_warp_mesh(points, faces, device=self.device)
                    carb.log_info(
                        f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(faces)} faces."
                    )
                else:
                    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
                    carb.log_info(f"Created {mesh_prim.GetTypeName()} mesh prim: {mesh_prim.GetPath()}.")
            else:
                # create mesh from primitive shape
                mesh = create_mesh_from_geom_shape(mesh_prim)
                points, faces = mesh.vertices, mesh.faces
                main_prim = sim_utils.find_first_matching_prim(path)
                if mesh_prim is None or not mesh_prim.IsValid():
                    raise RuntimeError(f"Invalid mesh prim path: {paths}")

                mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
                scale = resolve_prim_scale(mesh_prim)
                mesh.apply_scale(scale)

                relative_pos, relative_quat = resolve_prim_pose(mesh_prim, main_prim)
                relative_pos = torch.tensor(relative_pos, dtype=torch.float32)
                relative_quat = torch.tensor(relative_quat, dtype=torch.float32)
                rotation = matrix_from_quat(relative_quat)
                transform = np.eye(4)
                transform[:3, :3] = rotation.numpy()
                transform[:3, 3] = relative_pos.numpy()
                mesh.apply_transform(transform)
                points, faces = mesh.vertices, mesh.faces

                wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
                carb.log_info(f"Created {mesh_prim.GetTypeName()} mesh prim: {mesh_prim.GetPath()}.")

            registered_idx = _registered_points_idx(points, loaded_vertices)
            if registered_idx != -1:
                # Found a duplicate mesh, only reference the mesh.
                loaded_vertices.append(None)
                wp_mesh = wp_meshes[registered_idx]
            else:
                loaded_vertices.append(points)
                wp_mesh = convert_to_warp_mesh(points, faces, device=self.device)
            wp_meshes.append(wp_mesh)

        # split up the meshes for each environment. Little bit ugly, since
        # the current order is interleaved (env1_obj1, env1_obj2, env2_obj1, env2_obj2, ...)
        RigidObjectModel.meshes[prim_path] = []
        mesh_idx = 0
        n_meshes_per_env = len(wp_meshes) // self.num_instances
        self._num_meshes_per_env[prim_path] = n_meshes_per_env

        surface_pts = []
        surface_normals = []
        for i in range(self.num_instances):
            RigidObjectModel.meshes[prim_path].append(wp_meshes[mesh_idx : mesh_idx + n_meshes_per_env])
            mesh_idx += n_meshes_per_env

            if self.cfg.n_surface_pts > 0:
                # sample points from the mesh
                warp_mesh = RigidObjectModel.meshes[prim_path][i][0]
                pts = wp.to_torch(warp_mesh.points).to(self.device)
                vertices = wp.to_torch(warp_mesh.indices).to(self.device)
                from graspqp.core.pytorch3d_compat import sample_points_from_meshes, Meshes

                mesh = Meshes(verts=[pts], faces=[vertices.view(-1, 3).float()])
                sampled_pts, sampled_normals = sample_points_from_meshes(
                    mesh, self.cfg.n_surface_pts, return_normals=True
                )
                surface_pts.append(sampled_pts)
                surface_normals.append(sampled_normals)

        self._data.surface_pts_b = torch.cat(surface_pts)
        self._data.surface_normals_b = torch.cat(surface_normals)
        self._mesh_view = RigidObjectModel.mesh_views[prim_path]

        self._meshes = RigidObjectModel.meshes[prim_path]
        self._object_mesh_ids = [[m.id for m in b] for b in self._meshes]

    # ===========================================================================
    # ============================== VISUALIZATION ==============================
    # ===========================================================================
    def _setup_vis_terms(self, terms=[]):
        self._terms = {}
        for term in terms:
            self._terms[term] = {"state": False}

    @property
    def data(self) -> RigidObjectModelData:
        return self._data

    def _create_data_struct(self):
        """Create data for storing information."""
        self._data = RigidObjectModelData(self.root_physx_view, self.device)

    def _vis_callback(self, event, tasks=[]):
        """Callback function for the debug visualization."""

        if "Surface Pointcloud" in tasks:
            draw_interface = sim_utils.SimulationContext.instance().draw_interface
            pts_w = self.data.surface_pts_w
            draw_interface.plot_points(
                pts_w.detach().cpu().reshape(-1, 3).numpy().tolist(),
                color=[0.0, 1.0, 0.0, 1.0],
                size=5,
            )

    def _debug_vis_callback(self, event):
        """Callback function for the debug visualization.

        Args:
            event: The event that triggered the callback.
        """
        tasks = [task for task in self._terms.keys() if self._terms[task]["state"]]
        self._vis_callback(event, tasks)

    def _set_vis_frame_impl(self, vis_frame: omni.ui.Window) -> None:
        """Sets the visualization frame.

        Args:
            vis_frame: The visualization frame.
        """
        print("Setting the visualization frame!!!!!!!!!!!!!!!!!!!!!!!")
        self._vis_frame = vis_frame
        self._term_visualizers = []
        self._set_debug_vis_impl(False)

    def _set_debug_vis_impl(self, debug_vis: bool):

        try:
            from omni.kit.window.extensions import SimpleCheckBox
        except ImportError:
            return
        # from omni.kit.window.extensions import SimpleCheckBox
        import isaacsim

        """Set the debug visualization implementation.

        Args:
            debug_vis: Whether to enable or disable debug visualization.
        """

        if not hasattr(self, "_vis_frame"):
            raise RuntimeError("No frame set for debug visualization.")

        if self._vis_frame is None:
            return

        # Clear internal visualizers
        self._term_visualizers = []
        self._vis_frame.clear()

        if debug_vis:
            # if enabled create a subscriber for the post update event if it doesn't exist
            if not hasattr(self, "_debug_vis_handle") or self._debug_vis_handle is None:
                app_interface = omni.kit.app.get_app_interface()
                self._debug_vis_handle = app_interface.get_post_update_event_stream().create_subscription_to_pop(
                    lambda event, obj=weakref.proxy(self): obj._debug_vis_callback(event)
                )
        else:
            # if disabled remove the subscriber if it exists
            if self._debug_vis_handle is not None:
                self._debug_vis_handle.unsubscribe()
                self._debug_vis_handle = None

            self._vis_frame.visible = False
            return

        self._vis_frame.visible = True

        with self._vis_frame:
            with omni.ui.VStack():
                for name in self._terms.keys():

                    frame = SimpleCheckBox(
                        model=omni.ui.SimpleBoolModel(),
                        enabled=True,
                        checked=False,
                        text=name,
                        on_checked_fn=lambda value, e=name: self._on_checked(e, value),
                    )
                    isaacsim.gui.components.ui_utils.add_line_rect_flourish()

                    # with frame:
                    #     self._term_visualizers.append(plot)
                    frame.collapsed = True

        self._debug_vis = debug_vis

    def _on_checked(self, name, value):
        self._terms[name]["state"] = value
