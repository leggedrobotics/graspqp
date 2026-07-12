# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

from __future__ import annotations

import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import omni.log
from pxr import UsdPhysics

import isaaclab.sim as sim_utils

from isaaclab.assets.articulation import Articulation
from .articulation_model_data import ArticulationModelData

if TYPE_CHECKING:
    from .articulation_model_cfg import ArticulationModelCfg

from isaaclab.ui.components.list.list_component import ListComponent
import trimesh
import weakref

from typing import TYPE_CHECKING, ClassVar

import carb
import warp as wp
import hashlib
import isaaclab.utils.string as string_utils

from isaaclab.utils.math import transform_points, quat_inv, quat_apply, quat_apply, matrix_from_quat
# These mesh helpers were removed from isaaclab.utils.mesh; graspqp keeps local copies in
# object_model (which already defines them for the rigid-object path).
from graspqp_isaaclab.models.object_model import (
    PRIMITIVE_MESH_TYPES,
    create_mesh_from_geom_shape,
    create_trimesh_from_geom_mesh,
)
# Use graspqp's own convert_to_warp_mesh (supports `support_winding_number`); the stock
# isaaclab.utils.warp version does not accept that argument. Matches object_model.py.
from graspqp_isaaclab.utils.warp import convert_to_warp_mesh

import numpy as np

# no-op profiling context (python_utils dependency dropped)
from contextlib import nullcontext as timer
from isaacsim.core.utils.stage import get_current_stage

try:
    import omni.ui
except ImportError:
    pass

from graspqp_isaaclab.utils import warp as wp_mesh


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


def _registered_points_idx(
    points: np.ndarray, registered_points: dict[str, list[tuple[np.ndarray, int]]]
) -> tuple[int, str]:
    """Check if the points are already registered in the list of registered points.

    Args:
        points: The points to check.
        registered_points: The list of registered points.

    Returns:
        The index of the registered points if found, otherwise -1.
    """
    hashstr = hashlib.md5(points.tobytes()).hexdigest()
    if hashstr not in registered_points or len(registered_points[hashstr]) == 0:
        registered_points[hashstr] = []
        return -1, hashstr

    for reg_points, idx in registered_points[hashstr]:
        if reg_points.shape == points.shape and (reg_points == points).all():
            return idx, hashstr
    return -1, hashstr


class ArticulationModel(Articulation, ListComponent):
    cfg: ArticulationModelCfg

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

    local_sdf_tfs: ClassVar[dict[str, torch.Tensor]] = {}
    """The views of the meshes available for raycasting.

    The keys correspond to the prim path for the meshes, and values are the corresponding views of the prims.

    .. note::
           We store a global dictionary of all views to prevent re-loading for different ray-cast sensor instances.
    """

    def __init__(self, cfg: ArticulationModelCfg):
        # Ugly visualizer solution.
        self._debug_vis_cb_fnc = None
        self._debug_vis_toggle_fnc = None
        self._sdf_views = {}

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        self._setup_vis_terms(terms=["Surface Pointcloud", "SDF", "Net Handle Force"])
        self._vis_frame = None
        self._num_meshes_per_env = {}

        super(Articulation, self).__init__(cfg)

        # self._n_surface_pts = cfg.n_surface_pts
        # self._mesh_target = cfg.mesh_target_cfg

    def _initialize_impl(self):
        super()._initialize_impl()
        # load the meshes by parsing the stage
        self._initialize_warp_meshes()
        self._initialize_sdf_views()

        # process min values joint limits
        if self.cfg.joint_pos_limits_min is not None:
            indices_list, _, values_list = string_utils.resolve_matching_names_values(
                self.cfg.joint_pos_limits_min, self.joint_names
            )
            if len(indices_list) > 0:
                self._data.joint_limits[:, indices_list, 0] = torch.tensor(values_list, device=self.device)
                self.write_joint_limits_to_sim(self._data.joint_limits)

        # process max values joint limits
        if self.cfg.joint_pos_limits_max is not None:
            indices_list, _, values_list = string_utils.resolve_matching_names_values(
                self.cfg.joint_pos_limits_max, self.joint_names
            )
            if len(indices_list) > 0:
                self._data.joint_limits[:, indices_list, 1] = torch.tensor(values_list, device=self.device)
                self.write_joint_limits_to_sim(self._data.joint_limits)

        # Update soft joint limits
        joint_pos_mean = (self._data.joint_limits[..., 0] + self._data.joint_limits[..., 1]) / 2
        joint_pos_range = self._data.joint_limits[..., 1] - self._data.joint_limits[..., 0]
        soft_limit_factor = self.cfg.soft_joint_pos_limit_factor
        # add to data
        self._data.soft_joint_pos_limits[..., 0] = joint_pos_mean - 0.5 * joint_pos_range * soft_limit_factor
        self._data.soft_joint_pos_limits[..., 1] = joint_pos_mean + 0.5 * joint_pos_range * soft_limit_factor

    def _calc_sdf_chuncked(self, contact_pts_b: torch.Tensor, sdf_view: object):
        if contact_pts_b.shape[1] == 0:
            return contact_pts_b.new_empty((*contact_pts_b.shape[:2], 4))

        results = []
        for i in range(0, contact_pts_b.shape[1], sdf_view.max_num_points):
            chunk = contact_pts_b[:, i : i + sdf_view.max_num_points]

            padding = -chunk.shape[1]
            if sdf_view.max_num_points > chunk.shape[1]:
                # we don't have enough query points to fill the SDF view,
                # randomly add zeros to fill the view
                padding = sdf_view.max_num_points - chunk.shape[1]
                missing_pts = torch.zeros(
                    chunk.shape[0],
                    padding,
                    chunk.shape[-1],
                    device=chunk.device,
                    dtype=chunk.dtype,
                )
                chunk = torch.cat([chunk, missing_pts], dim=1)
            result = sdf_view.get_sdf_and_gradients(chunk)[..., :-padding, :]
            results.append(result)

        return torch.cat(results, dim=1)

        # return sdf_view.get_sdf_and_gradients(contact_pts_b)

    def calc_contact_normals(
        self, contact_pts_w: torch.Tensor, env_ids: Sequence[int] | None = None, pose=None, collision=False
    ):
        if contact_pts_w.shape[1] == 0:
            return (
                contact_pts_w.new_empty((*contact_pts_w.shape[:2], 3)),
                contact_pts_w.new_empty(contact_pts_w.shape[:2]),
            )

        if len(self._sdf_views) != 0:
            values = []
            for name, sdf_view in self._sdf_views.items():
                body_pos = self.data.body_state_w[:, self.data.body_names.index(name), :7] if pose is None else pose

                contact_pts_b = transform_points(contact_pts_w - body_pos[:, None, :3], quat=quat_inv(body_pos[:, 3:]))
                result = self._calc_sdf_chuncked(contact_pts_b, sdf_view)
                # convert normals
                result[..., :-1] = quat_apply(
                    body_pos[:, None, 3:].expand(-1, result.shape[1], -1),
                    result[..., :-1],
                )
                values.append(result)

            values = torch.stack(values, dim=1)
            sdf, gradients = values[..., -1], values[..., :-1]
            sdf_min, sdf_min_idx = torch.min(sdf, dim=1, keepdim=True)
            # convert normals back to world space
            normals = torch.gather(
                gradients,
                dim=1,
                index=sdf_min_idx.unsqueeze(-1).expand(-1, -1, -1, 3),
            ).squeeze(1)
            return normals, sdf_min.squeeze(1)

        body_poses = self._data.body_state_w[:, self._data.tracked_body_mask, :7]
        if env_ids is None:
            env_ids = torch.arange(len(body_poses), device=body_poses.device)

        mesh_pos = body_poses[env_ids, :, :3]
        mesh_rot = body_poses[env_ids, :, 3:7]

        mesh_ids = torch.tensor(self._object_mesh_ids, dtype=torch.long, device=self.device)
        if not collision:
            mesh_ids = mesh_ids[-1, None]
            mesh_pos = mesh_pos[:, -1, :].unsqueeze(1)
            mesh_rot = mesh_rot[:, -1, :].unsqueeze(1)

        mesh_ids = mesh_ids.permute(1, 0, 2).flatten(1, 2)[env_ids]
        sdf, triangle_normals = wp_mesh.calc_obj_distances(
            wp.array2d(mesh_ids.tolist(), dtype=wp.uint64, device=self.device),
            mesh_pos,
            mesh_rot,
            contact_pts_w,
            max_dist=1e6,
            env_ids=None,
        )

        sdf_min, sdf_min_idx = torch.min(sdf, dim=1, keepdim=True)
        triangle_normals = torch.gather(
            triangle_normals,
            dim=1,
            index=sdf_min_idx.unsqueeze(-1).expand(-1, -1, -1, 3),
        ).squeeze(1)
        return triangle_normals, sdf_min.squeeze(1)

    def calc_contact_normals_warp(self, contact_pts_w: torch.Tensor, env_ids: Sequence[int] | None = None):
        body_poses = self._data.body_state_w[:, self._data.tracked_body_mask, :7]
        if env_ids is None:
            env_ids = torch.arange(len(body_poses), device=body_poses.device)

        mesh_pos = body_poses[env_ids, :, :3]
        mesh_rot = body_poses[env_ids, :, 3:7]

        mesh_ids = torch.tensor(self._object_mesh_ids, dtype=torch.uint64).permute(1, 0, 2).flatten(1, 2)

        sdf, triangle_normals = wp_mesh.calc_obj_distances(
            wp.array2d(mesh_ids.tolist(), dtype=wp.uint64, device=self.device),
            mesh_pos,
            mesh_rot,
            contact_pts_w,
            max_dist=1e6,
            env_ids=env_ids,
        )

        sdf_min, sdf_min_idx = torch.min(sdf, dim=1, keepdim=True)
        triangle_normals = torch.gather(
            triangle_normals,
            dim=1,
            index=sdf_min_idx.unsqueeze(-1).expand(-1, -1, -1, 3),
        ).squeeze(1)
        return triangle_normals, sdf_min.squeeze(1)

    def _initialize_sdf_views(self):
        # for name, cfg in self.cfg.sdf_targets_cfg.items():
        #     # get the mesh target sd
        #     if name not in self._data.body_names:
        #         raise ValueError(
        #             f"Mesh target {name} not found in the articulation body names. Body Names: {self._data.body_names}"
        #         )

        # sort to make sure mesh order aligns with body names
        for name, cfg in self.cfg.sdf_targets_cfg.items():
            try:
                self._sdf_views[name] = self._physics_sim_view.create_sdf_shape_view(
                    cfg.target_prim_expr.replace(".*", "*"), cfg.max_sdf_pts
                )
                if self._sdf_views[name] is None or not self._sdf_views[name].check():
                    raise AttributeError(f"Failed to create SDF view for prim path: {cfg.target_prim_expr}")

            except AttributeError as e:
                if self.device.startswith("cpu"):
                    # PhysX SDF shape views are GPU-only (CpuSimulationView::createSdfShapeView
                    # is unimplemented). Fall back to the warp-mesh SDF path, which
                    # calc_contact_normals uses whenever _sdf_views is empty.
                    carb.log_warn(
                        "SDF shape views are not supported on CPU physics; falling back to "
                        "warp-mesh SDF for all targets."
                    )
                    self._sdf_views = {}
                    return
                print(f"Failed to create SDF view for prim path: {cfg.target_prim_expr}")
                print(
                    "Available prim paths with collision api:",
                    sim_utils.get_all_matching_child_prims(
                        "/",
                        lambda x: x.HasAPI(UsdPhysics.CollisionAPI) and "env_0" in str(x),
                    ),
                )

                raise e

    def _initialize_warp_meshes(self):

        mesh_cache = {}
        mesh_cache_normals = {}
        for name, cfg in self.cfg.mesh_targets_cfg.items():
            # get the mesh target prim
            if name not in self._data.body_names:
                raise ValueError(
                    f"Mesh target {name} not found in the articulation body names. Body Names: {self._data.body_names}"
                )

        # sort to make sure mesh order aligns with body names
        self.cfg.mesh_targets_cfg = {
            k: v
            for k, v in sorted(
                self.cfg.mesh_targets_cfg.items(),
                key=lambda item: self._data.body_names.index(item[0]),
            )
        }

        for name, cfg in self.cfg.mesh_targets_cfg.items():
            wp_meshes = []

            prim_path = cfg.target_prim_expr

            paths = sim_utils.find_matching_prim_paths(prim_path)
            if len(paths) == 0:
                raise RuntimeError(f"Failed to find a prim at path expression: {prim_path}")

            ArticulationModel.mesh_views[prim_path] = _get_prim_view(prim_path, self._physics_sim_view)
            loaded_vertices: dict[str, list[tuple[np.ndarray, int]]] = {}

            for path_idx, path in enumerate(paths):

                # check if the prim is a primitive object - handle these as special types
                mesh_prim = sim_utils.get_first_matching_child_prim(
                    path, lambda prim: prim.GetTypeName() in PRIMITIVE_MESH_TYPES
                )
                with timer("mesh_parsing"):
                    # print("Mesh prim:", mesh_prim)
                    if mesh_prim is None:
                        # obtain the mesh prim
                        mesh_prim = sim_utils.get_first_matching_child_prim(
                            path, lambda prim: prim.GetTypeName() == "Mesh"
                        )
                        main_prim = get_current_stage().GetPrimAtPath(path)
                        if mesh_prim is None or not mesh_prim.IsValid():
                            raise RuntimeError(f"Invalid mesh prim path: {paths}")

                        points, faces = create_trimesh_from_geom_mesh(mesh_prim)
                        mesh = trimesh.Trimesh(vertices=points, faces=faces, process=False)
                        scale = sim_utils.resolve_prim_scale(mesh_prim)
                        mesh.apply_scale(scale)

                        relative_pos, relative_quat = sim_utils.resolve_prim_pose(mesh_prim, main_prim)
                        relative_pos = torch.tensor(relative_pos, dtype=torch.float32)
                        relative_quat = torch.tensor(relative_quat, dtype=torch.float32)
                        rotation = matrix_from_quat(relative_quat)
                        transform = np.eye(4)
                        transform[:3, :3] = rotation.numpy()
                        transform[:3, 3] = relative_pos.numpy()
                        mesh.apply_transform(transform)
                        points, faces = mesh.vertices, mesh.faces

                        # if mesh_prim is None or not mesh_prim.IsValid():
                        #     raise RuntimeError(f"Invalid mesh prim path: {paths}")

                        # # print("Processing mesh prim:", mesh_prim.GetPath())
                        # # print("Full paths:", paths)

                        # points, faces = create_trimesh_from_geom_mesh(mesh_prim)
                        # points *= np.array(sim_utils.resolve_world_scale(mesh_prim))
                        # # print("applying scale", sim_utils.resolve_world_scale(mesh_prim))

                        if False and (
                            str(ArticulationModel.mesh_views[prim_path].prim_paths[path_idx])
                            != str(mesh_prim.GetPath())
                            and cfg.offset_compensation
                        ):
                            # find relative path
                            parent_prim = sim_utils.find_matching_prims(
                                ArticulationModel.mesh_views[prim_path].prim_paths[path_idx]
                            )[0]
                            import pdb

                            pdb.set_trace()
                            pos, orientation = sim_utils.get_relative_chain_pose_from_usd(mesh_prim, parent_prim)
                            # print("Set local tf for paath:,", prim_path)
                            # ArticulationModel.local_tfs[path] = torch.cat([pos, orientation], dim=-1)
                            points = (
                                transform_points(
                                    torch.from_numpy(points).to(pos.device, dtype=pos.dtype),
                                    pos,
                                    orientation,
                                )
                                .cpu()
                                .numpy()
                            )
                            wp_mesh = convert_to_warp_mesh(
                                points, faces, device=self.device, support_winding_number=True
                            )
                            print(
                                f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(faces)} faces."
                            )
                        else:
                            wp_mesh = convert_to_warp_mesh(
                                points, faces, device=self.device, support_winding_number=True
                            )
                    else:
                        # print("primitive mesh prim:", mesh_prim)
                        # create mesh from primitive shape
                        main_prim = get_current_stage().GetPrimAtPath(path)
                        mesh = create_mesh_from_geom_shape(mesh_prim)

                        scale = sim_utils.resolve_prim_scale(mesh_prim)
                        relative_pos, relative_quat = sim_utils.resolve_prim_pose(mesh_prim, main_prim)

                        mesh.apply_scale(scale)
                        relative_pos = torch.tensor(relative_pos, dtype=torch.float32)
                        relative_quat = torch.tensor(relative_quat, dtype=torch.float32)
                        rotation = matrix_from_quat(relative_quat)
                        transform = np.eye(4)
                        transform[:3, :3] = rotation.numpy()
                        transform[:3, 3] = relative_pos.numpy()
                        mesh.apply_transform(transform)
                        points, faces = mesh.vertices, mesh.faces

                        if False and str(ArticulationModel.mesh_views[prim_path].prim_paths[path_idx]) != str(
                            mesh_prim.GetPath()
                        ):

                            # find relative path
                            parent_prim = sim_utils.find_matching_prims(
                                ArticulationModel.mesh_views[prim_path].prim_paths[path_idx]
                            )[0]

                            points = (
                                sim_utils.convert_points_from_usd(
                                    mesh_prim,
                                    parent_prim,
                                    torch.from_numpy(points).to(self.device, dtype=torch.float32),
                                )
                                .cpu()
                                .numpy()
                            )
                            wp_mesh = convert_to_warp_mesh(
                                points, faces, device=self.device, support_winding_number=False
                            )
                            carb.log_info(
                                f"Read mesh prim: {mesh_prim.GetPath()} with {len(points)} vertices and {len(faces)} faces."
                            )

                        wp_mesh = convert_to_warp_mesh(points, faces, device=self.device, support_winding_number=False)
                        carb.log_info(f"Created {mesh_prim.GetTypeName()} mesh prim: {mesh_prim.GetPath()}.")

                registered_idx, hashstr = _registered_points_idx(points, loaded_vertices)

                if registered_idx != -1:
                    # Found a duplicate mesh, only reference the mesh.
                    wp_mesh = wp_meshes[registered_idx]
                else:
                    loaded_vertices[hashstr].append((points, len(wp_meshes)))
                    wp_mesh = convert_to_warp_mesh(points, faces, device=self.device, support_winding_number=True)
                wp_meshes.append(wp_mesh)

            # split up the meshes for each environment. Little bit ugly, since
            # the current order is interleaved (env1_obj1, env1_obj2, env2_obj1, env2_obj2, ...)
            ArticulationModel.meshes[prim_path] = []
            mesh_idx = 0
            n_meshes_per_env = len(wp_meshes) // self.num_instances
            self._num_meshes_per_env[prim_path] = n_meshes_per_env

            surface_pts = []
            surface_normals = []
            for i in range(self.num_instances):
                ArticulationModel.meshes[prim_path].append(wp_meshes[mesh_idx : mesh_idx + n_meshes_per_env])
                mesh_idx += n_meshes_per_env

                if cfg.n_surface_pts > 0:
                    # sample points from the mesh
                    warp_mesh = ArticulationModel.meshes[prim_path][i][0]
                    pts = wp.to_torch(warp_mesh.points).to(self.device)
                    vertices = wp.to_torch(warp_mesh.indices).to(self.device)
                    # Use the compat shim so this works without pytorch3d (WARP-native path);
                    # it falls back to a pure-PyTorch implementation when pytorch3d is absent.
                    from graspqp.core.pytorch3d_compat import Meshes, sample_points_from_meshes

                    mesh = Meshes(verts=[pts], faces=[vertices.view(-1, 3).float()])
                    sampled_pts, sampled_normals = sample_points_from_meshes(
                        mesh, cfg.n_surface_pts, return_normals=True
                    )
                    surface_pts.append(sampled_pts)
                    surface_normals.append(sampled_normals)

            mesh_cache[prim_path] = torch.cat(surface_pts, dim=0).to(self.device)
            mesh_cache_normals[prim_path] = torch.cat(surface_normals, dim=0).to(self.device)
        self._data._surface_pts_b = list(mesh_cache.values())
        self._data._surface_normals_b = list(mesh_cache_normals.values())

        self._data.tracked_body_mask = torch.tensor(
            [name in self.cfg.mesh_targets_cfg for name in self._data.body_names],
            device=self.device,
        )

        # self._data.surface_pts_b = torch.cat(surface_pts)

        mesh_ids = []
        for cfg in self.cfg.mesh_targets_cfg.values():
            prim_path = cfg.target_prim_expr
            meshes = ArticulationModel.meshes[prim_path]
            mesh_ids.append([[m.id for m in b] for b in meshes])
        self._object_mesh_ids = mesh_ids

    # ===========================================================================
    # ============================== VISUALIZATION ==============================
    # ===========================================================================
    def _setup_vis_terms(self, terms=[]):
        self._terms = {}
        for term in terms:
            self._terms[term] = {"state": False}

    @property
    def data(self) -> ArticulationModelData:
        return self._data

    def _create_data(self):
        return ArticulationModelData(self.root_physx_view, self.device)

    def _vis_callback(self, event, tasks=[]):
        """Callback function for the debug visualization."""

        if "Surface Pointcloud" in tasks:
            draw_interface = sim_utils.SimulationContext.instance().draw_interface
            pts_w = self.data.surface_pts_w
            pt_idx = 0
            colors = [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 0.5, 0.5, 1.0],
                [1.0, 1.5, 1.5, 1.0],
            ]

            for idx, cfg in enumerate(self.cfg.mesh_targets_cfg.values()):

                n_pts = cfg.n_surface_pts
                pts_w_local = pts_w[:, pt_idx : pt_idx + n_pts]
                pt_idx += n_pts
                color = colors[idx % len(colors)]

                draw_interface.plot_points(
                    pts_w_local.detach().cpu().reshape(-1, 3).numpy().tolist(),
                    color=color,
                    size=5,
                )

        if "SDF" in tasks:
            draw_interface = sim_utils.SimulationContext.instance().draw_interface
            pt_idx = 0
            colors = [
                [1.0, 0.0, 0.0, 1.0],
                [0.0, 1.0, 0.0, 1.0],
                [0.0, 0.0, 1.0, 1.0],
                [1.0, 1.0, 0.0, 1.0],
                [1.0, 0.5, 0.5, 1.0],
                [1.0, 1.5, 1.5, 1.0],
            ]
            surface_pts = self.data.surface_pts_w + torch.randn_like(self.data.surface_pts_w) * 0.005
            normals, sdf = self.calc_contact_normals(surface_pts)
            start_pts = surface_pts - normals * sdf[..., None]
            end_pts = surface_pts
            inside = sdf < 0.0
            # outside = sdf > 0
            # plot outside in green
            # draw_interface.plot_lines(
            #     start_pts[~inside].detach().cpu().reshape(-1, 3).numpy().tolist(),
            #     end_pts[~inside].detach().cpu().reshape(-1, 3).numpy().tolist(),
            #     color=[0.0, 1.0, 0.0, 1.0],
            #     size=1.0,
            # )
            # plot inside in red
            draw_interface.plot_lines(
                start_pts[inside].detach().cpu().reshape(-1, 3).numpy().tolist(),
                end_pts[inside].detach().cpu().reshape(-1, 3).numpy().tolist(),
                color=[1.0, 0.0, 0.0, 1.0],
                size=5.0,
            )
            query_pc_inside = surface_pts[inside]
            draw_interface.plot_points(
                query_pc_inside.detach().cpu().reshape(-1, 3).numpy().tolist(),
                color=[1.0, 0.0, 0.0, 1.0],
                size=5,
            )
            # query_pc_outside = surface_pts[~inside]
            # draw_interface.plot_points(
            #     query_pc_outside.detach().cpu().reshape(-1, 3).numpy().tolist(),
            #     color=[0.0, 1.0, 0.0, 1.0],
            #     size=5,
            # )

        if "Net Handle Force" in tasks:
            draw_interface = sim_utils.SimulationContext.instance().draw_interface
            handle_ids = [i for i, name in enumerate(self.body_names) if "handle" in name]
            if len(handle_ids) > 0:
                # net wrench at each link's incoming joint, in the link frame: (num_envs, num_links, 6)
                wrench = self.root_physx_view.get_link_incoming_joint_force()
                force_b = wrench[:, handle_ids, :3]
                body_pos_w = self.data.body_pos_w[:, handle_ids]
                body_quat_w = self.data.body_quat_w[:, handle_ids]
                force_w = quat_apply(body_quat_w.reshape(-1, 4), force_b.reshape(-1, 3)).reshape(force_b.shape)
                # 2 cm per N, capped at 30 N so large forces stay readable
                norm = force_w.norm(dim=-1, keepdim=True)
                force_vis = force_w / norm.clamp(min=1e-6) * norm.clamp(max=30.0) * 0.02
                starts = body_pos_w.detach().cpu().reshape(-1, 3).numpy().tolist()
                ends = (body_pos_w + force_vis).detach().cpu().reshape(-1, 3).numpy().tolist()
                draw_interface.plot_lines(starts, ends, color=[1.0, 0.1, 0.8, 1.0], size=6.0)
                draw_interface.plot_points(ends, color=[1.0, 0.1, 0.8, 1.0], size=10)

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
