# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, Sequence

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
import omni.log
import torch
from isaaclab.assets.articulation import Articulation
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.ui.components import ListComponent
from pxr import UsdPhysics

from .hand_model_data import HandModelData

if TYPE_CHECKING:
    from .hand_model_cfg import HandModelCfg

import re
from graspqp.hands import get_hand_model

import warp as wp
from graspqp_isaaclab.utils.utils import ortho_6_from_quat


class HandModel(ListComponent, Articulation):
    cfg: HandModelCfg

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

    def __init__(self, cfg: HandModelCfg):
        self._num_meshes_per_env = {}

        ListComponent.__init__(self, cfg)
        Articulation.__init__(self, cfg)
        self.register_callbacks(self._vis_callback)

        self._root_body_index = None
        self._root_body_offset = None

    def _initialize_impl(self):
        super()._initialize_impl()

        self._data.hand_model = get_hand_model(
            self.cfg.hand_model_name,
            self.device,  # , contact_points_path=json_data, n_surface_points=2
            n_surface_points=1024 if self.cfg.surface_pts is None else self.cfg.surface_pts,
            grasp_type=self.cfg.grasp_type,
        )

        self._data.isaac_sim_to_urdf_joint_mapping = []
        self._data.urdf_to_isaac_sim_joint_mapping = []
        self._data.actuated_joint_ids = []
        for joint_name in self._data.hand_model.actuated_joints_names:
            self._data.isaac_sim_to_urdf_joint_mapping.append(self._data.joint_names.index(joint_name))
            self._data.actuated_joint_ids.append(self._data.joint_names.index(joint_name))

        for joint_name in self._data.joint_names:
            try:
                self._data.urdf_to_isaac_sim_joint_mapping.append(
                    self._data.hand_model.actuated_joints_names.index(joint_name)
                )
            except ValueError:
                pass

        print("Joint mappings initialized")
        print("Actuated joint names (Isaac Sim)", self._data.actuated_joint_names)
        print(
            "Actuated joint names (Analytical Simulator)",
            self._data.hand_model.actuated_joints_names,
        )
        # load the meshes by parsing the stage
        # self._initialize_warp_meshes()

    def calc_penetration_depth(
        self, query_points, hand_pose=None, joint_positions=None, env_ids=None, with_self_penetration=False
    ):
        if env_ids is None:
            env_ids = slice(None)

        hand_state = self._get_urdf_hand_state(pose=hand_pose, joint_positions=joint_positions, env_ids=env_ids)
        hand_model = self._data.hand_model
        if (
            hand_model.hand_pose is None
            or len(hand_model.hand_pose) != len(hand_state)
            or not (hand_model.hand_pose == hand_state).all().item()
        ):
            hand_model.set_parameters(hand_state, contact_point_indices=self.cfg.contact_mode)
        distances = hand_model.cal_distance(query_points)
        if with_self_penetration:
            return distances, hand_model.self_penetration().abs()
        return distances

    def get_surface_points(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)

        hand_state = self._get_urdf_hand_state(env_ids=env_ids)
        hand_model = self._data.hand_model
        if (
            hand_model.hand_pose is None
            or len(hand_model.hand_pose) != len(hand_state)
            or not (hand_model.hand_pose == hand_state).all().item()
        ):
            hand_model.set_parameters(hand_state, contact_point_indices=self.cfg.contact_mode)
        return hand_model.get_surface_points()

    def get_internal_hand_model(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)

        hand_state = self._get_urdf_hand_state(env_ids=env_ids)
        hand_model = self._data.hand_model
        if (
            hand_model.hand_pose is None
            or len(hand_model.hand_pose) != len(hand_state)
            or not (hand_model.hand_pose == hand_state).all().item()
        ):
            hand_model.set_parameters(hand_state, contact_point_indices=self.cfg.contact_mode)
        return hand_model

    def get_contact_points(self, env_ids=None, hand_pose=None, joint_positions=None, return_normals=False):
        if env_ids is None:
            env_ids = slice(None)

        hand_state = self._get_urdf_hand_state(env_ids=env_ids, pose=hand_pose, joint_positions=joint_positions)
        hand_model = self._data.hand_model
        if (
            hand_model.hand_pose is None
            or len(hand_model.hand_pose) != len(hand_state)
            or not (hand_model.hand_pose == hand_state).all().item()
        ):
            hand_model.set_parameters(hand_state, contact_point_indices=self.cfg.contact_mode)
        return hand_model.get_contact_points(return_normals=return_normals)

    def calc_joint_vel(self, contact_idxs, interaction_forces, env_ids=None, hand_pose=None, joint_positions=None):
        hand_model = self._data.hand_model
        hand_state = self._get_urdf_hand_state(env_ids=env_ids, pose=hand_pose, joint_positions=joint_positions)
        if (
            hand_model.hand_pose is None
            or len(hand_model.hand_pose) != len(hand_state)
            or not (hand_model.hand_pose == hand_state).all().item()
        ):
            hand_model.set_parameters(hand_state, contact_point_indices=contact_idxs)
        delta_theta_full, residuals, ee_vel = hand_model.get_req_joint_velocities(
            interaction_forces, contact_idxs, return_ee_vel=True
        )

        delta_theta_full_isaac_sim = delta_theta_full[..., self._data.urdf_to_isaac_sim_joint_mapping]
        # self.calc_ee_vel(contact_idxs, delta_theta_full, env_ids=env_ids)

        # debug
        return delta_theta_full_isaac_sim, ee_vel

    def calc_ee_vel(self, contact_idxs, joint_vel, hand_poses=None, joint_positions=None, env_ids=None):
        hand_model = self._data.hand_model
        hand_state = self._get_urdf_hand_state(env_ids=env_ids, joint_positions=joint_positions, pose=hand_poses)
        if (
            hand_model.hand_pose is None
            or len(hand_model.hand_pose) != len(hand_state)
            or not (hand_model.hand_pose == hand_state).all().item()
        ):
            hand_model.set_parameters(hand_state, contact_point_indices=contact_idxs)
        ee_vel = hand_model.get_ee_vel(joint_vel[:, self._data.urdf_to_isaac_sim_joint_mapping], contact_idxs)
        contact_pts = hand_model.get_contact_points()
        return contact_pts, ee_vel

    # ===========================================================================
    # ============================== VISUALIZATION ==============================
    # ===========================================================================
    @property
    def data(self) -> HandModelData:
        return self._data

    def _create_data(self):
        """Create data for storing information."""
        return HandModelData(self.root_physx_view, self.device)

    def _create_buffers(self):
        super()._create_buffers()

        # load actuated joint indices
        actuated_joints_expr = self.cfg.actuated_joints_expr

        if actuated_joints_expr is None:
            actuated_joints_expr = [".*"]
        elif isinstance(actuated_joints_expr, str):
            actuated_joints_expr = [actuated_joints_expr]

        mimic_joints = self.cfg.mimic_joints
        if mimic_joints is None:
            mimic_joints = {}

        self._data.actuated_joint_names = []
        self._data.actuated_joint_indices = []

        self._data.mimic_joint_names = []
        self._data.mimic_joint_indices = []
        self._data.mimic_joint_parents_indices = []

        self._data.mimic_joint_assignements = torch.zeros(self.num_joints, dtype=torch.long, device=self.device) - 1
        self._data.mimic_joint_infos = torch.zeros(self.num_joints, 2, device=self.device)

        for joint_name in self.joint_names:
            for expr in actuated_joints_expr:
                if re.fullmatch(expr, joint_name):
                    if joint_name in self._data.actuated_joint_names:
                        omni.log.warn(
                            f"Joint '{joint_name}' is already in the actuated joints list. Multiple expressions are"
                            " matching the same joint. Ignoring."
                        )
                        continue
                    self._data.actuated_joint_names.append(joint_name)
                    self._data.actuated_joint_indices.append(self.joint_names.index(joint_name))

            for mimic_joint_name in mimic_joints:
                if mimic_joint_name == joint_name:
                    omni.log.info(f"Joint '{joint_name}' is a mimic joint.")
                    omni.log.info(f"Parent: {mimic_joints[mimic_joint_name]['parent']}")
                    omni.log.info(f"Multiplier: {mimic_joints[mimic_joint_name].get('multiplier', 1.0)}")
                    omni.log.info(f"Offset: {mimic_joints[mimic_joint_name].get('offset', 0.0)}")

                    parent = mimic_joints[mimic_joint_name]["parent"]
                    parent_idx = self.joint_names.index(parent)
                    child_idx = self.joint_names.index(mimic_joint_name)
                    multiplier = mimic_joints[mimic_joint_name].get("multiplier", 1.0)
                    offset = mimic_joints[mimic_joint_name].get("offset", 0.0)
                    self._data.mimic_joint_names.append(mimic_joint_name)
                    self._data.mimic_joint_indices.append(child_idx)
                    self._data.mimic_joint_parents_indices.append(parent_idx)

                    self._data.mimic_joint_infos[child_idx, 0] = multiplier
                    self._data.mimic_joint_infos[child_idx, 1] = offset
                    self._data.mimic_joint_assignements[parent_idx] = child_idx

                    break

        if len(self._data.mimic_joint_names) != len(mimic_joints):
            raise ValueError("Mimic joint names do not match the number of mimic joints.")

        # convert everything to tensors
        self._data.actuated_joint_indices = torch.tensor(self._data.actuated_joint_indices, device=self.device)
        self._data.mimic_joint_indices = torch.tensor(self._data.mimic_joint_indices, device=self.device)
        self._data.mimic_joint_parents_indices = torch.tensor(
            self._data.mimic_joint_parents_indices, device=self.device
        )

    def set_default_joint_positions(
        self,
        joint_positions: torch.Tensor,
        joint_ids: Sequence[int] | None = None,
        env_ids: Sequence[int] | None = None,
    ):
        """Set the default joint positions for the articulation.

        Args:
            joint_positions: Default joint positions. Shape is (len(env_ids), len(joint_ids)).
            joint_ids: The joint indices to set the default positions for. Defaults to None (all joints).
            env_ids: The environment indices to set the default positions for. Defaults to None (all environments).
        """
        # resolve indices
        if env_ids is None:
            env_ids = slice(None)
        if joint_ids is None:
            joint_ids = slice(None)

        if self.cfg.init_fnc is not None:
            self._data.default_joint_pos[env_ids, :] = self.cfg.init_fnc(joint_positions)
        elif isinstance(joint_ids, slice):
            self._data.default_joint_pos[env_ids, :] = joint_positions
        else:
            env_ids_index = env_ids
            if not isinstance(env_ids_index, slice) and env_ids_index.ndim == 1:
                env_ids_index = env_ids_index[:, None]
            self._data.default_joint_pos[env_ids_index, joint_ids] = joint_positions

        if not isinstance(env_ids, slice) and not isinstance(joint_ids, slice):
            env_ids = env_ids[:, None]
        # clip to joint limits
        self._data.default_joint_pos[env_ids, joint_ids] = torch.clamp(
            self._data.default_joint_pos[env_ids, joint_ids],
            self._data.joint_pos_limits[env_ids, joint_ids, 0] - 0.1,
            self._data.joint_pos_limits[env_ids, joint_ids, 1],
        )
        if len(self._data.mimic_joint_names) > 0:
            # check for mimic joints
            controlled_joints = self._data.mimic_joint_assignements[joint_ids]
            valid_mask = controlled_joints != -1

            child_joint_ids = controlled_joints[valid_mask]
            mimic_params = self._data.mimic_joint_infos[child_joint_ids]
            ref_joint_pos = self._data.default_joint_pos[env_ids, joint_ids][:, valid_mask]
            mimic_pos = mimic_params[:, 0] * ref_joint_pos + mimic_params[:, 1]
            # broadcast env_ids if needed to allow double indexing
            if not isinstance(env_ids, slice) and env_ids.ndim == 1:
                env_ids = env_ids[:, None]
            self._data.default_joint_pos[env_ids, child_joint_ids] = mimic_pos

        # update actuator models
        for actuator in self.actuators.values():
            actuator.reset(env_ids)

    def _resolve_root_body(self):
        """Resolve ``cfg.root_body`` to a rigid-body index, and cache the static root->body offset.

        ``self._root_body_offset`` (static root->root_body transform) is taken from the graspqp
        hand model's ``_root_frame_inv`` and cached whenever that transform exists -- independent
        of whether ``root_body`` is a separate rigid body in the USD. This makes both energy
        reconstruction (merged-frame case) and load-time pose conversion
        (:meth:`to_articulation_root_pose`, either case) work.

        ``self._root_body_index`` is the real articulation-body index when ``root_body`` is a
        rigid body, else ``-1`` (the frame was collapsed into its parent during USD import; energy
        eval then reconstructs it from the articulation root + the cached offset).
        """
        hand_model = self._data.hand_model
        root_frame_inv = getattr(hand_model, "_root_frame_inv", None)
        if root_frame_inv is not None:
            # _root_frame_inv maps root_body -> root; invert it to get root -> root_body.
            matrix = root_frame_inv.inverse().get_matrix()[0].detach().to(device=self.device, dtype=torch.float32)
            self._root_body_offset = (matrix[:3, 3], math_utils.quat_from_matrix(matrix[:3, :3]))

        try:
            self._root_body_index = self.data.body_names.index(self.cfg.root_body)
            return
        except ValueError:
            pass

        if root_frame_inv is None:
            raise ValueError(
                f"root_body '{self.cfg.root_body}' is not a rigid body of {self.cfg.root_body!r} and "
                f"the hand model has no root_frame transform to reconstruct it. Either add the body "
                f"to the USD or set root_frame on the graspqp hand model."
            )
        self._root_body_index = -1

    def to_articulation_root_pose(self, poses, source_frame):
        """Re-express saved grasp poses in the articulation root frame.

        Saved grasp files tag ``root_pose`` with the frame it is expressed in
        (``data["root_frame"]``). graspqp seeds are authored in the hand's optimization frame
        (e.g. a ``"grasp_frame"`` -- a grip point offset ahead of the mounting-plate root),
        whereas the articulation is placed by writing to its *root* body. Placing a grasp-frame
        pose straight onto the root offsets the gripper by the root->grasp_frame distance, which
        is the "floating grasp" bug. This converts grasp_frame -> root using the static
        ``root -> root_body`` offset (see :meth:`_resolve_root_body`), the exact inverse of the
        composition :meth:`_get_urdf_hand_state` applies for energy evaluation.

        Poses already in the gripper root frame (``source_frame`` in ``{None, "gripper_root",
        "root", "base"}``) are returned unchanged -- so mined/eval outputs (tagged
        ``"gripper_root"``) and untagged legacy files are placed as-is.

        Args:
            poses: ``[N, 7]`` position + quaternion (wxyz), expressed in ``source_frame``.
            source_frame: the file's ``root_frame`` field (``None`` if absent).

        Returns:
            ``[N, 7]`` poses expressed in the articulation root frame.
        """
        if source_frame in (None, "gripper_root", "root", "base"):
            return poses
        if self.cfg.root_body != source_frame:
            raise ValueError(
                f"Grasp file is tagged root_frame='{source_frame}', but the hand model's "
                f"root_body is '{self.cfg.root_body}'. Set HandModelCfg.root_body to "
                f"'{source_frame}' (the frame the seeds were optimized in) so the static "
                f"root->{source_frame} offset is known, or retag the file."
            )
        if self._root_body_index is None:
            self._resolve_root_body()
        if self._root_body_offset is None:
            # No static offset available: the graspqp hand model has no root_frame transform
            # (_root_frame_inv) to derive root->source_frame. This only happens for hands that
            # bake the ee offset into the URDF root (root_frame=None) -- such hands should never
            # tag seeds with a non-root frame, so reaching here means a mislabeled file.
            raise ValueError(
                f"Cannot convert poses tagged root_frame='{source_frame}': the graspqp hand model "
                f"has no root_frame transform to derive the static root->{source_frame} offset. "
                f"Set root_frame='{source_frame}' on the graspqp hand model, or retag the file."
            )
        off_pos, off_quat = self._root_body_offset  # static root -> source_frame
        off_pos = off_pos.to(device=poses.device, dtype=poses.dtype)
        off_quat = off_quat.to(device=poses.device, dtype=poses.dtype)
        # invert to source_frame -> root
        inv_quat = math_utils.quat_inv(off_quat)
        inv_pos = math_utils.quat_apply(inv_quat, -off_pos)
        n = poses.shape[0]
        pos, quat = math_utils.combine_frame_transforms(
            poses[:, :3],
            poses[:, 3:7],
            inv_pos.unsqueeze(0).expand(n, 3),
            inv_quat.unsqueeze(0).expand(n, 4),
        )
        return torch.cat([pos, quat], dim=-1)

    def _get_urdf_hand_state(self, pose=None, joint_positions=None, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)

        if pose is None:
            if self.cfg.root_body is None:
                pose = self.data.root_state_w[env_ids]
            else:
                if self._root_body_index is None:
                    self._resolve_root_body()
                if self._root_body_index >= 0:
                    pose = self.data.body_state_w[:, self._root_body_index][env_ids]
                else:
                    # root_body (e.g. "grasp_frame") is a URDF frame that was merged into its
                    # parent during USD import, so it is not a separate rigid body. Reconstruct
                    # its world pose from the articulation root and the hand model's static
                    # root->root_body transform, so the energy is evaluated at the authored
                    # grasp frame instead of the mounting-plate root.
                    root = self.data.root_state_w[env_ids]
                    off_pos, off_quat = self._root_body_offset
                    off_pos = off_pos.to(device=root.device, dtype=root.dtype).expand(root.shape[0], 3)
                    off_quat = off_quat.to(device=root.device, dtype=root.dtype).expand(root.shape[0], 4)
                    pos, quat = math_utils.combine_frame_transforms(root[:, :3], root[:, 3:7], off_pos, off_quat)
                    pose = torch.cat([pos, quat], dim=-1)

        if joint_positions is None:
            joint_positions = self.data.joint_pos[env_ids]

        joint_positions = joint_positions[:, self._data.isaac_sim_to_urdf_joint_mapping]
        return torch.cat([pose[:, :3], ortho_6_from_quat(pose[:, 3:7]), joint_positions], dim=-1)

    def _update_hand_model_state(self):
        hand_model = self._data.hand_model
        hand_state = self._get_urdf_hand_state()
        hand_model.set_parameters(hand_state, contact_point_indices=self.cfg.contact_mode)
        return hand_model

    def _visualize_marker_points(self, attr_name: str, marker_name: str, color: tuple[float, float, float], points):
        if hasattr(self, attr_name):
            getattr(self, attr_name).set_visibility(True)
        else:
            setattr(
                self,
                attr_name,
                VisualizationMarkers(
                    VisualizationMarkersCfg(
                        prim_path=f"/Visuals/Robot/{marker_name}",
                        markers={
                            marker_name: sim_utils.SphereCfg(
                                radius=0.005,
                                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
                            )
                        },
                    )
                ),
            )

        points = points.reshape(-1, 3)
        if len(points) == 0:
            return
        getattr(self, attr_name).visualize(
            translations=points,
            marker_indices=[0] * len(points),
        )

    def _set_marker_visibility(self, attr_name: str, value: bool):
        if hasattr(self, attr_name):
            getattr(self, attr_name).set_visibility(value)

    def _get_hand_model_collision_spheres(self, hand_model) -> torch.Tensor:
        spheres = []
        batch_size = hand_model.global_translation.shape[0]
        for link_name, mesh_data in hand_model.mesh.items():
            vertices = mesh_data.get("vertices")
            if vertices is None or vertices.numel() == 0:
                continue
            vertices = hand_model.current_status[link_name].transform_points(vertices)
            if vertices.ndim == 2:
                vertices = vertices.unsqueeze(0).expand(batch_size, -1, -1)
            vertices = (
                vertices @ hand_model.global_rotation.transpose(1, 2)
                + hand_model.global_translation.unsqueeze(1)
            )
            center = vertices.mean(dim=1)
            radius = torch.linalg.norm(vertices - center[:, None, :], dim=-1).amax(dim=1)
            spheres.append(torch.cat([center, radius[:, None]], dim=-1))
        if len(spheres) == 0:
            return torch.empty((0, 4), device=self.device)
        return torch.stack(spheres, dim=1).reshape(-1, 4)

    def _vis_callback(self, key: str, value: bool):
        """Callback function for the robot debug visualization list entries."""

        if key == "contact_points":
            self._vis_contact_points(value)
        elif key == "contact_normals":
            self._vis_contact_normals(value)
        elif key == "surface_points":
            self._vis_surface_points(value)
        elif key == "collision_spheres":
            self._vis_collision_spheres(value)
        elif key == "projected_gravity":
            self._vis_projected_gravity(value)

    def _vis_contact_points(self, value: bool):
        self._set_marker_visibility("_contact_points_vis", value)
        if not value:
            return
        contact_pts_w = self._update_hand_model_state().get_contact_points()
        self._visualize_marker_points(
            "_contact_points_vis",
            "ContactPoints",
            (0.0, 1.0, 0.0),
            contact_pts_w,
        )

    def _vis_contact_normals(self, value: bool):
        if not value:
            return
        draw_interface = sim_utils.SimulationContext.instance().draw_interface
        if not draw_interface.enabled:
            return
        contact_pts_w, normals_w = self._update_hand_model_state().get_contact_points(return_normals=True)
        contact_pts = contact_pts_w.reshape(-1, 3)
        normals = normals_w.reshape(-1, 3)
        if len(contact_pts) == 0:
            return
        normal_length = 0.02
        draw_interface.plot_lines(
            contact_pts.detach().cpu(),
            (contact_pts + normals * normal_length).detach().cpu(),
            color=(0.0, 1.0, 0.0, 1.0),
            size=2.0,
        )

    def _vis_surface_points(self, value: bool):
        self._set_marker_visibility("_surface_points_vis", value)
        if not value:
            return
        surface_pts_w = self._update_hand_model_state().get_surface_points()
        self._visualize_marker_points(
            "_surface_points_vis",
            "SurfacePoints",
            (0.0, 0.4, 1.0),
            surface_pts_w,
        )

    def _vis_projected_gravity(self, value: bool):
        """Visualize the gripper forward axis and its gravity component (what reg_gravity penalizes).

        Gray line: world gravity direction at the ee_link. Blue line: the gripper's forward
        axis (``cfg.forward_axis``) in world. Red line: the forward axis scaled by its cosine
        to gravity — the quantity squared in the reg_gravity reward. Zero red line = level
        forward axis = no penalty.
        """
        if not value:
            return
        draw_interface = sim_utils.SimulationContext.instance().draw_interface
        if not draw_interface.enabled:
            return
        ee_name = "ee_link" if "ee_link" in self.data.body_names else self.data.body_names[0]
        ee_idx = self.data.body_names.index(ee_name)
        ee_pos_w = self.data.body_pos_w[:, ee_idx]
        wrist_quat = self.data.body_quat_w[:, ee_idx]

        scale = 0.2  # meters at |cos| = 1
        # same math as mdp.reg_gravity
        axis_b = torch.tensor(getattr(self.cfg, "forward_axis", (0.0, 0.0, 1.0)), device=self.device)
        axis_b = axis_b / axis_b.norm()
        forward_w = math_utils.quat_apply(wrist_quat, axis_b.expand(len(wrist_quat), 3))
        cos_to_gravity = (forward_w * self.data.GRAVITY_VEC_W).sum(dim=-1, keepdim=True)
        # gravity direction (gray)
        draw_interface.plot_lines(
            ee_pos_w.detach().cpu(),
            (ee_pos_w + self.data.GRAVITY_VEC_W * scale).detach().cpu(),
            color=(0.5, 0.5, 0.5, 1.0),
            size=3.0,
        )
        # forward axis (blue)
        draw_interface.plot_lines(
            ee_pos_w.detach().cpu(),
            (ee_pos_w + forward_w * scale).detach().cpu(),
            color=(0.0, 0.4, 1.0, 1.0),
            size=3.0,
        )
        # penalized gravity component of the forward axis (red)
        draw_interface.plot_lines(
            ee_pos_w.detach().cpu(),
            (ee_pos_w + forward_w * cos_to_gravity * scale).detach().cpu(),
            color=(1.0, 0.0, 0.0, 1.0),
            size=5.0,
        )

    def _vis_collision_spheres(self, value: bool):
        self._set_marker_visibility("_collision_spheres_vis", value)
        if not value:
            return

        sphere_data = self._get_hand_model_collision_spheres(self._update_hand_model_state())
        if hasattr(self, "_collision_spheres_vis"):
            self._collision_spheres_vis.set_visibility(True)
        elif len(sphere_data) > 0:
            self._collision_spheres_vis = VisualizationMarkers(
                VisualizationMarkersCfg(
                    prim_path="/Visuals/Robot/CollisionSpheres",
                    markers={
                        "collision_spheres": sim_utils.SphereCfg(
                            radius=1.0,
                            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
                        )
                    },
                )
            )
        if len(sphere_data) > 0:
            self._collision_spheres_vis.visualize(
                translations=sphere_data[:, :3],
                scales=sphere_data[:, 3:4].repeat(1, 3),
                marker_indices=[0] * len(sphere_data),
            )
