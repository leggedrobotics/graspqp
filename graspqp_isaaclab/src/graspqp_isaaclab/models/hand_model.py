from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Sequence

import omni.log
from pxr import UsdPhysics

import isaaclab.sim as sim_utils

from isaaclab.assets.articulation import Articulation
from .hand_model_data import HandModelData

if TYPE_CHECKING:
    from .hand_model_cfg import HandModelCfg

import weakref

import torch
from typing import TYPE_CHECKING, ClassVar

import warp as wp
import re

import isaaclab.sim as sim_utils

import warp as wp
from graspqp.hands import get_hand_model

try:
    import omni.ui
except ImportError:
    pass

from typing import TYPE_CHECKING, ClassVar
import warp as wp

from graspqp_isaaclab.utils.utils import ortho_6_from_quat


class HandModel(Articulation):
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
        # Ugly visualizer solution.
        self._debug_vis_cb_fnc = None
        self._debug_vis_toggle_fnc = None

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self._debug_vis_handle = None
        self._setup_vis_terms(terms=["Contact Points"])
        self._vis_frame = None
        self._num_meshes_per_env = {}

        super(Articulation, self).__init__(cfg)

        self._root_body_index = None

    def _initialize_impl(self):
        super()._initialize_impl()
        print("Data:", self._data)

        print("Done initializing articulation, now initializing hand model")

        self._data.hand_model = get_hand_model(
            self.cfg.hand_model_name,
            self.device,  # , contact_points_path=json_data, n_surface_points=2
            n_surface_points=1024 if self.cfg.surface_pts is None else self.cfg.surface_pts,
        )
        print("Hand model initialized")

        self._data.isaac_sim_to_urdf_joint_mapping = []
        self._data.urdf_to_isaac_sim_joint_mapping = []
        for joint_name in self._data.hand_model.actuated_joints_names:
            self._data.isaac_sim_to_urdf_joint_mapping.append(self._data.joint_names.index(joint_name))

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

    def get_contact_points(self, env_ids=None):
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
        return hand_model.get_contact_points()

    def calc_joint_vel(self, contact_idxs, interaction_forces, env_ids=None):
        hand_model = self._data.hand_model
        hand_state = self._get_urdf_hand_state(env_ids=env_ids)
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
    def _setup_vis_terms(self, terms=[]):
        self._terms = {}
        for term in terms:
            self._terms[term] = {"state": False}

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

    # def write_joint_position_to_sim(
    #     self,
    #     position: torch.Tensor,
    #     joint_ids: Sequence[int] | slice | None = None,
    #     env_ids: Sequence[int] | slice | None = None,
    # ):
    #     """Write joint positions to the simulation.

    #     Args:
    #         position: Joint positions. Shape is (len(env_ids), len(joint_ids)).
    #         joint_ids: The joint indices to set the targets for. Defaults to None (all joints).
    #         env_ids: The environment indices to set the targets for. Defaults to None (all environments).
    #     """
    #     # resolve indices
    #     physx_env_ids = env_ids
    #     if env_ids is None:
    #         env_ids = slice(None)
    #         physx_env_ids = self._ALL_INDICES
    #     if joint_ids is None:
    #         joint_ids = slice(None)
    #     # broadcast env_ids if needed to allow double indexing
    #     if env_ids != slice(None) and joint_ids != slice(None):
    #         env_ids = env_ids[:, None]
    #     # set into internal buffers
    #     self._data.joint_pos[env_ids, joint_ids] = position
    #     # Need to invalidate the buffer to trigger the update with the new root pose.
    #     self._data._body_com_vel_w.timestamp = -1.0
    #     self._data._body_link_vel_w.timestamp = -1.0
    #     self._data._body_com_pose_b.timestamp = -1.0
    #     self._data._body_com_pose_w.timestamp = -1.0
    #     self._data._body_link_pose_w.timestamp = -1.0

    #     self._data._body_state_w.timestamp = -1.0
    #     self._data._body_link_state_w.timestamp = -1.0
    #     self._data._body_com_state_w.timestamp = -1.0

    #     if len(self._data.mimic_joint_names) > 0:
    #         # check for mimic joints
    #         controlled_joints = self._data.mimic_joint_assignements[joint_ids]
    #         valid_mask = controlled_joints != -1

    #         child_joint_ids = controlled_joints[valid_mask]
    #         mimic_params = self._data.mimic_joint_infos[child_joint_ids]
    #         ref_joint_pos = self._data.joint_pos[env_ids, joint_ids][:, valid_mask]
    #         mimic_pos = mimic_params[:, 0] * ref_joint_pos + mimic_params[:, 1]
    #         # broadcast env_ids if needed to allow double indexing
    #         if not isinstance(env_ids, slice) and env_ids.ndim == 1:
    #             env_ids = env_ids[:, None]
    #         self._data.joint_pos[env_ids, child_joint_ids] = mimic_pos

    #     # set into simulation
    #     self.root_physx_view.set_dof_positions(self._data.joint_pos, indices=physx_env_ids)

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
        # broadcast env_ids if needed to allow double indexing

        if not isinstance(env_ids, slice) and not isinstance(joint_ids, slice):
            env_ids = env_ids[:, None]
        self._data.default_joint_pos[env_ids, joint_ids] = joint_positions

        if self.cfg.init_fnc is not None:
            self._data.default_joint_pos[env_ids, :] = self.cfg.init_fnc(self._data.default_joint_pos[env_ids])

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

    def _get_urdf_hand_state(self, pose=None, joint_positions=None, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)

        if pose is None:
            if self.cfg.root_body is None:
                pose = self.data.root_state_w[env_ids]
            else:
                if self._root_body_index is None:
                    self._root_body_index = self.data.body_names.index(self.cfg.root_body)
                pose = self.data.body_state_w[:, self._root_body_index]

        if joint_positions is None:
            joint_positions = self.data.joint_pos[env_ids]

        joint_positions = joint_positions[:, self._data.isaac_sim_to_urdf_joint_mapping]
        return torch.cat([pose[:, :3], ortho_6_from_quat(pose[:, 3:7]), joint_positions], dim=-1)

    def _vis_callback(self, event, tasks=[]):
        """Callback function for the debug visualization."""

        if "Contact Points" in tasks:
            print("Visualizing contact points")
            draw_interface = sim_utils.SimulationContext.instance().draw_interface
            hand_model = self._data.hand_model
            hand_state = self._get_urdf_hand_state()
            hand_model.set_parameters(hand_state, contact_point_indices=self.cfg.contact_mode)
            contact_pts_w = hand_model.get_contact_points()
            contact_pts = contact_pts_w.view(-1, 3)
            draw_interface.plot_points(
                contact_pts.detach().cpu().numpy().tolist(),
                color=[0.0, 1.0, 0.0, 1.0],
                size=15,
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
