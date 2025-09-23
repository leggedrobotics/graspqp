"""
Based on Dexgraspnet: https://pku-epic.github.io/DexGraspNet/
"""

import os
import json
import numpy as np
import torch
from graspqp.utils.transforms import (
    robust_compute_rotation_matrix_from_ortho6d,
)
import contextlib

with contextlib.suppress(ImportError):
    import plotly.graph_objects as go

import pytorch_kinematics as pk
import pytorch3d.structures
import pytorch3d.ops
import trimesh as tm
import contextlib

SDF_BACKEND = os.environ.get("SDF_BACKEND", "TORCHSDF").upper()

if SDF_BACKEND == "WARP":
    import warp as wp
    from graspqp.utils import warp as wp_utils
elif SDF_BACKEND == "TORCHSDF":
    from torchsdf import index_vertices_by_faces, compute_sdf
elif SDF_BACKEND == "KAOLIN":
    import kaolin

import trimesh

with contextlib.suppress(ImportError):
    import open3d as o3d
import roma

from pytorch_kinematics.transforms.rotation_conversions import matrix_to_quaternion


@torch.jit.script
def pinv(A: torch.Tensor, l: float = 1e-3):
    m, n = A.shape[-2:]
    if m < n:
        # right inverse (full row rank)
        return A.mT @ torch.linalg.inv(A @ A.mT + l * torch.eye(m, device=A.device))
    else:
        # left inverse (full column rank)
        return torch.linalg.inv(A.mT @ A + l * torch.eye(n, device=A.device)) @ A.mT


class HandModel:

    @staticmethod
    def estimate_static_frame_from_hand_points(
        keypoint_3d_array: np.ndarray,
    ) -> np.ndarray:
        """
        Compute the 3D coordinate frame (orientation only) from detected 3d key points
        :param points: keypoint3 detected from MediaPipe detector. Order: [wrist, index, middle, pinky]
        :return: the coordinate frame of wrist in MANO convention
        """
        assert keypoint_3d_array.shape == (21, 3)
        points = keypoint_3d_array[[0, 5, 9], :]

        # Compute vector from palm to the first joint of middle finger
        x_vector = points[0] - points[2]

        # Normal fitting with SVD
        points = points - np.mean(points, axis=0, keepdims=True)
        u, s, v = np.linalg.svd(points)

        normal = v[2, :]

        # Gram–Schmidt Orthonormalize
        x = x_vector - np.sum(x_vector * normal) * normal
        x = x / np.linalg.norm(x)
        z = np.cross(x, normal)

        # We assume that the vector from pinky to index is similar the z axis in MANO convention
        if np.sum(z * (points[1] - points[2])) < 0:
            normal *= -1
            z *= -1
        frame = np.stack([x, normal, z], axis=1)
        return frame

    def retarget(self, mano_keypoints):
        from dex_retargeting.misc.constants import OPERATOR2MANO_RIGHT, ManoModel

        if not hasattr(self, "_retargeter") or self._retargeter is None:
            raise ValueError("Retargeter not loaded. Please load retargeter first.")

        if isinstance(mano_keypoints, torch.Tensor):
            mano_keypoints = mano_keypoints.cpu().numpy()

        static_orientation = self.estimate_static_frame_from_hand_points(mano_keypoints)
        self._static_orientation = np.eye(3, 3)  # static_orientation @ OPERATOR2MANO_RIGHT
        joint_pos_static = mano_keypoints  # @ static_orientation @ OPERATOR2MANO_RIGHT

        anchors = mano_keypoints  # (21, 3) joints
        wrist_pos = anchors[0]  # (3,) wrist position

        # zero center$
        mano_keypoints = anchors.copy()  # - wrist_pos  # (20, 3) finger joints relative to wrist
        joint_pos_static = mano_keypoints

        # Unique retargeting indices
        indices = self._retargeter.optimizer.target_link_human_indices
        origin_indices = indices[0, :]
        task_indices = indices[1, :]
        ref_value = joint_pos_static[task_indices, :] - joint_pos_static[origin_indices, :]
        self._retargeter.optimizer._start_kp = joint_pos_static[origin_indices, :]
        self._retargeter.optimizer._end_kp = joint_pos_static[task_indices, :]

        qpos, root_pose = self._retargeter.retarget(ref_value, keypoints=joint_pos_static)
        optimizer_link_order = self._retargeter.optimizer.all_joint_names
        for name in self._actuated_joints_names:
            if name not in optimizer_link_order:
                print(f"Warning: {name} not in optimizer link order. Skipping.")
                print(f"Available joints: {optimizer_link_order}")
                print("All joints: ", self._actuated_joints_names)
        target_idxs = [optimizer_link_order.index(name) for name in self._actuated_joints_names]
        position = qpos[target_idxs]
        target_joint_pos = torch.from_numpy(position).to(self.device).float()
        # figure out global alignment

        return target_joint_pos, torch.from_numpy(root_pose).to(self.device).float()

    def load_retargeter(self, retargeter_path, urdf_root_dir=None):
        import yaml
        from dex_retargeting.retarget.retargeting_config import RetargetingConfig

        RetargetingConfig.set_default_urdf_dir(urdf_root_dir)
        cfg = RetargetingConfig.load_from_dict(yaml.safe_load(open(retargeter_path, "r"))["retargeting"])
        self._retargeter = cfg.build()
        self._cfg = cfg

    def _parse_mjcf(
        self,
        mjcf_path,
        mesh_path,
        contact_points,
        penetration_points,
        use_collision=False,
        only_use_collision=False,
        simplify_mesh=False,
        contact_links=None,
    ):
        self.mesh = {}
        areas = {}
        device = self.device
        self._used_links_mask = []

        def _get_mesh_for_visual(visual, simplify_mesh):
            scale = torch.tensor([1, 1, 1], dtype=torch.float, device=device)
            if visual.geom_type == "box":
                return tm.primitives.Box(extents=2 * np.array(visual.geom_param)), torch.tensor(
                    [1, 1, 1], dtype=torch.float, device=device
                )
            elif visual.geom_type == "capsule":
                return tm.primitives.Capsule(
                    radius=visual.geom_param[0], height=visual.geom_param[1] * 2
                ).apply_translation((0, 0, -visual.geom_param[1]))
            elif visual.geom_type == "cylinder":
                return tm.primitives.Cylinder(
                    radius=visual.geom_param[0], height=visual.geom_param[1]
                ).apply_translation((0, 0, -visual.geom_param[1] / 2))
            elif visual.geom_type == "sphere":
                return tm.primitives.Sphere(radius=visual.geom_param)
            elif visual.geom_type == "mesh":
                if isinstance(visual.geom_param, str):
                    visual.geom_param = [visual.geom_param, None]
                visual.geom_param = list(visual.geom_param)
                if "package://" in visual.geom_param[0]:
                    visual.geom_param[0] = visual.geom_param[0].replace("package://", "")  # TODO: fix this

                if not os.path.exists(os.path.join(mesh_path, visual.geom_param[0])):
                    if os.path.exists(os.path.join(mesh_path, os.path.basename(visual.geom_param[0]))):
                        visual.geom_param[0] = os.path.basename(visual.geom_param[0])
                    else:
                        raise ValueError("File not found", os.path.join(mesh_path, visual.geom_param[0]))

                if not (
                    visual.geom_param[0].endswith(".STL")
                    or visual.geom_param[0].endswith(".stl")
                    or visual.geom_param[0].endswith(".obj")
                    or visual.geom_param[0].endswith(".dae"),
                ):
                    visual.geom_param[0] = visual.geom_param[0] + ".STL"
                if not os.path.exists(os.path.join(mesh_path, visual.geom_param[0])):
                    raise ValueError("File not found", visual.geom_param[0])

                link_mesh = tm.load_mesh(os.path.join(mesh_path, visual.geom_param[0]), process=False)
                # print("Loaded mesh from", os.path.join(mesh_path, visual.geom_param[0]))

                if simplify_mesh:
                    import fast_simplification

                    vertices, faces = link_mesh.vertices, link_mesh.faces
                    vertices, faces = fast_simplification.simplify(vertices, faces, 0.1)
                    link_mesh = tm.Trimesh(vertices=vertices, faces=faces)
                if visual.geom_param[1] is not None and visual.geom_param[1] != "":
                    scale = torch.tensor(visual.geom_param[1], dtype=torch.float, device=device)

            else:
                raise NotImplementedError("Unsupported geom type: " + visual.geom_type)
            return link_mesh, scale

        def build_mesh_recurse(body):
            if (len(body.link.visuals) > 0 and not only_use_collision) or len(body.link.collisions) > 0:
                link_name = body.link.name
                link_vertices = []
                link_faces = []
                n_link_vertices = 0

                visuals = []

                if not only_use_collision and (len(body.link.collisions) == 0 or not use_collision):
                    for v in body.link.visuals:
                        if isinstance(v, list):
                            visuals.extend(v)
                        else:
                            visuals.append(v)
                else:
                    for v in body.link.collisions:
                        if isinstance(v, list):
                            visuals += v
                        else:
                            visuals.append(v)

                for visual in visuals:
                    link_mesh, scale = _get_mesh_for_visual(visual, simplify_mesh)
                    vertices = torch.tensor(link_mesh.vertices, dtype=torch.float, device=device)
                    faces = torch.tensor(link_mesh.faces, dtype=torch.long, device=device)
                    pos = visual.offset.to(self.device)
                    vertices = vertices * scale
                    vertices = pos.transform_points(vertices)
                    link_vertices.append(vertices)
                    link_faces.append(faces + n_link_vertices)
                    n_link_vertices += len(vertices)

                link_vertices = (
                    torch.cat(link_vertices, dim=0)
                    if len(link_vertices) > 0
                    else torch.tensor([], dtype=torch.float, device=device).reshape(0, 3)
                )
                link_faces = (
                    torch.cat(link_faces, dim=0)
                    if len(link_faces) > 0
                    else torch.tensor([], dtype=torch.long, device=device).reshape(0, 3)
                )
                contact_candidates = torch.tensor([], dtype=torch.float, device=device).reshape(0, 3)

                if not (link_name in contact_points):
                    print("No contact points for", link_name)
                elif contact_links is not None and link_name not in contact_links:
                    print("Skipping", link_name, "because it is not in contact_links")
                else:
                    candidates = contact_points[link_name]
                    if not isinstance(candidates, list):
                        candidates = [candidates]

                    for candidate in candidates:
                        if isinstance(candidate, list) and len(candidate) == 2 and isinstance(candidate[0], str):

                            old_seed = torch.randint(0, 100, (1,)).item()
                            new_seed = 42
                            # seed everything
                            torch.manual_seed(new_seed)
                            np.random.seed(new_seed)
                            mash_file, num_points = candidate
                            if contact_links is not None and link_name in contact_links:
                                num_points = contact_links[link_name].get("n_points", num_points)
                            # load from file
                            mesh = tm.load_mesh(os.path.join(mesh_path, mash_file), process=False)
                            # sample surface points
                            points = tm.sample.sample_surface_even(mesh, 1000)[0]
                            # subsample with fps sampling
                            from pytorch3d.ops.sample_farthest_points import (
                                sample_farthest_points,
                            )

                            points = torch.tensor(points, dtype=torch.float, device=device)
                            points = sample_farthest_points(points.unsqueeze(0), K=num_points)[0].squeeze(0)

                            # seed everything
                            torch.manual_seed(new_seed)
                            np.random.seed(new_seed)

                            # Convert with offset
                            pos = visual.offset.to(self.device)
                            points = pos.transform_points(points * scale)
                            # Sample points
                            contact_candidates = torch.cat([contact_candidates, points], dim=0)
                        elif isinstance(candidate, list) and len(candidate) == 3:
                            points = torch.tensor(candidate, dtype=torch.float, device=device).reshape(-1, 3)
                            pos = visual.offset.to(self.device)
                            points = pos.transform_points(points)
                            # Loading contact points from list
                            contact_candidates = torch.cat(
                                [contact_candidates, points],
                                dim=0,
                            )
                        else:
                            raise ValueError("Unsupported contact points type. Got", type(candidate))

                if penetration_points is None or not (link_name in penetration_points):
                    penetration_keypoints = torch.tensor([], dtype=torch.float, device=device).reshape(0, 3)
                else:
                    penetration_keypoints = torch.tensor(
                        penetration_points[link_name], dtype=torch.float, device=device
                    )
                scales = torch.ones(len(penetration_keypoints), device=device) * 0.01

                if len(penetration_keypoints) != 0:
                    if penetration_keypoints.shape[-1] == 4:
                        scales = penetration_keypoints[:, 3]
                        penetration_keypoints = penetration_keypoints[:, :3]
                        # Transform with mesh
                        if hasattr(visual, "offset"):
                            pos = visual.offset.to(penetration_keypoints.device)
                            penetration_keypoints = pos.transform_points(penetration_keypoints)

                if penetration_keypoints is not None:
                    penetration_keypoints = penetration_keypoints.view(-1, 3)
                areas[link_name] = tm.Trimesh(link_vertices.cpu().numpy(), link_faces.cpu().numpy()).area.item()
                mesh = tm.Trimesh(link_vertices.cpu().numpy(), link_faces.cpu().numpy())

                if len(contact_candidates) > 0:
                    contacts_np = contact_candidates.cpu().numpy()
                    pt, dist, triangle_id = tm.proximity.closest_point(mesh, contacts_np)
                    normal_candidates = mesh.face_normals[triangle_id]

                self.mesh[link_name] = {
                    "vertices": link_vertices,
                    "faces": link_faces,
                    "contact_candidates": contact_candidates,
                    "penetration_keypoints": penetration_keypoints,
                    "scales": scales,
                    "normal_candidates": (
                        torch.tensor(normal_candidates, dtype=torch.float, device=device).reshape(-1, 3)
                        if len(contact_candidates) > 0
                        else torch.tensor([], dtype=torch.float, device=device).reshape(0, 3)
                    ),
                }
                self._used_links_mask.append(True)
                # import pdb; pdb.set_trace()
                if SDF_BACKEND == "TORCHSDF":
                    link_face_verts = index_vertices_by_faces(link_vertices, link_faces)
                elif SDF_BACKEND == "WARP":
                    # create warp mesh from vertices and faces
                    link_vertices, link_faces = np.array(mesh.vertices), mesh.faces
                    verts_wp = wp.from_numpy(
                        np.ascontiguousarray(link_vertices),
                        device=str(device),
                        dtype=wp.vec3,
                    )
                    faces_wp = wp.from_numpy(
                        np.ascontiguousarray(link_faces.flatten()),
                        device=str(device),
                        dtype=wp.int32,
                    )
                    wp_mesh = wp.Mesh(points=verts_wp, indices=faces_wp)
                    link_face_verts = wp_mesh
                elif SDF_BACKEND == "KAOLIN":
                    link_face_verts = kaolin.ops.mesh.index_vertices_by_faces(link_vertices.unsqueeze(0), link_faces)
                else:
                    raise ValueError(f"Unknown SDF_BACKEND: {SDF_BACKEND}")
                self.mesh[link_name]["face_verts"] = link_face_verts
            else:
                self._used_links_mask.append(False)

            for children in body.children:
                build_mesh_recurse(children)

        build_mesh_recurse(self.chain._root)

        data_dict = {}
        for link_name, infos in self.mesh.items():
            data_dict[link_name] = {
                "contact_candidates": infos["contact_candidates"].cpu().numpy().tolist(),
                "normal_candidates": infos["normal_candidates"].cpu().numpy().tolist(),
            }

        # save it
        # import json
        # json.dump(data_dict, open(os.path.join(mesh_path, "contact_infos.json"), "w"))
        # print("Contact infos saved to", os.path.join(mesh_path, "contact_infos.json"))

        return areas

    def __init__(
        self,
        mjcf_path,
        mesh_path,
        contact_points_path,
        penetration_points_path,
        n_surface_points=0,
        device="cpu",
        joint_calc_fnc=None,
        jacobian_fnc=None,
        joint_filter=None,
        default_state=None,
        forward_axis="x",
        up_axis="z",
        only_use_collision=False,
        use_collision_if_possible=True,
        contact_links=None,
        grasp_type=None,
        grasp_axis=None,
    ):
        """
        Create a Hand Model for a MJCF robot

        Parameters
        ----------
        mjcf_path: str
            path to mjcf file
        mesh_path: str
            path to mesh directory
        contact_points_path: str
            path to hand-selected contact candidates
        penetration_points_path: str
            path to hand-selected penetration keypoints
        n_surface_points: int
            number of points to sample from surface of hand, use fps
        device: str | torch.Device
            device for torch tensors
        """

        self.device = device
        self.joint_calc_fnc = joint_calc_fnc
        self.jacobian_fnc = jacobian_fnc

        if grasp_type != None and grasp_type != "all" and contact_links is None and grasp_type != "default":
            eigengrasp_file = os.path.join(os.path.dirname(mesh_path), "eigengrasps.json")
            print("Loading grasp type from", eigengrasp_file)

            if not os.path.exists(eigengrasp_file):
                raise ValueError(f"eigengrasps.json not found at {eigengrasp_file}")
            json_data = json.load(open(eigengrasp_file))
            if grasp_type not in json_data:
                raise ValueError(
                    f"grasp type {grasp_type} not found in eigengrasps.json. Available grasp types are {list(json_data.keys())}"
                )
            contact_links = json_data[grasp_type]

        self._contact_links = contact_links

        # load articulation
        if mjcf_path.endswith(".urdf"):
            self.chain = pk.build_chain_from_urdf(open(mjcf_path).read()).to(dtype=torch.float, device=device)
        else:
            self.chain = pk.build_chain_from_mjcf(open(mjcf_path).read()).to(dtype=torch.float, device=device)

        axis = {
            "x": torch.tensor([1, 0, 0], dtype=torch.float, device=device),
            "y": torch.tensor([0, 1, 0], dtype=torch.float, device=device),
            "z": torch.tensor([0, 0, 1], dtype=torch.float, device=device),
            "-x": torch.tensor([-1, 0, 0], dtype=torch.float, device=device),
            "-y": torch.tensor([0, -1, 0], dtype=torch.float, device=device),
            "-z": torch.tensor([0, 0, -1], dtype=torch.float, device=device),
        }
        self.forward_axis = axis[forward_axis]
        self.up_axis = axis[up_axis]

        self.grasp_axis = axis[grasp_axis] if grasp_axis is not None else axis[forward_axis]

        self._joint_mask = None  # [0,2,4,6,8, 9]
        self._joint_filter = joint_filter
        self._actuated_joints_names = [
            name
            for name in self.chain.get_joint_parameter_names()
            if self._joint_filter is None or name in self._joint_filter
        ]
        self.n_dofs = len(self._actuated_joints_names)
        # self.n_dofs = len(self.chain.get_joint_parameter_names()) if self._joint_mask is None else len(self._joint_mask)

        self.default_state = (
            default_state if default_state is not None else torch.zeros(self.n_dofs, dtype=torch.float, device=device)
        )

        # load contact points and penetration points

        if contact_points_path is not None:
            if isinstance(contact_points_path, str):
                contact_points = json.load(open(contact_points_path, "r"))
            elif isinstance(contact_points_path, dict):
                contact_points = contact_points_path
            else:
                raise ValueError("Unsupported contact_points_path type")
        else:
            contact_points = None

        # contact_points = (
        #     json.load(open(contact_points_path, "r"))
        #     if contact_points_path is not None
        #     else None
        # )
        penetration_points = (
            json.load(open(penetration_points_path, "r")) if penetration_points_path is not None else None
        )

        # build mesh
        if mjcf_path.endswith(".urdf"):
            areas = self._parse_mjcf(
                mjcf_path,
                mesh_path,
                contact_points,
                penetration_points,
                use_collision=use_collision_if_possible,
                only_use_collision=only_use_collision,
                contact_links=self._contact_links,
            )

        else:
            areas = self._parse_mjcf(
                mjcf_path,
                mesh_path,
                contact_points,
                penetration_points,
                use_collision=use_collision_if_possible,
                only_use_collision=only_use_collision,
                contact_links=self._contact_links,
            )

        # set joint limits
        self.joints_names = []
        self.joints_lower = []
        self.joints_upper = []

        def set_joint_range_recurse(body):
            if body.joint.joint_type != "fixed":
                self.joints_names.append(body.joint.name)
                if body.joint.range is None:
                    print("No range for joint", body.joint.name)
                    self.joints_lower.append(torch.tensor(-np.inf, dtype=torch.float, device=device))
                    self.joints_upper.append(torch.tensor(np.inf, dtype=torch.float, device=device))
                else:
                    self.joints_lower.append(torch.tensor(body.joint.range[0], dtype=torch.float, device=device))
                    self.joints_upper.append(torch.tensor(body.joint.range[1], dtype=torch.float, device=device))
            for children in body.children:
                set_joint_range_recurse(children)

        set_joint_range_recurse(self.chain._root)
        # drop mimic joitns
        if self._joint_mask is not None:
            self.joints_lower = self.joints_lower[self._joint_mask]
            self.joints_upper = self.joints_upper[self._joint_mask]

        if grasp_type is not None and grasp_type != "all":
            print(self.joints_names)
            if "allegro" in mjcf_path:
                if grasp_type == "pinch":
                    # set the other fingers to the lower limit.
                    for i, joint_name in enumerate(self._actuated_joints_names):
                        if "middle" in joint_name or "ring" in joint_name:
                            if "joint_0" in joint_name:
                                continue
                            self.default_state[i] = self.joints_upper[i]
                elif grasp_type == "precision":
                    for i, joint_name in enumerate(self._actuated_joints_names):
                        if "ring" in joint_name:
                            if "joint_0" in joint_name:
                                continue
                            self.default_state[i] = self.joints_upper[i]
            elif "shadow_hand" in mjcf_path:
                if grasp_type == "pinch":
                    # set the other fingers to the lower limit.
                    for i, joint_name in enumerate(self._actuated_joints_names):
                        if "MF" in joint_name or "RF" in joint_name or "LF" in joint_name:
                            if "J3" in joint_name or "LFJ4" in joint_name:
                                continue
                            self.default_state[i] = self.joints_upper[i]
                elif grasp_type == "precision":
                    for i, joint_name in enumerate(self._actuated_joints_names):
                        if "RF" in joint_name or "LF" in joint_name:
                            if "J3" in joint_name or "LFJ4" in joint_name:
                                continue
                            self.default_state[i] = self.joints_upper[i]
            elif "ability_hand" in mjcf_path:
                if grasp_type == "pinch":
                    # set the other fingers to the lower limit.
                    for i, joint_name in enumerate(self._actuated_joints_names):
                        if "middle" in joint_name or "ring" in joint_name or "pinky" in joint_name:
                            self.default_state[i] = self.joints_upper[i]
                elif grasp_type == "precision":
                    for i, joint_name in enumerate(self._actuated_joints_names):
                        if "ring" in joint_name or "pinky" in joint_name:
                            self.default_state[i] = self.joints_upper[i]

        print("==== Creating Hand Model ====")
        print("Number of actuated joints", self.n_dofs)
        print("Actuated joints", self._actuated_joints_names)
        print("All joints", self.joints_names)
        print("Joint filter", self._joint_filter)
        print("Joint lower limits", [f"{l.cpu().item():.2f}" for l in self.joints_lower])
        print("Joint upper limits", [f"{l.cpu().item():.2f}" for l in self.joints_upper])
        print("=============================")
        # self.joints_lower = self.joints_lower[[0,2,4,6,8, 9]]
        # self.joints_upper = self.joints_upper[[0,2,4,6,8, 9]]
        # print("Joint Names", self.joints_names)
        # sample surface points

        total_area = sum(areas.values())
        num_samples = dict(
            [(link_name, int(areas[link_name] / total_area * n_surface_points)) for link_name in self.mesh]
        )
        num_samples[list(num_samples.keys())[0]] += n_surface_points - sum(num_samples.values())

        for link_name in self.mesh:
            if num_samples[link_name] == 0:
                self.mesh[link_name]["surface_points"] = torch.tensor([], dtype=torch.float, device=device).reshape(
                    0, 3
                )
                continue
            mesh = pytorch3d.structures.Meshes(
                self.mesh[link_name]["vertices"].unsqueeze(0),
                self.mesh[link_name]["faces"].unsqueeze(0),
            )
            old_seed = torch.randint(0, 100, (1,))
            new_seed = 42
            # seed everything
            torch.manual_seed(new_seed)
            np.random.seed(new_seed)

            dense_point_cloud = pytorch3d.ops.sample_points_from_meshes(mesh, num_samples=100 * num_samples[link_name])
            surface_points = pytorch3d.ops.sample_farthest_points(dense_point_cloud, K=num_samples[link_name])[0][0]
            surface_points.to(dtype=float, device=device)
            # seed everything
            torch.manual_seed(old_seed)
            np.random.seed(old_seed)

            self.mesh[link_name]["surface_points"] = surface_points

        # indexing

        self.link_name_to_link_index = dict(zip([link_name for link_name in self.mesh], range(len(self.mesh))))

        self.contact_candidates = [self.mesh[link_name]["contact_candidates"] for link_name in self.mesh]
        self.contact_normals_candidates = [self.mesh[link_name]["normal_candidates"] for link_name in self.mesh]

        self.global_index_to_link_index = sum(
            [[i] * len(contact_candidates) for i, contact_candidates in enumerate(self.contact_candidates)],
            [],
        )
        self.contact_candidates = torch.cat(self.contact_candidates, dim=0)
        self.r_contact_body_frames = self.contact_candidates.clone()

        self.contact_normals_candidates = torch.cat(self.contact_normals_candidates, dim=0)

        self.global_index_to_link_index = torch.tensor(self.global_index_to_link_index, dtype=torch.long, device=device)
        self.n_contact_candidates = self.contact_candidates.shape[0]

        self.penetration_keypoints = [self.mesh[link_name]["penetration_keypoints"] for link_name in self.mesh]
        self.global_index_to_link_index_penetration = sum(
            [[i] * len(penetration_keypoints) for i, penetration_keypoints in enumerate(self.penetration_keypoints)],
            [],
        )
        self.penetration_keypoints = torch.cat(self.penetration_keypoints, dim=0)

        self.penetration_keypoints_expanded = None

        self.global_index_to_link_index_penetration = torch.tensor(
            self.global_index_to_link_index_penetration, dtype=torch.long, device=device
        )
        self.global_index_to_link_index_penetration_epanded = None

        self.n_keypoints = self.penetration_keypoints.shape[0]
        self.sphere_scales = []
        for link_name in self.mesh:
            if "scales" in self.mesh[link_name]:
                self.sphere_scales.append(self.mesh[link_name]["scales"])

        self.sphere_scales = torch.cat(self.sphere_scales, dim=0).to(device)

        self.thumb_indices = None

        # parameters

        self.hand_pose = None
        self.contact_point_indices = None
        self.global_translation = None
        self.global_rotation = None
        self.current_status = None

        self.contact_points = None
        self.contact_normals = None

        self.joints_lower = [
            self.joints_lower[i]
            for i in range(len(self.joints_lower))
            if self._joint_filter is None or self.joints_names[i] in self._joint_filter
        ]
        self.joints_upper = [
            self.joints_upper[i]
            for i in range(len(self.joints_upper))
            if self._joint_filter is None or self.joints_names[i] in self._joint_filter
        ]
        self.joints_lower = torch.stack(self.joints_lower).float().to(device)
        self.joints_upper = torch.stack(self.joints_upper).float().to(device)

    def joint_entropy(self):
        """
        Calculate joint entropy

        Returns
        -------
        entropy: (B,) torch.Tensor
            joint entropy
        """

        joints_entropy = 0
        n_bins = 32

        joint_values = self.hand_pose[:, 9:]
        for joint_idx, joint_position in enumerate(self.hand_pose[:, 9:].T):
            limits = (self.joints_lower[joint_idx], self.joints_upper[joint_idx])
            counts = joint_position.histc(n_bins, limits[0], limits[1])
            dist = counts / counts.sum()
            logs = torch.log(torch.where(dist > 0, dist, 1))
            joints_entropy += -(dist * logs).sum() / joint_values.shape[-1]
        return joints_entropy

    def pose_entropy(self):
        """
        Calculate pose entropy

        Returns
        -------
        entropy: (B,) torch.Tensor
            pose entropy
        """
        # translation entropy
        translation_entropy = 0
        rotation_entropy = 0
        for i in range(3):
            limits = (-0.1, 0.1)
            counts = self.global_translation[:, i].histc(32, limits[0], limits[1])
            dist = counts / counts.sum()
            logs = torch.log(torch.where(dist > 0, dist, 1))
            translation_entropy += -(dist * logs).sum() / 3

        poses = self.hand_pose[:, 3:9]
        x_raw = poses[:, 0:3]  # batch*3
        y_raw = poses[:, 3:6]  # batch*3
        rotmat = roma.special_gramschmidt(torch.stack([x_raw, y_raw], -1))
        rotvec = roma.rotmat_to_rotvec(rotmat)
        # convert to spherical coordinates
        r = torch.norm(rotvec, dim=-1)  # [0, pi]
        theta = torch.acos(rotvec[:, 2] / r)  # [0, pi]
        phi = torch.sign(rotvec[:, 1]) * torch.acos(rotvec[:, 0] / torch.norm(rotvec[:, :2], dim=-1))  # [-pi, pi]
        spherical_coordinates = torch.stack([r, theta, phi], -1)

        limits = [(0, np.pi), (0, np.pi), (-np.pi, np.pi)]

        # rotation entropy
        for i in range(3):
            limit = limits[i]
            counts = spherical_coordinates[..., i].histc(32, limit[0], limit[1])
            dist = counts / counts.sum()
            logs = torch.log(torch.where(dist > 0, dist, 1))
            rotation_entropy += -(dist * logs).sum() / 3

        return translation_entropy, rotation_entropy

    def fk(self, joint_angles):
        if self.joint_calc_fnc is not None:
            return self.joint_calc_fnc(joint_angles, self)

        return self.chain.forward_kinematics(joint_angles)

    @property
    def actuated_joints_names(self):
        return self._actuated_joints_names

    def jacobian(self, joint_angles):
        if self.jacobian_fnc is not None:
            jacobian = self.jacobian_fnc(joint_angles, self)
        else:
            jacobian = self.chain.jacobian(joint_angles)
        return jacobian[..., self._used_links_mask, :, :]

    @property
    def n_actutated_joints(self):
        return self.n_dofs

    def get_contact_points(self):
        return self.contact_points
        # return torch.cat([self.contact_points,  self.thumb_contacts], dim=1)

    def _set_contact_idxs(self, contact_point_indices, env_mask=None):
        if contact_point_indices is not None:
            if contact_point_indices == "all":
                contact_point_indices = (
                    torch.arange(self.n_contact_candidates, dtype=torch.long, device=self.device)
                    .unsqueeze(0)
                    .expand(self.hand_pose.shape[0], -1)
                )

            elif isinstance(contact_point_indices, str) and "random" in contact_point_indices:
                if not hasattr(self, "rand_contact_candidates"):
                    try:
                        n_pts = int(contact_point_indices.split("_")[-1])
                    except ValueError:
                        print("Invalid contact point indices format, using default 12 points")
                        n_pts = 12

                    # fix seed to make sure its reproducible
                    old_seed = torch.randint(0, 100, (1,)).item()
                    new_seed = 42
                    torch.manual_seed(new_seed)
                    self.rand_contact_candidates = torch.randint(
                        self.n_contact_candidates,
                        (n_pts,),
                        device=self.device,
                    ).unsqueeze(0)
                    torch.manual_seed(old_seed)
                contact_point_indices = self.rand_contact_candidates.clone().expand(self.hand_pose.shape[0], -1)

            if env_mask is None:
                self.contact_point_indices = contact_point_indices.clone()
            else:
                self.contact_point_indices = torch.where(
                    env_mask.unsqueeze(-1),
                    contact_point_indices,
                    self.contact_point_indices,
                )
            self.all_contact_points, self._all_contact_normals = self.get_contact_candidates(with_normals=True)
            self.contact_candidates = self.all_contact_points

            # self.all_contact_points is shape (B, C, 3)
            # contact_point_indices is shape (B, n_contact)

            self.contact_points = self.all_contact_points.gather(
                1, contact_point_indices.unsqueeze(-1).expand(-1, -1, 3)
            )
            self.contact_normals = self._all_contact_normals.gather(
                1, contact_point_indices.unsqueeze(-1).expand(-1, -1, 3)
            )

    def set_parameters(self, hand_pose, contact_point_indices=None, env_mask=None):
        """
        Set translation, rotation, joint angles, and contact points of grasps

        Parameters
        ----------
        hand_pose: (B, 3+6+`n_dofs`) torch.FloatTensor
            translation, rotation in rot6d, and joint angles
        contact_point_indices: (B, `n_contact`) [Optional]torch.LongTensor
            indices of contact candidates
        """

        if env_mask is not None:
            with torch.no_grad():
                # self.hand_pose = hand_pose.clone() * env_mask.unsqueeze(-1).clone().detach()# + self.hand_pose * (~env_mask).unsqueeze(-1)
                self.hand_pose = torch.where(env_mask.unsqueeze(-1), hand_pose, self.hand_pose)
            # fix grad
            self.hand_pose.requires_grad = True
            self.hand_pose.retain_grad()
            # torch.where(env_mask.unsqueeze(-1), hand_pose, self.hand_pose)
            # self.hand_pose._requires_grad = True
            # self.hand_pose[env_mask] = hand_pose.clone().detach()
        else:
            self.hand_pose = hand_pose.clone()

        if self.hand_pose.requires_grad:
            self.hand_pose.retain_grad()

        self.global_translation = self.hand_pose[:, 0:3]
        if self.hand_pose.isnan().any():
            raise ValueError("nan in hand_pose")
        self.global_rotation = robust_compute_rotation_matrix_from_ortho6d(self.hand_pose[:, 3:9])
        self.current_status = self.fk(self.hand_pose[:, 9:])

        self.penetration_keypoints_expanded = self.penetration_keypoints.unsqueeze(0).expand(
            self.hand_pose.shape[0], -1, -1
        )
        self.global_index_to_link_index_penetration_expanded = self.global_index_to_link_index_penetration.unsqueeze(
            0
        ).expand(self.hand_pose.shape[0], -1)
        self._set_contact_idxs(contact_point_indices, env_mask=env_mask)

        self.closing_force_des = self.contact_normals.clone()

    def cal_distance(self, x, return_link_lengths=False):
        """
        Calculate signed distances from object point clouds to hand surface meshes

        Interiors are positive, exteriors are negative

        Use analytical method and our modified Kaolin package

        Parameters
        ----------
        x: (B, N, 3) torch.Tensor
            point clouds sampled from object surface
        """

        dis = []
        x = (x - self.global_translation.unsqueeze(1)) @ self.global_rotation

        if SDF_BACKEND == "WARP":
            n_batch = x.shape[0]
            link_poses = torch.stack(
                [self.current_status[link_name].get_matrix().expand(n_batch, -1, -1) for link_name in self.mesh],
                dim=1,
            )

            link_poses_quat = matrix_to_quaternion(link_poses[:, :, :3, :3])[
                ..., [1, 2, 3, 0]
            ].contiguous()  # .detach().clone()
            link_positions = link_poses[:, :, :3, 3].contiguous()  # .detach().clone()

            meshes_torch = torch.tensor(
                [self.mesh[link_name]["face_verts"].id for link_name in self.mesh],
                dtype=torch.uint64,
                device=self.device,
            )
            meshes = wp.array2d(meshes_torch.unsqueeze(0).expand(n_batch, -1), dtype=wp.uint64)
            dis_local, normals = wp_utils.CalcObjDistances.apply(meshes, link_positions, link_poses_quat, x)

            return (-dis_local).max(dim=1).values

        for idx, link_name in enumerate(self.mesh):
            matrix = self.current_status[link_name].get_matrix()
            x_local = (x - matrix[:, :3, 3].unsqueeze(1)) @ matrix[:, :3, :3]
            x_local = x_local.reshape(-1, 3)  # (total_batch_size * num_samples, 3)

            # if "middle" not in link_name:
            #     continue-
            # checks = ["palm", "link_0", "link_1", "link_2", "link_3"]
            checks = [
                # "right_hand_link",
                # "right_hand_back_link",
                #   "right_hand_thumb_bend_link",
                #   "right_hand_thumb_rota_link1",
                #   "right_hand_thumb_rota_link2",
                #   "right_hand_index_rota_link1",
                #   "right_hand_index_rota_link2",
                #   "right_hand_mid_link1",
                #   "right_hand_ring_link1",
                #   "right_hand_pinky_link1",
                #   "right_hand_mid_link2",
                #   "right_hand_ring_link2",
                #   "right_hand_pinky_link2",
                # "right_hand_index_bend_link",
            ]

            matches = [check in link_name for check in checks]
            # print("Available links:", link_name)
            if any(matches):
                print("Skipping", link_name)
                continue

            if len(self.mesh[link_name]["vertices"]) == 0:
                # print("No vertices for link", link_name, "skipping")
                continue

            if "geom_param" not in self.mesh[link_name]:

                face_verts = self.mesh[link_name]["face_verts"]
                if SDF_BACKEND == "TORCHSDF":
                    dis_local, dis_signs, _, _ = compute_sdf(x_local, face_verts)
                elif SDF_BACKEND == "WARP":
                    # face_verts

                    pts = wp.from_torch(x_local, dtype=wp.vec3)

                    distance_wp, normal_wp = wp_utils.calc_sdf_field(points_wp=pts, mesh_id=face_verts.id)
                    dis_local = wp.to_torch(distance_wp)

                    dis_signs = torch.where(dis_local > 0, 1, -1)
                    dis_local = dis_local**2
                # ):
                elif SDF_BACKEND == "KAOLIN":
                    face_indexes = self.mesh[link_name]["faces"]
                    verts = self.mesh[link_name]["vertices"]
                    dis_local, _, _ = kaolin.metrics.trianglemesh.point_to_mesh_distance(
                        x_local.unsqueeze(0), face_verts
                    )
                    dis_signs = kaolin.ops.mesh.check_sign(verts.unsqueeze(0), face_indexes, x_local.unsqueeze(0))
                    dis_signs = torch.where(
                        dis_signs,
                        -1 * torch.ones_like(dis_signs, dtype=torch.int32),
                        torch.ones_like(dis_signs, dtype=torch.int32),
                    )
                dis_local = torch.sqrt(dis_local + 1e-8)
                dis_local = dis_local * (-dis_signs)
            else:
                height = self.mesh[link_name]["geom_param"][1] * 2
                radius = self.mesh[link_name]["geom_param"][0]
                nearest_point = x_local.detach().clone()
                nearest_point[:, :2] = 0
                nearest_point[:, 2] = torch.clamp(nearest_point[:, 2], 0, height)
                dis_local = radius - (x_local - nearest_point).norm(dim=1)

            dis.append(dis_local.reshape(x.shape[0], x.shape[1]))
        dis = torch.max(torch.stack(dis, dim=0), dim=0)[0]
        return dis

    def self_penetration(self):
        """
        Calculate self penetration energy

        Returns
        -------
        E_spen: (N,) torch.Tensor
            self penetration energy
        """
        batch_size = self.global_translation.shape[0]
        if self.penetration_keypoints_expanded.shape[1] == 0:
            return torch.zeros(batch_size, device=self.device)

        # penetration_keypoints = self.get_penetraion_keypoints()
        # points = self.penetration_keypoints.clone().repeat(batch_size, 1, 1)
        # link_indices = self.global_index_to_link_index_penetration.clone().repeat(batch_size,1)

        points = self.penetration_keypoints_expanded
        link_indices = self.global_index_to_link_index_penetration_expanded

        points = []
        lengths = []
        batch_size = self.global_translation.shape[0]

        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]["penetration_keypoints"].shape[0]
            if n_surface_points == 0:
                continue
            points.append(
                self.current_status[link_name].transform_points(self.mesh[link_name]["penetration_keypoints"])
            )
            points[-1] = points[-1] @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
            lengths.append(n_surface_points)

        all_pts = torch.cat(points, dim=-2)

        counter = 0
        distances = torch.zeros(all_pts.shape[0], len(points) - 1, device=self.device)
        for idx, link_spheres in enumerate(points[:-1]):

            link_spheres_r = self.sphere_scales[counter : counter + link_spheres.shape[1]]
            counter += link_spheres.shape[1]

            other_spheres = all_pts[:, counter:]
            other_spheres_r = self.sphere_scales[counter : all_pts.shape[1]]
            dis = (link_spheres.unsqueeze(1) - other_spheres.unsqueeze(2) + 1e-13).norm(dim=-1)
            th = link_spheres_r.view(1, -1) + other_spheres_r.view(-1, 1)
            pen_dis = dis - th

            distances[:, idx] = pen_dis.min(1)[0].min(1)[0]

        # plot everything using plotly
        coll = -distances.clamp(max=0).sum(1)
        return coll

    def get_surface_points(self):
        """
        Get surface points

        Returns
        -------
        points: (N, `n_surface_points`, 3)
            surface points
        """
        points = []
        batch_size = self.global_translation.shape[0]

        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]["surface_points"].shape[0]
            pts_transformed = self.current_status[link_name].transform_points(self.mesh[link_name]["surface_points"])
            if pts_transformed.ndim == 2:
                pts_transformed = pts_transformed.unsqueeze(0)
            if batch_size != pts_transformed.shape[0]:
                pts_transformed = pts_transformed.expand(batch_size, n_surface_points, 3)

            points.append(pts_transformed)
        try:
            points = torch.cat(points, dim=-2).to(self.device)
        except:
            import pdb

            pdb.set_trace()

        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points

    def get_manipulability(self, moving_directions, contact_point_indices=None, coupled=True):
        _, residuals = self.get_req_joint_velocities(moving_directions, contact_point_indices, coupled=coupled)
        if not coupled:
            residuals = residuals.mean(-1)
        return residuals

    def get_contact_jacobian(self, contact_point_indices=None):
        jacobian = self.jacobian(self.hand_pose[:, 9:])
        if contact_point_indices is None:
            contact_point_indices = (
                torch.arange(self.n_contact_candidates, dtype=torch.long, device=self.device)
                .unsqueeze(0)
                .expand(self.hand_pose.shape[0], -1)
            )
        links_for_contacts = self.global_index_to_link_index[
            contact_point_indices
        ]  # Shape (B, n_contacts) index for body of contact points
        r_contact_local = self.r_contact_body_frames[contact_point_indices]
        # self.contact_points = self.all_contact_points[self.contact_point_indices]
        transforms = []
        for idx, link_name in enumerate(self.mesh):
            cur = self.current_status[link_name].get_matrix()[..., :3, :3].expand(len(self.hand_pose), -1, -1)
            transforms.append(cur)
        body_transforms = torch.stack(transforms, dim=1)  # Shape (B, n_bodies, 3, 3)
        # gather correct transform for each contact point as specified by links_for_contacts
        body_transforms = body_transforms.gather(1, links_for_contacts.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3))
        r_contact_local = (body_transforms @ r_contact_local.unsqueeze(-1)).squeeze(-1)
        # gather relevant jacobians
        jacobian = jacobian.gather(
            1,
            links_for_contacts.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 6, self.n_dofs),
        )
        j_contact_lin = jacobian[..., :3, :] + torch.cross(jacobian[..., 3:, :], r_contact_local.unsqueeze(-1), dim=-2)
        # j_contact_lin is a matrix of shape
        # (B, n_contacts, 3, n_dofs), is a matrix
        # joint_vel has shape (B, n_dofs), is a vector
        # vector matrix multiplication
        j_contact_lin = j_contact_lin.view(-1, body_transforms.shape[1], 3, self.n_dofs)  # (B, n_contacts, 3, n_dofs)
        j_contact_lin = self.global_rotation.unsqueeze(1) @ j_contact_lin
        return j_contact_lin

    def get_ee_vel(self, joint_vel, contact_point_indices):

        jacobian = self.jacobian(self.hand_pose[:, 9:])
        if contact_point_indices is None:
            contact_point_indices = (
                torch.arange(self.n_contact_candidates, dtype=torch.long, device=self.device)
                .unsqueeze(0)
                .expand(self.hand_pose.shape[0], -1)
            )

        links_for_contacts = self.global_index_to_link_index[
            contact_point_indices
        ]  # Shape (B, n_contacts) index for body of contact points
        r_contact_local = self.r_contact_body_frames[contact_point_indices]
        # self.contact_points = self.all_contact_points[self.contact_point_indices]

        transforms = []
        for idx, link_name in enumerate(self.mesh):
            cur = self.current_status[link_name].get_matrix()[..., :3, :3].expand(len(self.hand_pose), -1, -1)
            transforms.append(cur)
        body_transforms = torch.stack(transforms, dim=1)  # Shape (B, n_bodies, 3, 3)
        # gather correct transform for each contact point as specified by links_for_contacts
        body_transforms = body_transforms.gather(1, links_for_contacts.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3))
        r_contact_local = (body_transforms @ r_contact_local.unsqueeze(-1)).squeeze(-1)
        # gather relevant jacobians
        jacobian = jacobian.gather(
            1,
            links_for_contacts.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 6, self.n_dofs),
        )

        j_contact_lin = jacobian[..., :3, :] + torch.cross(jacobian[..., 3:, :], r_contact_local.unsqueeze(-1), dim=-2)
        # j_contact_lin is a matrix of shape
        # (B, n_contacts, 3, n_dofs), is a matrix
        # joint_vel has shape (B, n_dofs), is a vector
        # vector matrix multiplication
        ee_vel = (torch.matmul(j_contact_lin, joint_vel.unsqueeze(-1).unsqueeze(1))).squeeze(-1)

        ee_vel = ee_vel.view(-1, body_transforms.shape[1], 3)
        ee_vel = (self.global_rotation.unsqueeze(1) @ ee_vel.unsqueeze(-1)).squeeze(-1)
        return ee_vel

    def get_req_joint_velocities(
        self,
        moving_directions,
        contact_point_indices=None,
        coupled=True,
        return_ee_vel=False,
    ):

        # convert with glocal rotation and translation
        # import pdb; pdb.set_trace()
        moving_directions = (self.global_rotation.mT.unsqueeze(1) @ moving_directions.unsqueeze(-1)).squeeze(-1)
        # moving_directions = self.global_rotation.mT @ moving_directions
        # moving_directions
        jacobian = self.jacobian(self.hand_pose[:, 9:])
        if contact_point_indices is None:
            contact_point_indices = (
                torch.arange(self.n_contact_candidates, dtype=torch.long, device=self.device)
                .unsqueeze(0)
                .expand(self.hand_pose.shape[0], -1)
            )

        links_for_contacts = self.global_index_to_link_index[
            contact_point_indices
        ]  # Shape (B, n_contacts) index for body of contact points
        r_contact_local = self.r_contact_body_frames[contact_point_indices]
        # self.contact_points = self.all_contact_points[self.contact_point_indices]

        transforms = []
        for idx, link_name in enumerate(self.mesh):
            cur = self.current_status[link_name].get_matrix()[..., :3, :3].expand(moving_directions.shape[0], -1, -1)
            transforms.append(cur)

        body_transforms = torch.stack(transforms, dim=1)  # Shape (B, n_bodies, 3, 3)
        # gather correct transform for each contact point as specified by links_for_contacts
        body_transforms = body_transforms.gather(1, links_for_contacts.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 3, 3))
        r_contact_local = (body_transforms @ r_contact_local.unsqueeze(-1)).squeeze(-1)
        # gather relevant jacobians
        jacobian = jacobian.gather(
            1,
            links_for_contacts.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 6, self.n_dofs),
        )

        j_contact_lin = jacobian[..., :3, :] + torch.cross(jacobian[..., 3:, :], r_contact_local.unsqueeze(-1), dim=-2)

        # j_contact_lin is the linear contact jacobian in world frame. Shape (B, n_contacts, 3, n_dofs)

        # Flatten jacobian to shape (B, 3*n_contacts, n_dofs)
        if coupled:
            j_contact_lin = j_contact_lin.flatten(1, 2)
            moving_directions = moving_directions.flatten(1, 2)

        moving_directions = moving_directions.unsqueeze(-1)

        theta_sol = pinv(j_contact_lin) @ moving_directions
        # theta_sol = 0*theta_sol
        # theta_sol[..., 5, :] = -0.1
        ee_vel = j_contact_lin @ (theta_sol)
        residuals = (ee_vel - moving_directions) ** 2
        if return_ee_vel:
            if coupled:
                ee_vel = ee_vel.view(-1, body_transforms.shape[1], 3)
            ee_vel = (self.global_rotation.unsqueeze(1) @ ee_vel.unsqueeze(-1)).squeeze(-1)
            return theta_sol.squeeze(-1), residuals.squeeze(-1), ee_vel
        return theta_sol.squeeze(-1), residuals.squeeze(-1)

    def get_contact_candidates(self, with_normals=False):
        """
        Get all contact candidates

        Returns
        -------
        points: (N, `n_contact_candidates`, 3) torch.Tensor
            contact candidates
        """
        points = []
        normals = []

        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            contacts = self.mesh[link_name]["contact_candidates"]
            n_surface_points = contacts.shape[0]
            if n_surface_points == 0:
                continue

            transformed = self.current_status[link_name].transform_points(contacts)
            if transformed.ndim == 2:
                transformed = transformed.unsqueeze(0).expand(batch_size, -1, -1)
            points.append(transformed)

            if with_normals:
                normals_p = self.current_status[link_name].transform_normals(self.mesh[link_name]["normal_candidates"])
                if normals_p.ndim == 2:
                    normals_p = normals_p.unsqueeze(0).expand(batch_size, -1, -1)
                normals.append(normals_p)

            # if 1 < batch_size != points[-1].shape[0]:
            #     points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
            #     if with_normals:
            #         normals[-1] = normals[-1].expand(batch_size, n_surface_points, 3)
        if len(points) == 0:
            if with_normals:
                return (
                    torch.zeros(batch_size, 0, 3, device=self.device),
                    torch.zeros(batch_size, 0, 3, device=self.device),
                )
            return torch.zeros(batch_size, 0, 3, device=self.device)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        if with_normals:
            normals = torch.cat(normals, dim=-2).to(self.device)
            normals = normals @ self.global_rotation.transpose(1, 2)
            return points, normals
        return points

    def get_penetraion_keypoints(self):
        """
        Get penetration keypoints

        Returns
        -------
        points: (N, `n_keypoints`, 3) torch.Tensor
            penetration keypoints
        """
        points = []
        batch_size = self.global_translation.shape[0]
        for link_name in self.mesh:
            n_surface_points = self.mesh[link_name]["penetration_keypoints"].shape[0]
            points.append(
                self.current_status[link_name].transform_points(self.mesh[link_name]["penetration_keypoints"])
            )
            if 1 < batch_size != points[-1].shape[0]:
                points[-1] = points[-1].expand(batch_size, n_surface_points, 3)
        points = torch.cat(points, dim=-2).to(self.device)
        points = points @ self.global_rotation.transpose(1, 2) + self.global_translation.unsqueeze(1)
        return points

    def get_plotly_data(
        self,
        i,
        opacity=0.8,
        color="#aaa",
        with_contact_points=False,
        pose=None,
        with_surface_points=False,
        with_penetration_points=False,
        simplify=False,
        offset=[0, 0, 0],
    ):
        """
        Get visualization data for plotly.graph_objects

        Parameters
        ----------
        i: int
            index of data
        opacity: float
            opacity
        color: str
            color of mesh
        with_contact_points: bool
            whether to visualize contact points
        pose: (4, 4) matrix
            homogeneous transformation matrix

        Returns
        -------
        data: list
            list of plotly.graph_object visualization data
        """
        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
        data = []
        for idx, link_name in enumerate(self.mesh):
            v = self.current_status[link_name].transform_points(self.mesh[link_name]["vertices"])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]["faces"].detach().cpu()
            if pose is not None:
                v = v @ pose[:3, :3].T + pose[:3, 3]

            if simplify and len(f) > 100:
                # o3d_mesh = o3d.geometry.TriangleMesh(vertices=o3d.utility.Vector3dVector(vertices), triangles=o3d.utility.Vector3iVector(mesh.faces))
                # o3d_mesh = o3d_mesh.simplify_quadric_decimation(1000)
                # mesh = tm.Trimesh(vertices=o3d_mesh.vertices, faces=o3d_mesh.triangles)
                mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(v),
                    triangles=o3d.utility.Vector3iVector(f),
                )
                mesh = mesh.simplify_quadric_decimation(100)
                v = np.array(mesh.vertices)
                f = np.array(mesh.triangles)

            if offset is not None:
                v += np.array(offset)

            data.append(
                go.Mesh3d(
                    x=v[:, 0],
                    y=v[:, 1],
                    z=v[:, 2],
                    i=f[:, 0],
                    j=f[:, 1],
                    k=f[:, 2],
                    color=color,
                    opacity=opacity,
                    name=link_name,
                    showlegend=True,
                    # legendgroup="hand",
                )
            )
        if with_contact_points:
            contact_points = self.contact_points[i].detach().cpu()
            if pose is not None:
                contact_points = contact_points @ pose[:3, :3].T + pose[:3, 3]
            if offset is not None:
                contact_points += np.array(offset)
            data.append(
                go.Scatter3d(
                    x=contact_points[:, 0],
                    y=contact_points[:, 1],
                    z=contact_points[:, 2],
                    mode="markers",
                    marker=dict(color="red", size=5),
                    legendgroup="contact_points",
                    name="contact_points",
                )
            )

            # vel_norm = self.get_req_joint_velocities(self.contact_normals)
            # plot contact normals
            contact_normals = self.contact_normals[i].detach().cpu()
            # vel_norm = vel_norm[i].detach().cpu()
            if pose is not None:
                contact_normals = contact_normals @ pose[:3, :3].T
            contact_normals *= 0.01
            # data.append(go.Scatter3d(
            #     x=[contact_points[:, 0], contact_points[:, 0] + contact_normals[:, 0]],
            #     y=[contact_points[:, 1], contact_points[:, 1] + contact_normals[:, 1]],
            #     z=[contact_points[:, 2], contact_points[:, 2] + contact_normals[:, 2]],
            #     mode='lines', line=dict(color='red', width=5), legendgroup='contact_normals', name='contact_normals', showlegend=True))
            # vel_norm = (vel_norm.abs() / (vel_norm.max())).clamp(max=1)
            # for j in range(contact_points.shape[0]):
            #     print("j", j, vel_norm[j].item())
            #     if vel_norm[j].item() == 0:
            #         continue
            #     color_string = f'rgb({int(255*vel_norm[j].item())}, {int(255*(1-vel_norm[j].item()))}, 0)'

            #     print("color_string", color_string)
            # data.append(go.Scatter3d(x=[contact_points[j, 0], contact_points[j, 0] + contact_normals[j, 0]], y=[contact_points[j, 1], contact_points[j, 1] + contact_normals[j, 1]], z=[contact_points[j, 2], contact_points[j, 2] + contact_normals[j, 2]],
            #                          mode='lines', line=dict(color=color_string, width=5), legendgroup='contact_normals', name='contact_normals', showlegend=j==0))

        if with_penetration_points:
            penetration_points = self.get_penetraion_keypoints().detach().cpu()
            penetration_points = penetration_points[i]
            # create sphere mesh
            for penetration_keypoint, scales in zip(penetration_points, self.sphere_scales):
                mesh = tm.primitives.Sphere(radius=scales.item())
                v = mesh.vertices + penetration_keypoint.cpu().numpy()
                f = mesh.faces
                if offset is not None:
                    if isinstance(offset, torch.Tensor):
                        offset = offset.cpu().numpy()
                    elif isinstance(offset, list):
                        offset = np.array(offset)
                    v += offset
                data += [
                    go.Mesh3d(
                        x=v[:, 0],
                        y=v[:, 1],
                        z=v[:, 2],
                        i=f[:, 0],
                        j=f[:, 1],
                        k=f[:, 2],
                        color="burlywood",
                        opacity=0.8,
                    )
                ]

        if with_surface_points:
            surface_points = self.get_surface_points().detach().cpu()
            surface_points = surface_points[i]
            if offset is not None:
                if isinstance(offset, torch.Tensor):
                    offset = offset.cpu().numpy()
                elif isinstance(offset, list):
                    offset = np.array(offset)
                surface_points += offset
            # if pose is not None:
            #     surface_points = surface_points @ pose[:3, :3].T + pose[:3, 3]
            data.append(
                go.Scatter3d(
                    x=surface_points[:, 0],
                    y=surface_points[:, 1],
                    z=surface_points[:, 2],
                    mode="markers",
                    marker=dict(color="green", size=5),
                    legendgroup="surface_points",
                    name="surface_points",
                )
            )
        return data

    def get_open3d_data(self, i, with_contact_points=True, pose=None):

        if pose is not None:
            pose = np.array(pose, dtype=np.float32)
        data = []
        mesh = trimesh.Trimesh()
        for link_name in self.mesh:
            v = self.current_status[link_name].transform_points(self.mesh[link_name]["vertices"])
            if len(v.shape) == 3:
                v = v[i]
            v = v @ self.global_rotation[i].T + self.global_translation[i]
            v = v.detach().cpu()
            f = self.mesh[link_name]["faces"].detach().cpu()
            if pose is not None:
                v = v @ pose[:3, :3].T + pose[:3, 3]
            mesh += trimesh.Trimesh(v, f)

        vertices, faces = mesh.vertices, mesh.faces
        mesh = o3d.geometry.TriangleMesh()
        mesh.vertices = o3d.utility.Vector3dVector(vertices)
        mesh.triangles = o3d.utility.Vector3iVector(faces)
        mesh.compute_vertex_normals()
        return mesh

    def show(
        self,
        with_contact_points=True,
        with_surface_points=False,
        with_penetration_points=False,
        additional_pts=None,
        with_grasp_vel=True,
        ee_vel=None,
        idx=0,
        others=[],
    ):
        """
        Visualize hand model

        Parameters
        ----------
        with_contact_points: bool
            whether to visualize contact points
        pose: (4, 4) matrix
            homogeneous transformation matrix
        with_surface_points: bool
            whether to visualize surface points
        with_penetration_points: bool
            whether to visualize penetration points
        """
        data = [] + others
        data += self.get_plotly_data(
            idx,
            with_contact_points=with_contact_points,
            with_surface_points=with_surface_points,
            with_penetration_points=with_penetration_points,
            simplify=False,
        )

        if additional_pts is not None:
            if additional_pts.ndim == 3:
                additional_pts = additional_pts[idx]
            data.append(
                go.Scatter3d(
                    x=additional_pts[:, 0],
                    y=additional_pts[:, 1],
                    z=additional_pts[:, 2],
                    mode="markers",
                    marker=dict(color="black", size=5),
                )
            )

        x_axis = go.Scatter3d(
            x=[0, 0.1],
            y=[0, 0],
            z=[0, 0],
            mode="lines",
            line=dict(color="red", width=5),
        )
        y_axis = go.Scatter3d(
            x=[0, 0],
            y=[0, 0.1],
            z=[0, 0],
            mode="lines",
            line=dict(color="green", width=5),
        )
        z_axis = go.Scatter3d(
            x=[0, 0],
            y=[0, 0],
            z=[0, 0.1],
            mode="lines",
            line=dict(color="blue", width=5),
        )
        data += [x_axis, y_axis, z_axis]
        if with_penetration_points:
            penetration_keypoints = self.get_penetraion_keypoints().detach().cpu()
            for penetration_keypoint in penetration_keypoints:
                mesh = tm.primitives.Capsule(radius=0.01, height=0)
                v = mesh.vertices + penetration_keypoint
                f = mesh.faces
                data += [
                    go.Mesh3d(
                        x=v[:, 0],
                        y=v[:, 1],
                        z=v[:, 2],
                        i=f[:, 0],
                        j=f[:, 1],
                        k=f[:, 2],
                        color="burlywood",
                        opacity=0.5,
                    )
                ]

        if with_surface_points:
            surface_points = self.get_surface_points().detach().cpu()
            data += [
                go.Scatter3d(
                    x=surface_points[:, 0],
                    y=surface_points[:, 1],
                    z=surface_points[:, 2],
                    mode="markers",
                    marker=dict(color="green", size=2),
                )
            ]

        if with_contact_points:
            contact_points = self.get_contact_points().detach().cpu()
            data += [
                go.Scatter3d(
                    x=contact_points[:, 0],
                    y=contact_points[:, 1],
                    z=contact_points[:, 2],
                    mode="markers",
                    marker=dict(color="red", size=5),
                )
            ]

        if with_grasp_vel:
            closing_force = self.closing_force_des.detach().cpu()[idx] * 0.01

            ee_vel_ = ee_vel.detach().cpu()[idx] * 0.01 if ee_vel is not None else None

            start_pts = self.get_contact_points().detach().cpu()[idx]
            end_pts = start_pts + closing_force
            for start_pt, end_pt in zip(start_pts, end_pts):
                data += [
                    go.Scatter3d(
                        x=[start_pt[0], end_pt[0]],
                        y=[start_pt[1], end_pt[1]],
                        z=[
                            start_pt[2],
                            end_pt[2],
                        ],
                        mode="lines",
                        line=dict(color="red", width=5),
                    )
                ]
            if ee_vel_ is not None:
                for start_pt, end_pt in zip(start_pts, start_pts + ee_vel_):
                    data += [
                        go.Scatter3d(
                            x=[start_pt[0], end_pt[0]],
                            y=[start_pt[1], end_pt[1]],
                            z=[
                                start_pt[2],
                                end_pt[2],
                            ],
                            mode="lines",
                            line=dict(color="gray", width=5),
                        )
                    ]

        fig = go.Figure(data=data)
        fig.show()
