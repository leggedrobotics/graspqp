"""
Based on Dexgraspnet: https://pku-epic.github.io/DexGraspNet/
"""

import os

import numpy as np
import torch
import trimesh as tm
import transforms3d
import plotly.graph_objects as go
from graspqp.hands import get_hand_model, AVAILABLE_HANDS
import open3d as o3d
import argparse
import plotly.io as pio

pio.renderers.default = "browser"

torch.manual_seed(1)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Visualize hand model")
    arg_parser.add_argument("--device", type=str, default="cuda", help="device to run the model")
    arg_parser.add_argument("--hand_name", type=str, default="robotiq3", help="name of the hand model", choices=AVAILABLE_HANDS + ["all"])
    arg_parser.add_argument("--show_jacobian", action="store_true", help="show jacobian")
    arg_parser.add_argument("--show_joint_axes", action="store_true", help="show joint axes")
    arg_parser.add_argument("--show_penetration_points", action="store_true", help="show penetration points")
    arg_parser.add_argument("--show_occupancy_grid", action="store_true", help="show occupancy grid")
    arg_parser.add_argument("--randomize_joints", action="store_true", help="randomize joint angles")
    arg_parser.add_argument("--spacing", type=float, default=0.25, help="spacing for visualization")
    arg_parser.add_argument("--grasp_type", type=str, default="all", help="grasp type")

    args = arg_parser.parse_args()
    device = args.device

    if args.hand_name == "all":
        hand_names = AVAILABLE_HANDS
    else:
        hand_names = [args.hand_name]

    data = []
    for idx, hand_name in enumerate(hand_names):
        hand_model = get_hand_model(hand_name, args.device, grasp_type=args.grasp_type)

        joint_angles = hand_model.default_state
        if args.randomize_joints:
            joint_angles = torch.rand_like(hand_model.joints_lower) * (hand_model.joints_upper - hand_model.joints_lower) + hand_model.joints_lower

        rotation = torch.tensor(transforms3d.euler.euler2mat(0.0, -np.pi / 3 * 0, 0, axes="rzxz"), dtype=torch.float, device=device)

        hand_pose = torch.cat([torch.tensor([idx * args.spacing, 0.0, 0.0], dtype=torch.float, device=device), rotation.T.ravel()[:6], joint_angles]).clone().detach()
        hand_pose.grad = None
        hand_pose.requires_grad = True
        hand_model.set_parameters(hand_pose.unsqueeze(0), contact_point_indices="all")

        # info
        surface_points = hand_model.get_surface_points()[0].detach().cpu().numpy()
        contact_candidates = hand_model.get_contact_candidates()[0].detach().cpu().numpy()
        penetration_keypoints = hand_model.get_penetraion_keypoints()[0].detach().cpu().numpy()

        hand_plotly = hand_model.get_plotly_data(i=0, opacity=0.8, color="lightblue", with_contact_points=True, with_penetration_points=False, simplify=False)

        x_axis = go.Scatter3d(x=[idx * args.spacing, idx * args.spacing + 0.1], y=[0, 0], z=[0, 0], mode="lines", line=dict(color="red", width=5))
        y_axis = go.Scatter3d(x=[idx * args.spacing, idx * args.spacing], y=[0, 0.1], z=[0, 0], mode="lines", line=dict(color="green", width=5))
        z_axis = go.Scatter3d(x=[idx * args.spacing, idx * args.spacing], y=[0, 0], z=[0, 0.1], mode="lines", line=dict(color="blue", width=5))

        coordinate_system = [x_axis, y_axis, z_axis]
        surface_points_plotly = [go.Scatter3d(x=surface_points[:, 0], y=surface_points[:, 1], z=surface_points[:, 2], mode="markers", marker=dict(color="yellow", size=2))]
        contact_candidates_plotly = [go.Scatter3d(x=contact_candidates[:, 0], y=contact_candidates[:, 1], z=contact_candidates[:, 2], mode="markers", marker=dict(color="green", size=5))]
        # calc contact normals

        if args.show_penetration_points:
            # data += [go.Scatter3d(x=penetration_keypoints[:, 0], y=penetration_keypoints[:, 1], z=penetration_keypoints[:, 2], mode='markers', marker=dict(color='red', size=3))]
            for penetration_keypoint, scale in zip(penetration_keypoints, hand_model.sphere_scales):
                mesh = tm.primitives.Sphere(radius=scale)
                v = mesh.vertices + penetration_keypoint
                f = mesh.faces
                data += [go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2], color="burlywood", opacity=0.5)]

        if args.show_jacobian:
            J = hand_model.jacobian(joint_angles)

            for mesh_idx, link_name in enumerate(hand_model.mesh):

                jacobian = J[mesh_idx]
                linear_jacobian = jacobian[:3]
                angular_jacobian = jacobian[3:]

                contact_points = hand_model.mesh[link_name]["contact_candidates"]
                # points = points @ self.global_rotation.transpose(
                #     1, 2
                # ) + self.global_translation.unsqueeze(1)

                v_pts = hand_model.mesh[link_name]["vertices"]

                cog = hand_model.mesh[link_name]["vertices"].mean(0, keepdim=True) * 0
                cog = hand_model.current_status[link_name].transform_points(cog).detach().cpu().squeeze(0).numpy()
                cog += hand_model.global_translation.squeeze().detach().cpu().numpy()

                contacts_hand_frame = hand_model.current_status[link_name].transform_points(contact_points)
                contacts_world_frame = contacts_hand_frame + hand_model.global_translation
                # draw coordinate system for each link, with respect to the cog
                axis = torch.tensor([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 0.1]], device=args.device)
                axis = hand_model.current_status[link_name].transform_normals(axis)

                axis = axis.detach().cpu().numpy()
                # draw axis lines
                for i in range(3):
                    c = ["red", "green", "blue"][i]
                    end_pt = cog + axis[i]
                    x, y, z = [cog[0], end_pt[0]], [cog[1], end_pt[1]], [cog[2], end_pt[2]]
                    scatter = go.Scatter3d(x=x, y=y, z=z, mode="lines", line=dict(color=c), name=f"axis_{link_name}", legendgroup=f"axis_{link_name}")
                    data += [scatter]

                # add line from 3d scatter in direction of jacobian
                for idx, (lin_vel, ang_vel) in enumerate(zip(linear_jacobian.T, angular_jacobian.T)):
                    if len(contact_points) == 0:
                        continue

                    contact_points_f = hand_model.current_status[link_name].transform_normals(contact_points)
                    point_vel = torch.cross(ang_vel[None], contact_points_f) + lin_vel[None]
                    point_vel = (hand_model.global_rotation @ point_vel.unsqueeze(-1)).squeeze(-1)
                    if point_vel.norm(dim=1).max() < 1e-3:
                        continue

                    for pt, vec in zip(contacts_world_frame, point_vel):
                        pt = pt.detach().cpu().numpy()
                        v = vec.detach().cpu().numpy()
                        col = ["orange", "purple", "pink", "blue"][idx % 4]
                        scatter = go.Scatter3d(x=[pt[0], pt[0] + v[0]], y=[pt[1], pt[1] + v[1]], z=[pt[2], pt[2] + v[2]], mode="lines", line=dict(color=col, width=6), name=f"jacobian_{link_name}", legendgroup=f"jacobian_{link_name}")
                        data += [scatter]

        if args.show_occupancy_grid:
            # sample points in 5cm cube, evenly grid
            n = 80
            scale = 0.25
            x = torch.linspace(-scale, scale, n) + 0.01
            y = torch.linspace(-scale, scale, n) + 0.01
            z = torch.linspace(-scale, scale, n)
            x, y, z = torch.meshgrid(x, y, z)
            pts = torch.stack([x.flatten(), y.flatten(), z.flatten()], dim=1).to(device)

            # calc distances
            pts_distances = hand_model.cal_distance(pts)
            # print("Hand Pose:", hand_pose)
            # print("Min Distance:", pts_distances)
            # loss = pts_distances.sum()
            # print("Loss:", loss)
            # # loss = hand_pose.sum()
            # loss.backward()
            # print("Gradient:", hand_pose.grad)

            pts_distances = pts_distances.detach().cpu()
            th = -0.002
            pts = pts[pts_distances[0] > th]
            pts_distances = pts_distances[pts_distances > th]
            # pts_distances = pts_distances[pts_distances < 0.005]
            # pts = pts[torch.logical_and(pts_distances[0] > 0 , pts_distances[0] < 0.05)]
            # pts = pts[pts_distances[0] > -10]
            # pts_distances[pts_distances <= 0] = -0.5
            # pts_distances[pts_distances > 0] = 0.5
            colors = ((pts_distances - pts_distances.min()) / (pts_distances.max() - pts_distances.min() + 1e-6) * 100).squeeze()
            pts = pts.detach().cpu().numpy()
            data += [go.Scatter3d(x=pts[:, 0], y=pts[:, 1], z=pts[:, 2], mode="markers", marker=dict(size=5, color=colors, colorscale="RdBu", opacity=1.0))]
        data = hand_plotly + surface_points_plotly + contact_candidates_plotly + coordinate_system + data

    fig = go.Figure(data)
    fig.update_layout(scene_aspectmode="data")
    fig.show()
