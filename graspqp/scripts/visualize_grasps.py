# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Visualize mined grasps for a hand on the handle mesh (graspqp native frame).

Renders the gripper meshes at each saved grasp pose together with the object/handle mesh
into a self-contained interactive HTML file. This shows the grasp in graspqp's own frame
(object at the origin, gripper at ``root_pose``), independent of any IsaacLab placement --
so it isolates whether the *grasps* are correct from whether the IsaacLab spawn frame is.

Runs in plain python (no Isaac Sim needed):

    /isaac-sim/python.sh source/custom/graspqp/graspqp/scripts/visualize_grasps.py \
        --hand allegro --handle <object_id> --energy_type dexgrasp --grasp_type default \
        --n_contacts 12 --num 8 --out /workspace/grasps.html
"""
import argparse
import glob
import os

import numpy as np
import roma
import torch
import trimesh
import plotly.graph_objects as go

from graspqp.hands import get_hand_model

# Dataset root: /media/zrene/data inside the container, /media/zrene/data1 on the host.
_CLEANED_CANDIDATES = [
    "/media/zrene/data/GraspGen/cleaned_handles",
    "/media/zrene/data1/GraspGen/cleaned_handles",
]


def _resolve_cleaned(override=None):
    for c in ([override] if override else []) + _CLEANED_CANDIDATES:
        if c and os.path.isdir(c):
            return c
    raise FileNotFoundError(
        f"cleaned_handles dir not found. Tried: {_CLEANED_CANDIDATES}. Pass --cleaned_dir."
    )


def quat_wxyz_to_rot6d(q):
    R = roma.unitquat_to_rotmat(q[..., [1, 2, 3, 0]])  # wxyz -> xyzw -> matrix
    return torch.cat([R[..., :, 0], R[..., :, 1]], dim=-1)  # first two columns (graspqp rot6d)


def add_frame(fig, origin, Rmat, length=0.04, width=6, group=None, show_legend=False, on=True):
    """Draw an RGB coordinate frame (x=red, y=green, z=blue) at origin, oriented by Rmat.

    Traces share ``legendgroup=group`` so one legend click toggles this frame across all grasps.
    ``on=False`` starts the frame hidden ("legendonly") -- click the legend entry to reveal it.
    """
    o = np.asarray(origin, dtype=float)
    Rm = np.asarray(Rmat)
    for j, (color, k) in enumerate((("#ff0000", 0), ("#00cc00", 1), ("#0000ff", 2))):
        tip = o + length * Rm[:, k]
        fig.add_trace(go.Scatter3d(
            x=[o[0], tip[0]], y=[o[1], tip[1]], z=[o[2], tip[2]],
            mode="lines", line=dict(color=color, width=width),
            legendgroup=group, name=group, showlegend=(show_legend and j == 0),
            visible=(True if on else "legendonly")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand", default="allegro")
    ap.add_argument("--handle", default=None)
    ap.add_argument("--n_contacts", type=int, default=12)
    ap.add_argument("--energy_type", default="dexgrasp")
    ap.add_argument("--grasp_type", default="default")
    ap.add_argument("--pt", default=None, help="explicit .pt path (overrides the selectors)")
    ap.add_argument("--num", type=int, default=8, help="number of grasps to draw")
    ap.add_argument("--spacing", type=float, default=0.25, help="grid spacing between grasps (m)")
    ap.add_argument("--full", action="store_true", help="full-res meshes (bigger file, nicer)")
    ap.add_argument("--closed", action="store_true",
                    help="render closed_parameters (fingers-closed pose from Isaac eval) instead of "
                         "parameters; only present in succ_grasps/failed_grasps files")
    ap.add_argument("--frame", default=None,
                    help="override the pose frame (e.g. 'grasp_frame' or 'gripper_root'); "
                         "default reads the .pt's 'root_frame' field, falling back to gripper_root")
    ap.add_argument("--cleaned_dir", default=None, help="override path to cleaned_handles dir")
    ap.add_argument("--out", default="grasps.html")
    args = ap.parse_args()

    cleaned = _resolve_cleaned(args.cleaned_dir)

    if args.pt is None:
        pat = os.path.join(cleaned, args.handle, "grasp_predictions", args.hand,
                           f"{args.n_contacts}_contacts", args.energy_type, args.grasp_type, "*.pt")
        files = sorted(glob.glob(pat))
        if not files:
            raise FileNotFoundError(f"No .pt found at {pat}")
        args.pt = files[-1]  # final checkpoint
    print(f"[viz] grasp file: {args.pt}")

    data = torch.load(args.pt, map_location="cpu", weights_only=False)

    # Frame the saved root_pose is in. Seeds from graspqp fit.py are tagged with the hand's
    # optimization frame (e.g. a re-rooted hand's "grasp_frame"); mined/eval outputs are tagged "gripper_root"
    # (written straight to the articulation root). Absent field => gripper root, by convention.
    # Build the hand model in that frame so the drawn gripper matches how the pose is defined
    # (otherwise the gripper renders offset by the grasp-frame->root distance and looks "floating").
    frame = args.frame or data.get("root_frame", "gripper_root")
    root_frame = None if frame == "gripper_root" else frame
    print(f"[viz] root_frame = {frame}  (hand model root_frame={root_frame})")

    # use visual meshes (directly under the hand's meshes/ dir) rather than collision meshes (in a
    # remeshed/ subdir graspqp's loader doesn't resolve) -- and they render nicer.
    hm = get_hand_model(args.hand, device="cpu", use_collision_if_possible=False, root_frame=root_frame)
    params = data["parameters"]
    if args.closed:
        cp = data.get("closed_parameters")
        if isinstance(cp, dict) and cp.get("root_pose") is not None and len(cp["root_pose"]) > 0:
            params = cp
            print("[viz] using closed_parameters (fingers-closed pose)")
        else:
            print("[viz] --closed requested but closed_parameters is missing/empty "
                  "(GA/survivor files don't have it -- use a succ_grasps/failed_grasps file); "
                  "falling back to parameters")
    root = params["root_pose"].float()                       # [N, 7] pos + quat(wxyz)
    # Joint angles: one column per actuated joint, in the hand model's fk order. fit.py exports
    # them as parameters[actuated_joints_names[idx]] = joint_positions[:, idx], so multi-DOF hands
    # (e.g. allegro's 16 DOF) need ALL joints concatenated in that order -- not just the first
    # (which happened to work only for single-DOF grippers).
    joint_order = getattr(hm, "actuated_joints_names", None) or ["finger_left_slide"]
    cols = []
    for name in joint_order:
        j = params[name].float()
        cols.append(j[:, None] if j.ndim == 1 else j)
    finger = torch.cat(cols, dim=-1)                         # [N, n_dofs]
    hand_pose = torch.cat([root[:, :3], quat_wxyz_to_rot6d(root[:, 3:7]), finger], dim=-1)

    n = min(args.num, hand_pose.shape[0])
    print(f"[viz] root height (z) per grasp: {[round(float(z), 4) for z in root[:n, 2].tolist()]}")
    hm.set_parameters(hand_pose[:n])

    # handle mesh (canonical frame = origin, same frame the grasps are relative to)
    obj_path = os.path.join(cleaned, args.handle, "remeshed_simplified.obj")
    handle_mesh = trimesh.load(obj_path, force="mesh") if os.path.exists(obj_path) else None

    # lay the grasps out in a grid so they don't overlap on the small knob (one cell each)
    colors = ["#e6550d", "#1d91c0", "#31a354", "#756bb1", "#d62728", "#17becf", "#bcbd22", "#8c564b",
              "#e377c2", "#2ca02c", "#9467bd", "#ff9896"]
    cols = int(np.ceil(np.sqrt(n)))
    sp = args.spacing

    # per-grasp global transform (root_pose) and per-link FK (relative to root), for the frames
    Rg = hm.global_rotation.detach().cpu().numpy()       # [n,3,3]  root rotation
    Tg = hm.global_translation.detach().cpu().numpy()    # [n,3]    root translation
    link_names = list(hm.current_status.keys())          # all URDF frames (frontend, tcp, fingers, ...)
    link_M = {L: hm.current_status[L].get_matrix().detach().cpu().numpy() for L in link_names}  # [n,4,4]
    # which frames are visible by default; the rest start hidden and are one legend-click away
    default_on = {"frontend", "world", "tcp"}

    offsets = [np.array([(i % cols) * sp, -(i // cols) * sp, 0.0]) for i in range(n)]

    fig = go.Figure()
    # gripper meshes + handles (each a single toggleable legend group)
    for i in range(n):
        traces = hm.get_plotly_data(  # simplify=True needs open3d (not installed) -> full-res
            i, opacity=0.75, color=colors[i % len(colors)], with_contact_points=False,
            simplify=False, offset=list(offsets[i]), legendgroup="grippers", showlegend=False)
        for t in traces:
            t.legendgroup = "grippers"
            t.showlegend = (i == 0 and t is traces[0])
            t.name = "grippers"
        fig.add_traces(traces)
        if handle_mesh is not None:
            v = np.asarray(handle_mesh.vertices) + offsets[i]
            f = np.asarray(handle_mesh.faces)
            fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                    color="#888", opacity=1.0, legendgroup="handles",
                                    name="handles", showlegend=(i == 0)))
    # one coordinate frame per link, per grasp -- grouped by link so the legend toggles all cells
    for L in link_names:
        M = link_M[L]
        nb = M.shape[0]  # fixed links (no joint dependence) come back batch-1; broadcast them
        for i in range(n):
            Mi = M[i] if nb == n else M[0]
            world_pos = Tg[i] + Rg[i] @ Mi[:3, 3] + offsets[i]
            world_rot = Rg[i] @ Mi[:3, :3]
            add_frame(fig, world_pos, world_rot, length=0.03, group=f"frame:{L}",
                      show_legend=(i == 0), on=(L in default_on))

    # graspqp GRASP FRAME: the frontend frame re-labelled by (forward, up, grasp) axes via look_at's
    # basis = [forward, -(forward x up), up]. This is the frame the grasp pose is actually defined in
    # (approach = forward_axis); it differs from the raw frontend/ee_link axes IsaacLab places at.
    fa = np.asarray(hm.forward_axis).reshape(3)
    ua = np.asarray(hm.up_axis).reshape(3)
    grasp_basis = np.stack([fa, -np.cross(fa, ua), ua], axis=-1)  # columns: forward, -(f x u), up
    for i in range(n):
        add_frame(fig, Tg[i] + offsets[i], Rg[i] @ grasp_basis, length=0.06, width=9,
                  group="frame:GRASP", show_legend=(i == 0), on=True)
    fig.update_layout(
        scene_aspectmode="data",
        title=f"{args.hand} grasps on handle {args.handle} - {n} grasps (toggle frames in legend)",
        legend=dict(groupclick="togglegroup"),
        margin=dict(l=0, r=0, t=30, b=0),
    )
    fig.write_html(args.out)
    print(f"[viz] wrote {args.out}  ({n} grasps, full-res meshes)")


if __name__ == "__main__":
    main()
