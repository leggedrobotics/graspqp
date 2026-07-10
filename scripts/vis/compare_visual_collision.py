# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Overlay a hand's visual meshes against its collision meshes in one 3D view.

Builds the hand twice (once from <visual> geometry, once from <collision>) at the same
default pose and overlays them with Plotly: visual in solid blue, collision as a red
wireframe-ish translucent surface. If a collider is mis-transformed (e.g. a bad OBJ axis
round-trip), the red shell will be visibly rotated/offset from the blue visual.

Usage:
    python scripts/vis/compare_visual_collision.py --hand_name allegro
"""

import argparse

import plotly.graph_objects as go
import torch

from graspqp.hands import get_hand_model


def _link_meshes(hand_model):
    """Return {link_name: (vertices[N,3], faces[M,3])} at the current pose."""
    out = {}
    for link_name, m in hand_model.mesh.items():
        v = m["vertices"].detach().cpu().numpy()
        f = m["faces"].detach().cpu().numpy()
        out[link_name] = (v, f)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hand_name", required=True)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()

    # Same neutral pose for both builds: identity rotation (6D), zero translation, default joints.
    def build(only_collision):
        hm = get_hand_model(args.hand_name, args.device, only_use_collision=only_collision)
        pose = torch.zeros(1, 9 + hm.n_dofs, device=args.device)
        pose[:, 3] = 1.0  # rot6d col-0
        pose[:, 7] = 1.0  # rot6d col-1
        pose[:, 9:] = hm.default_state  # default joint angles
        hm.set_parameters(pose, torch.zeros(1, 1, dtype=torch.long, device=args.device))
        return _link_meshes(hm)

    visual = build(only_collision=False)
    collision = build(only_collision=True)

    fig = go.Figure()
    for link, (v, f) in visual.items():
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color="royalblue", opacity=0.45, name=f"visual:{link}", showlegend=True))
    for link, (v, f) in collision.items():
        fig.add_trace(go.Mesh3d(x=v[:, 0], y=v[:, 1], z=v[:, 2], i=f[:, 0], j=f[:, 1], k=f[:, 2],
                                color="crimson", opacity=0.35, name=f"collision:{link}", showlegend=True))

    fig.update_layout(scene_aspectmode="data", title=f"{args.hand_name}: visual (blue) vs collision (red)")
    fig.show()


if __name__ == "__main__":
    main()
