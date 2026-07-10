# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Grasp energy terms used to optimize hand poses against an object.

This module bundles the individual energy terms that drive the grasp
optimization (contact distance, force closure, joint-limit violation, hand
object penetration, self penetration and several optional priors) into a
single :func:`calculate_energy` call. The returned per-term energies are
weighted and summed by the caller to form the scalar objective that the
optimizers in :mod:`graspqp.core.optimizer` minimize.
"""

import torch


def calculate_energy(
    hand_model,
    object_model,
    energy_fnc: any = None,
    energy_names=[],
    method="gendexgrasp",
    svd_gain=0.1,
):
    """Compute the dictionary of grasp energy terms for a batch of grasps.

    Every term is returned as a per-grasp tensor of shape ``(batch_size,)`` so
    the caller can weight and sum them into a scalar objective. Which optional
    terms are produced is controlled by ``energy_names``.

    Args:
        hand_model (HandModel): Hand whose ``contact_points``, ``contact_normals``,
            ``hand_pose`` and joint limits have already been set via
            :meth:`HandModel.set_parameters`.
        object_model (ObjectModel): Initialized object providing the signed
            distance field (``cal_distance``), sampled surface points and
            center of gravity.
        energy_fnc (callable): Force-closure / grasp-quality metric. Called as
            ``energy_fnc(contact_pts, contact_normals, sdf, cog, with_solution=True,
            svd_gain=...)`` and expected to return ``(E_fc, solution)``.
        energy_names (list[str]): Names of optional terms to additionally
            compute. Recognized values are ``"E_wall"``, ``"E_prior"`` and
            ``"E_manipulativity"``.
        method (str): Contact-distance formulation. ``"dexgraspnet"`` uses the
            plain absolute signed distance; ``"gendexgrasp"`` (default)
            additionally weights it by the alignment between the object surface
            normal and the hand contact normal.
        svd_gain (float): Regularization gain forwarded to ``energy_fnc`` for
            the force-closure solve.

    Returns:
        dict[str, torch.Tensor]: Mapping of energy name to a ``(batch_size,)``
        tensor. Always contains ``"E_dis"`` (contact distance), ``"E_fc"``
        (force closure), ``"E_joints"`` (joint-limit violation), ``"E_pen"``
        (hand-object penetration) and ``"E_spen"`` (self penetration). May also
        contain ``"E_prior"``, ``"E_wall"`` and ``"E_manipulativity"`` depending
        on ``energy_names``.

    Raises:
        ValueError: If ``method`` is not ``"dexgraspnet"`` or ``"gendexgrasp"``.
    """

    batch_size, n_contact, _ = hand_model.contact_points.shape
    device = object_model.device

    losses = {}

    if method == "dexgraspnet":
        distance, contact_normal = object_model.cal_distance(hand_model.contact_points)
        E_dis = torch.sum(distance.abs(), dim=-1, dtype=torch.float).to(device)
        losses["E_dis"] = E_dis
    elif method == "gendexgrasp":
        distance, contact_normal = object_model.cal_distance(hand_model.contact_points)

        if "E_wall" in energy_names:
            good_contacts = hand_model.contact_points[:, :, -1] > 0.0
            contact_normal *= good_contacts[..., None] + 1e-3
            distance[~good_contacts] *= 10.0

        vC = contact_normal
        nH = hand_model.contact_normals
        E_dis = ((1 - torch.sum((-vC) * nH, dim=-1)).exp() * distance.abs()).sum(-1)
    else:
        raise ValueError(f"Unknown method: {method}")
    losses["E_dis"] = E_dis

    # Points that are opposite to the object surface are also invalid

    E_fc, _lambda = energy_fnc(
        contact_pts=hand_model.contact_points,  # hand_model.contact_points,
        contact_normals=contact_normal,
        sdf=distance,
        cog=object_model.cog,
        with_solution=True,
        svd_gain=svd_gain,
    )

    losses["E_fc"] = E_fc

    # E_joints
    E_joints = torch.sum(
        (hand_model.hand_pose[:, 9:] > hand_model.joints_upper)
        * (hand_model.hand_pose[:, 9:] - hand_model.joints_upper),
        dim=-1,
    ) + torch.sum(
        (hand_model.hand_pose[:, 9:] < hand_model.joints_lower)
        * (hand_model.joints_lower - hand_model.hand_pose[:, 9:]),
        dim=-1,
    )
    losses["E_joints"] = E_joints

    # E_pen
    object_scale = object_model.object_scale_tensor.flatten().unsqueeze(1).unsqueeze(2)
    object_surface_points = (
        object_model.surface_points_tensor * object_scale
    )  # (n_objects * batch_size_each, num_samples, 3)
    distances = hand_model.cal_distance(object_surface_points)
    distances[distances <= 0] = 0
    E_pen = distances.sum(-1)
    losses["E_pen"] = E_pen

    # E_spen
    E_spen = hand_model.self_penetration()
    losses["E_spen"] = E_spen

    if "E_prior" in energy_names:
        forward_axis = (hand_model.global_rotation @ hand_model.grasp_axis.view(1, -1, 1)).view(-1, 3)
        # This should point downwards,
        axis_prior = torch.tensor([0, 0, -1], dtype=torch.float, device=device).view(1, 3)
        E_prior = 1 - torch.sum((forward_axis * axis_prior), dim=-1)
        # E_joints += w_prior * E_prior / w_joints
        losses["E_prior"] = E_prior

    if "E_wall" in energy_names:
        z_height = hand_model.get_surface_points()[..., -1].clamp(max=0.0)
        losses["E_wall"] = z_height.abs().sum(-1)

    if "E_manipulativity" in energy_names:
        # Jacobian
        E_jacobian = hand_model.get_manipulability(
            contact_normal * distance.unsqueeze(-1).abs().clamp(min=5e-3),
            hand_model.contact_point_indices,
        )
        E_manipulativity = E_jacobian.mean(-1)
        losses["E_manipulativity"] = E_manipulativity

    return losses
