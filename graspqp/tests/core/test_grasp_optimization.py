# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""End-to-end smoke test for the GraspQP optimization pipeline.

Runs the real MALA* optimizer (the one used by ``scripts/fit.py``) for a handful of
iterations on a small bundled object mesh and asserts that the grasp energy goes down.

This exercises the full stack -- hand model, object SDF (Warp backend), the DexGraspNet
energy terms, and the MALA* accept/reject loop -- on a tiny problem so it stays fast.

The bundled asset lives at ``tests/assets/dummy_sphere/coacd/decomposed.obj`` in the
layout ``ObjectModel`` expects (``<data_root>/<object_code>/coacd/decomposed.obj``).
"""

import math
import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

# tests/core/ -> tests/assets/
ASSET_ROOT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")
DUMMY_OBJECT = "dummy_sphere"
BATCH_SIZE = 32
N_ITERS = 10


def _fit_args():
    """Namespace of the ``scripts/fit.py`` defaults consumed by ``initialize_convex_hull``."""
    return SimpleNamespace(
        n_contact=12,
        jitter_strength=0.1,
        distance_lower=0.05,
        distance_upper=0.1,
        rotate_lower=-math.pi,
        rotate_upper=math.pi,
        pitch_lower=-15 * math.pi / 180,
        pitch_upper=15 * math.pi / 180,
        tilt_lower=-45 * math.pi / 180,
        tilt_upper=45 * math.pi / 180,
    )


def _make_energy_fnc(energy_type):
    """Build the E_fc metric for an energy type, mirroring ``scripts/fit.py``.

    Both variants below run in the lightweight (WARP) image: ``dexgrasp`` is analytic and
    ``graspqp`` uses the qpth ``SQPLsqSolver`` (shipped via the ``opt`` extra) -- neither
    needs the optional ``proxsuite``/TorchSDF backends.
    """
    from graspqp.metrics import GraspSpanMetricFactory

    if energy_type == "dexgrasp":
        return GraspSpanMetricFactory.create(GraspSpanMetricFactory.MetricType.DEXGRASP)
    if energy_type == "graspqp":
        return GraspSpanMetricFactory.create(
            GraspSpanMetricFactory.MetricType.GRASPQP,
            solver_kwargs={"friction": 0.2, "max_limit": 20.0, "n_cone_vecs": 4},
        )
    raise ValueError(energy_type)


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="grasp optimization needs a CUDA device for the SDF backend",
)
@pytest.mark.parametrize("energy_type", ["dexgrasp", "graspqp"])
def test_graspqp_optimization_reduces_energy(energy_type):
    from graspqp.core import ObjectModel
    from graspqp.core.energy import calculate_energy
    from graspqp.core.initializations import initialize_convex_hull
    from graspqp.core.optimizer import MalaStar
    from graspqp.hands import get_hand_model

    torch.manual_seed(0)
    np.random.seed(0)
    device = "cuda"

    # --- build hand + object -------------------------------------------------
    hand_model = get_hand_model("allegro", device, grasp_type="all")

    object_model = ObjectModel(
        data_root_path=ASSET_ROOT,
        batch_size_each=BATCH_SIZE,
        num_samples=2500,
        device=device,
    )
    object_model.initialize([DUMMY_OBJECT])

    # convex-hull initialization places the hand around the object (as in fit.py)
    initialize_convex_hull(hand_model, object_model, _fit_args())

    # --- optimizer + energy (fit.py defaults) --------------------------------
    optimizer = MalaStar(
        hand_model,
        switch_possibility=0.4,
        starting_temperature=18.0,
        temperature_decay=0.95,
        annealing_period=30,
        step_size=0.005,
        stepsize_period=50,
        mu=0.98,
        device=device,
        batch_size=BATCH_SIZE,
        clip_grad=False,
    )

    energy_fnc = _make_energy_fnc(energy_type)
    weight_dict = {"E_dis": 100.0, "E_fc": 1.0, "E_pen": 100.0, "E_spen": 10.0, "E_joints": 1.0}
    energy_names = [name for name, w in weight_dict.items() if w > 0.0]

    def total_energy():
        losses = calculate_energy(
            hand_model,
            object_model,
            energy_fnc=energy_fnc,
            energy_names=energy_names,
            method="gendexgrasp",
            svd_gain=0.1,
        )
        e = 0
        for name, value in losses.items():
            e = e + weight_dict[name] * value
        return e

    # initial energy + gradient (mirrors fit.py's pre-loop backward/zero_grad)
    energy = total_energy()
    energy.sum().backward()
    optimizer.zero_grad()
    energy = energy.detach()

    initial_mean = energy.mean().item()
    assert np.isfinite(initial_mean)
    trajectory = [initial_mean]

    # --- optimization loop (mirrors scripts/fit.py) --------------------------
    for _ in range(N_ITERS):
        optimizer.try_step()
        optimizer.zero_grad()

        new_energy = total_energy()
        new_energy.sum().backward()

        with torch.no_grad():
            batched = energy.view(-1, BATCH_SIZE)
            z_score = ((batched - batched.mean(-1, keepdim=True)) / (batched.std(-1, keepdim=True) + 1e-9)).view(-1)
            accept, _ = optimizer.accept_step(energy, new_energy, None, z_score, 2.0)
            energy = energy.clone()
            energy[accept] = new_energy[accept].detach()
        trajectory.append(energy.mean().item())

    final_mean = energy.mean().item()

    assert np.isfinite(final_mean), f"[{energy_type}] energy became non-finite: {trajectory}"
    assert final_mean < initial_mean, (
        f"[{energy_type}] optimization did not reduce the grasp energy over {N_ITERS} iterations: "
        f"{initial_mean:.3f} -> {final_mean:.3f} (trajectory: {[round(t, 2) for t in trajectory]})"
    )
