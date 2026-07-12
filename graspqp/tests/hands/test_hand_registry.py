# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Tests for the graspqp.hands registry (dispatch + released-hand set)."""

import pytest
import torch

from graspqp.hands import AVAILABLE_HANDS, get_hand_model

# The exact set of hands shipped in the public release.
EXPECTED_HANDS = {
    "ability_hand",
    "allegro",
    "panda",
    "robotiq2",
    "robotiq3",
    "schunk2",
    "shadow_hand",
}

# Copyright-restricted hands that were removed and must never reappear.
RESTRICTED_HANDS = {"xhand", "dex3", "hoi"}


def test_registry_exposes_exactly_the_released_hands():
    assert set(AVAILABLE_HANDS) == EXPECTED_HANDS


def test_no_restricted_hands_are_registered():
    assert RESTRICTED_HANDS.isdisjoint(set(AVAILABLE_HANDS))


def test_unknown_hand_raises():
    with pytest.raises(KeyError):
        get_hand_model("does_not_exist", "cpu")


@pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="building a hand model needs a CUDA device for the SDF backend",
)
@pytest.mark.parametrize("hand_name", sorted(EXPECTED_HANDS))
def test_every_registered_hand_loads(hand_name):
    hand_model = get_hand_model(hand_name, "cuda")
    assert hand_model.n_dofs > 0
    assert len(hand_model.actuated_joints_names) > 0
    assert hand_model.n_contact_candidates > 0
