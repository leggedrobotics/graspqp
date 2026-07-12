# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Registry of supported robotic hands / grippers.

Each supported end-effector is implemented in its own module (``allegro``,
``shadow``, ``robotiq2``, ...) and exposes a ``getHandModel(device, asset_dir, **kwargs)``
factory that builds a :class:`graspqp.core.HandModel` from the URDF/mesh assets shipped
under ``assets/``. This package collects those factories into a single name-keyed
``_REGISTRY`` so callers can instantiate any hand by string name via
:func:`get_hand_model`.

Attributes:
    ASSET_DIR (str): Absolute path to the bundled ``assets`` directory containing the
        per-hand URDFs, meshes and contact/penetration point definitions.
    AVAILABLE_HANDS (list[str]): Names of all registered hands accepted by
        :func:`get_hand_model` (e.g. ``"allegro"``, ``"shadow_hand"``, ``"robotiq2"``).
"""

import os

ASSET_DIR = os.path.join(os.path.dirname(__file__), "../../../assets")


from .ability_hand import getHandModel as getAbilityHandModel
from .allegro import getHandModel as getAllegroHandModel
from .panda import getHandModel as getPandaHandModel
from .robotiq2 import getHandModel as getRobotiq2HandModel
from .robotiq3 import getHandModel as getRobotiq3HandModel
from .schunk import getHandModel as getSchunkHandModel
from .shadow import getHandModel as getShadowHandModel

_REGISTRY = {
    "robotiq3": getRobotiq3HandModel,
    "panda": getPandaHandModel,
    "allegro": getAllegroHandModel,
    "ability_hand": getAbilityHandModel,
    "shadow_hand": getShadowHandModel,
    "robotiq2": getRobotiq2HandModel,
    "schunk2": getSchunkHandModel,
}

AVAILABLE_HANDS = list(_REGISTRY.keys())


def get_hand_model(hand_name: str, device: str, **kwargs):
    """Build a :class:`graspqp.core.HandModel` for a registered hand by name.

    Args:
        hand_name: Name of the hand to instantiate. Must be one of
            :data:`AVAILABLE_HANDS` (e.g. ``"allegro"``, ``"shadow_hand"``,
            ``"ability_hand"``, ``"robotiq2"``, ``"robotiq3"``, ``"panda"``,
            ``"schunk2"``).
        device: Torch device the model tensors are created on (e.g. ``"cuda"``
            or ``"cpu"``).
        **kwargs: Extra keyword arguments forwarded to the hand's ``getHandModel``
            factory, overriding its defaults (e.g. ``grasp_type`` for the Allegro,
            Ability and Shadow hands, or ``n_surface_points``).

    Returns:
        HandModel: The configured hand model.

    Raises:
        KeyError: If ``hand_name`` is not a registered hand.
    """
    return _REGISTRY[hand_name](device, ASSET_DIR, **kwargs)
