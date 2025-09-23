# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This script demonstrates how to spawn multiple objects in multiple environments.
.. code-block:: bash
    # Usage
    ././isaaclab.sh -p source/standalone/demos/multi_object.py --num_envs 2048
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from pxr import Usd

from .multi_asset import (default_selection_func,
                          spawn_multi_object_randomly_sdf)


@configclass
class MultiAssetCfg(sim_utils.SpawnerCfg):
    """Configuration parameters for loading multiple assets randomly."""

    func: sim_utils.SpawnerCfg.func = spawn_multi_object_randomly_sdf

    assets_cfg: Iterable[sim_utils.SpawnerCfg] = MISSING
    """Iterable of asset configurations to spawn."""

    postprocess_func: callable[[str, str, Usd.Stage], None] | None = None
    """Optional postprocess function to call after spawning each assets.
        The function should take two arguments: The path to the prototype prim, the path to the spawned prim in the environment, and the stage.
    """

    randomize: bool = True
    """Whether to randomly spawn assets, or spawn them in the order they are defined in the configuration.
    If True, the assets are spawned in random order by using random.choice (i.e. it is not guaranteed that all assets will be spawned the same number of times).
    """

    selection_func: callable[[int, int], int] = default_selection_func
