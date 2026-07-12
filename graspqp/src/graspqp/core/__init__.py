# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Core grasp-synthesis building blocks.

Re-exports the two central models used throughout GraspQP:

* :class:`~graspqp.core.hand_model.HandModel` -- a differentiable articulated
  hand (forward kinematics, surface/contact sampling, SDF penetration).
* :class:`~graspqp.core.object_model.ObjectModel` -- batched object meshes with
  a differentiable signed distance field.

Companion modules in this package provide the grasp energy terms
(:mod:`graspqp.core.energy`), pose initialization
(:mod:`graspqp.core.initializations`) and the annealing optimizers
(:mod:`graspqp.core.optimizer`).
"""

from .hand_model import HandModel
from .object_model import ObjectModel

__all__ = ["HandModel", "ObjectModel"]
