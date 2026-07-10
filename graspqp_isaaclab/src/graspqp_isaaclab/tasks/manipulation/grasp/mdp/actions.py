# Copyright (c) 2025 ETH Zurich, René Zurbrügg
#
# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

from isaaclab.envs.mdp.actions.joint_actions import JointPositionAction

# NOTE: every grasp hand config sets ``class_type=FixedJointPositionAction`` (the project workaround
# for "resets in isaacsim are broken", i.e. isaac-sim/IsaacLab#1965). So the offset-reset fix lives
# here, on the single class all hands use -- no monkey-patch of the Isaac Lab core base class needed.


class FixedJointPositionAction(JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)

        # reset offsets (isaac-sim/IsaacLab#1965): re-read default_joint_pos so a changed default
        # (the miner sets a new candidate's finger pose per env at reset) doesn't leave a stale offset.
        if self.cfg.use_default_offset:
            # env_ids=None means "reset all"; self._offset[None] would insert a spurious axis and
            # silently corrupt the offset, so map None -> all envs.
            ids = slice(None) if env_ids is None else env_ids
            self._offset[ids] = self._asset.data.default_joint_pos[ids][:, self._joint_ids].clone()
