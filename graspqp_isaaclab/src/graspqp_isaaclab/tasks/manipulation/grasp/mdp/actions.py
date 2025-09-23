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


class FixedJointPositionAction(JointPositionAction):
    """Joint action term that applies the processed actions to the articulation's joints as position commands."""

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)

        # reset offsets
        if self.cfg.use_default_offset:
            self._offset[env_ids] = self._asset.data.default_joint_pos[env_ids][:, self._joint_ids].clone()
