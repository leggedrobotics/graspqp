import os
from typing import Tuple

import pandas as pd
import roma
import torch
from prettytable import PrettyTable


class RunningStatistics:
    def __init__(self, n_envs: int, asset_mapping: torch.Tensor):
        self._n_envs = n_envs
        self._asset_mapping = asset_mapping
        self._device = asset_mapping.device
        self.reset()

        self._infos = {}
        self._configs = None
        self._paths = None

    def reset(self):
        self.sucesses = torch.zeros(self._n_envs, device=self._device)
        self.trials = torch.zeros(self._n_envs, device=self._device)

    def update_info(self, env_ids, fix=False, **kwargs):
        for var, value in kwargs.items():
            if var not in self._infos:
                self._infos[var] = {
                    "values": torch.zeros(self._n_envs, device=self._device),
                    "count": torch.zeros(self._n_envs, device=self._device),
                }
            if not fix:
                self._infos[var]["values"][env_ids] += value
                self._infos[var]["count"][env_ids] += 1
            else:
                self._infos[var]["values"][env_ids] = value
                self._infos[var]["count"][env_ids] = 1

    def update(self, env_ids, sucesses):
        self.sucesses[env_ids] += sucesses
        self.trials[env_ids] += 1

    def _get_overview_table(self):
        table = PrettyTable()
        table.field_names = ["Asset", "Success Rate", "Successes", "Trials"] + list(self._infos.keys()) + ["Path"]
        table.float_format = "0.2"

        for asset_id in range(self._asset_mapping.max().item() + 1):
            mask = self._asset_mapping == asset_id
            ratio = self.sucesses[mask] / (self.trials[mask] + 1e-3)
            row_data = [
                asset_id,
                ratio.mean().item(),
                self.sucesses[mask].sum().item(),
                self.trials[mask].mean().item(),
            ]

            for info_data in self._infos.values():
                info_values = info_data["values"][mask] / (info_data["count"][mask] + 1e-3)
                row_data.append(info_values.mean().item())
            row_data.append(self._paths[asset_id] if self._paths is not None else "")
            table.add_row(row_data)
        return table

    def _get_details_table(self):
        table = PrettyTable()
        table.field_names = ["Env", "Asset ID", "Success Rate", "Successes", "Trials"] + list(self._infos.keys()) + ["Path"]
        table.float_format = "0.2"
        for env_id in range(self._n_envs):
            ratio = self.sucesses[env_id] / (self.trials[env_id] + 1e-3)
            data = [
                env_id,
                self._asset_mapping[env_id].item(),
                ratio.item(),
                self.sucesses[env_id].item(),
                self.trials[env_id].item(),
            ]
            for info_data in self._infos.values():
                info_values = info_data["values"][env_id] / (info_data["count"][env_id] + 1e-3)
                data.append(info_values.item())
            data.append(self._paths[self._asset_mapping[env_id].item()] if self._paths is not None else "")
            table.add_row(data)
        return table

    def get_df(self):
        table = self._get_overview_table()

        columns, data = table.field_names, table.rows
        df = pd.DataFrame(data, columns=columns)
        if "good_grasps_2cm_050deg" in df.columns and "joints_entropy" in df.columns:
            df["Score"] = df["good_grasps_2cm_050deg"] * (
                0.5 + 1 / 2.5 * (df["joints_entropy"] + 0.2 * df["position_entropy"] + 0.2 * df["orientation_entropy"])
            )
        df = df.astype({"Asset": str})
        number_fields_mask = df.map(lambda x: isinstance(x, (int, float))).all()
        df_mean = df.loc[:, number_fields_mask].mean()
        df = pd.concat(
            [
                df,
                pd.DataFrame([df_mean.to_list()], columns=list(df.columns[number_fields_mask])),
            ]
        )
        df.iloc[-1, 0] = "Mean"
        return df

    def print_statistics(self, full_statistics=False):
        table = self._get_overview_table()
        if full_statistics:
            table = self._get_details_table()
            print(table)

    def to_csv(self, file_name="eval_isaac_sim.csv", folders=None):
        import io

        import pandas as pd

        details = self._get_details_table()
        csv = details.get_csv_string()
        df = pd.read_csv(io.StringIO(csv))
        groups = df.groupby("Path")
        dfs = [groups.get_group(x) for x in groups.groups]
        for df in dfs:
            if folders is not None:
                folder = folders[df["Asset ID"].iloc[0]]
            else:
                folder = os.path.dirname(df["Path"].iloc[0])
            out_file = f"{folder}/{file_name}"
            os.makedirs(folder, exist_ok=True)
            df.to_csv(out_file)
            print(f"Saved to {out_file}")

    def set_configs(self, configs):
        self._configs = configs
        self._paths = [cfg.usd_path for cfg in configs]


def calc_unique_grasps(
    joint_positions: torch.Tensor,
    hand_poses: torch.Tensor,
    limits_pos: torch.Tensor,
    limits_joints: torch.Tensor,
    limits_deg: torch.Tensor,
    valid_envs: torch.Tensor,
) -> Tuple[int, int]:
    euler_angles = roma.unitquat_to_euler("xyz", hand_poses[:, [4, 5, 6, 3]])
    full_state = torch.cat(
        [
            (hand_poses[:, :3] / limits_pos).floor(),
            (euler_angles / limits_deg).floor(),
            (joint_positions / limits_joints).floor(),
        ],
        dim=-1,
    ).long()

    n_unique_grasps = full_state.unique(dim=0).shape[0]
    n_unique_working_grasps = full_state[valid_envs].unique(dim=0).shape[0]
    print(f"Unique grasps: {n_unique_grasps}, Unique working grasps: {n_unique_working_grasps}")
    return n_unique_grasps, n_unique_working_grasps


def calc_entropy_for_grasps(
    joint_positions: torch.Tensor, hand_poses: torch.Tensor, env: object, n_bins: int = 32
) -> Tuple[torch.Tensor | float, torch.Tensor, torch.Tensor]:
    joints_entropy = 0
    actuated_joint_ids = env.scene["robot"].data.actuated_joint_indices
    joint_pos_limits = env.scene["robot"].data.joint_pos_limits

    for joint_idx, joint_position in enumerate(joint_positions.T):
        limits = joint_pos_limits[0, actuated_joint_ids[joint_idx]]
        joints_entropy += entropy(joint_position, n_bins, limits[0], limits[1]) / joint_positions.shape[-1]

    position_entropy = entropy(hand_poses[:, :3].T, n_bins, -0.1, 0.1)
    orientation_entropy = entropy(hand_poses[:, 3:].T, n_bins, -1, 1)
    return joints_entropy, position_entropy, orientation_entropy


def entropy(distribution: torch.Tensor, n_bins: int, min_limit: float, max_limit: float) -> torch.Tensor:
    if distribution.ndim == 1:
        distribution = distribution.unsqueeze(0)

    entropy = 0
    for samples in distribution:
        counts = samples.histc(n_bins, min_limit, max_limit)
        dist = counts / counts.sum()
        logs = torch.log(torch.where(dist > 0, dist, 1))
        ent = -(dist * logs).sum()
        ent[torch.isnan(ent)] = 0
        entropy += ent
    return entropy / distribution.shape[0]
