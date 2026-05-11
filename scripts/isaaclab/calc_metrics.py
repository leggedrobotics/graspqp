import os
import re
import glob
import torch
import roma
import numpy as np
import pandas as pd
import tqdm

from graspqp.core import ObjectModel
from graspqp.hands import get_hand_model


def do_eval(checkpoint_path, object_file, hand_type):
    root_path = object_file

    checkpoint_data = torch.load(checkpoint_path)
    device = "cuda"
    hand_model = get_hand_model(
        hand_type,
        "cuda",
        use_collision_if_possible=True,
        grasp_type=checkpoint_data.get("grasp_type", None),
    )
    params = checkpoint_data["parameters"]
    joint_states = []

    for joint_name in hand_model._actuated_joints_names:
        joint_states.append(params[joint_name])

    joint_states = torch.stack(joint_states, dim=-1).to(device)
    root_pose = params["root_pose"].to(device)

    energies = -checkpoint_data["values"]

    _, sorted_indices = torch.sort(energies, descending=True)
    root_pose = root_pose[sorted_indices]
    energies = energies[sorted_indices]
    joint_states = joint_states[sorted_indices]

    root_orientation = roma.unitquat_to_rotmat(root_pose[..., [4, 5, 6, 3]]).mT.flatten(1, 2)
    hand_params = torch.cat([root_pose[..., :3], root_orientation[..., :6], joint_states], dim=-1).to(device)
    if len(hand_params) == 0:
        print(f"No grasps found for {checkpoint_path}")
        return None

    hand_model.set_parameters(hand_params, contact_point_indices="all")

    object_model = ObjectModel(
        data_root_path=root_path,
        batch_size_each=len(hand_params),
        num_samples=3000,
        device=device,
    )
    object_model.initialize([root_path])

    assert len(object_model.object_mesh_list) == 1

    e_pen_max = hand_model.cal_distance(object_model.surface_points_tensor).max(dim=1).values.clamp(min=0)
    e_spen = hand_model.self_penetration()

    filename = os.path.basename(checkpoint_path)
    step_match = re.search(r"step_(\d+)", filename)
    step = int(step_match.group(1)) if step_match else 0

    object_type = os.path.basename(object_file)

    # Path structure: .../grasp_predictions/{hand}/{contacts}/{energy_type}/{grasp_type}/{filename}
    path_parts = checkpoint_path.split(os.sep)
    grasp_predictions_idx = path_parts.index("grasp_predictions")
    energy_type = path_parts[grasp_predictions_idx + 3]

    num_samples = len(e_pen_max)
    df = pd.DataFrame(
        {
            "Path": [checkpoint_path] * num_samples,
            "object_type": [object_type] * num_samples,
            "energy": [energy_type] * num_samples,
            "step": [step] * num_samples,
            "e_pen_max": e_pen_max.detach().cpu().numpy(),
            "e_spen": e_spen.detach().cpu().numpy(),
            "energies": energies.detach().cpu().numpy(),
        }
    )
    out_path = checkpoint_path.replace(".pt", "_analytical_eval.csv")
    df.to_csv(out_path, index=False)
    print(f"Saved analytical evaluation to {out_path}")
    return df


dataset = "/media/zrene/data1/GraspGen/release/test"
template = "{}/*/grasp_predictions/{}/{}/{}/{}/{}"
hands = ["ability_hand", "allegro", "robotiq2", "robotiq3", "shadow_hand"]
contacts = ["12_contacts"]
energy_types = ["graspqp"]
grasp_types = ["default", "pinch", "precision"]

for hand in hands:
    for contact in contacts:
        for energy_type in energy_types:
            for grasp_type in grasp_types:
                print(f"Processing {hand} with {contact} and {energy_type} and {grasp_type}")

                all_files = list(set(glob.glob(
                    template.format(dataset, hand, contact, energy_type, grasp_type, "*.dexgrasp.pt")
                )))
                print(f"Found {len(all_files)} files to process")

                for file in tqdm.tqdm(all_files):
                    if "survivors" in file:
                        continue

                    dataset_file = os.path.dirname(file.split("grasp_predictions")[0])
                    do_eval(file, dataset_file, hand_type=hand)
