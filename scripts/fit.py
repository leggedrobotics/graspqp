"""
Based on Dexgraspnet: https://pku-epic.github.io/DexGraspNet/
"""

import os


import argparse
import numpy as np
import torch
from tqdm import tqdm
import math

from graspqp.hands import get_hand_model, AVAILABLE_HANDS
from graspqp.core import ObjectModel

from graspqp.core.initializations import initialize_convex_hull
from graspqp.core.energy import calculate_energy

from graspqp.core.optimizer import (
    MalaStar,
    AnnealingDexGraspNet,
)
from graspqp.utils.transforms import (
    robust_compute_rotation_matrix_from_ortho6d,
)

from graspqp.utils.wandb_wrapper import WandbMockup
from graspqp.utils.plot_utils import (
    get_plotly_fig,
    show_initialization,
)


from graspqp.metrics import GraspSpanMetricFactory

import datetime
import roma


import torch


# prepare arguments
parser = argparse.ArgumentParser()

parser.add_argument("--seed", default=1, type=int)
parser.add_argument("--gpu", default="0", type=str)
parser.add_argument(
    "--object_code_list",
    default=[],
    nargs="+",
)

parser.add_argument("--energy_name", default=None, type=str)
parser.add_argument("--n_contact", default=12, type=int)
parser.add_argument("--batch_size", default=32, type=int)
parser.add_argument("--n_iter", default=7000, type=int)
# hyper parameters (** Magic, don't touch! **)
parser.add_argument("--switch_possibility", default=0.4, type=float)
parser.add_argument("--mu", default=0.98, type=float)
parser.add_argument("--step_size", default=0.005, type=float)
parser.add_argument("--stepsize_period", default=50, type=int)
parser.add_argument("--starting_temperature", default=18, type=float)
parser.add_argument("--annealing_period", default=30, type=int)
parser.add_argument("--temperature_decay", default=0.95, type=float)


parser.add_argument("--w_dis", default=100.0, type=float)
parser.add_argument("--w_fc", default=1.0, type=float)
parser.add_argument("--w_pen", default=100, type=float)
parser.add_argument("--w_spen", default=10, type=float)
parser.add_argument("--w_joints", default=1, type=float)
parser.add_argument("--clip_grad", action="store_true")


# initialization settings
parser.add_argument("--jitter_strength", default=0.1, type=float)  # In percentage of joint limits
parser.add_argument("--distance_lower", default=0.05, type=float)
parser.add_argument("--distance_upper", default=0.1, type=float)

# rotation settings
parser.add_argument("--rotate_lower", default=-180 * math.pi / 180, type=float)
parser.add_argument("--rotate_upper", default=180 * math.pi / 180, type=float)

parser.add_argument("--pitch_lower", default=-15 * math.pi / 180, type=float)
parser.add_argument("--pitch_upper", default=15 * math.pi / 180, type=float)
parser.add_argument("--tilt_lower", default=-45 * math.pi / 180, type=float)
parser.add_argument("--tilt_upper", default=45 * math.pi / 180, type=float)

parser.add_argument("--reset_epochs", default=600, type=int)
parser.add_argument("--optimizer", default="mala_star", type=str, choices=["mala_star", "dexgraspnet"])
parser.add_argument("--initialization", default="convex_hull", type=str, choices=["random", "convex_hull"])

parser.add_argument("--w_prior", default=0.0, type=float)
parser.add_argument("--w_wall", default=0.0, type=float)

parser.add_argument(
    "--energy_type",
    default="graspqp",
    type=str,
    choices=["dexgrasp", "graspqp", "tdg"],
)

parser.add_argument("--debug", action="store_true")
parser.add_argument("--selected_object", default=None, type=str)
parser.add_argument("--dataset", default="debug", type=str)
parser.add_argument(
    "--data_root_path",
    default=None,
    type=str,
)


parser.add_argument("--show_initialization", action="store_true")

parser.add_argument("--object_code_file", default=None, type=str)

parser.add_argument("--wandb_name", default=None, type=str)
parser.add_argument("--wandb_project", default="graspqp", type=str)
parser.add_argument("--log_to_wandb", action="store_true")
parser.add_argument("--no_plotly", action="store_true")

parser.add_argument("--hand_name", default="allegro", type=str, choices=AVAILABLE_HANDS)

parser.add_argument("--norm_sampling", action="store_true")
parser.add_argument("--w_svd", default=0.1, type=float)
parser.add_argument("--z_score_threshold", default=1.0, type=float)
parser.add_argument("--grasp_type", default="all", type=str)
parser.add_argument("--log_entropy", action="store_true")
parser.add_argument("--friction", default=0.2, type=float)
parser.add_argument("--max_lambda_limit", default=20.0, type=float)

# Specific energy arguments
parser.add_argument("--torque_weight", default=5.0, type=float)
parser.add_argument("--n_friction_cone", default=4, type=int)
parser.add_argument("--use_gendexgrasp", default=True, type=bool)
parser.add_argument("--no_exp_term", action="store_true")  # Disable exploration term

args = parser.parse_args()

if args.data_root_path is None:
    args.data_root_path = os.path.join("/data/release", args.dataset)

# Check if PYTHON_EULER_ROOT env exists
if "PYTHON_EULER_ROOT" in os.environ:
    pass
    # print("EULER CLUSTER DETECTED, will re-map data path")
    # args.data_root_path = args.data_root_path.replace(
    #     "/data", "/cluster/scratch/zrene/data"
    # )


if args.energy_name is None:
    args.energy_name = args.energy_type

if args.object_code_file is not None:
    print("File with object codes provided")
    if not os.path.exists(args.object_code_file):
        raise ValueError(f"File {args.object_code_file} does not exist")
    with open(args.object_code_file, "r") as f:
        # trim whistespace and split by space or new line
        args.object_code_list = f.read().replace("\n", " ").strip().split(" ")

if len(args.object_code_list) == 0:
    print("Using all objects in the data root path")
    args.object_code_list = [o for o in os.listdir(args.data_root_path) if "captures" not in o]
    print(args.object_code_list)

if isinstance(args.object_code_list, str):
    args.object_code_list = [args.object_code_list]

# args.object_code_list =  args.object_code_list[:5]
os.environ["KMP_DUPLICATE_LIB_OK"] = "True"

np.seterr(all="raise")
np.random.seed(args.seed)
torch.manual_seed(args.seed)


num_objects = len(args.object_code_list)
total_batch_size = num_objects * args.batch_size

print("=====================================")
print("Starting grasp optimization on", torch.cuda.get_device_name(0))
print("Batch size: ", total_batch_size)
print("Energy type: ", args.energy_type)
print("Energy name: ", args.energy_name)
print("Data root path: ", args.data_root_path)
print("Number of objects: ", num_objects)
print("=====================================")

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(
    "Starting grasp optimization on",
    device,
    "saving results to",
    os.path.join(args.data_root_path, args.object_code_list[0], "grasp_predictions"),
)

args.selected_object = args.object_code_list[0]
wandb = WandbMockup(enabled=args.log_to_wandb)

# Log args to wandb
wandb.init(
    project=args.wandb_project,
    config=args,
    name=("asset_" + args.selected_object if args.wandb_name is None else args.wandb_name),
)

timrestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

os.makedirs(
    os.path.join(args.data_root_path, args.object_code_list[0], "grasp_predictions"),
    exist_ok=True,
)
print("Logging to:", os.path.join(args.data_root_path, args.object_code_list[0], "grasp_predictions"))


def get_result_path(asset_id):
    path = os.path.join(
        os.path.join(
            args.data_root_path,
            args.object_code_list[asset_id],
            "grasp_predictions",
            args.hand_name,
            f"{args.n_contact}_contacts",
            args.energy_name,
        )
    )

    if args.grasp_type in [None, "all"]:
        path = os.path.join(path, "default")
    else:
        path = os.path.join(path, args.grasp_type)

    os.makedirs(path, exist_ok=True)
    return path


def export_poses(hand_model, energy, object_model, suffix="None"):

    full_hand_poses = hand_model.hand_pose.detach().cpu()
    energies = energy.detach().cpu()

    old_contact_indices = hand_model.contact_point_indices.clone().detach()
    distance, contact_normal = object_model.cal_distance(hand_model.contact_points)
    contact_normal = 5 * (contact_normal * distance.unsqueeze(-1).abs())  # .clamp(min=5e-3)

    delta_theta, residuals, ee_vel = hand_model.get_req_joint_velocities(
        -contact_normal, hand_model.contact_point_indices, return_ee_vel=True
    )
    #
    hand_model._set_contact_idxs("all")
    distance_full, contact_normal_full = object_model.cal_distance(hand_model.contact_points)
    contact_normal_full = 5 * (contact_normal_full * distance_full.unsqueeze(-1).abs())  # .clamp(min=5e-3)
    delta_theta_full, _ = hand_model.get_req_joint_velocities(
        -contact_normal_full, hand_model.contact_point_indices, return_ee_vel=False
    )
    hand_model._set_contact_idxs(old_contact_indices)

    # hand_model.closing_force_des
    distance, contact_normal = object_model.cal_distance(hand_model.contact_points)
    contact_normal = 5 * contact_normal * (distance.unsqueeze(-1).abs() + 0.005)  # .clamp(min=5e-3)

    delta_theta_off, residuals, ee_vel = hand_model.get_req_joint_velocities(
        -contact_normal, hand_model.contact_point_indices, return_ee_vel=True
    )

    for asset_idx in range(len(args.object_code_list)):
        data = {"values": energies[asset_idx * args.batch_size : (asset_idx + 1) * args.batch_size]}
        start_idx = asset_idx * args.batch_size
        end_idx = (asset_idx + 1) * args.batch_size

        joint_delta = delta_theta[start_idx:end_idx]

        hand_poses = robust_compute_rotation_matrix_from_ortho6d(full_hand_poses[start_idx:end_idx, 3:9])
        hand_qxyzw = roma.rotmat_to_unitquat(hand_poses)
        hand_qwxyz = hand_qxyzw[:, [3, 0, 1, 2]]
        hand_poses = torch.cat([full_hand_poses[start_idx:end_idx, :3], hand_qwxyz], dim=1)
        joint_positions = full_hand_poses[start_idx:end_idx, 9:]
        parameters = {}
        for idx in range(joint_positions.shape[1]):
            parameters[hand_model._actuated_joints_names[idx]] = joint_positions[:, idx].detach().cpu()
        parameters["root_pose"] = hand_poses.detach().cpu()

        grasp_velocities = {}
        full_grasp_velocities = {}
        grasp_velocities_off = {}
        for idx in range(joint_delta.shape[1]):
            grasp_velocities[hand_model._actuated_joints_names[idx]] = joint_delta[:, idx].detach().cpu()
            full_grasp_velocities[hand_model._actuated_joints_names[idx]] = (
                delta_theta_full[start_idx:end_idx, idx].detach().cpu()
            )
            grasp_velocities_off[hand_model._actuated_joints_names[idx]] = (
                delta_theta_off[start_idx:end_idx, idx].detach().cpu()
            )

        SHOW = False
        if SHOW:
            plot = object_model.get_plotly_data(0)
            hand_model.show(others=plot, ee_vel=ee_vel)
            input("Press Enter to continue...")

        file_path = os.path.join(
            get_result_path(asset_idx),
            args.object_code_list[asset_idx] + f"{suffix}.dexgrasp.pt",
        )
        data["parameters"] = parameters
        data["grasp_velocities"] = grasp_velocities
        data["full_grasp_velocities"] = full_grasp_velocities
        data["grasp_velocities_off"] = grasp_velocities_off
        data["contact_idx"] = hand_model.contact_point_indices[start_idx:end_idx].detach().cpu()
        data["grasp_type"] = args.grasp_type
        data["contact_links"] = hand_model._contact_links
        torch.save(data, file_path)
        print(f"\033[94m==> Exported to {os.path.abspath(file_path)}\033[0m")


# hand_model = get_allegro_model(device)
hand_model = get_hand_model(args.hand_name, device, grasp_type=args.grasp_type)

object_model = ObjectModel(
    data_root_path=args.data_root_path,
    batch_size_each=args.batch_size,
    num_samples=2500,
    device=device,
)
object_model.initialize(args.object_code_list)

if args.initialization == "convex_hull":
    initialize_convex_hull(hand_model, object_model, args)

print("n_contact_candidates", hand_model.n_contact_candidates)
print("total batch size", total_batch_size)
hand_pose_st = hand_model.hand_pose.detach()

optim_config = {
    "switch_possibility": args.switch_possibility,
    "starting_temperature": args.starting_temperature,
    "temperature_decay": args.temperature_decay,
    "annealing_period": args.annealing_period,
    "step_size": args.step_size,
    "stepsize_period": args.stepsize_period,
    "mu": args.mu,
    "device": device,
    "batch_size": args.batch_size,
    "clip_grad": args.clip_grad,
}

if args.optimizer == "mala_star":
    optimizer = MalaStar(hand_model, **optim_config)
elif args.optimizer == "dexgraspnet":
    optimizer = AnnealingDexGraspNet(hand_model, **optim_config)

else:
    raise NotImplementedError("Optimizer not implemented")


energy_fnc = None
if args.energy_type == "dexgrasp" or args.energy_type == "gendexgrasp":
    energy_fnc = GraspSpanMetricFactory.create(GraspSpanMetricFactory.MetricType.DEXGRASP)
elif args.energy_type == "tdg":
    energy_fnc = GraspSpanMetricFactory.create(GraspSpanMetricFactory.MetricType.TDG)
elif args.energy_type == "graspqp":
    energy_fnc = GraspSpanMetricFactory.create(
        GraspSpanMetricFactory.MetricType.GRASPQP,
        solver_kwargs={
            "friction": args.friction,
            "max_limit": args.max_lambda_limit,
            "n_cone_vecs": args.n_friction_cone,
        },
    )
elif args.energy_type == "handle":
    energy_fnc = GraspSpanMetricFactory.create(GraspSpanMetricFactory.MetricType.HANDLE)
else:
    raise NotImplementedError("Energy type not implemented")


weight_dict = {
    "E_dis": args.w_dis,
    "E_fc": args.w_fc,
    "E_pen": args.w_pen,
    "E_spen": args.w_spen,
    "E_joints": args.w_joints,
    "E_prior": args.w_prior,
    "E_wall": args.w_wall,
}

energy_names = [e for e in weight_dict.keys() if weight_dict[e] > 0.0]

energy_kwargs = {}
if args.use_gendexgrasp:
    energy_kwargs["method"] = "gendexgrasp"
energy_kwargs["svd_gain"] = args.w_svd


losses = calculate_energy(
    hand_model,
    object_model,
    energy_names=energy_names,
    energy_fnc=energy_fnc,
    **energy_kwargs,
)

energy = 0
for loss_name, loss_value in losses.items():
    if loss_name not in weight_dict:
        raise ValueError(f"Loss name {loss_name} not in weight_dict")
    energy += weight_dict[loss_name] * loss_value

energy.sum().backward()
optimizer.zero_grad()


for step in tqdm(range(1, args.n_iter + 1), desc="optimizing"):
    s = optimizer.try_step()
    reset_mask = None

    E_fc_batch = energy.view(-1, args.batch_size)
    mean = E_fc_batch.mean(-1)
    std = E_fc_batch.std(-1)
    z_score = ((E_fc_batch - mean.unsqueeze(-1)) / std.unsqueeze(-1)).view(-1)

    if args.reset_epochs is not None and step % args.reset_epochs == 0 and (step < args.n_iter - 2 * args.reset_epochs):
        reset_mask = z_score > args.z_score_threshold

        # log distribution to wandb
        if reset_mask.sum() > 0:
            wandb.log(
                {"optimizer/reset": reset_mask.sum() / reset_mask.shape[0]},
                step=step,
                commit=False,
            )

            print("Resetting", reset_mask.sum(), "envs")

            initialize_convex_hull(hand_model, object_model, args, env_mask=reset_mask)
            optimizer.reset_envs(reset_mask)

    optimizer.zero_grad()

    new_energies = calculate_energy(
        hand_model,
        object_model,
        energy_fnc=energy_fnc,
        energy_names=energy_names,
        **energy_kwargs,
    )

    new_energy = 0
    for loss_name, loss_value in new_energies.items():
        if loss_name not in weight_dict:
            raise ValueError(f"Loss name {loss_name} not in weight_dict")
        new_energy += weight_dict[loss_name] * loss_value

    new_energy.sum().backward()

    if args.show_initialization:
        show_initialization(object_model, hand_model, args.batch_size)

    with torch.no_grad():
        accept, t = optimizer.accept_step(
            energy,
            new_energy,
            reset_mask,
            z_score,
            args.z_score_threshold,
        )

        energy[accept] = new_energy[accept]
        for loss_name, loss_value in new_energies.items():
            if loss_name not in weight_dict:
                raise ValueError(f"Loss name {loss_name} not in weight_dict")
            losses[loss_name][accept] = loss_value[accept]

        wandb.log({"optimizer/step": s}, step=step, commit=False)

        if args.log_entropy:
            joints_entropy = hand_model.joint_entropy()
            translation_entropy, rotation_entropy = hand_model.pose_entropy()
            score = -energy + joints_entropy
            data = {
                "entropy/joints_entropy": joints_entropy.mean(),
                "entropy/translation_entropy": translation_entropy.mean(),
                "entropy/rotation_entropy": rotation_entropy.mean(),
                "entropy/total": 0.5 * joints_entropy.mean()
                + 0.5 * (translation_entropy.mean() + rotation_entropy.mean()),
                "stats/score": score.mean(),
            }
            wandb.log(data, step=step, commit=False)

        data = {}
        for entry in losses:
            data[f"energy/{entry}"] = losses[entry].mean()
            data[f"energy_weight/{entry}"] = weight_dict[entry] * losses[entry].mean()
        wandb.log(
            {
                "energy/mean": energy.mean().item(),
                **data,
            },
            step=step,
        )

        if args.debug and step % 50 == 1:
            show_initialization(object_model, hand_model, args.batch_size, len(args.object_code_list))
            test = input("Continue? (y/n)")
            if test == "n":
                exit()

        if args.log_to_wandb and not args.no_plotly:

            # log best env
            if step % 100 == 1 or step == args.n_iter - 1:
                for batch_idx in range(len(args.object_code_list)):

                    selected_e = energy[batch_idx * args.batch_size : (batch_idx + 1) * args.batch_size]
                    n_plot = min(args.batch_size, 5)
                    env_idxs = torch.topk(-selected_e, n_plot).indices
                    # sort env idxs
                    env_idxs = torch.arange(n_plot, device=device)
                    for i, env_idx in enumerate(env_idxs):
                        fig = get_plotly_fig(
                            object_model,
                            hand_model,
                            env_idx + batch_idx * args.batch_size,
                        )
                        # asset
                        asset_name = args.object_code_list[batch_idx]
                        # Create a table
                        wandb.log(
                            {f"vis_{asset_name}/mesh_best_{i}": wandb.Plotly(fig)},
                            step=step,
                        )

    if step % 500 == 0:
        export_poses(hand_model, energy, object_model=object_model, suffix=f"_step_{step}")

export_poses(hand_model, energy, object_model=object_model, suffix="")
