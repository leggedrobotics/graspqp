"""Parser utilities for command line argument handling and asset processing.

This module provides utilities for configuring argument parsers with default
parameters and processing command line arguments for grasp simulation tasks.
It handles asset discovery, prediction file matching, and USD file resolution.
"""

import glob
import os
import random
import re


def add_default_args(parser):
    """Add default command line arguments for grasp simulation tasks.

    Configures parser with standard arguments for environment setup,
    asset loading, grasp parameters, and simulation options.

    Args:
        parser: ArgumentParser instance to configure

    Returns:
        ArgumentParser: The configured parser with added arguments
    """
    parser.add_argument(
        "--disable_fabric",
        action="store_true",
        default=False,
        help="Disable Fabric acceleration and use standard USD I/O operations for compatibility.",
    )
    parser.add_argument(
        "--num_envs", type=int, default=32, help="Number of parallel simulation environments to run simultaneously."
    )
    parser.add_argument(
        "--object_type",
        type=str,
        default="Object",
        help="Type of manipulation target: 'Object' for general objects.",
        choices=["Object"],
    )
    parser.add_argument(
        "--task",
        type=str,
        default="Isaac-%OBJ_TYPE%-Grasp-Mining-%HANDTYPE%-v0",
        help="Isaac Lab task name template. %OBJ_TYPE% and %HANDTYPE% will be automatically replaced.",
    )

    parser.add_argument(
        "--n_grasps_per_env",
        type=int,
        default=4,
        help="Number of grasp proposals to test for each object.",
    )
    parser.add_argument(
        "--assets",
        default="",
        nargs="+",
        help="List of asset names to load. If empty, all assets from data_path will be used.",
    )
    parser.add_argument(
        "--num_assets",
        type=int,
        default=-1,
        help="Maximum number of assets to load. -1 means load all available assets.",
    )

    parser.add_argument(
        "--prediction_files",
        nargs="+",
        default=None,
        help="Pre-computed grasp prediction files (.pt format) to use instead of auto-discovery.",
    )
    parser.add_argument(
        "--usd_files", nargs="+", default=None, help="USD geometry files for assets. Auto-discovered if not specified."
    )

    parser.add_argument(
        "--prediction_folder",
        default="grasp_predictions",
        help="Subdirectory name within asset folders containing grasp prediction files.",
    )

    parser.add_argument(
        "--data_path",
        type=str,
        default="/data/dexgrasp_remeshed/test",
        help="Root directory path containing asset folders and grasp data.",
    )
    parser.add_argument(
        "--energy_type",
        type=str,
        default="span_overall_cone_sqp_default_longer_gendex",
        help="Energy function type for grasp quality evaluation and ranking.",
    )
    parser.add_argument(
        "--static_show",
        action="store_true",
        help="Use static grasp agent for visualization instead of dynamic execution.",
    )

    parser.add_argument(
        "--grasp_iteration",
        type=int,
        default=-1,
        help="Specific grasp training iteration to load. -1 uses the most recent iteration.",
    )

    parser.add_argument(
        "--hand_type",
        type=str,
        default="robotiq3",
        help="Type of robotic hand model (e.g., ability_hand, allegro, shadow_hand).",
    )
    parser.add_argument(
        "--n_contacts",
        type=int,
        default=12,
        help="Number of contact points for grasp. Auto-detected if not specified.",
    )

    parser.add_argument(
        "--train_energy_type",
        default=None,
        help="Energy function type used during training phase (overrides energy_type).",
    )

    parser.add_argument(
        "--grasp_type", default="default", help="Grasp category or type identifier for filtering specific grasp sets."
    )
    parser.add_argument("--file_pattern", default=None, help="Regex pattern for matching specific prediction files by name.")

    return parser


def parse_args(parser):
    """Parse command line arguments and resolve asset files.

    Processes arguments to discover assets, match prediction files,
    and resolve USD file paths. Handles automatic asset discovery
    from data directories and validates file existence.

    Args:
        parser: Configured ArgumentParser instance

    Returns:
        Namespace: Processed arguments with resolved file paths and
                  validated asset configurations

    Raises:
        FileNotFoundError: If required asset directories or files are missing
    """
    args_cli = parser.parse_args()
    args_cli.task = args_cli.task.replace("%HANDTYPE%", args_cli.hand_type)
    args_cli.task = args_cli.task.replace("%OBJ_TYPE%", args_cli.object_type)

    if args_cli.assets is None or len(args_cli.assets) == 0 or args_cli.assets[0] == "":
        print("No assets provided. Loading all assets stored in the data path.")

        args_cli.assets = []
        for file in os.listdir(args_cli.data_path):
            if "captures" in file:
                continue

            args_cli.assets.append(file)
        args_cli.assets = sorted(args_cli.assets)
        random.seed(42)
        random.shuffle(args_cli.assets)

    if isinstance(args_cli.assets, str):  # Convert to list if string argument is given
        args_cli.assets = [args_cli.assets.split(",")]

    # weird case when called from a shell
    if len(args_cli.assets) == 1 and " " in args_cli.assets[0]:
        args_cli.assets = args_cli.assets[0].split(" ")

    assets = [asset.strip() for asset in args_cli.assets]
    print("Checking assets: ", assets)

    if args_cli.prediction_files is None:
        args_cli.prediction_files = []
        args_cli.assets = []
        for asset in assets:
            asset_dir = os.path.join(args_cli.data_path, asset)
            if not os.path.exists(asset_dir):
                raise FileNotFoundError(f"Asset directory '{asset_dir}' not found.")

            grasp_folder = os.path.join(asset_dir, args_cli.prediction_folder)
            if not os.path.exists(grasp_folder):
                # print warning in yellow
                print("\033[93m" + f"Grasp folder '{grasp_folder}' not found." + "\033[0m")
                continue

            # find matching dexgrasp file
            grasp_folder = os.path.join(grasp_folder, args_cli.hand_type)
            if not os.path.exists(grasp_folder):
                print("\033[93m" + f"Grasp folder '{grasp_folder}' not found." + "\033[0m")
                continue

            if args_cli.n_contacts is not None:
                grasp_folder = os.path.join(grasp_folder, f"{args_cli.n_contacts}_contacts")
            else:
                grasp_folder = os.path.join(grasp_folder, "*_contacts")

            grasp_folders = glob.glob(os.path.join(grasp_folder, args_cli.energy_type, args_cli.grasp_type, "*.pt"))

            if args_cli.file_pattern is not None:
                folders = []
                pattern = re.compile(args_cli.file_pattern)
                print(f"Pattern: {pattern}")
                for folder in grasp_folders:
                    name = os.path.basename(folder)
                    print("Checking: ", name)
                    if pattern.match(name):
                        folders.append(folder)
                        print(f"Matched: {name}")
                grasp_folders = folders
            else:

                # Filter grasp folders
                grasp_folders = [folder for folder in grasp_folders if ".dexgrasp.pt" in folder]

            # find newest file
            if len(grasp_folders) == 0:
                print(
                    "\033[93m"
                    + f"No grasp files found in '{grasp_folder}' using energy type '{args_cli.energy_type}'"
                    + "\033[0m"
                )
                continue

            prediction_file = sorted(grasp_folders, key=os.path.getmtime)[-1]
            if args_cli.grasp_iteration != -1:
                # matching folder
                matched_folders = [folder for folder in grasp_folders if f"step_{args_cli.grasp_iteration}." in folder]
                if len(matched_folders) == 0:
                    raise FileNotFoundError(
                        f"No grasp files found for grasp iteration {args_cli.grasp_iteration} in '{grasp_folder}'. \n"
                        f"Available folders: {[os.path.basename(folder) for folder in grasp_folders]}"
                    )
                prediction_file = matched_folders[0]

            args_cli.prediction_files.append(prediction_file)
            args_cli.assets.append(asset)

        if args_cli.num_assets is not None and args_cli.num_assets != -1:
            args_cli.assets = args_cli.assets[: args_cli.num_assets]
            args_cli.prediction_files = args_cli.prediction_files[: args_cli.num_assets]

    else:
        args_cli = args_cli.assets

    if isinstance(args_cli.prediction_files, str):  # Convert to list if string argument is given
        args_cli.prediction_files = [args_cli.prediction_files]
    # weird case when called from a shell
    if len(args_cli.prediction_files) == 1 and " " in args_cli.prediction_files[0]:
        args_cli.prediction_files = args_cli.prediction_files[0].split(" ")

    args_cli.prediction_files = [asset.strip() for asset in args_cli.prediction_files]

    if args_cli.usd_files is None:
        args_cli.usd_files = []
        # Load usd paths
        for asset in args_cli.assets:
            asset_dir = os.path.join(args_cli.data_path, asset)
            if not os.path.exists(asset_dir):
                raise FileNotFoundError(f"Asset directory '{asset_dir}' not found.")

            usd_file = glob.glob(asset_dir + "/*" + os.path.basename(asset_dir) + ".usd")
            if len(usd_file) > 0:
                usd_file = usd_file[0]
            else:
                usd_file = None

            if usd_file is None:
                # find matching usd file.
                # Dexgrasp
                usd_file = os.path.join(asset_dir, "coacd", f"{asset}.usd")
            if not os.path.exists(usd_file):
                # direct access
                usd_file = os.path.join(asset_dir, f"{asset}.usd")
                if not os.path.exists(usd_file):
                    raise FileNotFoundError(f"No USD file found in '{asset_dir}' at 'coacd/{asset}.usd' nor '{asset}.usd'")
            args_cli.usd_files.append(usd_file)
    else:
        if isinstance(args_cli.usd_files, str):
            args_cli.usd_files = [args_cli.usd_files]
    args_cli.usd_files = [asset.strip() for asset in args_cli.usd_files]

    if args_cli.num_envs < args_cli.n_grasps_per_env * len(args_cli.assets):
        print("[WARNING] Number of environments is less than the number of grasps per environment!")
        print(
            f"[WARNING] Setting number of environments to number of grasps per environment. ({args_cli.n_grasps_per_env * len(args_cli.assets)})"
        )

    args_cli.num_envs = args_cli.n_grasps_per_env * len(args_cli.assets)

    return args_cli
