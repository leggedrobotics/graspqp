from __future__ import annotations

import argparse
import glob
import os
import subprocess

# Command line argument parser setup
parser = argparse.ArgumentParser(description="Chunk and process assets for grasp evaluation in IsaacLab")

# Required arguments
parser.add_argument("python_file", type=str, help="Path to the Python file to execute for each chunk")

# Data path and processing arguments
parser.add_argument("--data_path", type=str, default="/data/handles/eval", help="Root folder path containing asset directories")
parser.add_argument("--n_grasps_per_obj", type=int, default=32, help="Number of grasps to evaluate per object")
parser.add_argument("--max_envs", type=int, default=4096, help="Maximum number of simulation environments")
parser.add_argument("--prediction_folder", default="grasp_predictions", help="Name of the folder containing grasp predictions")
parser.add_argument("--n_contacts", type=int, default=12, help="Number of contact points for grasps")

# Simulation and model configuration
parser.add_argument("--headless", action="store_true", help="Run simulation in headless mode (no GUI)")
parser.add_argument("--energy_type", type=str, default="dexgrasp", help="Type of grasp energy model to use")
parser.add_argument("--hand_type", type=str, default="robotiq3", help="Type of robotic hand model")
parser.add_argument("--object_type", type=str, default="Object", help="Type of objects to grasp")
parser.add_argument(
    "--train_energy_type",
    type=str,
    default=None,
    help="Energy type used during training (if different from evaluation)",
)
parser.add_argument("--n_grasps_per_env", type=int, default=32, help="Number of grasps to evaluate per simulation environment")

# Evaluation control arguments
parser.add_argument("--force_reevaluate", action="store_true", help="Force re-evaluation even if results already exist")
parser.add_argument("--selected_step", type=int, default=-1, help="Specific training step to evaluate (-1 for latest)")
parser.add_argument("--grasp_type", type=str, default="default", help="Type/category of grasps to evaluate")

# Parse command line arguments (known and unknown for pass-through)
args_cli, unknownargs = parser.parse_known_args()

# Detect if running in Docker environment
IS_DOCKER = "LOCAL_MODE" not in os.environ


def resolve_assets(assets: list[str]) -> list[str]:
    """
    Resolve grasp prediction files for given assets.

    Args:
        assets: List of asset names to find grasp files for

    Returns:
        List of paths to grasp prediction files

    Raises:
        FileNotFoundError: If asset directory or grasp folder is not found
    """
    prediction_files = []

    for asset in assets:
        # Build path to asset directory
        asset_dir = os.path.join(args_cli.data_path, asset)
        if not os.path.exists(asset_dir):
            raise FileNotFoundError(f"\033[91mAsset directory '{asset_dir}' not found.\033[0m")

        # Build path to grasp predictions folder
        grasp_folder = os.path.join(asset_dir, args_cli.prediction_folder)
        if not os.path.exists(grasp_folder):
            raise FileNotFoundError(f"\033[91mGrasp folder '{grasp_folder}' not found.\033[0m")
        # Search for grasp prediction files matching the specified criteria
        grasp_folders = glob.glob(
            os.path.join(
                grasp_folder,
                args_cli.hand_type,
                f"{args_cli.n_contacts}_contacts",
                args_cli.energy_type,
                args_cli.grasp_type,
                "*.pt",
            )
        )

        # Organize grasp files by training step
        folder_by_step = {}
        for folder in grasp_folders:
            try:
                # Extract step number from filename (e.g., "grasp_model_1000.dexgrasp.pt")
                name = folder.split(".dexgrasp.pt")[0]
                step = int(name.split("_")[-1])
            except ValueError:
                # If no step number found, use -1 as default
                step = -1
            folder_by_step[step] = folder

        # Validate that grasp files were found
        if len(grasp_folders) == 0:
            search_path = os.path.join(
                grasp_folder,
                args_cli.hand_type,
                f"{args_cli.n_contacts}_contacts",
                args_cli.energy_type,
                args_cli.grasp_type,
                "*.pt",
            )
            raise FileNotFoundError(f"\033[91mNo grasp files found in '{grasp_folder}'. " f"Checked {search_path}\033[0m")

        # Select the appropriate grasp file based on selected step
        if args_cli.selected_step != -1:
            # Use specific training step if specified
            prediction_file = folder_by_step[args_cli.selected_step]
        else:
            # Use the most recently modified file
            prediction_file = sorted(grasp_folders, key=os.path.getmtime)[-1]

        prediction_files.append(prediction_file)
        print(f"    \033[96m Using grasp file: {prediction_file}\033[0m")

    return prediction_files


# =============================================================================
# MAIN SCRIPT: Asset Discovery and Processing
# =============================================================================

print("🔍 Discovering available assets...")

# Find all available assets in the data directory
assets = []
usd_files = []

for file in os.listdir(args_cli.data_path):
    file_path = os.path.join(args_cli.data_path, file)

    # First, try to find USD files in the standard location
    matches = glob.glob(file_path + "/*" + os.path.basename(file_path) + ".usd")
    if len(matches) > 1:
        matches = [matches[0]]  # Take only the first match if multiple found

    if len(matches) == 0:
        # Alternative search patterns for USD files
        matches = glob.glob(os.path.join(file_path, "coacd", f"{file}.usd")) + glob.glob(os.path.join(file_path, f"{file}.usd"))

    # Skip assets without USD files
    if len(matches) == 0:
        print(f"\033[93m⚠️  Skipping {file} - No USD assets found\033[0m")
        continue

    if len(matches) > 1:
        print(f"\033[93m⚠️  [WARNING] Multiple USD assets found for {file}. " f"Selecting the first one. {matches}\033[0m")

    # Check if evaluation results already exist (skip if not forcing re-evaluation)
    eval_file = os.path.join(
        file_path,
        args_cli.prediction_folder,
        args_cli.hand_type,
        f"{args_cli.n_contacts}_contacts",
        args_cli.energy_type,
        args_cli.grasp_type,
        "dexgrasp_eval_isaac_sim.csv",
    )

    if os.path.exists(eval_file) and not args_cli.force_reevaluate:
        print(f"\033[90m  Skipping {file} - Evaluation results already exist\033[0m")
        continue

    # Add valid asset to processing list
    assets.append(file)
    usd_files.append(matches[0])
    print(f"\033[92m✓ Added asset: {file}\033[0m")

print(f"\033[1m\033[94m Summary: Found {len(assets)} assets to process\033[0m")

# Resolve grasp prediction files for all discovered assets
print("\033[96m🔍 Looking for grasp prediction files...\033[0m")
grasps = resolve_assets(assets)
print(f"\033[92m✓ Found {len(grasps)} grasp files\033[0m")

# Validate that we have matching numbers of assets and grasp files
if len(assets) != len(grasps):
    print(f"\033[91m❌ Error: Found {len(assets)} assets but only {len(grasps)} grasp files. " f"Exiting\033[0m")
    exit(1)

# =============================================================================
# CHUNKING: Split assets into batches for parallel processing
# =============================================================================

# Calculate optimal batch size based on available simulation environments
batch_size = args_cli.max_envs // args_cli.n_grasps_per_obj
print("\033[1m\033[95m📦 Chunking Configuration:\033[0m")
print(f"   • Grasps per object: {args_cli.n_grasps_per_obj}")
print(f"   • Max environments: {args_cli.max_envs}")
print(f"   • Batch size: {batch_size} assets per chunk")

# Split assets and grasps into chunks for processing
asset_chunks = [assets[i : i + batch_size] for i in range(0, len(assets), batch_size)]
grasp_chunks = [grasps[i : i + batch_size] for i in range(0, len(grasps), batch_size)]

# =============================================================================
# PROCESSING: Execute evaluation for each chunk
# =============================================================================

print(f"\033[1m\033[92m🚀 Starting processing of {len(asset_chunks)} chunks...\033[0m")

# dont printz in color from here on
for i, (asset_chunk, grasp_chunk) in enumerate(zip(asset_chunks, grasp_chunks)):
    print(f"\n\033[1m\033[94m📦 Processing Chunk {i+1}/{len(asset_chunks)} " f"({len(asset_chunk)} assets)\033[0m")

    # Prepare asset list for subprocess
    assets_str = " ".join(asset_chunk)
    print(f"   Assets: {assets_str}")

    # Select appropriate Python interpreter based on environment
    if args_cli.python_file.endswith(".py"):
        interpreter = "/isaac-sim/python.sh" if IS_DOCKER else "python"
    else:
        interpreter = "bash"

    print(f"   Using interpreter: {interpreter}")

    # Build command arguments for subprocess
    cmd_args = [
        interpreter,
        args_cli.python_file,
        "--assets",
        assets_str,
        "--hand_type",
        args_cli.hand_type,
        "--n_contacts",
        str(args_cli.n_contacts),
        "--energy_type",
        args_cli.energy_type,
        "--data_path",
        args_cli.data_path,
        "--object_type",
        args_cli.object_type,
        "--n_grasps_per_env",
        str(args_cli.n_grasps_per_env),
        "--grasp_iteration",
        str(args_cli.selected_step),
        "--grasp_type",
        args_cli.grasp_type,
    ] + unknownargs

    # Add optional arguments if specified
    if args_cli.train_energy_type is not None:
        cmd_args.extend(["--train_energy_type", args_cli.train_energy_type])

    if args_cli.headless:
        cmd_args.append("--headless")

    # Log the command being executed
    print(f"   Command: {' '.join(cmd_args)}")

    # Execute the subprocess
    print("\033[92m⚡ Executing evaluation...\033[0m")
    try:
        result = subprocess.run(cmd_args, check=True)
        print(f"\033[92m✓ Chunk {i+1} completed successfully\033[0m")
    except subprocess.CalledProcessError as e:
        print(f"\033[91m❌ Chunk {i+1} failed with return code {e.returncode}\033[0m")
        # Continue with next chunk instead of failing completely
        continue

print("\n\033[1m\033[92m🎉 All chunks processed!\033[0m")
