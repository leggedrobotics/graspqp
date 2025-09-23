#!/bin/bash

# Default values
DEFAULT_DATA_PATH="/data"
DEFAULT_FOLDER_SUFFIX="release/debug"
DEFAULT_HAND="allegro"
DEFAULT_GRASP_TYPE="default"
DEFAULT_ENERGY_METHOD="graspqp"
DEFAULT_N_CONTACTS=12
DEFAULT_N_GRASPS_PER_OBJ=32
DEFAULT_NUM_ASSETS=64

# Function to display help
show_help() {
    cat << EOF
Usage: $0 [OPTIONS]

Process grasp fitting for multiple assets with configurable parameters.

OPTIONS:
    -d, --data-path PATH        Base data path (default: $DEFAULT_DATA_PATH)
    -f, --folder-suffix PATH    Folder suffix to append to data path (default: $DEFAULT_FOLDER_SUFFIX)
    -h, --hand NAME             Hand type (default: $DEFAULT_HAND)
    -g, --grasp-type TYPE       Grasp type (default: $DEFAULT_GRASP_TYPE)
    -e, --energy-method METHOD  Energy method (default: $DEFAULT_ENERGY_METHOD)
    -c, --n-contacts NUM        Number of contacts (default: $DEFAULT_N_CONTACTS)
    -n, --n-grasps-per-obj NUM  Number of grasps per object (default: $DEFAULT_N_GRASPS_PER_OBJ)
    -a, --num-assets NUM        Number of assets to process per batch (default: $DEFAULT_NUM_ASSETS)
    --help                      Show this help message and exit

EXAMPLES:
    $0                                          # Use all default values
    $0 --hand robotiq3 --num-assets 5         # Use robotiq3 hand with 5 assets per batch
    $0 -d /custom/data -e dexgrasp             # Custom data path and energy method
EOF
}

# Initialize variables with defaults
data_path="$DEFAULT_DATA_PATH"
folder_suffix="$DEFAULT_FOLDER_SUFFIX"
HAND="$DEFAULT_HAND"
GRASP_TYPE="$DEFAULT_GRASP_TYPE"
ENERGY_METHOD="$DEFAULT_ENERGY_METHOD"
N_CONTACTS="$DEFAULT_N_CONTACTS"
n_grasps_per_obj="$DEFAULT_N_GRASPS_PER_OBJ"
num_assets="$DEFAULT_NUM_ASSETS"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--data-path)
            data_path="$2"
            shift 2
            ;;
        -f|--folder-suffix)
            folder_suffix="$2"
            shift 2
            ;;
        -h|--hand)
            HAND="$2"
            shift 2
            ;;
        -g|--grasp-type)
            GRASP_TYPE="$2"
            shift 2
            ;;
        -e|--energy-method)
            ENERGY_METHOD="$2"
            shift 2
            ;;
        -c|--n-contacts)
            N_CONTACTS="$2"
            shift 2
            ;;
        -n|--n-grasps-per-obj)
            n_grasps_per_obj="$2"
            shift 2
            ;;
        -a|--num-assets)
            num_assets="$2"
            shift 2
            ;;
        --help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option $1"
            show_help
            exit 1
            ;;
    esac
done

# Check if running on cluster (env var set)
if [ -z "$PYTHON_EULER_ROOT" ]; then
    echo "Running locally"
    CLUSTER=0
else
    echo "Running on cluster"
    data_path="/cluster/scratch/zrene/data"
    CLUSTER=1
fi

# Construct full folder path
folder="$data_path/$folder_suffix"

# Display configuration
echo "Configuration:"
echo "  Data path: $data_path"
echo "  Folder: $folder"
echo "  Hand: $HAND"
echo "  Grasp type: $GRASP_TYPE"
echo "  Energy method: $ENERGY_METHOD"
echo "  Number of contacts: $N_CONTACTS"
echo "  Grasps per object: $n_grasps_per_obj"
echo "  Assets per batch: $num_assets"
echo ""

# Validate inputs
if [[ ! -d "$folder" ]]; then
    echo "Error: Folder '$folder' does not exist!"
    exit 1
fi

if [[ ! "$N_CONTACTS" =~ ^[0-9]+$ ]] || [[ "$N_CONTACTS" -le 0 ]]; then
    echo "Error: Number of contacts must be a positive integer!"
    exit 1
fi

if [[ ! "$n_grasps_per_obj" =~ ^[0-9]+$ ]] || [[ "$n_grasps_per_obj" -le 0 ]]; then
    echo "Error: Number of grasps per object must be a positive integer!"
    exit 1
fi

if [[ ! "$num_assets" =~ ^[0-9]+$ ]] || [[ "$num_assets" -le 0 ]]; then
    echo "Error: Number of assets must be a positive integer!"
    exit 1
fi

# Create folder for executable files if not existing
bin_folder="bin/${HAND}/${GRASP_TYPE}/${ENERGY_METHOD}/${N_CONTACTS}"
mkdir -p "$bin_folder"
echo "Created bin folder: $bin_folder"

# Process assets in batches
curr_asset_cnt=0
total_asset_cnt=0
skipped_cnt=0
asset_str=""

echo "Scanning for assets to process..."

for file in "$folder"/*; do
    if [[ ! -d "$file" ]]; then
        continue  # Skip non-directories
    fi

    filename=$(basename -- "$file")
    # Construct path to check for existing grasp prediction file
    CHECK_FILE_PATH="$file/grasp_predictions/$HAND/${N_CONTACTS}_contacts/$ENERGY_METHOD/$GRASP_TYPE/${filename}_step_7000.dexgrasp.pt"

    # Check if file exists
    if [[ -f "$CHECK_FILE_PATH" ]]; then
        echo "  Skipping $filename - grasp prediction already exists"
        skipped_cnt=$((skipped_cnt + 1))
        continue
    fi

    echo "  Adding $filename to processing queue"
    curr_asset_cnt=$((curr_asset_cnt + 1))

    # Add to asset string
    if [[ -z "$asset_str" ]]; then
        asset_str="$filename"
    else
        asset_str="$asset_str $filename"
    fi

    # Check if we have enough assets for a batch
    if [[ $curr_asset_cnt -eq $num_assets ]]; then
        echo ""
        echo "Creating batch with assets: $asset_str"
        dt=$(date '+%H_%M_%S')
        total_asset_cnt=$((total_asset_cnt + curr_asset_cnt))

        # Create a file in bin/ with the asset names
        name="assets_${HAND}_${GRASP_TYPE}_${dt}_${total_asset_cnt}"
        asset_file="$bin_folder/$name.txt"
        run_script="$bin_folder/run_$name.sh"

        echo "  Creating asset file: $asset_file"
        echo "$asset_str" > "$asset_file"

        # Create corresponding executable file
        echo "  Creating run script: $run_script"
        cat > "$run_script" << EOF
#!/bin/bash
python scripts/fit.py \\
    --object_code_file "$asset_file" \\
    --grasp_type "$GRASP_TYPE" \\
    --hand_name "$HAND" \\
    --energy_type "$ENERGY_METHOD" \\
    --n_contact "$N_CONTACTS" \\
    --data_root_path "$(dirname "$folder")" \\
    --dataset "$(basename "$folder")" \\
EOF
        chmod +x "$run_script"

        # Reset counters for next batch
        curr_asset_cnt=0
        asset_str=""
        echo ""
    fi
done

# Handle remaining assets if any
if [[ $curr_asset_cnt -gt 0 ]]; then
    echo ""
    echo "Creating final batch with remaining assets: $asset_str"
    dt=$(date '+%H_%M_%S')
    total_asset_cnt=$((total_asset_cnt + curr_asset_cnt))

    name="assets_${HAND}_${GRASP_TYPE}_${dt}_${total_asset_cnt}"
    asset_file="$bin_folder/$name.txt"
    run_script="$bin_folder/run_$name.sh"

    echo "  Creating asset file: $asset_file"
    echo "$asset_str" > "$asset_file"

    echo "  Creating run script: $run_script"
    cat > "$run_script" << EOF
#!/bin/bash
python scripts/fit.py \\
    --object_code_file "$asset_file" \\
    --grasp_type "$GRASP_TYPE" \\
    --hand_name "$HAND" \\
    --energy_type "$ENERGY_METHOD" \\
    --n_contact "$N_CONTACTS" \\
    --data_root_path "$(dirname "$folder")" \\
    --dataset "$(basename "$folder")"
EOF
    chmod +x "$run_script"
fi

echo ""
echo "Summary:"
echo "  Total assets processed: $total_asset_cnt"
echo "  Assets skipped (already processed): $skipped_cnt"
echo "  Batch files created in: $bin_folder"
echo ""
echo "To run the generated scripts:"
echo "  find $bin_folder -name 'run_*.sh' -exec {} \\;"
echo "To run all scripts on the cluster (if applicable):"
echo "  for script in $bin_folder/run_*.sh; do sbatch --time=23:59:00 --gpus=rtx_4090:1 --mem-per-cpu=24 --wrap=\"bash \$script\"; done"
echo ""
