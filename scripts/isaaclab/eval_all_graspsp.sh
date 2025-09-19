#!/bin/bash
# Arguments are "run directory" "profile" others
run_dir=$1
profile=$2
asset_type=$3
arg=$4
mined_flag=""

object_type=Object
if [ "$asset_type" == "handles" ]; then
    object_type=Handle
fi
RERUN=0
# check LOCAL_MODE is set
if [ -z "$LOCAL_MODE" ]; then
    echo "[ERROR] LOCAL_MODE is not set. Exiting."
    # folder=/cluster/scratch/zrene/data/GraspGen/dexgrasp/test
    if [ "$asset_type" == "handles" ]; then
        folder=/cluster/scratch/zrene/data/GraspGen/handles/val
        local_folder=/data/GraspGen/handles/val
    else
        folder=/cluster/scratch/zrene/data/GraspGen/dexgrasp_remeshed/test
        local_folder=/data/GraspGen/dexgrasp_remeshed/test
    fi
    module load eth_proxy

else
    if [ "$asset_type" == "handles" ]; then
        folder=/data/GraspGen/handles/val
        local_folder=/data/GraspGen/handles/val
    else
        folder=/data/GraspGen/dexgrasp_remeshed/test
        local_folder=/data/GraspGen/dexgrasp_remeshed/test
    fi
    run_dir=/cluster/home/zrene/isaaclab_static
    profile="isaac-lab-base"
fi

if [ "$arg" == "train" ]; then
    echo "Training grasps"

    # iterate over energy_types 
    declare -a energy_types=("dexgrasp" "span_overall_cone_sqp" "tdg")
    
    for eng in "${energy_types[@]}"; do
        echo "running: bash docker/cluster/submit_job_slurm.sh $asset_type-$eng" $eng
        docker/cluster/submit_training_slurm.sh $run_dir $profile  "$asset_type-$eng" $eng 46
        # sleep for 1s
        sleep 2
    done

    # check if RERUN is set
    exit 0
fi

# check if arg is refine
if [ "$arg" == "refine" ]; then
    echo "Refining grasps"
elif [ "$arg" == "eval" ]; then
    echo "Evaluating grasps"
elif [ "$arg" == "eval_mined" ]; then
    echo "Evaluating mined grasps"
    mined_flag="--mined"
else
    echo "Invalid argument. Use 'refine' or 'eval'."
    exit 1
fi

n_grasps_per_obj=32
# drwxr-xr-x 3 zrene zrene-group 4096 Apr 22 10:00 dexgrasp_default_longer
# drwxr-xr-x 3 zrene zrene-group 4096 Apr 22 10:00 dexgrasp_baseline_default_longer

# declare -a hand_names=("shadow_hand" "allegro") #"panda" "allegro" "ability_hand")  # "ability_hand")
declare -a hand_names=("shadow_hand" "ability_hand" "shadow_hand" "allegro" "robotiq2" "robotiq3")  # "ability_hand")
declare -a grasp_types=("pinch" "precision") # "pinch" "precision") # "precision" "pinch")                  # "precision" "pinch")  # "pinch" "precision" "full")
declare -a energy_methods=("span_overall_cone_sqp") # "dexgrasp_default_longer") # "span_overall_cone_sqp")
declare -a ablations=("default_longer_gendex") #"density" "more_density" "no_temp")
# declare -a ablations=("") #"density" "more_density" "no_temp")

declare -a n_contacts=(12)
# cleaned_assets/ddg-kit_SoftCheese/grasp_predictions/robotiq2/12_contacts/span_overall_cone_sqp_default_longer_gendex/default/

# declare -a hand_names=("robotiq3" "ability_hand" "robotiq2" "allegro" "shadow_hand")
# # declare -a energy_methods=("dexgrasp_baseline" "dexgrasp" "span_overall_cone_theseus")
# declare -a energy_methods=("span_overall_cone_sqp") # "span_overall_cone_theseus")
# declare -a n_contacts=(12)
# declare -a grasp_types=("all" "precision" "pinch") # "pinch")
# declare -a ablations=("default_longer_manip")

for hand_name in "${hand_names[@]}"; do
    for energy_method in "${energy_methods[@]}"; do
        for n_contact in "${n_contacts[@]}"; do

            for grasp_type in "${grasp_types[@]}"; do
                for abl in "${ablations[@]}"; do
                    # check that the folder exists
                    folder_pattern=$folder/*/grasp_predictions/$hand_name/"$n_contact"_contacts/"$energy_method"_"$abl"/$grasp_type

                    # check folder is val, Ignore error stream
                    glob_array=($folder_pattern)
                    n_files=${#glob_array[@]}
                    # n_files=$(ls -1q $folder_pattern | wc -l)
                    if [ $n_files -le 1 ]; then
                        # print this message in light gray
                        echo -e "\e[90mNo files found in folder: $folder_pattern. Skipping\e[0m"
                        continue
                    fi
                    # check if eval file is already present
                    if [ "$arg" == "refine" ]; then
                        eval_file=$folder_pattern/refine_isaac_sim*.csv
                    else
                        eval_file=$folder_pattern/eval_isaac_sim*.csv
                    fi

                    if compgen -G "$eval_file" >/dev/null; then
                        # convert eval_file glob into array
                        glob_array=($eval_file)
                        num_eval_files=${#glob_array[@]}
                        if [ $num_eval_files -eq $n_files ]; then
                            if [ $RERUN -eq 0 ]; then
                                echo -e "\e[34mRun already Checked! Found eval file: $eval_file. \e[0m"
                                continue
                            fi
                        fi
                        echo -e "\e[34mMissmatch in number of eval files found: $num_eval_files != $n_files\e[0m"
                    fi

                    # print this in green
                    echo -e "\e[32mFound  Running Evaluation for: $folder_pattern\e[0m"
                    echo -e "\e[32mFound Hand: $hand_name, Energy: $energy_method, Contacts: $n_contact, Grasp Type: $grasp_type\e[0m"
                    # check local mode

                    if [ "$energy_method" == "span_overall_cone_sqp" ]; then
                        train_energy_type=span_overall_cone_qp
                    else
                        train_energy_type=dexgrasp
                    fi

                    # check if refine flag is set
                    if [ "$arg" == "refine" ]; then
                        echo "Refining grasps"
                        if [ -z "$LOCAL_MODE" ]; then

                            bash docker/cluster/submit_job_slurm.sh $run_dir $profile source/custom/GraspMiningIsaacLab/scripts/mine_object_grasp.py \
                                --n_grasps_per_env $n_grasps_per_obj --hand_type $hand_name --object_type $object_type --data_path $local_folder \
                                --energy_type "$energy_method"_"$abl" --n_contacts $n_contact --train_energy_type $train_energy_type --headless \
                                --wandb_name "$energy_method"_"$abl"_"$n_contact"_"$grasp_type"_"$hand_name"_co_02_mr_075_full --wandb_project object_mine --grasp_type $grasp_type \
                                --crossover_probability 0.2 --mutation_rate 0.5

                        else

                            bash docker/cluster/submit_job_local.sh $run_dir $profile source/custom/GraspMiningIsaacLab/scripts/mine_object_grasp.py \
                                --n_grasps_per_env $n_grasps_per_obj --hand_type $hand_name --object_type $object_type --data_path $local_folder \
                                --energy_type $energy_method --n_contacts $n_contact --train_energy_type $train_energy_type --headless \
                                --wandb_name "$energy_method"_"$n_contact"_"$grasp_type"_"$hand_name"_co_02_mr_075_full --wandb_project object_mine --grasp_type $grasp_type \
                                --crossover_probability 0.0 --mutation_rate 0.5
                        fi

                    else
                        echo "Evaluating grasps"
                        if [ -z "$LOCAL_MODE" ]; then

                            bash docker/cluster/submit_job_slurm.sh $run_dir $profile source/custom/GraspMiningIsaacLab/scripts/eval_object_grasp.sh \
                                --n_grasps_per_env $n_grasps_per_obj --hand_type $hand_name --object_type $object_type --data_path $local_folder \
                                --energy_type "$energy_method"_"$abl" --n_contacts $n_contact --headless \
                                --wandb_name "$energy_method"_"$abl"_"$n_contact"_"$grasp_type" --wandb_project handles_eval --grasp_type $grasp_type $mined_flag # --log_to_wandb
                        else
                            bash docker/cluster/submit_job_local.sh $run_dir $profile source/custom/GraspMiningIsaacLab/scripts/eval_object_grasp.sh \
                                --n_grasps_per_env $n_grasps_per_obj --hand_type $hand_name --object_type $object_type --data_path $local_folder --energy_type "$energy_method"_"$abl" --n_contacts $n_contact --headless \
                                --wandb_name "$energy_method"_"$abl"_"$n_contact"_"$grasp_type" --wandb_project handles_eval --grasp_type $grasp_type $mined_flag # --log_to_wandb
                        fi
                    fi

                done
            done
        done
    done

done
