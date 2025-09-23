
declare -a hand_names=("allegro" "shadow_hand" "ability_hand" "robotiq2" "robotiq3")

declare -a grasp_types=("default" "pinch" "precision")
declare -a energies=("span_overall_cone_sqp_default_longer_gendex")
for hand_name in "${hand_names[@]}"; do
    for grasp_type in "${grasp_types[@]}"; do
        for energy in "${energies[@]}"; do
            echo "python vis/color_meshes.py  --dataset /data/GraspGen/dexgrasp_remeshed/test --grasp_type $grasp_type --max_grasps 32 --hand_name $hand_name --num_contact 12_contacts --energy $energy"
            python vis/color_meshes.py  --dataset /data/GraspGen/dexgrasp_remeshed/test --grasp_type $grasp_type --max_grasps 32 --hand_name $hand_name --num_contact 12_contacts --energy $energy
        done
    done
done
