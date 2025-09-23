#!/bin/bash
HAND_TYPES=("ability_hand" "allegro" "robotiq2" "robotiq3" "shadow_hand")
MAX_GRASPS=9
# GRASP_TYPES=("default" "pinch" "precision")
GRASP_TYPES=("pinch" "precision")
DATASET_PATH="/media/zrene/data/GraspGen/dexgrasp_remeshed/test"
for HAND in "${HAND_TYPES[@]}"; do
  echo "Processing hand type: $HAND"
  for GRASP_TYPE in "${GRASP_TYPES[@]}"; do
    echo "  Grasp type: $GRASP_TYPE"
    python scripts/visualize_result.py  \
    --dataset $DATASET_PATH \
    --max_grasps $MAX_GRASPS --hand_name $HAND --grasp_type $GRASP_TYPE \
    --num_contact 12_contacts --energy span_overall_cone_sqp_default_longer_gendex  --calc_energy
   done
done

# Compress with draco
find . -type f -name "*.glb" -exec sh -c '
  for file; do
    out="${file%.glb}_draco.glb"
    echo "Compressing $file → $out"
    npx gltf-pipeline -i "$file" -o "$out" -d
  done
' sh {} +

# Create json file listing all results
python list.py
