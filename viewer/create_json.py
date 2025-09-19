import os
import json

objects = os.listdir("_vis/objects")
energy = "12_contacts/span_overall_cone_sqp_default_longer_gendex"
data = {}

for obj in objects:
    hands = os.listdir(f"_vis/objects/{obj}")
    for hand in ["shadow_hand", "ability_hand", "robotiq3", "robotiq2", "allegro"]:
        entry = data.get(hand, {})
        for grasp_type in os.listdir(f"_vis/objects/{obj}/{hand}"):
            file_entry = entry.get(grasp_type, {})
            if os.path.exists(f"_vis/objects/{obj}/{hand}/{grasp_type}/{energy}/render_draco.glb"):
                file_entry[obj] = f"_vis/objects/{obj}/{hand}/{grasp_type}/{energy}/render_draco.glb"
                entry[grasp_type] = file_entry
            else:
                print("Missing", f"_vis/objects/{obj}/{hand}/{grasp_type}/{energy}/render_draco.glb")
        data[hand] = entry

json.dump(data, open("models.json", "w"), indent=4)
