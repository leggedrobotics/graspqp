# Scripts overview

- vis/visualize_hand_model.py — Plotly visualization of a hand model
- vis/visualize_result.py — Visualize predicted grasps from files
- isaaclab/eval_object_grasp.py — Batch evaluation of grasps in Isaac Lab
- isaaclab/show_object_grasp.py — Interactive playback in Isaac Lab

Common flags:

- `--hand_name` / `--hand_type` — select gripper
- `--n_contacts` — number of contacts
- `--n_grasps_per_env` — per-env grasps in simulator
- `--dataset` and `--data_root_path` — dataset selection

See `--help` on each script for full options.
