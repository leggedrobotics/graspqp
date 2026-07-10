# Scripts

Command-line entry points for GraspQP. This page is an index of everything under
`scripts/`, grouped by subfolder, with a one-line description and the most useful
flags for each. Run any Python script with `--help` for the full list of options.

**Prerequisites**

- The `graspqp` package installed (see the top-level `README.md`).
- The `vis/` scripts additionally need `plotly` / `trimesh` (pulled in by the
  standalone requirements).
- The `isaaclab/` scripts require a working Isaac Lab / Isaac Sim install and are
  launched through Isaac Lab's `AppLauncher`.

> **Note on flag naming.** The core fitter (`fit.py`) uses `--hand_name` and
> `--n_contact` (singular). The Isaac Lab scripts use `--hand_type` and
> `--n_contacts` (plural). This is intentional; match the flag to the script.

Registered hand names (`--hand_name` / `--hand_type`):
`ability_hand`, `allegro`, `panda`, `robotiq2`, `robotiq3`, `schunk2`, `shadow_hand`.

---

## Top level

### `fit.py`

Main grasp optimizer. Fits hand poses + contact points to one or more objects and
writes grasp predictions to disk.

Key flags:

- `--hand_name` — hand to use (one of the registered names above). Default `allegro`.
- `--energy_type` — grasp energy: `graspqp` (default), `dexgrasp`, `tdg`, `handle`.
- `--n_contact` — number of contact points (default `12`).
- `--data_root_path` — root folder containing object subfolders.
- `--dataset` — dataset name/subfolder (default `debug`).
- `--object_code_list` / `--object_code_file` — restrict to specific objects
  (space-separated list, or a text file of names).
- `--grasp_type` — grasp category (default `all`).
- `--batch_size`, `--n_iter`, `--seed`, `--gpu` — optimization controls.
- `--optimizer` — `mala_star` (default) or `dexgraspnet`.
- `--initialization` — `convex_hull` (default) or `random`.
- `--use_gendexgrasp` / `--no-use_gendexgrasp` — toggle GenDexGrasp initialization
  (on by default).
- `--log_to_wandb`, `--wandb_project`, `--wandb_name` — Weights & Biases logging.

```bash
python scripts/fit.py \
    --data_root_path /path/to/data --dataset debug \
    --hand_name allegro --energy_type graspqp --n_contact 12
```

### `fit_all.sh`

Batch driver around `fit.py`. Scans a dataset folder, skips objects that already
have predictions, and generates per-batch run scripts (handy for cluster / SLURM
submission).

Key flags: `-d/--data-path`, `-f/--folder-suffix`, `-h/--hand`, `-g/--grasp-type`,
`-e/--energy-method`, `-c/--n-contacts`, `-n/--n-grasps-per-obj`,
`-a/--num-assets`. Run `scripts/fit_all.sh --help` for details.

---

## `vis/` — visualization

### `vis/visualize_hand_model.py`

Plotly viewer for a single hand model: base origin, sampled contact points,
jacobian, and SDF occupancy grid. Used when bringing up a new hand.

Key flags: `--hand_name` (any registered name, or `all`), `--grasp_type`,
`--show_jacobian`, `--show_joint_axes`, `--show_penetration_points`,
`--show_occupancy_grid`, `--only_collision`, `--randomize_joints`, `--spacing`,
`--device`.

```bash
python scripts/vis/visualize_hand_model.py --hand_name allegro --show_occupancy_grid
```

### `vis/visualize_result.py`

Loads optimized grasp predictions from disk and renders them on their objects.

Key flags: `--hand_name`, `--dir` (prediction folder), `--dataset`,
`--num_contacts` (e.g. `12_contacts`), `--energy`, `--grasp_type`
(`default`/`pinch`/`precision`/`tips`), `--max_grasps`, `--calc_energy`,
`--headless`, `--show`, `--obj_path`, `--scale`, `--overwrite`, `--vis_dir`.

### `vis/color_meshes.py`

Colors object-mesh vertices by how often the hand interacts with them and exports
the colored meshes.

Key flags: `--hand_name`, `--dataset`, `--grasp_type`, `--num_contacts`,
`--energy`, `--max_grasps`, `--num_assets`, `--calc_energy`, `--headless`,
`--overwrite`.

### `vis/color_all.bash`

Convenience loop that runs `color_meshes.py` over several hands and grasp types.
Edit the arrays / dataset path at the top before running.

### `vis/blender.py`

Blender (`bpy`) script that imports exported interaction meshes and renders
screenshots. Run inside Blender; configure `HAND_TYPES`, `GRASP_TYPES`,
`ENERGY_METHODS`, and the folder paths at the top of the file (no CLI flags).

### `vis/parse_coll_spheres.py`

One-off helper that converts a hand's `penetration_points.json` collision spheres
into a config block. Edit the input path in-file before running.

---

## `isaaclab/` — Isaac Lab evaluation

These scripts require Isaac Lab / Isaac Sim and are launched through Isaac Lab's
`AppLauncher` (add `--headless` to run without a GUI). Most share a common set of
flags: `--data_path`, `--hand_type`, `--energy_type`, `--n_contacts`,
`--grasp_type`, `--object_type` (`Object`/`Handle`), `--task`, `--num_envs`,
`--n_grasps_per_env`, `--assets`, `--num_assets`, `--prediction_folder`.

### `isaaclab/show_hands.py`

Spawns every registered hand side-by-side in an Isaac Lab scene. Register new
hands by adding them to the `AVAILABLE_HANDS` dict at the top of the file. Accepts
`AppLauncher` args (e.g. `--headless`).

### `isaaclab/eval_object_grasp.py`

Batch physics-based evaluation of predicted grasps. In addition to the shared
flags: `--show`, `--min_evals`, `--step`, `--force_value`, `--log_to_wandb`,
`--wandb_project`, `--wandb_name`.

### `isaaclab/show_object_grasp.py`

Interactive playback of predicted grasps for a single object. Shared flags plus
`--show`, `--min_evals`, `--step`, `--force_value`.

### `isaaclab/chunk_assets.py`

Splits a dataset into chunks and runs a given Python file once per chunk (parallel
evaluation). Positional arg: `python_file`. Flags: `--data_path`,
`--n_grasps_per_obj`, `--max_envs`, `--prediction_folder`, `--n_contacts`,
`--headless`, `--energy_type`, `--hand_type`, `--object_type`,
`--n_grasps_per_env`, `--force_reevaluate`, `--selected_step`, `--grasp_type`.

### `isaaclab/calc_metrics.py`

Helper module that computes analytic grasp metrics from saved checkpoints. Imported
by the evaluation scripts; it has no standalone CLI.
