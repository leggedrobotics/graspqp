# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Batch-remesh collision meshes in Blender for GraspQP (adding_hand.md, step 6).

The GraspQP SDF occupancy check needs *watertight, manifold* collision meshes. Raw
collision meshes are often non-manifold, self-intersecting, or needlessly high-poly
(some collision meshes are several MB each), which makes the SDF report gaps/floating points.

This script reproduces the documented Blender fix for every matching mesh:

  1. Voxel remesh   -> rebuilds the surface as a single watertight manifold shell
  2. Smooth modifier-> removes the blocky voxel stair-stepping
  3. (opt) Decimate -> collapses the result to a sane triangle budget
  4. Triangulate + recalc normals outward, then export

Run it headless (Blender is NOT imported by the rest of the repo):

  blender --background --python scripts/blender/remesh_collisions.py -- \
      --input-dir graspqp/assets/shadow_hand/meshes \
      --pattern "*collisions*.obj" \
      --voxel-rel 0.025 --smooth-iters 8 --decimate 0.5 --in-place

By default it writes the cleaned meshes to <input-dir>/remeshed/ (originals untouched).
Pass --in-place to overwrite the originals so the existing URDF <collision> tags
(which reference meshes/..._collisions_..._mesh.obj) pick them up with no URDF edit.
Originals are recoverable from git; --in-place also writes a <name>.orig.obj backup.

Works with Blender 3.x and 4.x (the OBJ import/export operators were renamed in 4.0).
"""

import argparse
import glob
import os
import sys

import bpy


def _parse_args() -> argparse.Namespace:
    # Blender passes everything after the literal "--" to the script.
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else []
    p = argparse.ArgumentParser(description="Voxel-remesh + smooth collision meshes.")
    p.add_argument("--input-dir", required=True, help="Folder containing the meshes to remesh.")
    p.add_argument("--pattern", default="*collisions*.obj", help="Glob for meshes to process.")
    p.add_argument("--output-dir", default=None, help="Output folder (default: <input-dir>/remeshed).")
    p.add_argument("--in-place", action="store_true", help="Overwrite originals (writes a .orig.obj backup).")
    p.add_argument(
        "--convex-hull",
        action="store_true",
        help="Replace each mesh with its convex hull before the voxel remesh.",
    )
    # Remesh resolution: voxel size is the dominant quality/cost knob.
    p.add_argument("--voxel-size", type=float, default=None, help="Absolute voxel size [m]; overrides --voxel-rel.")
    p.add_argument("--voxel-rel", type=float, default=0.025, help="Voxel size as a fraction of the bbox diagonal.")
    p.add_argument("--min-voxel", type=float, default=0.0015, help="Lower clamp for adaptive voxel size [m].")
    p.add_argument("--max-voxel", type=float, default=0.006, help="Upper clamp for adaptive voxel size [m].")
    # Post-remesh cleanup.
    p.add_argument("--smooth-iters", type=int, default=8, help="Smooth modifier iterations (0 disables).")
    p.add_argument("--smooth-factor", type=float, default=0.5, help="Smooth modifier factor.")
    p.add_argument("--decimate", type=float, default=0.5, help="Decimate ratio in (0,1]; 1.0 disables.")
    return p.parse_args(argv)


# URDF meshes are stored in raw Z-up coordinates. Blender's OBJ importer/exporter
# default to the OBJ spec's Y-up convention (a 90deg rotation about X), while STL is
# Z-up native. Pinning OBJ import/export to up=Z/forward=Y disables that conversion so
# coordinates pass through unchanged for every format (an STL->OBJ round-trip otherwise
# silently rotates the mesh -90deg about X relative to the URDF <origin>).
def _import_mesh(filepath: str) -> None:
    ext = os.path.splitext(filepath)[1].lower()
    if ext == ".stl":
        # STL importer was renamed to the C++ `wm.stl_import` in Blender 4.0; STL is Z-up native.
        if bpy.app.version >= (4, 0, 0):
            bpy.ops.wm.stl_import(filepath=filepath, forward_axis="Y", up_axis="Z")
        else:
            bpy.ops.import_mesh.stl(filepath=filepath)
    elif ext == ".dae":
        bpy.ops.wm.collada_import(filepath=filepath)
    else:  # .obj
        if bpy.app.version >= (4, 0, 0):
            bpy.ops.wm.obj_import(filepath=filepath, forward_axis="Y", up_axis="Z")
        else:
            bpy.ops.import_scene.obj(filepath=filepath, axis_forward="Y", axis_up="Z")


def _export_obj(filepath: str) -> None:
    if bpy.app.version >= (4, 0, 0):
        bpy.ops.wm.obj_export(
            filepath=filepath,
            export_selected_objects=True,
            export_materials=False,
            export_triangulated_mesh=True,
            forward_axis="Y",
            up_axis="Z",
        )
    else:
        bpy.ops.export_scene.obj(
            filepath=filepath, use_selection=True, use_materials=False, axis_forward="Y", axis_up="Z"
        )


def _clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()
    # Purge orphan meshes so repeated imports don't leak memory across the batch.
    for block in bpy.data.meshes:
        if block.users == 0:
            bpy.data.meshes.remove(block)


def _active_single_mesh() -> bpy.types.Object:
    """Join all imported mesh objects into one and return it, active and selected."""
    meshes = [o for o in bpy.context.scene.objects if o.type == "MESH"]
    if not meshes:
        raise RuntimeError("No mesh objects were imported.")
    bpy.ops.object.select_all(action="DESELECT")
    for o in meshes:
        o.select_set(True)
    bpy.context.view_layer.objects.active = meshes[0]
    if len(meshes) > 1:
        bpy.ops.object.join()
    return bpy.context.view_layer.objects.active


def _bbox_diagonal(obj: bpy.types.Object) -> float:
    xs = [v[0] for v in obj.bound_box]
    ys = [v[1] for v in obj.bound_box]
    zs = [v[2] for v in obj.bound_box]
    dx, dy, dz = max(xs) - min(xs), max(ys) - min(ys), max(zs) - min(zs)
    return (dx * dx + dy * dy + dz * dz) ** 0.5


def _apply(obj: bpy.types.Object, modifier_name: str) -> None:
    bpy.context.view_layer.objects.active = obj
    bpy.ops.object.modifier_apply(modifier=modifier_name)


def remesh_one(filepath: str, out_path: str, args: argparse.Namespace) -> None:
    _clear_scene()
    _import_mesh(filepath)
    obj = _active_single_mesh()

    n_v0, n_f0 = len(obj.data.vertices), len(obj.data.polygons)

    # 0) Optional convex-hull approximation -> a guaranteed-convex, watertight collider
    #    before the voxel remesh cleans up topology.
    if args.convex_hull:
        bpy.ops.object.mode_set(mode="EDIT")
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.convex_hull(delete_unused=True, use_existing_faces=False, make_holes=False)
        bpy.ops.mesh.select_all(action="SELECT")
        bpy.ops.mesh.normals_make_consistent(inside=False)
        bpy.ops.object.mode_set(mode="OBJECT")

    # 1) Voxel remesh -> watertight manifold. Size adaptive to the part unless overridden.
    if args.voxel_size is not None:
        voxel = args.voxel_size
    else:
        voxel = max(args.min_voxel, min(args.max_voxel, _bbox_diagonal(obj) * args.voxel_rel))
    rem = obj.modifiers.new(name="Remesh", type="REMESH")
    rem.mode = "VOXEL"
    rem.voxel_size = voxel
    rem.use_smooth_shade = True
    _apply(obj, rem.name)

    # 2) Smooth modifier -> de-blockify the voxel surface.
    if args.smooth_iters > 0:
        sm = obj.modifiers.new(name="Smooth", type="SMOOTH")
        sm.factor = args.smooth_factor
        sm.iterations = args.smooth_iters
        _apply(obj, sm.name)

    # 3) Optional decimate to a triangle budget.
    if args.decimate < 1.0:
        dec = obj.modifiers.new(name="Decimate", type="DECIMATE")
        dec.ratio = args.decimate
        _apply(obj, dec.name)

    # 4) Triangulate + outward-consistent normals for a clean SDF input.
    tri = obj.modifiers.new(name="Triangulate", type="TRIANGULATE")
    _apply(obj, tri.name)
    bpy.ops.object.mode_set(mode="EDIT")
    bpy.ops.mesh.select_all(action="SELECT")
    bpy.ops.mesh.normals_make_consistent(inside=False)
    bpy.ops.object.mode_set(mode="OBJECT")

    n_v1, n_f1 = len(obj.data.vertices), len(obj.data.polygons)

    bpy.ops.object.select_all(action="DESELECT")
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    _export_obj(out_path)

    print(
        f"[remesh] {os.path.basename(filepath)}: voxel={voxel*1000:.2f}mm  "
        f"verts {n_v0}->{n_v1}  faces {n_f0}->{n_f1}  -> {out_path}",
        flush=True,
    )


def main() -> None:
    args = _parse_args()
    in_dir = os.path.abspath(args.input_dir)
    files = sorted(glob.glob(os.path.join(in_dir, args.pattern)))
    # Don't re-ingest our own backups.
    files = [f for f in files if not f.endswith(".orig.obj")]
    if not files:
        raise SystemExit(f"No meshes matched {args.pattern!r} in {in_dir}")

    out_dir = in_dir if args.in_place else (os.path.abspath(args.output_dir) if args.output_dir else os.path.join(in_dir, "remeshed"))
    os.makedirs(out_dir, exist_ok=True)

    print(f"[remesh] {len(files)} mesh(es) from {in_dir} -> {out_dir}{' (in place)' if args.in_place else ''}", flush=True)
    for f in files:
        name = os.path.basename(f)
        if args.in_place:
            backup = f[:-4] + ".orig.obj"
            if not os.path.exists(backup):
                os.replace(f, backup)  # move original aside; we read from the backup
                src = backup
            else:
                src = backup
            remesh_one(src, f, args)
        else:
            # Always export OBJ regardless of source extension (.stl/.dae/.obj).
            out_name = os.path.splitext(name)[0] + ".obj"
            remesh_one(f, os.path.join(out_dir, out_name), args)

    print("[remesh] done.", flush=True)


if __name__ == "__main__":
    main()
