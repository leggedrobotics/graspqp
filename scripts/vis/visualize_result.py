""" """

import os

import numpy as np
import torch
import plotly.graph_objects as go

from graspqp.hands import get_hand_model, AVAILABLE_HANDS
from graspqp.core import ObjectModel
from graspqp.core.energy import calculate_energy

import argparse
import glob

import roma

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union
import numpy as np
import plotly.io as pio

pio.renderers.default = "browser"

# Optional import guards keep this import light if you only build extra meshes
try:
    import plotly.graph_objects as go
except Exception:
    go = None  # type: ignore

from pygltflib import (
    GLTF2,
    Scene,
    Node,
    Mesh,
    Buffer,
    BufferView,
    Accessor,
    Asset,
    Primitive,
    Material,
)

try:
    # Newer naming in some releases
    from pygltflib import PBRMetallicRoughness as _PBRType
except Exception:
    # Canonical class name in most versions
    from pygltflib import PbrMetallicRoughness as _PBRType

# ---- GLTF constants (avoid version quirks) ----
T_ARRAY = 34962  # ARRAY_BUFFER
T_ELEMENT_ARRAY = 34963  # ELEMENT_ARRAY_BUFFER
CTYPE_FLOAT = 5126  # FLOAT
CTYPE_UINT = 5125  # UNSIGNED_INT
MODE_POINTS = 0
MODE_LINES = 1
MODE_TRIANGLES = 4


@dataclass
class SimpleMesh:
    """Minimal triangle mesh container for non-Plotly input."""

    vertices: np.ndarray  # (N,3) float32
    faces: np.ndarray  # (M,3) uint32
    color: Sequence[float] = (0.8, 0.8, 0.8, 1.0)  # RGBA in [0,1]
    name: str = "mesh"


# ------------------ Small utilities ------------------


def _align4(n: int) -> int:
    """Next multiple of 4."""
    return (n + 3) & ~0x03


def _minmax_vec(arr: np.ndarray):
    return arr.min(axis=0).tolist(), arr.max(axis=0).tolist()


def _rgba_from_plotly(color, opacity: Optional[float], fallback=(0.8, 0.8, 0.8, 1.0)):
    """
    Accepts:
      - '#rgb' '#rrggbb' '#rrggbbaa'
      - 'rgb(r,g,b)' 'rgba(r,g,b,a)'
      - CSS color names ('red', 'yellow', etc.)
      - sequences like [r,g,b] or [r,g,b,a] in 0..1 or 0..255
      - None
    Returns normalized (r,g,b,a) in 0..1.
    """
    CSS = {
        "black": (0, 0, 0),
        "white": (255, 255, 255),
        "red": (255, 0, 0),
        "green": (0, 128, 0),
        "blue": (0, 0, 255),
        "yellow": (255, 255, 0),
        "orange": (255, 165, 0),
        "purple": (128, 0, 128),
        "cyan": (0, 255, 255),
        "magenta": (255, 0, 255),
        "gray": (128, 128, 128),
        "grey": (128, 128, 128),
        "brown": (165, 42, 42),
        "pink": (255, 192, 203),
    }

    def _norm3(r, g, b):
        scale = 255.0 if max(abs(r), abs(g), abs(b)) > 1.0 else 1.0
        return (float(r) / scale, float(g) / scale, float(b) / scale)

    def _norm4(r, g, b, a):
        return (*_norm3(r, g, b), float(a))

    if color is None:
        r, g, b, a = fallback
    elif hasattr(color, "__iter__") and not isinstance(color, (str, bytes)):
        vals = list(color)
        if len(vals) == 3:
            r, g, b = _norm3(vals[0], vals[1], vals[2])
            a = 1.0
        elif len(vals) == 4:
            r, g, b, a = _norm4(vals[0], vals[1], vals[2], vals[3])
        else:
            r, g, b, a = fallback
    else:
        s = str(color).strip().lower()
        r = g = b = 1.0
        a = 1.0
        if s.startswith("#"):
            hexpart = s[1:]
            if len(hexpart) == 3:  # #rgb
                r = int(hexpart[0] * 2, 16) / 255.0
                g = int(hexpart[1] * 2, 16) / 255.0
                b = int(hexpart[2] * 2, 16) / 255.0
            elif len(hexpart) in (6, 8):
                r = int(hexpart[0:2], 16) / 255.0
                g = int(hexpart[2:4], 16) / 255.0
                b = int(hexpart[4:6], 16) / 255.0
                if len(hexpart) == 8:
                    a = int(hexpart[6:8], 16) / 255.0
            else:
                r, g, b, a = fallback
        elif s.startswith("rgba"):
            nums = s[s.find("(") + 1 : s.find(")")].split(",")
            if len(nums) == 4:
                r, g, b, a = _norm4(float(nums[0]), float(nums[1]), float(nums[2]), float(nums[3]))
            else:
                r, g, b, a = fallback
        elif s.startswith("rgb"):
            nums = s[s.find("(") + 1 : s.find(")")].split(",")
            if len(nums) == 3:
                r, g, b = _norm3(float(nums[0]), float(nums[1]), float(nums[2]))
                a = 1.0
            else:
                r, g, b, a = fallback
        elif s in CSS:
            rc, gc, bc = CSS[s]
            r, g, b = rc / 255.0, gc / 255.0, bc / 255.0
            a = 1.0
        else:
            r, g, b, a = fallback

    if opacity is not None:
        a = float(opacity)
    return (float(r), float(g), float(b), float(a))


# ------------------ GLTF Builder ------------------


class GLTFBuilder:
    def __init__(self, scene_name: str = "scene", up_axis: str = "Z"):
        self.scene_name = scene_name
        self.bin = bytearray()
        self.up_axis = (up_axis or "Z").upper()
        self.bufferViews: List[BufferView] = []
        self.accessors: List[Accessor] = []
        self.materials: List[Material] = []
        self.meshes: List[Mesh] = []
        self.nodes: List[Node] = []
        self.material_cache = {}

    def _push_blob(self, data: bytes, target: Optional[int] = None) -> int:
        """Append bytes to bin chunk, return BufferView index."""
        # glTF requires 4-byte alignment
        pad = _align4(len(self.bin)) - len(self.bin)
        if pad:
            self.bin.extend(b"\x00" * pad)
        offset = len(self.bin)
        self.bin.extend(data)
        bv = BufferView(buffer=0, byteOffset=offset, byteLength=len(data), target=target)
        self.bufferViews.append(bv)
        return len(self.bufferViews) - 1

    def _add_accessor(
        self,
        bufferView: int,
        componentType: int,
        count: int,
        type_str: str,
        minv=None,
        maxv=None,
        byteOffset: int = 0,
    ) -> int:
        acc = Accessor(
            bufferView=bufferView,
            byteOffset=byteOffset,
            componentType=componentType,
            count=count,
            type=type_str,
            min=minv,
            max=maxv,
        )
        self.accessors.append(acc)
        return len(self.accessors) - 1

    def get_or_create_material(self, rgba: Sequence[float], name: Optional[str] = None) -> int:
        rgba = [float(np.clip(c, 0.0, 1.0)) for c in rgba]
        key = tuple(rgba)
        if key in self.material_cache:
            return self.material_cache[key]

        mat = Material(
            name=name or f"mat_{len(self.materials)}",
            pbrMetallicRoughness=_PBRType(
                baseColorFactor=rgba,
                metallicFactor=0.0,
                roughnessFactor=1.0,
            ),
            doubleSided=True,
        )
        # Enable blending when alpha < 1
        mat.alphaMode = "BLEND" if rgba[3] < 0.999 else "OPAQUE"

        self.materials.append(mat)
        idx = len(self.materials) - 1
        self.material_cache[key] = idx
        return idx

    # ----------- High-level helpers -----------

    def add_points_or_lines(
        self,
        positions: np.ndarray,
        indices: Optional[np.ndarray],
        mode: int,
        rgba=(0.8, 0.8, 0.8, 1.0),
        name="scatter",
    ) -> int:
        """
        Adds a mesh with a single primitive (POINTS or LINES).
        If indices is None for POINTS, it draws all vertices.
        """
        assert positions.ndim == 2 and positions.shape[1] == 3
        positions = positions.astype(np.float32, copy=False)

        # Upload positions
        pos_bytes = positions.tobytes()
        bv_pos = self._push_blob(pos_bytes, target=T_ARRAY)
        pos_min, pos_max = _minmax_vec(positions)
        acc_pos = self._add_accessor(bv_pos, CTYPE_FLOAT, len(positions), "VEC3", pos_min, pos_max)

        prim_kwargs = {"attributes": {"POSITION": acc_pos}, "mode": mode}

        # Upload indices if provided
        if indices is not None:
            indices = indices.astype(np.uint32, copy=False).ravel()
            bv_idx = self._push_blob(indices.tobytes(), target=T_ELEMENT_ARRAY)
            acc_idx = self._add_accessor(bv_idx, CTYPE_UINT, indices.size, "SCALAR")
            prim_kwargs["indices"] = acc_idx

        mat_idx = self.get_or_create_material(rgba, name=f"{name}_mat")
        prim = Primitive(material=mat_idx, **prim_kwargs)

        mesh = Mesh(name=name, primitives=[prim])
        self.meshes.append(mesh)
        mesh_idx = len(self.meshes) - 1

        node = Node(name=name, mesh=mesh_idx)
        self.nodes.append(node)
        return len(self.nodes) - 1

    def add_triangle_mesh(
        self,
        vertices: np.ndarray,
        faces: np.ndarray,
        rgba=(0.8, 0.8, 0.8, 1.0),
        name="mesh",
    ) -> int:
        assert vertices.ndim == 2 and vertices.shape[1] == 3
        assert faces.ndim == 2 and faces.shape[1] == 3
        vtx = vertices.astype(np.float32, copy=False)
        fcs = faces.astype(np.uint32, copy=False)

        # Positions
        bv_pos = self._push_blob(vtx.tobytes(), target=T_ARRAY)
        pos_min, pos_max = _minmax_vec(vtx)
        acc_pos = self._add_accessor(bv_pos, CTYPE_FLOAT, len(vtx), "VEC3", pos_min, pos_max)

        # Indices
        bv_idx = self._push_blob(fcs.ravel().tobytes(), target=T_ELEMENT_ARRAY)
        acc_idx = self._add_accessor(bv_idx, CTYPE_UINT, fcs.size, "SCALAR")

        prim = Primitive(attributes={"POSITION": acc_pos}, indices=acc_idx, mode=MODE_TRIANGLES)
        prim.material = self.get_or_create_material(rgba, name=f"{name}_mat")

        mesh = Mesh(name=name, primitives=[prim])
        self.meshes.append(mesh)
        mesh_idx = len(self.meshes) - 1

        node = Node(name=name, mesh=mesh_idx)
        self.nodes.append(node)
        return len(self.nodes) - 1

    def finish(self, out_path: str):
        # If inputs are Z-up, add a root node that rotates Z-up → Y-up (−90° around X)
        scene_nodes = list(range(len(self.nodes)))
        if self.up_axis == "Z":
            s = float(np.sqrt(0.5))  # cos(−90°/2) = √1/2
            qx = -s  # sin(−90°/2) * axis_x
            qw = s
            root = Node(
                name="Zup_to_Yup",
                rotation=[qx, 0.0, 0.0, qw],
                children=scene_nodes if scene_nodes else None,
            )
            self.nodes.append(root)
            scene_nodes = [len(self.nodes) - 1]

        gltf = GLTF2(
            asset=Asset(version="2.0"),
            scenes=[Scene(name=self.scene_name, nodes=scene_nodes)],
            nodes=self.nodes,
            meshes=self.meshes,
            materials=self.materials,
            buffers=[Buffer(byteLength=len(self.bin))],
            bufferViews=self.bufferViews,
            accessors=self.accessors,
        )

        try:
            gltf.set_binary_blob(bytes(self.bin))  # type: ignore[attr-defined]
            gltf.save_binary(out_path)
        except Exception:
            import os

            base, _ = os.path.splitext(out_path)
            bin_path = base + ".bin"
            with open(bin_path, "wb") as f:
                f.write(self.bin)
            gltf.buffers[0].uri = os.path.basename(bin_path)
            gltf.save(base + ".gltf")


# ------------------ Plotly converters ------------------


def _as_np(arr) -> np.ndarray:
    return np.asarray(arr, dtype=np.float64)


def _scatter3d_to_gltf(builder: GLTFBuilder, trace, scale: float = 1.0, name: str = "scatter"):
    x = _as_np(getattr(trace, "x", []))
    y = _as_np(getattr(trace, "y", []))
    z = _as_np(getattr(trace, "z", []))
    if not (len(x) and len(y) and len(z)) or not (len(x) == len(y) == len(z)):
        return  # ignore empty/bad trace

    # Build mask for valid points; NaNs break line segments
    valid = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    # Map original indices -> compact indices for valid points
    idx_map = -np.ones(len(x), dtype=np.int64)
    idx_map[valid] = np.arange(valid.sum())

    positions = np.column_stack([x[valid], y[valid], z[valid]]).astype(np.float32) * float(scale)

    mode = (getattr(trace, "mode", "") or "").lower()
    want_points = "markers" in mode
    want_lines = "lines" in mode

    # Colors (uniform per primitive)
    marker_rgba = _rgba_from_plotly(
        getattr(getattr(trace, "marker", None), "color", None),
        getattr(getattr(trace, "marker", None), "opacity", None),
        fallback=(0.9, 0.9, 0.9, 1.0),
    )
    line_rgba = _rgba_from_plotly(
        getattr(getattr(trace, "line", None), "color", None),
        getattr(getattr(trace, "line", None), "opacity", None),
        fallback=(0.2, 0.2, 0.2, 1.0),
    )

    # One mesh node per Scatter3d, with up to two primitives that share the same POSITION accessor
    # Implementation detail: add the POSITION once by creating a dummy POINTS primitive first, then reuse accessor.
    # To keep this simple and robust across pygltflib versions, we instead build two separate meshes that
    # duplicate the positions data minimally. (The file stays small; points/lines indices are tiny.)
    base_name = getattr(trace, "name", None) or name

    if want_points and positions.shape[0] > 0:
        builder.add_points_or_lines(
            positions=positions,
            indices=None,
            mode=MODE_POINTS,
            rgba=marker_rgba,
            name=f"{base_name}_points",
        )

    if want_lines and positions.shape[0] > 1:
        # Build line indices as pairs within segments (NaN breaks)
        pairs = []
        last_valid_orig = None
        for i in range(len(x)):
            if not valid[i]:
                last_valid_orig = None
                continue
            if last_valid_orig is not None:
                a = idx_map[last_valid_orig]
                b = idx_map[i]
                if a >= 0 and b >= 0:
                    pairs.append([a, b])
            last_valid_orig = i
        if pairs:
            line_indices = np.asarray(pairs, dtype=np.uint32)
            builder.add_points_or_lines(
                positions=positions,
                indices=line_indices,
                mode=MODE_LINES,
                rgba=line_rgba,
                name=f"{base_name}_lines",
            )


def _mesh3d_to_gltf(builder: GLTFBuilder, trace, scale: float = 1.0, name: str = "mesh"):
    x = _as_np(getattr(trace, "x", []))
    y = _as_np(getattr(trace, "y", []))
    z = _as_np(getattr(trace, "z", []))
    i = getattr(trace, "i", None)
    j = getattr(trace, "j", None)
    k = getattr(trace, "k", None)
    if i is None or j is None or k is None:
        return

    verts = np.column_stack([x, y, z]).astype(np.float32) * float(scale)
    faces = np.column_stack([_as_np(i), _as_np(j), _as_np(k)]).astype(np.uint32)

    tr_opacity = getattr(trace, "opacity", None)
    rgba = _rgba_from_plotly(getattr(trace, "color", None), tr_opacity, fallback=(0.7, 0.7, 0.8, 1.0))
    base_name = getattr(trace, "name", None) or name
    builder.add_triangle_mesh(verts, faces, rgba=rgba, name=base_name)


# ------------------ Public API ------------------


def convert_plotly_to_gltf(
    traces_or_fig: Union["go.Figure", Sequence[object]],
    out_path: str,
    *,
    extra_meshes: Optional[List[SimpleMesh]] = None,
    scale: float = 1.0,
    scene_name: str = "scene",
):
    """
    Convert Plotly 3D traces (Scatter3d lines/markers and Mesh3d) + optional SimpleMesh list to glTF (.glb).
    - traces_or_fig: a go.Figure or an iterable of Plotly traces
    - out_path: output file; if ends with .glb, writes GLB; otherwise writes .gltf + .bin fallback
    - extra_meshes: list of SimpleMesh(vertices(N,3) float32, faces(M,3) uint32, color RGBA)
    - scale: uniform scale applied to positions
    - scene_name: glTF scene name
    """
    builder = GLTFBuilder(scene_name=scene_name)

    # Resolve traces
    traces: List[object] = []
    if go is not None and hasattr(traces_or_fig, "data"):  # likely a Figure
        traces = list(traces_or_fig.data)  # type: ignore[attr-defined]
    else:
        traces = list(traces_or_fig)  # assume iterable

    for t in traces:
        ttype = t.__class__.__name__.lower()
        if "scatter3d" in ttype:
            _scatter3d_to_gltf(builder, t, scale=scale, name="scatter")
        elif "mesh3d" in ttype:
            _mesh3d_to_gltf(builder, t, scale=scale, name="mesh")
        else:
            # ignore other plotly trace types
            pass

    if extra_meshes:
        for m in extra_meshes:
            builder.add_triangle_mesh(
                m.vertices.astype(np.float32, copy=False) * float(scale),
                m.faces.astype(np.uint32, copy=False),
                rgba=tuple(m.color),
                name=m.name,
            )

    builder.finish(out_path)


torch.manual_seed(1)

os.environ["KMP_DUPLICATE_LIB_OK"] = "True"


def _draw_line_plotly(start, end, color="red", width=5):
    return go.Scatter3d(
        x=[start[0], end[0]],
        y=[start[1], end[1]],
        z=[start[2], end[2]],
        mode="lines",
        line=dict(color=color, width=width),
    )


def _flatten(arr):
    data = []
    for a in arr:
        data.extend(a)
    return data


def _show_dir(dir, args, device, origin=(0, 0)):
    data_path = os.path.join(dir, args.hand_name)
    glob_pattern = os.path.join(data_path, args.num_contacts, args.energy, args.grasp_type, "*.dexgrasp.pt")

    print(f"Loading from {glob_pattern}")
    if len(glob.glob(glob_pattern, recursive=True)) == 0:
        print(f"No files found for pattern {glob_pattern}")
        return None
    checkpoint_path = sorted(glob.glob(glob_pattern, recursive=True), key=os.path.getmtime)[-1]
    if "diffused" in checkpoint_path:
        checkpoint_path = sorted(glob.glob(glob_pattern, recursive=True), key=os.path.getmtime)[0]

    # print in green color
    print(f"\033[92mLoading {checkpoint_path}\033[0m")
    # print(f"Loading Files from {checkpoint_path}")
    checkpoint_data = torch.load(checkpoint_path)
    print("Loading data from", checkpoint_path)
    # put on cpu, detach grad

    hand_model = get_hand_model(
        args.hand_name,
        args.device,
        use_collision_if_possible=True,
        grasp_type=checkpoint_data.get("grasp_type", None),
    )
    params = checkpoint_data["parameters"]
    params = {k: v.detach().cpu() for k, v in params.items()}
    joint_states = []

    grasp_velocities = []
    for joint_name in hand_model._actuated_joints_names:
        joint_states.append(params[joint_name])
        if "grasp_velocities_off" in checkpoint_data:
            grasp_velocities.append(checkpoint_data["grasp_velocities_off"][joint_name])
        else:
            print("Warning: grasp velocities are not in the checkpoint data, using zeros instead")
            grasp_velocities.append(torch.zeros_like(params[joint_name]))

    grasp_velocities = torch.stack(grasp_velocities, dim=-1).to(device)
    joint_states = torch.stack(joint_states, dim=-1).to(device)
    root_pose = params["root_pose"].to(device).float()
    if len(root_pose) == 0:
        print("[Warning] No grasps found in the checkpoint data")
        return None

    if "offset" in checkpoint_data:
        print("Adding offset to root pose")
        pose = checkpoint_data["offset"].to(device)
        if pose.shape[-1] == 7:
            print("Pose shape is 7, converting to 4x4 matrix")
            # pose = pose.unsqueeze(0).expand(len(root_pose), -1)
            T_graspit_ours = torch.zeros(len(root_pose), 4, 4).to(device).float()
            T_graspit_ours[:, 3, 3] = 1.0
            # def rotmat_from_euler(roll, pitch, yaw):
            #     roll = torch.tensor([roll], device=device)
            #     pitch = torch.tensor([pitch], device=device)
            #     yaw = torch.tensor([yaw], device=device)

            #     R_x = torch.tensor([[1, 0, 0], [0, torch.cos(roll), -torch.sin(roll)], [0, torch.sin(roll), torch.cos(roll)]])
            #     R_y = torch.tensor([[torch.cos(pitch), 0, torch.sin(pitch)], [0, 1, 0], [-torch.sin(pitch), 0, torch.cos(pitch)]])
            #     R_z = torch.tensor([[torch.cos(yaw), -torch.sin(yaw), 0], [torch.sin(yaw), torch.cos(yaw), 0], [0, 0, 1]])
            #     return R_z @ R_y @ R_x
            # T_graspit_ours[:, :3, :3] = rotmat_from_euler(1.575,  0, 1.575).unsqueeze(0).expand(len(root_pose), -1, -1)[:, :3, :3]
            # # T_graspit_ours = torch.eye(4).unsqueeze(0).expand(len(root_pose), -1, -1).to(device).float()

            # # T_graspit_ours[:, :3, 3] = pose[:, :3]
            # # 0.037, 0.009, -0.011,
            # T_graspit_ours[:, 0, -1] += 0.028
            # T_graspit_ours[:, 1, -1] -= 0.002
            # T_graspit_ours[:, 2, -1] -= 0.029
            # T_graspit_ours[:, 3, 3] = 1.0

            if pose.ndim == 1:
                pose = pose.unsqueeze(0)

            if pose.shape[0] != len(root_pose):
                pose = pose.expand(len(root_pose), -1)

            R_graspit_ours = roma.unitquat_to_rotmat(pose[..., [4, 5, 6, 3]])
            t_graspit_ours = pose[:, :3]
            T_graspit_ours[:, :3, :3] = R_graspit_ours
            T_graspit_ours[:, :3, 3] = t_graspit_ours

            # ours_quat = roma.rotmat_to_unitquat(T_graspit_ours[:, :3, :3])[:, [3, 0, 1, 2]]
            # ours_trans = T_graspit_ours[:, :3, 3]
            # import pdb; pdb.set_trace()
            # T_graspit_ours = T_graspit_ours.inverse()
            # import pdb; pdb.set_trace()

        else:
            T_graspit_ours = checkpoint_data["offset"].to(device).expand(len(root_pose), -1, -1)

        t_w_graspit = root_pose[:, :3]
        R_w_graspit = roma.unitquat_to_rotmat(root_pose[..., [4, 5, 6, 3]])
        T_w_graspit = torch.zeros_like(T_graspit_ours)
        T_w_graspit[:, :3, :3] = R_w_graspit
        T_w_graspit[:, :3, 3] = t_w_graspit
        T_w_graspit[:, 3, 3] = 1.0

        T_w_ours = T_w_graspit @ T_graspit_ours
        R_w_ours = T_w_ours[:, :3, :3]
        t_w_ours = T_w_ours[:, :3, 3]
        quat = roma.rotmat_to_unitquat(R_w_ours)[..., [3, 0, 1, 2]]
        root_pose[:, :3] = t_w_ours.clone()
        root_pose[:, 3:7] = quat.clone()

    energies = checkpoint_data["values"]

    energies, indices = torch.sort(energies)

    if "contact_idx" in checkpoint_data:
        contact_idxs = checkpoint_data["contact_idx"].to(device)  # [:2]
        contact_idxs = contact_idxs[indices][: args.max_grasps]
    else:
        contact_idxs = "all"
    # sort by energy
    root_pose = root_pose[indices][: args.max_grasps]
    joint_states = joint_states[indices][: args.max_grasps]
    grasp_velocities = grasp_velocities[indices][: args.max_grasps]

    root_orientation = roma.unitquat_to_rotmat(root_pose[..., [4, 5, 6, 3]]).mT.flatten(1, 2)
    hand_params = torch.cat([root_pose[..., :3], root_orientation[..., :6], joint_states], dim=-1).to(device)
    hand_model.set_parameters(hand_params, contact_point_indices=contact_idxs)

    batch_size = len(hand_params)
    asset_path = os.path.dirname(dir)
    root_path = os.path.dirname(asset_path)
    if args.obj_path is not None:
        root_path = args.obj_path
        asset_path = args.obj_path

    object_model = ObjectModel(
        data_root_path=root_path,
        batch_size_each=batch_size,
        num_samples=1500,
        device=device,
        scale=args.scale,
    )
    object_model.initialize([asset_path])

    n_envs = min(args.max_grasps, len(joint_states))
    n_axis = np.sqrt(n_envs).astype(int).item()
    all_data = []

    distance, contact_normal = object_model.cal_distance(hand_model.contact_points)
    contact_normal = contact_normal * 0.025  # distance.abs().unsqueeze(-1)*3
    delta_theta, residuals, ee_vel = hand_model.get_req_joint_velocities(
        -contact_normal, hand_model.contact_point_indices, return_ee_vel=True
    )

    for env_id in range(min(args.max_grasps, len(joint_states))):
        data = []
        loc_x = env_id % n_axis
        loc_y = env_id // n_axis
        # info
        data += object_model.get_plotly_data(
            i=env_id,
            simplify=False,
            offset=[
                loc_x * args.spacing + origin[0],
                loc_y * args.spacing + origin[1],
                0,
            ],
            color="rgba(128, 255, 128, 1.0)",
            opacity=1.0,
        )
        offset = np.array([loc_x * args.spacing + origin[0], loc_y * args.spacing + origin[1], 0])
        # surface_points = hand_model.get_surface_points()[env_id].detach().cpu().numpy() + offset

        hand_plotly = hand_model.get_plotly_data(
            i=env_id,
            opacity=1.0,
            color="rgba(128, 128, 128, 1.0)",
            with_contact_points=False,
            with_penetration_points=args.show_penetration_points,
            with_surface_points=False,
            simplify=False,
            offset=[
                loc_x * args.spacing + origin[0],
                loc_y * args.spacing + origin[1],
                0,
            ],
        )
        idx = 0
        x_axis = go.Scatter3d(
            x=[
                loc_x * args.spacing + origin[0],
                loc_x * args.spacing + 0.1 + origin[0],
            ],
            y=[loc_y * args.spacing + origin[1], loc_y * args.spacing + origin[1]],
            z=[0, 0],
            mode="lines",
            line=dict(color="red", width=5),
        )
        y_axis = go.Scatter3d(
            x=[loc_x * args.spacing + origin[0], loc_x * args.spacing + origin[0]],
            y=[
                loc_y * args.spacing + origin[1],
                loc_y * args.spacing + 0.1 + origin[1],
            ],
            z=[0, 0],
            mode="lines",
            line=dict(color="green", width=5),
        )
        z_axis = go.Scatter3d(
            x=[loc_x * args.spacing + origin[0], loc_x * args.spacing + origin[0]],
            y=[loc_y * args.spacing + origin[1], loc_y * args.spacing + origin[1]],
            z=[0, 0.1],
            mode="lines",
            line=dict(color="blue", width=5),
        )

        # coordinate_system = [x_axis, y_axis, z_axis]
        data += hand_plotly

        # data += coordinate_system
        # surface_points_plotly = [go.Scatter3d(x=surface_points[:, 0], y=surface_points[:, 1], z=surface_points[:, 2], mode='markers', marker=dict(color='yellow', size=2))]
        # contact_candidates_plotly = [go.Scatter3d(x=contact_candidates[:, 0], y=contact_candidates[:, 1], z=contact_candidates[:, 2], mode='markers', marker=dict(color='green', size=5))]
        # data += surface_points_plotly
        # data = hand_plotly + surface_points_plotly + contact_candidates_plotly + coordinate_system + data
        # import pdb; pdb.set_trace()
        # draw contact normals

        contact_start_pt = hand_model.contact_points[env_id].detach().cpu().numpy() + offset

        contact_end_pt = contact_start_pt - 1.0 * contact_normal[env_id].detach().cpu().numpy()

        for start, end in zip(contact_start_pt, contact_end_pt):
            data.append(
                go.Scatter3d(
                    x=[start[0], end[0]],
                    y=[start[1], end[1]],
                    z=[start[2], end[2]],
                    mode="lines",
                    line=dict(color="green", width=5),
                )
            )

        # # show contact through cog
        # contact_start_pt = (object_model.cog[env_id].detach().cpu().numpy() + offset) + 0* contact_start_pt
        # contact_end_pt = contact_start_pt - 1.0 * contact_normal[env_id].detach().cpu().numpy()
        # for start, end in zip(contact_start_pt, contact_end_pt):
        #     data.append(go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode='lines', line=dict(color='green', width=5)))

        # Show CoG of the object
        cog = object_model.cog[env_id].detach().cpu().numpy() + offset
        data.append(
            go.Scatter3d(
                x=[cog[0]],
                y=[cog[1]],
                z=[cog[2]],
                mode="markers",
                marker=dict(color="red", size=20),
            )
        )

        # ee_vel_end_pt = contact_start_pt + 1.0 * (ee_vel[env_id].detach().cpu().numpy())
        # for start, end in zip(contact_start_pt, ee_vel_end_pt):
        #     data.append(go.Scatter3d(x=[start[0], end[0]], y=[start[1], end[1]], z=[start[2], end[2]], mode='lines', line=dict(color='blue', width=5)))
        # data.append(go.Scatter3d(x=[contact_start_pt[0, 0], contact_end_pt[0, 0]], y=[contact_start_pt[0, 1], contact_end_pt[0, 1]], z=[contact_start_pt[0, 2], contact_end_pt[0, 2]], mode='lines', line=dict(color='red', width=5)))

        if args.show_jacobian:
            J = hand_model.jacobian(joint_states)
            ee_vel = (J @ grasp_velocities.unsqueeze(1).unsqueeze(1).mT).squeeze(-1)[env_id]

            for mesh_idx, link_name in enumerate(hand_model.mesh):

                jacobian = ee_vel[mesh_idx]

                linear_jacobian = jacobian[:3]
                angular_jacobian = jacobian[3:]

                contact_points = hand_model.mesh[link_name]["contact_candidates"]
                if 0 in contact_points.shape:
                    continue

                #  batch_size = self.global_translation.shape[0]
                # for link_name in self.mesh:
                #     contacts = self.mesh[link_name]["contact_candidates"]
                #     n_surface_points = contacts.shape[0]
                #     if n_surface_points == 0:
                #         continue

                #     transformed = self.current_status[link_name].transform_points(contacts)
                #     if transformed.ndim == 2:
                #         transformed = transformed.unsqueeze(0).expand(batch_size, -1, -1)
                #     points.append(transformed)

                #     if with_normals:
                #         normals_p = self.current_status[link_name].transform_normals(
                #             self.mesh[link_name]["normal_candidates"]
                #         )
                #         if normals_p.ndim == 2:
                #             normals_p = normals_p.unsqueeze(0).expand(batch_size, -1, -1)
                #         normals.append(normals_p)

                #         points = points @ self.global_rotation.transpose(
                #     1, 2
                # ) + self.global_translation.unsqueeze(1)
                contacts_local = hand_model.mesh[link_name]["contact_candidates"]
                transformed_contacts = hand_model.current_status[link_name].transform_points(contacts_local)
                transformed_contacts = transformed_contacts @ hand_model.global_rotation.transpose(
                    1, 2
                ) + hand_model.global_translation.unsqueeze(1)
                cog_links = transformed_contacts[env_id].mean(0, keepdim=True)
                cog_links_w = cog_links + torch.from_numpy(offset).to(args.device).float()

                motion_start = cog_links_w.cpu().numpy()[0]
                motion_end = (
                    motion_start
                    + (linear_jacobian[None] @ hand_model.global_rotation[env_id].transpose(0, 1)).cpu().numpy()[0]
                )
                # draw the line
                data.append(
                    go.Scatter3d(
                        x=[motion_start[0], motion_end[0]],
                        y=[motion_start[1], motion_end[1]],
                        z=[motion_start[2], motion_end[2]],
                        mode="lines",
                        line=dict(color="red", width=5),
                    )
                )

        if args.calc_energy:
            from graspqp.metrics import GraspSpanMetricFactory

            energy_fnc = GraspSpanMetricFactory.create(
                GraspSpanMetricFactory.MetricType.GRASPQP_EUCLIDIAN_SCIPY,
                # solver_kwargs={"max_limit": 10},
            )

            # for idx, energy_name in enumerate(["dexgrasp", "span_overall_scipy", "span_overall_cone_scipy", "span_eucledian_scipy", "span_overall_sqp"]):
            for idx, energy_name in enumerate(["span_eucledian_scipy"]):
                energy = calculate_energy(
                    hand_model,
                    object_model,
                    energy_fnc=energy_fnc,
                    # energy_kwargs=dict(reg=0.0, contact_threshold=0.0, svd_gain=0.0),
                )
                # convert dict to cpu
                energy = {k: v.detach().cpu() for k, v in energy.items()}

                energy = energy["E_fc"]
                # def calculate_energy(
                #     hand_model,
                #     object_model,
                #     energy_fnc: any = None,
                #     energy_names=[],
                #     method="dexgraspnet",
                #     svd_gain=0.1,

                import matplotlib.cm as cm

                color = cm.viridis(
                    (energy[env_id] - energy.min()).cpu().numpy() / (energy.max() - energy.min()).cpu().numpy()
                )
                color_rgb_str = f"rgb({int(color[0]*255)},{int(color[1]*255)},{int(color[2]*255)})"
                text = f"{energy_name}: {energy[env_id]:.2f}"
                spacing = 0.04
                offset_z = 0.1
                data.append(
                    go.Scatter3d(
                        x=[offset[0]],
                        y=[offset[1]],
                        z=[offset[2] - idx * spacing - offset_z],
                        mode="text",
                        text=[text],
                        textposition="top center",
                        textfont=dict(size=18, color=color_rgb_str),
                    )
                )

                if energy_name == "span_overall_cone_scipy":
                    ranking_idx = energy.argsort()
                    ranking = ranking_idx[env_id] + 1
                    # write text on top of the object
                    text = f"{ranking.item():d}"
                    # offset = object_model.cog[env_id].detach().cpu().numpy() + np.array([0, 0, 0.2])
                    off = object_model.cog[env_id].detach().cpu().numpy() + np.array([0, 0, 0.2]) + offset
                    data.append(
                        go.Scatter3d(
                            x=[off[0]],
                            y=[off[1]],
                            z=[off[2]],
                            mode="text",
                            text=[text],
                            textposition="top center",
                            textfont=dict(size=25, color="red"),
                        )
                    )

                SHOW_OVERALL = True
                SHOW_EUCLEDIAN = True
                # continue
                if energy_name == "span_overall_cone_scipy":
                    cache = energy_fnc.metric._cache
                    linear_forces = cache["linear_forces"][env_id].cpu().numpy()

                    coef = cache["results"][-1][env_id].cpu().numpy().squeeze()
                    r_contact = (
                        cache["r_cog_contact"][env_id].cpu().numpy()
                        + offset
                        + object_model.cog[env_id].detach().cpu().numpy()
                    )

                    r_end = r_contact + linear_forces  # * (coef - 1)[:, None] * 0.05

                    for start, end in zip(r_contact, r_end):
                        if np.linalg.norm(start - end) > 1e-3:
                            data.append(
                                go.Scatter3d(
                                    x=[start[0], end[0]],
                                    y=[start[1], end[1]],
                                    z=[start[2], end[2]],
                                    mode="lines",
                                    line=dict(color="red", width=5),
                                )
                            )

                if energy_name == "span_overall_scipy" and SHOW_OVERALL:
                    cache = energy_fnc.metric._cache
                    basis = cache["basis"]

                    # Show Forces
                    SHOW_FORCES = True
                    SHOW_NET_FORCE = False
                    SHOW_POINTS = True
                    SHOW_TORQUES = False
                    SHOW_NET_TORQUE = False

                    if SHOW_FORCES:
                        pass
                        # forces = cache["linear_forces"][env_id]
                        # forces = forces.view(-1, 4, 3).permute(1, 0, 2)
                        start_pts = cache["contact_pts"][env_id]
                        # end_pts = start_pts + forces * 0.05
                        # vertices = []
                        # for pairs in zip(end_pts[1:], end_pts[:-1]):
                        #     for triangle_id in range(len(start_pts)):
                        #         pt1, pt2, pt3 = (
                        #             start_pts[triangle_id],
                        #             pairs[0][triangle_id],
                        #             pairs[1][triangle_id],
                        #         )

                        #         vertices.append(torch.stack([pt1, pt2, pt3], dim=0))
                        # all_vertices = (
                        #     torch.cat(vertices, dim=1).view(-1, 3).cpu().numpy()
                        # )
                        # # all_faces = [[0,1,2], [3,4,5], ...]
                        # all_faces = [
                        #     list(range(i, i + 3))
                        #     for i in range(0, len(all_vertices), 3)
                        # ]

                        # for force in forces:
                        #     print("Drawing.....")
                        #     start = start_pts.detach().cpu().numpy() + offset
                        #     end = start - force.cpu().numpy() * 0.05 * 100
                        #     data.append(
                        #         _draw_line_plotly(start, end, color="brown", width=50)
                        #     )

                    if SHOW_NET_FORCE:
                        forces = cache["linear_forces"][env_id]
                        start = start_pts.detach().cpu().numpy() + offset
                        net_force = forces.sum(0)
                        end = start + net_force.cpu().numpy() * 0.05
                        data.append(_draw_line_plotly(start, end, color="orange", width=10))

                    if SHOW_TORQUES:
                        torques = cache["torques"][env_id]
                        for torque in torques:
                            start = object_model.cog[env_id].detach().cpu().numpy() + offset
                            end = start + torque.cpu().numpy() * 0.05
                            data.append(_draw_line_plotly(start, end, color="black"))

                    if SHOW_NET_TORQUE:
                        torques = cache["torques"][env_id]
                        start = object_model.cog[env_id].detach().cpu().numpy() + offset
                        net_torque = torques.sum(0)
                        end = start + net_torque.cpu().numpy() * 0.05
                        data.append(_draw_line_plotly(start, end, color="gray", width=10))

                    if SHOW_POINTS:
                        for point in cache["contact_pts"][env_id]:
                            point = point.detach().cpu().numpy() + offset
                            print("Showing point", point)
                            data.append(
                                go.Scatter3d(
                                    x=[point[0]],
                                    y=[point[1]],
                                    z=[point[2]],
                                    mode="markers",
                                    marker=dict(color="blue", size=15),
                                )
                            )

                    xyz = basis[env_id][0][:3].cpu().numpy() * 0.05  # vector with xyz, (,3)
                    moment = basis[env_id][0][3:].cpu().numpy() * 0.05  # vector with xyz, (,3)
                    start_pt_xyz = object_model.cog[env_id].detach().cpu().numpy() + offset
                    end_pt_xyz = start_pt_xyz + xyz
                    start_pt_moment = object_model.cog[env_id].detach().cpu().numpy() + offset
                    end_pt_moment = start_pt_moment + moment
                    data.append(_draw_line_plotly(start_pt_xyz, end_pt_xyz, color="red"))  # draw arrow pointing at xyz
                    data.append(_draw_line_plotly(start_pt_moment, end_pt_moment, color="blue"))
                    # draw arrow pointing at xyz

                elif energy_name == "span_eucledian_scipy" and SHOW_EUCLEDIAN:
                    cache = energy_fnc.metric._cache
                    basis = cache[
                        "basis"
                    ]  # Direction of the 12 basis vectors. First 6 are for forces, last 6 are for moments

                    basis_force = torch.cat([basis[env_id, :3, :3], basis[env_id, 6:9, :3]], dim=0).cpu().numpy()
                    basis_moment = torch.cat([basis[env_id, 3:6, 3:], basis[env_id, 9:, 3:]], dim=0).cpu().numpy()

                    error_force = (
                        torch.cat(
                            [
                                cache["results"][0][env_id][:3],
                                cache["results"][0][env_id][6:9],
                            ],
                            dim=0,
                        )
                        .cpu()
                        .numpy()
                    )
                    error_moment = (
                        torch.cat(
                            [
                                cache["results"][0][env_id][3:6],
                                cache["results"][0][env_id][9:],
                            ],
                            dim=0,
                        )
                        .cpu()
                        .numpy()
                    )

                    MAX_LENGTH = 0.04
                    length_force_axis = (1 - error_force * 4).clip(0, 1) * MAX_LENGTH
                    length_moment_axis = (1 - error_moment * 4).clip(0, 1) * MAX_LENGTH
                    force_axis_start = (object_model.cog[env_id].detach().cpu().numpy() + offset)[None].repeat(
                        len(length_force_axis), 0
                    ) + np.array([0, 0.05, -0.3])
                    force_axis_end = force_axis_start + length_force_axis[:, None] * basis_force
                    force_axis_end_gt = force_axis_start + 0.95 * MAX_LENGTH * basis_force

                    moment_axis_start = (object_model.cog[env_id].detach().cpu().numpy() + offset)[None].repeat(
                        len(length_moment_axis), 0
                    ) + np.array([0, -0.05, -0.3])
                    moment_axis_end = moment_axis_start + length_moment_axis[:, None] * basis_moment
                    moment_axis_end_gt = moment_axis_start + 0.95 * MAX_LENGTH * basis_moment

                    from trimesh.convex import convex_hull

                    force_hull = convex_hull(force_axis_end + np.random.randn(*force_axis_end.shape) * 1e-4)
                    force_hull_gt = convex_hull(force_axis_end_gt + np.random.randn(*force_axis_end_gt.shape) * 1e-4)
                    moment_hull = convex_hull(moment_axis_end + np.random.randn(*moment_axis_end.shape) * 1e-4)
                    moment_hull_gt = convex_hull(moment_axis_end_gt + np.random.randn(*moment_axis_end_gt.shape) * 1e-4)

                    def mesh_from_hull(hull, color="rgba(30, 30, 30, 128)", opacity=0.5):
                        return go.Mesh3d(
                            x=hull.vertices[:, 0],
                            y=hull.vertices[:, 1],
                            z=hull.vertices[:, 2],
                            i=hull.faces[:, 0],
                            j=hull.faces[:, 1],
                            k=hull.faces[:, 2],
                            opacity=opacity,
                            color=color,
                        )

                    data.append(mesh_from_hull(force_hull_gt, color="rgba(30, 30, 30, 128)", opacity=0.2))
                    data.append(mesh_from_hull(force_hull, color="rgba(255, 0, 0, 255)", opacity=1.0))
                    # Forces are red
                    data.append(mesh_from_hull(moment_hull_gt, color="rgba(30, 30, 30, 128)", opacity=0.2))
                    data.append(mesh_from_hull(moment_hull, color="rgba(0, 0, 255, 255)", opacity=1.0))
                    # Moments are blue

        all_data.append(data)

    # create plot to write to html
    output_dir = os.path.join(
        args.vis_dir,
        "objects",
        os.path.basename(os.path.dirname(dir)),
        args.hand_name,
        args.grasp_type,
        args.num_contacts,
        args.energy,
    )

    fig = go.Figure(_flatten(all_data))
    os.makedirs(output_dir, exist_ok=True)
    if args.show:
        # define axes for figure. Make sure no distortion, but to have the same aspect ratio
        lengths = args.spacing * np.sqrt(n_envs)
        fig.update_layout(
            scene=dict(aspectmode="data"),
        )
        fig.show()
        input("Press Enter to continue...")
    try:
        convert_plotly_to_gltf(fig, out_path=os.path.join(output_dir, "render.glb"))
        print("\033[94m=> Saved mesh to", os.path.join(output_dir, "render.glb"), "\033[0m")
    except Exception as e:
        print("Failed to convert to gltf:", e)

    return all_data


if __name__ == "__main__":
    arg_parser = argparse.ArgumentParser(description="Visualize hand model")
    arg_parser.add_argument("--device", type=str, default="cuda", help="device to run the model")
    arg_parser.add_argument(
        "--hand_name",
        type=str,
        default="ability_hand",
        help="name of the hand model",
        choices=AVAILABLE_HANDS + ["all"],
    )
    arg_parser.add_argument("--show_jacobian", action="store_true", help="show jacobian")
    arg_parser.add_argument("--show_joint_axes", action="store_true", help="show joint axes")
    arg_parser.add_argument("--show_penetration_points", action="store_true", help="show penetration points")
    arg_parser.add_argument("--show_occupancy_grid", action="store_true", help="show occupancy grid")
    arg_parser.add_argument("--randomize_joints", action="store_true", help="randomize joint angles")
    arg_parser.add_argument("--spacing", type=float, default=0.45, help="spacing for visualization")
    arg_parser.add_argument(
        "--grasp_type",
        type=str,
        default="default",
        help="grasp type",
        choices=["default", "pinch", "precision", "tips"],
    )
    arg_parser.add_argument(
        "--dir",
        type=str,
        default="/data/DexGraspNet/tiny/core-camera-5265ff657b9db80cafae29a76344a143/grasp_predictions",
        help="directory to save the images",
    )

    arg_parser.add_argument("--dataset", type=str, default=None, help="dataset to visualize")

    arg_parser.add_argument("--num_contacts", type=str, default="12_contacts", help="number of contacts")
    arg_parser.add_argument("--energy", type=str, default="graspqp", help="energy")
    arg_parser.add_argument(
        "--max_grasps",
        type=int,
        default=4,
        help="maximum number of grasps to visualize",
    )
    arg_parser.add_argument("--calc_energy", action="store_true", help="calculate energy")
    arg_parser.add_argument(
        "--vis_dir",
        type=str,
        default="/home/zrene/git/graspqp/_vis",
        help="directory to save visualization",
    )
    arg_parser.add_argument("--headless", action="store_true", help="run in headless mode")
    arg_parser.add_argument("--overwrite", action="store_true", help="overwrite existing files")
    arg_parser.add_argument("--num_assets", type=int, default=-1, help="number of assets to visualize")
    arg_parser.add_argument("--obj_path", type=str, default=None, help="object path")
    arg_parser.add_argument("--scale", type=float, default=1.0, help="scale of the object")
    arg_parser.add_argument("--show", action="store_true", help="show the plot")

    args = arg_parser.parse_args()
    if args.dataset is not None:
        print(f"Visualizing dataset {args.dataset}")
        print(f"Ignoring dir argument")
        # find all grasp predictions in the dataset
        data = sorted(glob.glob(f"{args.dataset}/**/grasp_predictions", recursive=True))
        if len(data) == 0:
            print(f"No grasp predictions found for path {args.dataset} and pattern {args.num_contacts}/{args.energy}")
            exit()
        print(f"Found {len(data)} grasp predictions")
        args.dir = data
    else:
        if isinstance(args.dir, str):
            args.dir = [args.dir]
    print(f"Visualizing for:")
    print("\n - ".join(args.dir))
    device = args.device

    data = []

    def _get_origin(idx, n):
        loc_x = idx % np.sqrt(n)
        loc_y = idx // np.sqrt(n)
        spacing = 0.75
        return (loc_x * spacing, loc_y * spacing)

    idx = 0
    for directory in args.dir:
        res = _show_dir(directory, args, device, origin=_get_origin(idx * 0, len(args.dir)))
        if res is not None:
            data += res
            idx += 1

        if idx >= args.num_assets and args.num_assets > 0:
            break

    output_dir = os.path.join(
        args.vis_dir,
        "hands",
        args.hand_name,
        args.num_contacts,
        args.energy,
        args.grasp_type,
    )
    os.makedirs(output_dir, exist_ok=True)
