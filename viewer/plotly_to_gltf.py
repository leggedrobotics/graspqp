# plotly_to_gltf.py
# Convert Plotly 3D traces (Scatter3d, Mesh3d) + optional custom meshes to a glTF scene.
# Usage:
#   import plotly.graph_objects as go
#   from plotly_to_gltf import convert_plotly_to_gltf, SimpleMesh
#
#   fig = go.Figure()
#   fig.add_trace(go.Scatter3d(x=[0,1,2], y=[0,1,0], z=[0,0,1], mode='lines+markers',
#                              line=dict(color='rgb(200,30,30)'),
#                              marker=dict(color='rgba(30,150,30,0.9)')))
#   fig.add_trace(go.Mesh3d(x=[0,1,0], y=[0,0,1], z=[0,0,0], i=[0], j=[1], k=[2],
#                           color='#3399ff', opacity=0.9))
#   convert_plotly_to_gltf(fig, "scene.glb")
#
# Notes/limits:
# - Scatter3d markers export as GLTF POINTS (rendered as 1px points in many viewers).
#   If you need sized spheres/billboards per point, you’d have to instance spheres instead.
# - Scatter3d "lines" become GLTF LINES (no per-trace thickness in core glTF).
# - Mesh3d expects i/j/k (explicit triangles). If omitted, we skip the mesh (or triangulate yourself).
# - Coordinates are passed through as-is (glTF is right-handed, +Y up). Add a transform if you need Z-up->Y-up.
# - Single, uniform RGBA color per trace is supported. (Per-vertex colors not included to keep it lean.)
#
# pip install numpy pygltflib plotly

from dataclasses import dataclass
from typing import List, Optional, Sequence, Union

import numpy as np

# Optional import guards keep this import light if you only build extra meshes
try:
    import plotly.graph_objects as go
except Exception:
    go = None  # type: ignore

from pygltflib import (GLTF2, Accessor, Asset, Buffer, BufferView, Material,
                       Mesh, Node, Primitive, Scene)

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
    def __init__(self, scene_name: str = "scene"):
        self.scene_name = scene_name
        self.bin = bytearray()
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
        key = tuple(np.clip(rgba, 0.0, 1.0).tolist())
        if key in self.material_cache:
            return self.material_cache[key]
        mat = Material(
            name=name or f"mat_{len(self.materials)}",
            pbrMetallicRoughness=_PBRType(baseColorFactor=list(key), metallicFactor=0.0, roughnessFactor=1.0),
            doubleSided=True,
        )
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
        gltf = GLTF2(
            asset=Asset(version="2.0"),
            scenes=[Scene(name=self.scene_name, nodes=list(range(len(self.nodes))))],
            nodes=self.nodes,
            meshes=self.meshes,
            materials=self.materials,
            buffers=[Buffer(byteLength=len(self.bin))],
            bufferViews=self.bufferViews,
            accessors=self.accessors,
        )
        # attach binary and write .glb
        try:
            # modern pygltflib
            gltf.set_binary_blob(bytes(self.bin))  # type: ignore[attr-defined]
            gltf.save_binary(out_path)
        except Exception:
            # Fallback: write external .bin + .gltf
            import os

            base, ext = os.path.splitext(out_path)
            bin_path = base + ".bin"
            with open(bin_path, "wb") as f:
                f.write(self.bin)
            gltf.buffers[0].uri = bin_path.split("/")[-1]
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
    # Needs x/y/z and i/j/k
    x = _as_np(getattr(trace, "x", []))
    y = _as_np(getattr(trace, "y", []))
    z = _as_np(getattr(trace, "z", []))
    i = getattr(trace, "i", None)
    j = getattr(trace, "j", None)
    k = getattr(trace, "k", None)
    if i is None or j is None or k is None:
        return  # skip if triangles not provided

    verts = np.column_stack([x, y, z]).astype(np.float32) * float(scale)
    faces = np.column_stack([_as_np(i), _as_np(j), _as_np(k)]).astype(np.uint32)

    rgba = _rgba_from_plotly(
        getattr(trace, "color", None),
        getattr(trace, "opacity", None),
        fallback=(0.7, 0.7, 0.8, 1.0),
    )
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
