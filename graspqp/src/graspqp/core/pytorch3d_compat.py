# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""
Thin compatibility layer for pytorch3d operations.

Priority order for each operation:
  1. pytorch3d  — CUDA-accelerated, preferred.
  2. pytorch_cluster — GPU-accelerated FPS (sample_farthest_points only;
     mesh sampling falls through to the native fallback).
  3. Pure-PyTorch fallbacks — correct but slower.

Public API
----------
Meshes
    Minimal mesh container matching pytorch3d.structures.Meshes.
sample_points_from_meshes(mesh, num_samples, return_normals=False)
    Uniform surface sampling weighted by triangle area.
    Returns Tensor(1, N, 3), or tuple (Tensor(1,N,3), Tensor(1,N,3)) with normals.
sample_farthest_points(points, K, random_start_point=True) -> (Tensor(B,K,3), Tensor(B,K))
    Greedy furthest-point sampling.
"""

import warnings
import torch

# ---------------------------------------------------------------------------
# Native (pure-PyTorch) fallback implementations
# ---------------------------------------------------------------------------

class _NativeMeshes:
    """Minimal mesh container (pytorch3d.structures.Meshes drop-in)."""

    def __init__(self, verts: torch.Tensor, faces: torch.Tensor):
        # verts: (1, V, 3), faces: (1, F, 3) — may be float or long
        self._verts = verts
        self._faces = faces


def _native_sample_points_from_meshes(
    mesh, num_samples: int, return_normals: bool = False
):
    """Sample *num_samples* points uniformly from the mesh surface.

    Returns ``(1, num_samples, 3)`` points, or a tuple
    ``((1, N, 3), (1, N, 3))`` when *return_normals* is True.
    """
    verts = mesh._verts[0]          # (V, 3)
    faces = mesh._faces[0].long()   # (F, 3)
    device = verts.device

    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    cross = torch.linalg.cross(v1 - v0, v2 - v0)
    areas = cross.norm(dim=-1).mul(0.5).clamp(min=0.0)
    total = areas.sum()
    if total == 0:
        pts = torch.zeros(1, num_samples, 3, device=device, dtype=verts.dtype)
        if return_normals:
            return pts, torch.zeros_like(pts)
        return pts

    face_ids = torch.multinomial(areas / total, num_samples, replacement=True)

    r = torch.rand(num_samples, 2, device=device, dtype=verts.dtype)
    sqrt_r0 = r[:, 0].sqrt()
    u = 1.0 - sqrt_r0
    v = sqrt_r0 * (1.0 - r[:, 1])
    w = sqrt_r0 * r[:, 1]

    points = (
        u.unsqueeze(1) * verts[faces[face_ids, 0]]
        + v.unsqueeze(1) * verts[faces[face_ids, 1]]
        + w.unsqueeze(1) * verts[faces[face_ids, 2]]
    )
    if return_normals:
        face_normals = cross / cross.norm(dim=-1, keepdim=True).clamp(min=1e-8)
        normals = face_normals[face_ids].unsqueeze(0)  # (1, num_samples, 3)
        return points.unsqueeze(0), normals
    return points.unsqueeze(0)  # (1, num_samples, 3)


def _native_sample_farthest_points(
    points: torch.Tensor, K: int, lengths: torch.Tensor = None, random_start_point: bool = True
):
    """Greedy furthest-point sampling.

    Args:
        points: ``(B, N, 3)`` input point cloud.
        K: Number of points to sample.
        lengths: Optional ``(B,)`` number of valid points per batch element (for padded
            clouds); points at index ``>= lengths[b]`` are never selected.
        random_start_point: If ``True`` the first seed is chosen randomly;
            otherwise index 0 is used.

    Returns:
        Tuple ``(sampled_points, sampled_indices)`` with shapes
        ``(B, K, 3)`` and ``(B, K)``.
    """
    B, N, _ = points.shape
    device = points.device
    if lengths is None:
        lengths = torch.full((B,), N, device=device, dtype=torch.long)
    valid = torch.arange(N, device=device).unsqueeze(0) < lengths.unsqueeze(1)  # (B, N)

    sampled_idx = torch.zeros(B, K, dtype=torch.long, device=device)
    if random_start_point:
        for b in range(B):
            sampled_idx[b, 0] = torch.randint(0, int(max(1, lengths[b].item())), (1,), device=device)

    # invalid (padded) points get dist -1 so argmax never picks them
    dist = torch.where(valid, torch.full((B, N), float("inf"), device=device, dtype=points.dtype),
                       torch.full((B, N), -1.0, device=device, dtype=points.dtype))

    for k in range(1, K):
        last = points[torch.arange(B, device=device), sampled_idx[:, k - 1]]  # (B, 3)
        d = ((points - last.unsqueeze(1)) ** 2).sum(-1)  # (B, N)
        dist = torch.where(valid, torch.minimum(dist, d), dist)
        sampled_idx[:, k] = dist.argmax(dim=-1)

    sampled_pts = points[torch.arange(B, device=device).unsqueeze(1), sampled_idx]
    return sampled_pts, sampled_idx


def _native_ball_query(p1, p2, K: int, radius: float, return_nn: bool = True):
    """Pure-PyTorch drop-in for ``pytorch3d.ops.ball_query``.

    For each point in ``p1`` finds up to ``K`` points of ``p2`` within ``radius``; returns
    ``(dists, idx, nn)`` with squared distances, indices (``-1`` where fewer than ``K`` were
    found), and the gathered neighbour points — matching pytorch3d's semantics/order-independent
    use for neighbour counting.

    Cost is one ``(B, N, M)`` distance matrix + ``topk`` per call. Distances are computed with a
    single matmul (``|a-b|^2 = |a|^2 + |b|^2 - 2 a.b``) — no sqrt — and the neighbour tensor is
    only materialised when ``return_nn`` is set (density callers pass ``return_nn=False``). For
    heavy use, install pytorch3d: the compat shim then dispatches to its fused CUDA kernel.

    Args:
        p1: ``(B, N, D)`` query points.
        p2: ``(B, M, D)`` reference points.
    """
    B, N, D = p1.shape
    M = p2.shape[1]
    # squared distances via matmul (avoids cdist's sqrt); clamp fp round-off to >= 0
    d2 = (p1.pow(2).sum(-1, keepdim=True) + p2.pow(2).sum(-1).unsqueeze(-2)
          - 2.0 * (p1 @ p2.transpose(-1, -2))).clamp_min_(0.0)  # (B, N, M)
    d2 = d2.masked_fill(d2 > radius * radius, float("inf"))
    Kc = min(K, M)
    dists, idx = d2.topk(Kc, dim=-1, largest=False)  # (B, N, Kc)
    missing = torch.isinf(dists)
    idx = torch.where(missing, torch.full_like(idx, -1), idx)
    dists = torch.where(missing, torch.zeros_like(dists), dists)
    nn = None
    if return_nn:
        idx_safe = idx.clamp(min=0)
        gathered = torch.gather(
            p2.unsqueeze(1).expand(B, N, M, D), 2, idx_safe.unsqueeze(-1).expand(B, N, Kc, D)
        )
        nn = torch.where(missing.unsqueeze(-1), torch.zeros_like(gathered), gathered)
    return dists, idx, nn


class _NativePointclouds:
    """Minimal ``pytorch3d.structures.Pointclouds`` drop-in (padding utility only)."""

    def __init__(self, points, features=None):
        self._points = list(points)
        self._features = list(features) if features is not None else None

    def points_padded(self):
        return torch.nn.utils.rnn.pad_sequence(self._points, batch_first=True)

    def features_padded(self):
        if self._features is None:
            return None
        return torch.nn.utils.rnn.pad_sequence(self._features, batch_first=True)


# ---------------------------------------------------------------------------
# Tier 1 — pytorch3d (preferred)
# ---------------------------------------------------------------------------

try:
    import pytorch3d.ops as _p3d_ops
    import pytorch3d.structures as _p3d_structs

    Meshes = _p3d_structs.Meshes
    Pointclouds = _p3d_structs.Pointclouds

    def sample_points_from_meshes(mesh, num_samples: int, return_normals: bool = False):
        return _p3d_ops.sample_points_from_meshes(
            mesh, num_samples=num_samples, return_normals=return_normals
        )

    def sample_farthest_points(
        points: torch.Tensor, K: int, lengths: torch.Tensor = None, random_start_point: bool = True
    ):
        return _p3d_ops.sample_farthest_points(
            points, lengths=lengths, K=K, random_start_point=random_start_point
        )

    def ball_query(p1, p2, K: int, radius: float, return_nn: bool = True):
        return _p3d_ops.ball_query(p1, p2, K=K, radius=radius, return_nn=return_nn)

except Exception:  # ImportError or undefined-symbol OSError
    warnings.warn(
        "pytorch3d not available — trying pytorch_cluster for FPS, "
        "native PyTorch for mesh sampling.",
        ImportWarning,
        stacklevel=2,
    )

    # Meshes/Pointclouds, mesh sampling and ball_query always use the native fallback.
    Meshes = _NativeMeshes  # type: ignore[misc]
    Pointclouds = _NativePointclouds  # type: ignore[misc]
    sample_points_from_meshes = _native_sample_points_from_meshes  # type: ignore[assignment]
    ball_query = _native_ball_query  # type: ignore[assignment]

    # -------------------------------------------------------------------
    # Tier 2 — pytorch_cluster for FPS
    # -------------------------------------------------------------------
    try:
        import torch_cluster as _tc

        def _cluster_sample_farthest_points(
            points: torch.Tensor, K: int, lengths: torch.Tensor = None, random_start_point: bool = True
        ):
            # torch_cluster.fps has no padded-length support; use the native path when lengths given.
            if lengths is not None:
                return _native_sample_farthest_points(
                    points, K=K, lengths=lengths, random_start_point=random_start_point
                )
            """FPS via torch_cluster.fps (GPU-accelerated).

            torch_cluster.fps works on a flat point list with a batch index
            vector.  We adapt its interface to match the pytorch3d signature
            ``(B, N, 3) -> ((B, K, 3), (B, K))``.

            Note: torch_cluster returns ``ceil(ratio * N)`` points per batch
            element; we truncate or pad to exactly K so the output shape is
            always deterministic.
            """
            B, N, C = points.shape
            device = points.device

            ratio = K / N
            x_flat = points.reshape(B * N, C)
            batch_vec = torch.arange(B, device=device).repeat_interleave(N)

            # Returns global indices into x_flat, length ≈ B*K
            global_idx = _tc.fps(x_flat, batch_vec, ratio=ratio, random_start=random_start_point)

            # Split into per-batch buckets and convert to local indices
            local_idx_list = []
            for b in range(B):
                mask = batch_vec[global_idx] == b
                loc = global_idx[mask] - b * N  # local index within batch b
                n_got = loc.shape[0]
                if n_got >= K:
                    loc = loc[:K]
                else:  # pad by repeating the last index
                    pad = loc[-1:].expand(K - n_got)
                    loc = torch.cat([loc, pad])
                local_idx_list.append(loc)

            sampled_idx = torch.stack(local_idx_list, dim=0)  # (B, K)
            sampled_pts = points[torch.arange(B, device=device).unsqueeze(1), sampled_idx]
            return sampled_pts, sampled_idx

        sample_farthest_points = _cluster_sample_farthest_points  # type: ignore[assignment]

    except Exception:
        warnings.warn(
            "pytorch_cluster not available — using slower pure-PyTorch FPS fallback.",
            ImportWarning,
            stacklevel=2,
        )
        _cluster_sample_farthest_points = None  # type: ignore[assignment]
        sample_farthest_points = _native_sample_farthest_points  # type: ignore[assignment]
