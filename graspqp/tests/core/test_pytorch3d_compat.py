# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""
Tests for graspqp.core.pytorch3d_compat.

Validates all three public symbols (Meshes, sample_points_from_meshes,
sample_farthest_points) regardless of which backend is active.  Each test
also exercises the native fallback implementations directly so they are
covered even when pytorch3d or pytorch_cluster is installed.
"""

import importlib.util
from pathlib import Path

import pytest
import torch

# Import the module directly to avoid graspqp/core/__init__.py, which pulls in
# TorchSDF (a C extension that may not be compiled for the current PyTorch ABI).
_MODULE_PATH = Path(__file__).parents[2] / "src" / "graspqp" / "core" / "pytorch3d_compat.py"
_spec = importlib.util.spec_from_file_location("pytorch3d_compat", _MODULE_PATH)
compat = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(compat)

_NativeMeshes = compat._NativeMeshes
_native_sample_farthest_points = compat._native_sample_farthest_points
_native_sample_points_from_meshes = compat._native_sample_points_from_meshes
# None when torch_cluster is not installed (tests are skipped in that case).
# The symbol only exists on the module when the pytorch3d import fails and the
# code falls back to the pytorch_cluster / native tiers, so access it defensively.
_cluster_sample_farthest_points = getattr(compat, "_cluster_sample_farthest_points", None)


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_unit_triangle_mesh(device=DEVICE):
    """Single triangle mesh with one face for deterministic area tests."""
    verts = torch.tensor([[[0.0, 0.0, 0.0],
                           [1.0, 0.0, 0.0],
                           [0.0, 1.0, 0.0]]], device=device)  # (1, 3, 3)
    faces = torch.tensor([[[0, 1, 2]]], device=device)         # (1, 1, 3)
    return verts, faces


def _make_cube_mesh(device=DEVICE):
    """Simple cube mesh with 8 verts and 12 triangles."""
    verts = torch.tensor([[
        [0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0],
        [0, 0, 1], [1, 0, 1], [1, 1, 1], [0, 1, 1],
    ]], dtype=torch.float32, device=device)  # (1, 8, 3)
    faces = torch.tensor([[
        [0, 1, 2], [0, 2, 3],
        [4, 5, 6], [4, 6, 7],
        [0, 1, 5], [0, 5, 4],
        [2, 3, 7], [2, 7, 6],
        [0, 3, 7], [0, 7, 4],
        [1, 2, 6], [1, 6, 5],
    ]], device=device)  # (1, 12, 3)
    return verts, faces


# ---------------------------------------------------------------------------
# Active-backend tests (whatever tier was resolved at import time)
# ---------------------------------------------------------------------------

class TestPublicAPI:

    def test_meshes_constructor(self):
        verts, faces = _make_cube_mesh()
        mesh = compat.Meshes(verts=verts, faces=faces)
        assert mesh is not None

    @pytest.mark.parametrize("num_samples", [1, 16, 256])
    def test_sample_points_from_meshes_shape(self, num_samples):
        verts, faces = _make_cube_mesh()
        mesh = compat.Meshes(verts=verts, faces=faces)
        pts = compat.sample_points_from_meshes(mesh, num_samples)
        assert pts.shape == (1, num_samples, 3), pts.shape

    def test_sample_points_from_meshes_device(self):
        verts, faces = _make_cube_mesh()
        mesh = compat.Meshes(verts=verts, faces=faces)
        pts = compat.sample_points_from_meshes(mesh, 64)
        assert str(pts.device).startswith(DEVICE)

    @pytest.mark.parametrize("B,N,K", [(1, 64, 8), (4, 128, 16), (2, 50, 50)])
    def test_sample_farthest_points_shape(self, B, N, K):
        points = torch.rand(B, N, 3, device=DEVICE)
        sampled_pts, sampled_idx = compat.sample_farthest_points(points, K)
        assert sampled_pts.shape == (B, K, 3), sampled_pts.shape
        assert sampled_idx.shape == (B, K), sampled_idx.shape

    def test_sample_farthest_points_indices_valid(self):
        B, N, K = 3, 100, 10
        points = torch.rand(B, N, 3, device=DEVICE)
        _, idx = compat.sample_farthest_points(points, K)
        assert (idx >= 0).all() and (idx < N).all()

    def test_sample_farthest_points_deterministic_start(self):
        """With random_start_point=False both calls should yield the same result."""
        points = torch.rand(2, 64, 3, device=DEVICE)
        pts1, idx1 = compat.sample_farthest_points(points, 8, random_start_point=False)
        pts2, idx2 = compat.sample_farthest_points(points, 8, random_start_point=False)
        assert torch.equal(idx1, idx2)


# ---------------------------------------------------------------------------
# Native-fallback tests (always run regardless of installed backends)
# ---------------------------------------------------------------------------

class TestNativeFallback:

    def test_meshes_stores_tensors(self):
        verts, faces = _make_unit_triangle_mesh()
        mesh = _NativeMeshes(verts=verts, faces=faces)
        assert torch.equal(mesh._verts, verts)
        assert torch.equal(mesh._faces, faces)

    def test_sample_points_shape(self):
        verts, faces = _make_cube_mesh()
        mesh = _NativeMeshes(verts=verts, faces=faces)
        pts = _native_sample_points_from_meshes(mesh, 128)
        assert pts.shape == (1, 128, 3)

    def test_sample_points_degenerate_mesh(self):
        """Zero-area mesh should return a zero tensor without error."""
        verts = torch.zeros(1, 3, 3, device=DEVICE)
        faces = torch.zeros(1, 1, 3, dtype=torch.long, device=DEVICE)
        mesh = _NativeMeshes(verts=verts, faces=faces)
        pts = _native_sample_points_from_meshes(mesh, 8)
        assert pts.shape == (1, 8, 3)
        assert pts.abs().max() == 0.0

    def test_sample_points_on_surface(self):
        """Sampled points must lie inside the triangle (z == 0 for unit triangle)."""
        verts, faces = _make_unit_triangle_mesh(device="cpu")
        mesh = _NativeMeshes(verts=verts, faces=faces)
        pts = _native_sample_points_from_meshes(mesh, 512)
        # All z coords should be (nearly) zero
        assert pts[0, :, 2].abs().max() < 1e-5
        # All x,y >= 0 and x+y <= 1
        x, y = pts[0, :, 0], pts[0, :, 1]
        assert (x >= -1e-5).all()
        assert (y >= -1e-5).all()
        assert ((x + y) <= 1.0 + 1e-5).all()

    @pytest.mark.parametrize("B,N,K", [(1, 32, 4), (3, 50, 10)])
    def test_fps_shape(self, B, N, K):
        points = torch.rand(B, N, 3, device=DEVICE)
        pts, idx = _native_sample_farthest_points(points, K)
        assert pts.shape == (B, K, 3)
        assert idx.shape == (B, K)

    def test_fps_indices_valid(self):
        B, N, K = 2, 64, 12
        points = torch.rand(B, N, 3, device=DEVICE)
        _, idx = _native_sample_farthest_points(points, K)
        assert (idx >= 0).all() and (idx < N).all()

    def test_fps_sampled_points_match_input(self):
        """Returned points must exactly match points[b, idx[b]]."""
        B, N, K = 2, 40, 8
        points = torch.rand(B, N, 3, device=DEVICE)
        pts, idx = _native_sample_farthest_points(points, K, random_start_point=False)
        expected = points[torch.arange(B, device=DEVICE).unsqueeze(1), idx]
        assert torch.allclose(pts, expected)

    def test_fps_spread(self):
        """FPS on a grid should pick points that are spread apart."""
        # 10 evenly spaced points on a line; K=3 should span the range
        N = 10
        points = torch.linspace(0, 1, N).reshape(1, N, 1).expand(1, N, 3).clone()
        pts, _ = _native_sample_farthest_points(points, K=3, random_start_point=False)
        # Min pairwise distance should be > 0.3 (spread > 1/3 of range)
        p = pts[0]  # (3, 3)
        dists = torch.pdist(p)
        assert dists.min() > 0.3


# ---------------------------------------------------------------------------
# pytorch_cluster FPS tests (skipped when torch_cluster is not installed)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def cluster_fps():
    if _cluster_sample_farthest_points is None:
        pytest.skip("torch_cluster not installed")
    return _cluster_sample_farthest_points


class TestClusterFallback:

    @pytest.mark.parametrize("B,N,K", [(1, 64, 8), (4, 128, 16), (2, 50, 50)])
    def test_shape(self, cluster_fps, B, N, K):
        points = torch.rand(B, N, 3, device=DEVICE)
        pts, idx = cluster_fps(points, K)
        assert pts.shape == (B, K, 3), pts.shape
        assert idx.shape == (B, K), idx.shape

    def test_indices_valid(self, cluster_fps):
        B, N, K = 3, 100, 10
        points = torch.rand(B, N, 3, device=DEVICE)
        _, idx = cluster_fps(points, K)
        assert (idx >= 0).all() and (idx < N).all()

    def test_sampled_points_match_input(self, cluster_fps):
        """Returned points must equal points[b, idx[b]] exactly."""
        B, N, K = 2, 64, 8
        points = torch.rand(B, N, 3, device=DEVICE)
        pts, idx = cluster_fps(points, K)
        expected = points[torch.arange(B, device=DEVICE).unsqueeze(1), idx]
        assert torch.allclose(pts, expected)

    def test_spread(self, cluster_fps):
        """FPS should return well-spread points, not clustered near one end."""
        N = 20
        points = torch.linspace(0, 1, N).reshape(1, N, 1).expand(1, N, 3).clone()
        pts, _ = cluster_fps(points, K=4)
        dists = torch.pdist(pts[0])
        assert dists.min() > 0.2

    def test_random_vs_fixed_start_differ(self, cluster_fps):
        """random_start_point=True should (usually) produce different results."""
        torch.manual_seed(0)
        points = torch.rand(1, 128, 3, device=DEVICE)
        _, idx_fixed = cluster_fps(points, K=10, random_start_point=False)
        results = {tuple(cluster_fps(points, K=10, random_start_point=True)[1][0].tolist())
                   for _ in range(10)}
        # At least one random run should differ from the fixed-start result
        fixed_tuple = tuple(idx_fixed[0].tolist())
        assert len(results) > 1 or fixed_tuple not in results

    def test_agrees_with_native(self, cluster_fps):
        """Both implementations must select indices that are a subset of valid points."""
        B, N, K = 2, 64, 8
        points = torch.rand(B, N, 3, device=DEVICE)
        _, c_idx = cluster_fps(points, K, random_start_point=False)
        _, n_idx = _native_sample_farthest_points(points, K, random_start_point=False)
        # Indices need not be identical (different tie-breaking), but both must be valid
        assert (c_idx >= 0).all() and (c_idx < N).all()
        assert (n_idx >= 0).all() and (n_idx < N).all()
