# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Cross-backend SDF value/gradient consistency test for ``ObjectModel.cal_distance``.

graspqp has three SDF backends (selected by the ``SDF_BACKEND`` env var, read at import time):
``WARP`` (warp mesh query + custom autograd ``CalcSdfField``), ``TORCHSDF`` (torchsdf CUDA op) and
``KAOLIN`` (``point_to_mesh_distance`` + ``check_sign``). This test runs the SAME mesh and the SAME
query points through each and checks that

  * the forward SIGNED distances agree, and
  * the per-query-point GRADIENTS point the same way (cosine ~= 1),

so that whichever backend a run uses, the gradient the optimizer sees is the same.

Because ``SDF_BACKEND`` is fixed at import, the driver re-spawns this script once per backend
(each child dumps its ``(dist, grad)``), then compares. Backends that aren't installed are skipped.

Run (needs a mesh dataset + CUDA), e.g. in the ``graspqp`` image with the patched source mounted::

    python graspqp/scripts/test_sdf_backends.py --data_root /data/cleaned_handles --object 00316047

Exit code 0 = all compared pairs agree; 1 = a mismatch (or <2 usable backends).
"""

from __future__ import annotations

import argparse
import itertools
import os
import subprocess
import sys
import tempfile

import torch
import torch.nn.functional as F

BACKENDS = ["WARP", "TORCHSDF", "KAOLIN"]


def _run_worker(args) -> None:
    """Child process: one backend (from ``SDF_BACKEND``), dump forward dist + query gradient + timings."""
    import time

    from graspqp.core import ObjectModel

    be = os.environ["SDF_BACKEND"].upper()
    dev = "cuda:0"
    om = ObjectModel(data_root_path=args.data_root, batch_size_each=1, device=dev)
    om.initialize([args.object], sdf_library=be, use_winding_number=args.use_winding_number)

    g = torch.Generator(device=dev).manual_seed(args.seed)
    x = ((torch.rand(1, args.n, 3, generator=g, device=dev) - 0.5) * args.scale).requires_grad_(True)
    dist, normal = om.cal_distance(x)  # (1, n) signed, (1, n, 3)
    dist.sum().backward()

    # --- speed test: forward-only and forward+backward, CUDA-synced, warmup-excluded ---
    cuda = dev.startswith("cuda")

    def _sync():
        if cuda:
            torch.cuda.synchronize()

    def _time(fn, iters):
        for _ in range(args.warmup):
            fn()
        _sync()
        t0 = time.perf_counter()
        for _ in range(iters):
            fn()
        _sync()
        return (time.perf_counter() - t0) / iters * 1e3  # ms/iter

    def _fwd():
        with torch.no_grad():
            om.cal_distance(x)

    def _fwd_bwd():
        xx = x.detach().requires_grad_(True)
        om.cal_distance(xx)[0].sum().backward()

    t_fwd = _time(_fwd, args.iters)
    t_fb = _time(_fwd_bwd, args.iters)

    torch.save(
        {"dist": dist.detach().cpu(), "grad": x.grad.detach().cpu(), "normal": normal.detach().cpu(),
         "t_fwd_ms": t_fwd, "t_fb_ms": t_fb},
        args.out,
    )
    print(f"[{be}] dist[min,max]=[{float(dist.min()):.4f},{float(dist.max()):.4f}] "
          f"grad|mean|={float(x.grad.abs().mean()):.5f} | fwd={t_fwd:.3f}ms fwd+bwd={t_fb:.3f}ms")


def _compare(results: dict, dist_tol: float, grad_cos_tol: float) -> bool:
    ok = True
    for a, b in itertools.combinations(results, 2):
        da, db = results[a], results[b]
        dd = (da["dist"] - db["dist"]).abs().max().item()
        ga, gb = da["grad"].reshape(-1, 3), db["grad"].reshape(-1, 3)
        gcos = (F.normalize(ga, dim=-1) * F.normalize(gb, dim=-1)).sum(-1)
        # ignore near-zero gradients (missed queries / medial-axis singularities) in the cos check
        valid = torch.minimum(ga.norm(dim=-1), gb.norm(dim=-1)) > 1e-4
        gcos_min = float(gcos[valid].min()) if valid.any() else 1.0
        passed = (dd <= dist_tol) and (gcos_min >= grad_cos_tol)
        ok &= passed
        print(f"  {a:8s} vs {b:8s}:  dist_maxdiff={dd:.6f}  grad_cos_min={gcos_min:.4f}  "
              f"({int(valid.sum())}/{valid.numel()} pts)  -> {'PASS' if passed else 'FAIL'}")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_root", default="/data/cleaned_handles")
    ap.add_argument("--object", default="00316047")
    ap.add_argument("--n", type=int, default=256, help="number of query points")
    ap.add_argument("--scale", type=float, default=0.20, help="query points sampled in [-scale/2, scale/2]^3")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dist_tol", type=float, default=1e-4)
    ap.add_argument("--grad_cos_tol", type=float, default=0.999)
    ap.add_argument("--iters", type=int, default=100, help="timed iterations for the speed test")
    ap.add_argument("--warmup", type=int, default=10, help="warmup iterations (excluded from timing)")
    ap.add_argument("--use_winding_number", action="store_true",
                    help="(WARP) build meshes with winding-number support and use it for the sign")
    ap.add_argument("--out", default=None, help=argparse.SUPPRESS)  # set => worker mode
    args = ap.parse_args()

    if args.out is not None:
        _run_worker(args)
        return

    tmp = tempfile.mkdtemp()
    results: dict = {}
    for be in BACKENDS:
        out = os.path.join(tmp, f"{be}.pt")
        cmd = [sys.executable, os.path.abspath(__file__),
               "--data_root", args.data_root, "--object", args.object,
               "--n", str(args.n), "--scale", str(args.scale), "--seed", str(args.seed),
               "--iters", str(args.iters), "--warmup", str(args.warmup), "--out", out]
        if args.use_winding_number:
            cmd.append("--use_winding_number")
        r = subprocess.run(cmd, env={**os.environ, "SDF_BACKEND": be}, capture_output=True, text=True)
        if r.returncode == 0 and os.path.exists(out):
            results[be] = torch.load(out, map_location="cpu")
            print(r.stdout.strip().splitlines()[-1] if r.stdout.strip() else f"[{be}] ok")
        else:
            last = (r.stderr.strip().splitlines() or ["no output"])[-1]
            print(f"[{be}] SKIP/FAIL: {last}")

    # --- speed table (relative to the fastest fwd+bwd) ---
    if results:
        fastest = min(r["t_fb_ms"] for r in results.values())
        print(f"\nspeed  (n={args.n} query pts, {args.iters} iters, warmup {args.warmup}):")
        print(f"  {'backend':10s} {'fwd (ms)':>10s} {'fwd+bwd (ms)':>14s} {'vs fastest':>12s}")
        for be, r in sorted(results.items(), key=lambda kv: kv[1]["t_fb_ms"]):
            print(f"  {be:10s} {r['t_fwd_ms']:>10.3f} {r['t_fb_ms']:>14.3f} {r['t_fb_ms']/fastest:>11.2f}x")

    print(f"\ncompared backends: {list(results)}")
    if len(results) < 2:
        print("Need >= 2 usable backends to compare.")
        sys.exit(1)
    ok = _compare(results, args.dist_tol, args.grad_cos_tol)
    print("\nALL PASS" if ok else "\nMISMATCH")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
