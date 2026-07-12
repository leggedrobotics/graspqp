# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

import os

import pytest

pytest.importorskip("proxsuite")  # ProxQP is an optional dependency

# These are ProxQP-vs-qpth benchmarks that replay a saved QP problem. The debug
# artifact is not shipped with the repository, so skip the module when it is absent.
if not os.path.exists("qp_debug.pt"):
    pytest.skip("qp_debug.pt debug artifact not shipped", allow_module_level=True)

import torch
from proxsuite.torch.qplayer import QPFunction as ProxQPFunction
import time
from qpth.qp import QPFunction

N_ITERS= 10

import os
import proxsuite
import numpy as np
from proxsuite.torch.utils import expandParam, extract_nBatch, bger

class OurProxQPFunction:
    def __init__(self, verbose=False, maxIter=12, eps=5e-2):
        self.verbose = verbose
        self.maxIter = maxIter
        self.eps = eps

    def __call__(self, Q, p, A, b, G, l, u, n_threads= int( 0.75 * os.cpu_count())):
        self.vector_of_qps = proxsuite.proxqp.dense.BatchQP()
        
        nBatch = extract_nBatch(Q, p, A, b, G, l, u)
        Q, _ = expandParam(Q, nBatch, 3)
        p, _ = expandParam(p, nBatch, 2)
        G, _ = expandParam(G, nBatch, 3)
        u, _ = expandParam(u, nBatch, 2)
        l, _ = expandParam(l, nBatch, 2)
        A, _ = expandParam(A, nBatch, 3)
        b, _ = expandParam(b, nBatch, 2)
        
        Q_np, p_np, A_np, b_np, G_np, l_np, u_np = (
            Q.cpu().numpy(),
            p.cpu().numpy(),
            A.cpu().numpy(),
            b.cpu().numpy(),
            G.cpu().numpy(),
            l.cpu().numpy(),
            u.cpu().numpy(),
        )
    
        _, nineq, nz = G.size()
        neq = A.size(1) if A.nelement() > 0 else 0
    
        for i in range(nBatch):
            qp = self.vector_of_qps.init_qp_in_place(nz, neq, nineq)
            qp.settings.primal_infeasibility_solving = False
            qp.settings.max_iter = self.maxIter
            qp.settings.max_iter_in = 100
            default_rho = 5.0e-5
            qp.settings.default_rho = default_rho
            qp.settings.refactor_rho_threshold = default_rho  # no refactorization
            qp.settings.eps_abs = self.eps
            qp.init(
                Q_np[i],
                p_np[i],
                A_np[i] if neq > 0 else None,
                b_np[i] if neq > 0 else None,
                G_np[i] if nineq > 0 else None,
                l_np[i] if nineq > 0 else None,
                u_np[i] if nineq > 0 else None,
            )
            
        proxsuite.proxqp.dense.solve_in_parallel(
            num_threads=n_threads,
            qps=self.vector_of_qps
        )
        
        # Shape is 936 x nz
        zhats = np.empty((nBatch, nz), dtype=np.float32)
        
        # nbatch is 936
        for i in range(nBatch):
            zhats[i] = self.vector_of_qps.get(i).results.x
        return torch.from_numpy(zhats).to(Q.device), None


def test_our_proxqp_solver():
    data = torch.load("qp_debug.pt")
    
    
    for eps in [1e-2, 1e-4, 1e-5, 1e-6]:
        time1 = time.time()
        qp_function = OurProxQPFunction(maxIter=12, eps=eps)
        for _ in range(N_ITERS):
            print(f"OurProxQPFunction Solve {_}/{N_ITERS}", end="\r")
            with torch.no_grad():
                qp_function(data["Q"], data["p"], torch.Tensor(), torch.Tensor(), data["G"][:data["l"].shape[-1]], data["l"], data["u"])
        total_time = time.time() - time1
        print("OurProxQPFunction Time: [", eps, "]", total_time / N_ITERS)
    
    
def test_proxqp_solver():
    data = torch.load("qp_debug.pt")
    time1 = time.time()
    qp_function = ProxQPFunction(maxIter=12, omp_parallel=True)
    for _ in range(N_ITERS):
        print(f"ProxQP Solve {_}/{N_ITERS}", end="\r")
        with torch.no_grad():
            qp_function(data["Q"], data["p"], torch.Tensor(), torch.Tensor(), data["G"][:data["l"].shape[-1]], data["l"], data["u"])
    total_time = time.time() - time1
    print("ProxQP Time:", total_time / N_ITERS)
    
def qpth_solver():
    data = torch.load("qp_debug.pt")
    time1 = time.time()
    qp_function = QPFunction()
    for _ in range(N_ITERS):
        print(f"QPTH Solve {_}/{N_ITERS}", end="\r")
        with torch.no_grad():
            qp_function(data["Q"], data["p"], data["G"], data["h"], torch.Tensor(), torch.Tensor())
    total_time = time.time() - time1
    print("QPTH Time:", total_time / N_ITERS)
    
if __name__ == "__main__":
    # pytest.main([__file__, "-s"])
    test_our_proxqp_solver()
    test_proxqp_solver()
    qpth_solver()