# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

import torch
import proxsuite
from proxsuite.torch.qplayer import QPFunction as ProxQPFunction

import pytest
import torch
from proxsuite.torch.qplayer import QPFunction as ProxQPFunction
import time
from qpth.qp import QPFunction
N_ITERS= 100

import os
import proxsuite
import numpy as np
from proxsuite.torch.utils import expandParam, extract_nBatch, bger

from .reluqp_impl import _RELUQP, TensorDeviceType


    
# if __name__ == "__main__":
#     # test on simple QP
#     # min 1/2 x' * H * x + g' * x
#     # s.t. l <= A * x <= u

      # qpth
      # min 1/2 x' * Q * x + p' * x
      # s.t. l <= G * x <= u

# i.e. H <= Q, g <= p, A <= G



# if __name__ == "__main__":
#     # test on simple QP
#     # min 1/2 x' * H * x + g' * x
#     # s.t. l <= A * x <= u
#     tensor_args = TensorDeviceType(dtype=torch.float32, device=torch.device("cuda", 0))
#     H = tensor_args.to_device([[6, 2, 1], [2, 5, 2], [1, 2, 4.0]])
#     # g = torch.tensor([-8.0, -3, -3], dtype=torch.double)
#     A = tensor_args.to_device([[1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
#     l = torch.tensor([3.0, 0, -10.0, -10, -10], dtype=torch.double).to(device=tensor_args.device)
#     u = tensor_args.to_device([3.0, 0, torch.inf, torch.inf, torch.inf])

#     batch = 1000
#     H = H.unsqueeze(0).repeat(batch, 1, 1)
#     A = A.unsqueeze(0).repeat(batch, 1, 1)
#     u = u.unsqueeze(0).repeat(batch, 1)
#     import time 
    
#     qp = _RELUQP(batch, H.shape[1], A.shape[1], tensor_args)
    
#     for i in range(10):
#         s = time.time()
#         x  = qp.solve(H=H, A=A, u=u, l=l)
#         print(x)
#         print(time.time() - s)

class OurProxQPFunction:
    def __init__(self, verbose=False, maxIter=12, eps=5e-2):
        self.verbose = verbose
        self.maxIter = maxIter

    def __call__(self, Q, p, A, b, G, l, u, n_threads= int( 0.75 * os.cpu_count())):
        
        nBatch = extract_nBatch(Q, p, A, b, G, l, u)
        Q, _ = expandParam(Q, nBatch, 3)
        p, _ = expandParam(p, nBatch, 2)
        G, _ = expandParam(G, nBatch, 3)
        u, _ = expandParam(u, nBatch, 3)
        l, _ = expandParam(l, nBatch, 3)
        A, _ = expandParam(A, nBatch, 3)
        b, _ = expandParam(b, nBatch, 2)

        qp_solver = _RELUQP(nBatch, Q.shape[1], G.shape[1], TensorDeviceType(dtype=Q.dtype, device=Q.device))
        import pdb; pdb.set_trace()
        
        x = qp_solver.solve(H=Q, A=G, u=u, l=l)
        import pdb; pdb.set_trace()

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
            qp.settings.eps_abs = 1e-6
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

class ForwardSQPLsqSolver:
    def __init__(self, sum_to_one=False):
        self._sum_to_one = sum_to_one
        self._qp_function = OurProxQPFunction(verbose=False, maxIter=12)

    @classmethod
    def from_mat(cls, A, b, step_size=0.15, solver_kwargs={}):
        solver = cls(
            solver_kwargs.pop("sum_to_one", False),
        )
        solver.build_solver_from_mat(A, b, step_size=step_size, solver_kwargs=solver_kwargs)
        return solver

    def build_solver_from_mat(self, A, b, step_size=0.15, solver_kwargs={}):
        if A.ndim == 2:
            A = A.unsqueeze(0)
        if b.ndim == 1:
            b = b.unsqueeze(0)

        if A.ndim == 4:
            # two batch dimensions. Lets flatten them
            batch_size = A.shape[0] * A.shape[1]
        else:
            batch_size = A.shape[0]

        self.build_solver(
            A.shape[-1],
            b.shape[-1],
            batch_size,
            device=A.device,
            step_size=step_size,
            solver_kwargs=solver_kwargs,
        )

    def to(self, device):
        self._device = device

    def build_solver(
        self,
        num_wrenches: int,
        wrench_dim: int,
        batch_size: int = 1,
        step_size=0.15,
        device="cuda",
        solver_kwargs={},
    ):
        self._num_wrenches = num_wrenches
        self._wrench_dim = wrench_dim
        self._batch_size = batch_size
        self._device = device
        self._step_size = step_size
        self._warned = False

    def __call__(self, A, b, **kwargs):
        return self.solve(A, b, **kwargs)

    def solve(self, A, b, init=None, min_bound=-1e4, max_bound=1e4, return_solution=False, **kwargs):
        """Solving ||A*X -B|| s.t. min_bound <= X <= max_bound"""
        # self._qp_function = QPFunction(verbose=True, eps = 1e-3, maxIter=5)

        if len(kwargs) > 0 and not self._warned:
            print("WARNING: Unknown kwargs passed to solver", ForwardSQPLsqSolver.__name__)
            print("These kwargs will be ignored:", kwargs.keys())
            self._warned = True

        if init is None:
            init = torch.ones((self._batch_size, self._num_wrenches))
        else:
            if isinstance(init, (int, float)):
                init = (
                    torch.tensor(init, device=self._device, dtype=A.dtype)
                    .view(1, 1)
                    .expand(self._batch_size, self._num_wrenches)
                    .clone()
                )
            if init.ndim == 1:
                init = init.unsqueeze(0).expand(self._batch_size, self._num_wrenches).clone()
            elif init.ndim == 2:
                init = init.clone().to(self._device)

        # clamp init to bounds
        init = init.clamp(min_bound + 1e-6, max_bound - 1e-6)

        batch_shape = (A.shape[0],)
        if A.ndim == 4:
            # two batch dimensions. Lets flatten them
            batch_shape = A.shape[0], A.shape[1]
            if b.shape[0] != A.shape[0]:
                b = b.expand(A.shape[0], -1, -1)

            A = A.flatten(0, 1)
            b = b.flatten(0, 1)

        u = torch.ones((A.shape[0], self._num_wrenches), device=self._device, dtype=A.dtype) * max_bound

        l = torch.ones((A.shape[0], self._num_wrenches), device=self._device, dtype=A.dtype) * min_bound

        # propare variables for sqp step
        Q = A.mT @ A
        Q += torch.eye(self._num_wrenches, device=self._device).unsqueeze(0) * 1e-3

        p = (-A.mT @ (b[..., None])).squeeze(-1)
        
        # G = torch.cat(
        #     [
        #         torch.eye(self._num_wrenches, device=self._device),
        #         -torch.eye(self._num_wrenches, device=self._device),
        #     ],
        #     dim=-2,
        # )
        # h = torch.cat([u, -l], dim=-1)
        G = torch.eye(self._num_wrenches, device=self._device)
        # Call the QP solver
        
        x = self._qp_function(Q, p, torch.Tensor(), torch.Tensor(), G, l, u)[0]
        
        value = 0.5 * torch.sum((b - (A @ x.unsqueeze(-1)).squeeze(-1)).pow(2), -1)

        x = x.view(*batch_shape, self._num_wrenches)
        value = value.view(*batch_shape)

        if return_solution:
            return value, x

        return value
