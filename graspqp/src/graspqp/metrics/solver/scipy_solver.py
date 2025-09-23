import numpy as np
import torch
from scipy.optimize import lsq_linear


def _solve_lsq(args):
    A_single, b_single, lower, upper = args
    res = lsq_linear(A_single, b_single, bounds=(lower, upper))
    return res.x, res.cost


class ScipyLsqSolver:
    def __init__(self):
        pass

    @classmethod
    def from_mat(cls, A, b, step_size=0.15, solver_kwargs={}):
        solver = cls()
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

    def __call__(self, A, b, **kwargs):
        return self.solve(A, b, **kwargs)

    def solve(
        self,
        A,
        b,
        reg=0.0,
        init=None,
        min_bound=-1e4,
        max_bound=1e4,
        return_solution=False,
        verbose=False,
    ):

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

        A_np = A.detach().cpu().numpy().astype(float)
        b_np = b.detach().cpu().numpy().astype(float)
        bounds = (l.cpu().numpy().astype(float), u.cpu().numpy().astype(float))
        solutions = []
        values = []

        for idx, (A_single, b_single) in enumerate(zip(A_np, b_np)):
            # solve with scipy
            res = lsq_linear(A_single, b_single, bounds=(bounds[0][idx], bounds[1][idx]))
            solutions.append(res.x)
            values.append(res.cost)

        solutions = torch.from_numpy(np.stack(solutions)).to(device=self._device, dtype=A.dtype)
        values = torch.from_numpy(np.stack(values)).to(device=self._device, dtype=A.dtype)
        x = solutions.view(*batch_shape, self._num_wrenches)
        value = values.view(*batch_shape)

        if return_solution:
            return value, x

        return value
