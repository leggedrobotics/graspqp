import torch
from qpth.qp import QPFunction


class SQPLsqSolver:
    def __init__(self, sum_to_one=False):
        self._sum_to_one = sum_to_one
        self._qp_function = QPFunction(verbose=False, maxIter=12, eps=5e-2)

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

    def __call__(self, A, b, **kwargs):
        return self.solve(A, b, **kwargs)

    def solve(self, A, b, init=None, min_bound=-1e4, max_bound=1e4, return_solution=False, **kwargs):
        """Solving ||A*X -B|| s.t. min_bound <= X <= max_bound"""
        # self._qp_function = QPFunction(verbose=True, eps = 1e-3, maxIter=5)

        if len(kwargs) > 0:
            print("WARNING: Unknown kwargs passed to solver", SQPLsqSolver.__name__)
            print("These kwargs will be ignored:", kwargs.keys())

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
        Q += torch.eye(self._num_wrenches, device=self._device).unsqueeze(0) * 1e-4

        p = (-A.mT @ (b[..., None])).squeeze(-1)
        G = torch.cat(
            [
                torch.eye(self._num_wrenches, device=self._device),
                -torch.eye(self._num_wrenches, device=self._device),
            ],
            dim=-2,
        )
        h = torch.cat([u, -l], dim=-1)

        # Call the QP solver

        # sum condition
        if self._sum_to_one:
            eq_bound = torch.ones((1, A.shape[-1]), device=self._device, dtype=A.dtype)  # * self._num_wrenches
            eq_value = torch.ones((1,), device=self._device, dtype=A.dtype) * self._num_wrenches
            h = torch.cat([u - 1, l - 1], dim=-1)
        else:
            eq_bound = torch.Tensor()
            eq_value = torch.Tensor()

        x = self._qp_function(Q, p, G, h, eq_bound, eq_value)
        value = 0.5 * torch.sum((b - (A @ x.unsqueeze(-1)).squeeze(-1)).pow(2), -1)

        x = x.view(*batch_shape, self._num_wrenches)
        value = value.view(*batch_shape)

        if return_solution:
            return value, x

        return value
