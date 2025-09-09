import torch
import math

from graspqp.metrics.solver.qp_solver import SQPLsqSolver as LsqSolver


@torch.jit.script
def cross_product(a: torch.Tensor, b: torch.Tensor):
    """Generalized cross product to also work in 2D."""
    if a.shape[-1] == 2:
        return (a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]).unsqueeze(-1)
    else:
        return torch.linalg.cross(a, b)


class GraspSpanMetric(torch.nn.Module):
    """Module that implicitly defines a grasp span metric as the solution of an optimization problem.

    Wraps around a QP solver and builds the FC Matrix based on the contact points and contact normals.
    """

    def __init__(
        self,
        warm_start=True,
        solver_cls=LsqSolver,
        cache=True,
        max_limit=50,
        svd_gain=1.0,
    ):
        super().__init__()

        self._warm_start = warm_start
        self._solver = None
        self._last_solution = None
        self._solver_cls = solver_cls

        self._cache = {}
        self._use_cache = cache

        self._max_limit_value = max_limit
        self._svd_gain = svd_gain

    @property
    def solver(self):
        if self._solver is None:
            raise ValueError("Solver not set. Call .compile() first")
        return self._solver

    @classmethod
    def from_mat(cls, A, b, solver_cls=LsqSolver):
        metric = cls(solver_cls=solver_cls)
        metric.compile_from_mat(A, b)
        return metric

    @classmethod
    def from_dim(
        cls,
        num_wrenches,
        wrench_dim,
        batch_size=1,
        device="cuda",
        solver_cls=LsqSolver,
        **kwargs,
    ):
        solver_kwargs = kwargs
        A = torch.zeros(batch_size, wrench_dim, num_wrenches).to(device)
        b = torch.zeros(batch_size, wrench_dim).to(device)
        metric = cls(solver_cls=solver_cls)
        metric.compile_from_mat(A, b, solver_kwargs=solver_kwargs)
        return metric

    def compile(self, num_wrenches, wrench_dim, batch_size=1, device="cuda"):
        self._solver = self._solver_cls()
        self._solver.build_solver(num_wrenches, wrench_dim, batch_size=batch_size, device=device)

    def compile_from_mat(self, A, b, solver_kwargs={}):
        # expand batch dimension for solver
        self.wrench_dim = A.shape[-2]
        A_rs = A.unsqueeze(1).expand(-1, self.n_basis_vectors, -1, -1).clone()
        b_rs = b.unsqueeze(1).expand(-1, self.n_basis_vectors, -1).clone()
        self._solver = self._solver_cls.from_mat(A_rs, b_rs, solver_kwargs=solver_kwargs)
        self._solver.to(A.device)

    def get_friction_cone(self, normals: torch.Tensor):
        return normals

    def _min_limit(self):
        return 0

    def _max_limit(self):
        return self._max_limit_value

    def forward(
        self,
        contact_pts: torch.Tensor,
        contact_normals: torch.Tensor,
        cog=torch.Tensor,
        contact_threshold: float = 0.0,
        reg=0.0,
        env_ids=None,
        return_solution=True,
        torque_weight=5,
    ):

        if self._use_cache:
            self._cache["contact_pts"] = contact_pts
            self._cache["contact_normals"] = contact_normals
            self._cache["cog"] = cog
            self._cache["contact_threshold"] = contact_threshold
            self._cache["reg"] = reg
            self._cache["env_ids"] = env_ids

        # Build grasp matrix F.
        r_cog_contact = contact_pts - cog.unsqueeze(1)
        linear_forces = self.get_friction_cone(contact_normals)
        friction_cone_size = linear_forces.shape[-2] // contact_normals.shape[-2]
        r_cog_contact = r_cog_contact.repeat_interleave(friction_cone_size, dim=-2)
        torques = cross_product(r_cog_contact, linear_forces) * torque_weight
        F = torch.cat([linear_forces, torques], dim=-1).mT

        basis = self._get_basis_vectors(F)

        lower_bound = self._min_limit()
        upper_bound = self._max_limit()

        if self._use_cache:
            self._cache["linear_forces"] = linear_forces
            self._cache["torques"] = torques

        F_batch = F.unsqueeze(1).expand(-1, self.n_basis_vectors, -1, -1)

        if self._use_cache:
            self._cache["linear_forces"] = linear_forces
            self._cache["F"] = F
            self._cache["basis"] = basis
            self._cache["F_batch"] = F_batch
            self._cache["r_cog_contact"] = r_cog_contact

        init = 1.5
        if self._warm_start and self._last_solution is not None:
            init = self._last_solution.to(F.device)

            if env_ids is not None:
                init = init[env_ids]
                init = init.squeeze(1)
            if init.shape[0] != F.shape[0]:
                init = 1.5
        # call solver
        # check nan in input

        res, values = self.solver.solve(
            F_batch,
            basis,
            reg=reg,
            init=init,
            min_bound=lower_bound,
            max_bound=upper_bound,
            verbose=False,
            return_solution=True,
        )
        values = values.to(F.device)

        # svd of F
        if self._warm_start:
            if env_ids is not None:
                if self._last_solution is None:
                    self._last_solution = torch.zeros(
                        self.solver._batch_size,
                        values.shape[-2],
                        values.shape[-1],
                        device=values.device,
                    )
                if self._last_solution.device != values.device:
                    self._last_solution = self._last_solution.to(values.device)
                self._last_solution[env_ids] = values
            else:
                self._last_solution = values

        svd_scales = (torch.linalg.svdvals(F)).prod(-1).unsqueeze(-1) ** (1 / (F.shape[-2]))
        if self._use_cache:
            self._cache["results"] = (
                res,
                basis.detach().clone(),
                (-self._svd_gain * svd_scales).exp(),
                values,
            )
        # import wandb
        # wandb.log({"svd_scales": svd_scales}, commit=False)s
        if not return_solution:
            return res, basis.detach().clone(), svd_scales
        values = values.view(*values.shape[:-1], -1, friction_cone_size).sum(-1).squeeze(1)
        return res, basis.detach().clone(), svd_scales, values

    @property
    def batch_size(self):
        return self.solver.batch_size

    @property
    def n_basis_vectors(self):
        raise NotImplementedError()

    def _get_basis_vectors(self, F: torch.Tensor):
        raise NotImplementedError()


class EucledianGraspSpanMetric(GraspSpanMetric):
    def __init__(self, solver_cls=LsqSolver):
        super().__init__(solver_cls=solver_cls)
        self._basis_vectors = None

    @property
    def n_basis_vectors(self):
        return 2 * self.wrench_dim

    def _get_basis_vectors(self, F: torch.Tensor):
        if self._basis_vectors is None:
            self._basis_vectors = torch.cat([torch.eye(F.shape[-2]), -torch.eye(F.shape[-2])]).to(F.device).unsqueeze(0).expand(1, -1, -1).contiguous().clone()

        if self._basis_vectors.shape[0] != F.shape[0]:
            return self._basis_vectors.expand(F.shape[0], -1, -1).contiguous()

        return self._basis_vectors


class EucledianFrictionConeSpanMetric(EucledianGraspSpanMetric):
    def __init__(self, solver_cls=LsqSolver, friction=0.2, n_cone_vecs=4):
        super().__init__(solver_cls=solver_cls)

        self._mu = friction if friction is not None else 0.2
        self.n_cone_vecs = n_cone_vecs

    @classmethod
    def from_dim(
        cls,
        num_wrenches,
        wrench_dim,
        batch_size=1,
        device="cuda",
        solver_cls=LsqSolver,
        **kwargs,
    ):
        solver_kwargs = kwargs
        friction = solver_kwargs.pop("friction", 0.2)
        n_cone_vecs = solver_kwargs.pop("n_cone_vecs", 4)
        if len(solver_kwargs) > 0:
            print("WARNING: Unknown kwargs", solver_kwargs.keys())

        friction_cone_size = 2 if wrench_dim == 3 else n_cone_vecs
        A = torch.zeros(batch_size, wrench_dim, friction_cone_size * num_wrenches).to(device)
        b = torch.zeros(batch_size, wrench_dim).to(device)
        metric = cls(solver_cls=solver_cls, friction=friction, n_cone_vecs=n_cone_vecs)
        metric.compile_from_mat(A, b, solver_kwargs=solver_kwargs)
        return metric

    def get_friction_cone(self, normals: torch.Tensor):
        tangential_basis = []
        if normals.shape[-1] == 2:
            v_t = torch.stack([normals[..., 1], -normals[..., 0]], dim=-1)
            tangential_basis.append(self._mu * v_t + math.sqrt(1 - self._mu**2) * normals)
            tangential_basis.append(-self._mu * v_t + math.sqrt(1 - self._mu**2) * normals)
        else:
            # first basis vector
            b1 = torch.tensor([1, 1, 1], device=normals.device).view(1, 1, 3).expand(normals.shape[0], normals.shape[1], 3) / math.sqrt(3)
            # check b1 too close to normals

            dot_product = torch.sum(b1 * normals, dim=-1) * 1 / (normals.norm(dim=-1) + 1e-6)
            b1[..., 1] -= 2 * (dot_product > 0.9).float()

            v_t1 = torch.linalg.cross(normals, b1)
            v_t2 = torch.linalg.cross(normals, v_t1)
            if self.n_cone_vecs == 4:
                tangential_basis.append(self._mu * v_t1 + math.sqrt(1 - self._mu**2) * normals)
                tangential_basis.append(self._mu * v_t2 + math.sqrt(1 - self._mu**2) * normals)

                tangential_basis.append(-self._mu * v_t1 + math.sqrt(1 - self._mu**2) * normals)
                tangential_basis.append(-self._mu * v_t2 + math.sqrt(1 - self._mu**2) * normals)
            else:
                angle_step = 2 * math.pi / self.n_cone_vecs
                for i in range(self.n_cone_vecs):
                    angle = angle_step * i
                    basis = math.cos(angle) * v_t1 + math.sin(angle) * v_t2
                    tangential_basis.append(self._mu * basis + math.sqrt(1 - self._mu**2) * normals)

        bases = torch.stack(tangential_basis, dim=-2).flatten(-3, -2) / len(tangential_basis)
        return bases


class OverallFrictionConeSpanMetric(EucledianFrictionConeSpanMetric):
    def __init__(self, solver_cls=LsqSolver, friction=0.2, n_cone_vecs=4):
        super().__init__(solver_cls=solver_cls, friction=friction, n_cone_vecs=n_cone_vecs)
        self._basis_vectors = None

    def _min_limit(self):
        return 0

    @property
    def n_basis_vectors(self):
        return 1

    def _get_basis_vectors(self, F: torch.Tensor):
        return -F.sum(-1).unsqueeze(1)

    def forward(
        self,
        contact_pts: torch.Tensor,
        contact_normals: torch.Tensor,
        cog=torch.Tensor,
        contact_threshold: float = 0.0,
        reg=0.0,
        env_ids=None,
        return_solution=True,
        torque_weight=5,
    ):

        if self._use_cache:
            self._cache["contact_pts"] = contact_pts
            self._cache["contact_normals"] = contact_normals
            self._cache["cog"] = cog
            self._cache["contact_threshold"] = contact_threshold
            self._cache["reg"] = reg
            self._cache["env_ids"] = env_ids

        basis = torch.zeros(
            contact_normals.shape[0],
            self.n_basis_vectors,
            6,
            device=contact_normals.device,
        )

        # Build grasp matrix F.
        r_cog_contact = contact_pts - cog.unsqueeze(1)
        linear_forces = self.get_friction_cone(contact_normals)
        friction_cone_size = linear_forces.shape[-2] // contact_normals.shape[-2]
        r_cog_contact = r_cog_contact.repeat_interleave(friction_cone_size, dim=-2)
        torques = cross_product(r_cog_contact, linear_forces) * torque_weight
        F = torch.cat([linear_forces, torques], dim=-1).mT

        lower_bound = self._min_limit() * 0 + 1
        upper_bound = self._max_limit() + 1

        if self._use_cache:
            self._cache["linear_forces"] = linear_forces
            self._cache["torques"] = torques

        F_batch = F.unsqueeze(1).expand(-1, self.n_basis_vectors, -1, -1)

        if self._use_cache:
            self._cache["linear_forces"] = linear_forces
            self._cache["F"] = F
            self._cache["basis"] = basis
            self._cache["F_batch"] = F_batch
            self._cache["r_cog_contact"] = r_cog_contact

        init = 1.5
        if self._warm_start and self._last_solution is not None:
            init = self._last_solution.to(F.device)

            if env_ids is not None:
                init = init[env_ids]
                init = init.squeeze(1)
            if init.shape[0] != F.shape[0]:
                init = 1.5
        # call solver
        # check nan in input

        res, values = self.solver.solve(
            F_batch,
            basis,
            init=init,
            min_bound=lower_bound,
            max_bound=upper_bound,
            return_solution=True,
        )
        values = values.to(F.device)

        # svd of F
        if self._warm_start:
            if env_ids is not None:
                if self._last_solution is None:
                    self._last_solution = torch.zeros(
                        self.solver._batch_size,
                        values.shape[-2],
                        values.shape[-1],
                        device=values.device,
                    )
                if self._last_solution.device != values.device:
                    self._last_solution = self._last_solution.to(values.device)
                self._last_solution[env_ids] = values
            else:
                self._last_solution = values

        svd_scales = (torch.linalg.svdvals(F)).prod(-1).unsqueeze(-1) ** (1 / (F.shape[-2]))
        if self._use_cache:
            self._cache["results"] = (
                res,
                basis.detach().clone(),
                (-self._svd_gain * svd_scales).exp(),
                values,
            )
        # import wandb
        # wandb.log({"svd_scales": svd_scales}, commit=False)s
        if not return_solution:
            return res, basis.detach().clone(), svd_scales
        values = values.view(*values.shape[:-1], -1, friction_cone_size).sum(-1).squeeze(1)
        return res, basis.detach().clone(), svd_scales, values
