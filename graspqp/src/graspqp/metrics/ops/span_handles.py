# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Handle-oriented GraspQP span metric.

Specialization of :class:`~graspqp.metrics.ops.span.EucledianFrictionConeSpanMetric`
for handle-style grasps: instead of probing all wrench axes, the test-wrench basis is
restricted to a small, task-relevant set of directions (default ``-z`` pull plus the
``mz``/``-mz`` torques) so the metric rewards grasps that resist the wrenches expected
when manipulating a handle. Selected via ``GraspSpanMetricFactory.MetricType.HANDLE``.
"""

import torch
import math

from graspqp.metrics.solver.qp_solver import SQPLsqSolver as LsqSolver
from graspqp.metrics.ops.span import EucledianFrictionConeSpanMetric


class HandleFrictionConeSpanMetric(EucledianFrictionConeSpanMetric):
    """Friction-cone span metric with a fixed, handle-oriented test-wrench basis.

    Args:
        solver_cls: QP / least-squares solver class used to solve the span problem.
        friction: Coulomb friction coefficient ``mu`` (defaults to ``0.2`` when
            ``None``).
        n_cone_vecs: Number of linear edges used to discretize each friction cone.
        basis_vectors: Names of the fixed test-wrench directions to span, each one of
            ``x``, ``y``, ``z`` (forces) or ``mx``, ``my``, ``mz`` (torques),
            optionally prefixed with ``-`` to flip the sign.
        weights: Per-basis weights applied to the corresponding ``basis_vectors``.
    """

    def __init__(
        self,
        solver_cls=LsqSolver,
        friction=0.2,
        n_cone_vecs=4,
        basis_vectors=["-z", "mz", "-mz"],  # , "y", "-y", "x", "-x"],
        weights=[1.0, 1.0, 1.0, 0.2, 0.2, 0.2, 0.2],
    ):
        super().__init__(solver_cls=solver_cls)

        self._mu = friction if friction is not None else 0.2
        self.n_cone_vecs = n_cone_vecs
        self._basis_vector_types = basis_vectors
        self.weights = weights
        self._max_limit_value = 5


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
        """Solve the handle span problem and return the (scaled) span energy.

        Delegates to the parent friction-cone forward pass and rescales the residual by
        50 so the handle energy is comparable in magnitude to the other metrics. Same
        arguments and return contract as
        :meth:`~graspqp.metrics.ops.span.GraspSpanMetric.forward`.
        """
        # wandb.log({"svd_scales": svd_scales}, commit=False)s
        if not return_solution:
            res, basis, svd_scales = super().forward(
                contact_pts,
                contact_normals,
                cog=cog,
                contact_threshold=contact_threshold,
                reg=reg,
                env_ids=env_ids,
                return_solution=return_solution,
                torque_weight=torque_weight,
            )
            res *= 50
            return res, basis, svd_scales
        
        res, basis, svd_scales, values = super().forward(
            contact_pts,
            contact_normals,
            cog=cog,
            contact_threshold=contact_threshold,
            reg=reg,
            env_ids=env_ids,
            return_solution=return_solution,
            torque_weight=torque_weight,
        )
        res *= 50
        return res, basis, svd_scales, values
    
    def _get_basis_vectors(self, F: torch.Tensor):
        if self._basis_vectors is None:
            vectors = []
            n_dim = F.shape[-2]

            for weight, basis_vector_type in zip(self.weights, self._basis_vector_types):
                sign = 1 * weight
                if "-" in basis_vector_type:
                    sign = -1 * weight
                    basis_vector_type = basis_vector_type[1:]
                if basis_vector_type == "x":
                    vectors.append([sign, 0, 0, 0, 0, 0]) if n_dim == 6 else vectors.append([sign, 0, 0])
                elif basis_vector_type == "y":
                    vectors.append([0, sign, 0, 0, 0, 0]) if n_dim == 6 else vectors.append([0, sign, 0])
                elif basis_vector_type == "z":
                    if n_dim == 6:
                        vectors.append([0, 0, sign, 0, 0, 0])
                    else:
                        raise ValueError("z basis vector not supported for 2D wrenches")
                elif basis_vector_type == "mx":
                    vectors.append([0, 0, 0, sign, 0, 0]) if n_dim == 6 else vectors.append([0, 0, 0])
                elif basis_vector_type == "my":
                    vectors.append([0, 0, 0, 0, sign, 0]) if n_dim == 6 else vectors.append([0, 0, 0])
                elif basis_vector_type == "mz":
                    if n_dim == 6:
                        vectors.append([0, 0, 0, 0, 0, sign])
                    else:
                        raise ValueError("mz basis vector not supported for 2D wrenches")

            self._basis_vectors = (
                torch.tensor(vectors, dtype=torch.float, device=F.device)
                .unsqueeze(0)
                # .expand(F.shape[0], -1, -1)
                .contiguous()
                .clone()
            )

        if self._basis_vectors.shape[0] != F.shape[0]:
            return self._basis_vectors.expand(F.shape[0], -1, -1).contiguous()


        return self._basis_vectors

    @property
    def n_basis_vectors(self):
        return len(self._basis_vector_types)
