# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Factory and lazy-initialization wrapper for the grasp span metrics.

Defines :class:`GraspSpanMetricFactory`, the single public entry point that maps a
:class:`GraspSpanMetricFactory.MetricType` to a concrete metric module, and
:class:`SpanMetricWrapper`, which defers construction of the underlying span metric /
QP solver until the contact dimensionality is known at the first ``forward`` call.

The QP-based backends (``SQPLsqSolver`` via ``qpth``, ``ForwardSQPLsqSolver`` via
ProxQP) are optional imports; if their dependencies are missing they are set to
``None`` and only the SciPy backend remains available.
"""

from enum import Enum

import torch

from ..solver.scipy_solver import ScipyLsqSolver
from .dexgrasp import DexgraspSpanMetric
from .span import EucledianFrictionConeSpanMetric, OverallFrictionConeSpanMetric
from .tdg import TDGSpanMetric
from .span_handles import HandleFrictionConeSpanMetric

try:
    from graspqp.metrics.solver.qp_solver import SQPLsqSolver
except ImportError:
    print("Error importing SQPLsqSolver. Make sure to install proxsuite")
    SQPLsqSolver = None
    pass
try:
    from graspqp.metrics.solver.forward_qp_solver import ForwardSQPLsqSolver
except ImportError:
    print("Error importing ForwardSQPLsqSolver. Make sure to install proxsuite")
    ForwardSQPLsqSolver = None
    pass


class SpanMetricWrapper(torch.nn.Module):
    """Lazily-initialized wrapper around a GraspQP span metric.

    Holds a span-metric *class* (not instance) and the kwargs needed to build it. On
    the first :meth:`forward` call it inspects the contact tensors, instantiates the
    metric (and its QP solver) with the matching number of wrenches / wrench dimension
    via ``metric.from_dim``, then reuses it on subsequent calls. The forward output is
    the aggregated, SVD-regularized span energy (optionally with the raw solver
    solution).

    Args:
        metric: Span-metric class to instantiate (e.g.
            :class:`~graspqp.metrics.ops.span.OverallFrictionConeSpanMetric`).
        metric_kwargs: Keyword arguments forwarded to ``metric.from_dim`` at
            initialization (e.g. ``solver_cls``, ``friction``, ``max_limit``). The key
            ``use_max_aggregation`` (default ``True``) is consumed here and controls
            whether per-basis energies are summed (``True``) or averaged (``False``).
    """

    def __init__(
        self,
        metric: OverallFrictionConeSpanMetric = OverallFrictionConeSpanMetric,
        metric_kwargs: dict = {},
    ):
        super().__init__()
        self.metric = metric
        self._initialized = False
        self.metric_kwargs = metric_kwargs
        self.use_max_aggregation = metric_kwargs.pop("use_max_aggregation", True)

    def forward(
        self,
        contact_pts: torch.Tensor,
        contact_normals: torch.Tensor,
        cog=torch.Tensor,
        contact_threshold: float = 0.0,
        torque_weight: float = 5.0,
        **kwargs,
    ):
        """Compute the aggregated grasp span energy.

        Args:
            contact_pts: Contact points, shape ``(batch, n_contact, 3)`` in meters.
            contact_normals: Inward/outward contact normals, shape
                ``(batch, n_contact, 3)``.
            cog: Object center of gravity, shape ``(batch, 3)`` in meters.
            contact_threshold: Distance threshold used by the metric to gate contacts.
            torque_weight: Scaling of the torque rows of the wrench matrix relative to
                the force rows (balances translational vs. rotational resistance).
            **kwargs: Optional ``svd_gain`` (default ``0.1``) weighting the SVD
                condition penalty, ``values_gain`` (default ``2.0``) scaling the span
                energy, and ``with_solution`` (default ``False``) to also return the
                raw solver solution.

        Returns:
            torch.Tensor: Per-batch span energy of shape ``(batch,)``. If
            ``with_solution`` is set, a ``(energy, solution)`` tuple is returned
            instead.
        """
        svd_gain = kwargs.pop("svd_gain", 0.1)
        values_gain = kwargs.pop("values_gain", 2.0)
        with_solution = kwargs.pop("with_solution", False)

        if not self._initialized:
            max_limit = None
            if "max_limit" in self.metric_kwargs:
                max_limit = self.metric_kwargs.pop("max_limit")

            print(
                "[SpanMetricWrapper] Initializing metric. Passing metric_kwargs: ",
                self.metric_kwargs,
            )

            self.metric = self.metric.from_dim(
                contact_normals.shape[1],
                6,
                batch_size=contact_normals.shape[0],
                device=contact_pts.device,
                **self.metric_kwargs,
            )

            if max_limit is not None:
                print("Updating max limit")
                self.metric._max_limit_value = max_limit

            self._initialized = True

        res = self.metric(
            contact_pts,
            contact_normals,
            cog,
            contact_threshold=contact_threshold,
            return_solution=with_solution,
            torque_weight=torque_weight,
        )
        if with_solution:
            values, basis, svd_scales, x = res
        else:
            values, basis, svd_scales = res

        if self.use_max_aggregation:
            final_values = values.sum(-1)
            # final_values = values.max(-1)[0]
        else:
            final_values = values.mean(-1)
            
        eps = 1e-2
        if with_solution:
            return (
                values_gain * (final_values + eps) * (-svd_gain * svd_scales.mean(-1)).exp(),
                x,
            )

        return values_gain * (final_values + eps) * (-svd_gain * svd_scales.mean(-1)).exp()


class GraspSpanMetricFactory:
    """Factory that instantiates a grasp-quality metric from a :class:`MetricType`.

    This is the public entry point of :mod:`graspqp.metrics`. See :meth:`create`.
    """

    # enum for different metric types
    class MetricType(Enum):
        """Selects which grasp-quality metric (and QP backend) to build.

        Members:
            DEXGRASP: DexGraspNet force-closure energy (:class:`DexgraspSpanMetric`).
            TDG: Task-oriented Dexterous Grasp / GWS energy (:class:`TDGSpanMetric`).
            GRASPQP: GraspQP span metric with the ``qpth`` QP solver (default backend).
            GRASPQP_SCIPY: GraspQP span metric with the SciPy least-squares backend.
            GRASPQP_EUCLIDIAN: GraspQP with a Euclidean (±identity) test-wrench basis,
                ``qpth`` backend.
            GRASPQP_EUCLIDIAN_SCIPY: Euclidean-basis GraspQP with the SciPy backend.
            HANDLE: Handle-grasp GraspQP variant with a fixed ``-z`` test-wrench basis,
                ``qpth`` backend.
            HANDLE_SCIPY: Handle-grasp GraspQP variant with the SciPy backend.
            GRASPQP_PROXQP: GraspQP span metric with the ProxQP/forward QP backend.
            GRASPQP_EUCLIDIAN_PROXQP: Euclidean-basis GraspQP with the ProxQP backend.
        """

        DEXGRASP = 1
        TDG = 2
        GRASPQP = 3
        GRASPQP_SCIPY = 4
        GRASPQP_EUCLIDIAN_SCIPY = 5
        GRASPQP_EUCLIDIAN = 6
        HANDLE = 7
        HANDLE_SCIPY = 8
        GRASPQP_PROXQP = 9
        GRASPQP_EUCLIDIAN_PROXQP = 10

    @staticmethod
    def create(metric_type: MetricType, solver_kwargs: dict = {}):
        """Instantiate the metric module for the given :class:`MetricType`.

        For the GraspQP variants this returns a :class:`SpanMetricWrapper` configured
        with the appropriate span-metric class and QP backend; for DEXGRASP and TDG it
        returns the corresponding standalone metric module.

        Args:
            metric_type: Which metric / backend to build (see :class:`MetricType`).
            solver_kwargs: Extra keyword arguments passed through to the span metric /
                solver. Recognized keys include ``friction`` (friction coefficient)
                and ``max_limit`` (upper bound on the per-wrench force magnitudes);
                any remaining keys are forwarded to the metric's ``from_dim``.

        Returns:
            torch.nn.Module: The configured metric module.

        Raises:
            ValueError: If ``metric_type`` is not a recognized :class:`MetricType`.
        """

        if metric_type == GraspSpanMetricFactory.MetricType.DEXGRASP:
            return DexgraspSpanMetric()
        elif metric_type == GraspSpanMetricFactory.MetricType.TDG:
            return TDGSpanMetric()
        elif metric_type == GraspSpanMetricFactory.MetricType.GRASPQP:
            return SpanMetricWrapper(
                OverallFrictionConeSpanMetric,
                metric_kwargs={
                    "solver_cls": SQPLsqSolver,
                    "friction": solver_kwargs.pop("friction", None),
                    "max_limit": solver_kwargs.pop("max_limit", None),
                    **solver_kwargs,
                },
            )
        elif metric_type == GraspSpanMetricFactory.MetricType.GRASPQP_SCIPY:
            return SpanMetricWrapper(
                OverallFrictionConeSpanMetric,
                metric_kwargs={
                    "solver_cls": ScipyLsqSolver,
                    "friction": solver_kwargs.pop("friction", None),
                    "max_limit": solver_kwargs.pop("max_limit", None),
                    **solver_kwargs,
                },
            )
        elif metric_type == GraspSpanMetricFactory.MetricType.GRASPQP_EUCLIDIAN:
            return SpanMetricWrapper(
                EucledianFrictionConeSpanMetric,
                metric_kwargs={
                    "solver_cls": SQPLsqSolver,
                    "friction": solver_kwargs.pop("friction", None),
                    "max_limit": solver_kwargs.pop("max_limit", None),
                    "solver_kwargs": solver_kwargs,
                },
            )
        elif metric_type == GraspSpanMetricFactory.MetricType.GRASPQP_EUCLIDIAN_SCIPY:
            return SpanMetricWrapper(
                EucledianFrictionConeSpanMetric,
                metric_kwargs={
                    "solver_cls": ScipyLsqSolver,
                    "friction": solver_kwargs.pop("friction", None),
                    "max_limit": solver_kwargs.pop("max_limit", None),
                    "solver_kwargs": solver_kwargs,
                },
            )
        elif metric_type == GraspSpanMetricFactory.MetricType.GRASPQP_EUCLIDIAN_PROXQP:
            return SpanMetricWrapper(
                EucledianFrictionConeSpanMetric,
                metric_kwargs={
                    "solver_cls": ForwardSQPLsqSolver,
                    "friction": solver_kwargs.pop("friction", None),
                    "max_limit": solver_kwargs.pop("max_limit", None),
                    "solver_kwargs": solver_kwargs,
                },
            )
        elif metric_type == GraspSpanMetricFactory.MetricType.HANDLE:
            return SpanMetricWrapper(
                HandleFrictionConeSpanMetric,
                metric_kwargs={
                    "solver_cls": SQPLsqSolver,
                    "friction": solver_kwargs.pop("friction", None),
                    "max_limit": solver_kwargs.pop("max_limit", None),
                    "basis_vectors": ["-z"],
                    **solver_kwargs,
                },
            )
        elif metric_type == GraspSpanMetricFactory.MetricType.HANDLE_SCIPY:
            return SpanMetricWrapper(
                HandleFrictionConeSpanMetric,
                metric_kwargs={
                    "solver_cls": ScipyLsqSolver,
                    "friction": solver_kwargs.pop("friction", None),
                    "max_limit": solver_kwargs.pop("max_limit", None),
                    "basis_vectors": ["-z"],
                    **solver_kwargs,
                },
            )
        elif metric_type == GraspSpanMetricFactory.MetricType.GRASPQP_PROXQP:
            return SpanMetricWrapper(
                OverallFrictionConeSpanMetric,
                metric_kwargs={
                    "solver_cls": ForwardSQPLsqSolver,
                    "friction": solver_kwargs.pop("friction", None),
                    "max_limit": solver_kwargs.pop("max_limit", None),
                    **solver_kwargs,
                },
            )
        else:
            raise ValueError(f"Invalid metric type {metric_type}")
