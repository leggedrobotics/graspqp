from enum import Enum

import torch

from ..solver.scipy_solver import ScipyLsqSolver
from .dexgrasp import DexgraspSpanMetric
from .span import EucledianFrictionConeSpanMetric, OverallFrictionConeSpanMetric
from .tdg import TDGSpanMetric

try:
    from graspqp.metrics.solver.qp_solver import SQPLsqSolver
except ImportError:
    print("Error importing SQPLsqSolver. Make sure to install proxsuite")
    SQPLsqSolver = None
    pass


class SpanMetricWrapper(torch.nn.Module):
    """Smart Wrapper that initializes the metric with the correct dimensions."""

    def __init__(
        self,
        metric: OverallFrictionConeSpanMetric = OverallFrictionConeSpanMetric,
        metric_kwargs: dict = {},
    ):
        super().__init__()
        self.metric = metric
        self._initialized = False
        self.metric_kwargs = metric_kwargs

    def forward(
        self,
        contact_pts: torch.Tensor,
        contact_normals: torch.Tensor,
        cog=torch.Tensor,
        contact_threshold: float = 0.0,
        torque_weight: float = 5.0,
        **kwargs,
    ):

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

        eps = 1e-2
        if with_solution:
            return (
                values_gain * (values.mean(-1) + eps) * (-svd_gain * svd_scales.mean(-1)).exp(),
                x,
            )

        return values_gain * (values.mean(-1) + eps) * (-svd_gain * svd_scales.mean(-1)).exp()


class GraspSpanMetricFactory:
    # enum for different metric types
    class MetricType(Enum):
        DEXGRASP = 1
        TDG = 2
        GRASPQP = 3
        GRASPQP_SCIPY = 4
        GRASPQP_EUCLIDIAN_SCIPY = 5

    @staticmethod
    def create(metric_type: MetricType, solver_kwargs: dict = {}):

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
        else:
            raise ValueError(f"Invalid metric type {metric_type}")
