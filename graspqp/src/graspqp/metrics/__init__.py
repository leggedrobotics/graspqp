# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Differentiable grasp-quality metrics.

This package provides the grasp-quality energies used to score and optimize grasps
from batched contact points, contact normals and the object center of gravity.

The public entry point is :class:`~graspqp.metrics.GraspSpanMetricFactory`. Its
:meth:`~graspqp.metrics.GraspSpanMetricFactory.create` classmethod takes a
:class:`~graspqp.metrics.GraspSpanMetricFactory.MetricType` and returns a ready-to-use
``torch.nn.Module`` metric. The available metrics are:

* **GraspQP** (``GRASPQP*`` variants): our span metric. It builds a friction-cone
  grasp (wrench) matrix and measures how well a target wrench set can be spanned by
  solving a bounded least-squares / QP problem. The QP backend is selected by the
  variant suffix -- ``qpth`` (default), SciPy (``*_SCIPY``) or ProxQP/forward
  (``*_PROXQP``) -- and the Euclidean variants change the basis (test-wrench) set.
* **HANDLE**: a GraspQP variant specialized for handle-style grasps, restricting the
  test wrenches to a fixed basis (e.g. ``-z``).
* **DEXGRASP** (:class:`DexgraspSpanMetric`): the DexGraspNet force-closure energy.
* **TDG** (:class:`TDGSpanMetric`): the Task-oriented Dexterous Grasp / GWS energy.

The GraspQP metrics are wrapped by :class:`SpanMetricWrapper` (aliased here as
``GraspQPSpanMetric``), which lazily initializes the underlying solver to the correct
number of contacts on the first ``forward`` call.
"""

from .ops.registry import (DexgraspSpanMetric, GraspSpanMetricFactory,
                           SpanMetricWrapper, TDGSpanMetric)

GraspQPSpanMetric = SpanMetricWrapper

__all__ = [
    "GraspSpanMetricFactory",
    "DexgraspSpanMetric",
    "TDGSpanMetric",
    "GraspQPSpanMetric",
]
