# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

"""Bounded least-squares / QP solver backends used by the span metrics.

Provides interchangeable solvers for the span optimization problem: ``SQPLsqSolver``
(``qpth``, differentiable, default), ``ForwardSQPLsqSolver`` (ProxQP), and
``ScipyLsqSolver`` (SciPy CPU reference). The backend is chosen via the
``GraspSpanMetricFactory.MetricType`` variant.
"""

# from .solver import LsqSolver
