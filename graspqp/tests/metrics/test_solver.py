import torch
import pytest


def test_qpth_solver():
    from graspqp.metrics.solver.qp_solver import SQPLsqSolver as LsqSolver

    A = torch.Tensor(
        [
            [-0.0, 0.25819889, 0.25819889, -0.25819889, -0.25819889, -0.25819889],
            [
                0.31622777,
                -0.18257419,
                -0.18257419,
                -0.18257419,
                -0.18257419,
                -0.18257419,
            ],
            [0.02810913, -0.09128826, 0.10345754, 0.13997471, -0.05477109, -0.10345754],
        ]
    ).unsqueeze(0)
    b = torch.Tensor([0.25819889, 0.18257419, 0.00608464]).unsqueeze(0)
    solver = LsqSolver.from_mat(A, b)
    solution = solver(A, b, min_bound=-10.0, max_bound=1e3, init=0.1)
    # ensure solution is zero
    assert torch.allclose(solution, torch.zeros_like(solution), atol=1e-4)


def test_scipy_solver():
    # test scipy

    from graspqp.metrics.solver.scipy_solver import ScipyLsqSolver

    A = torch.Tensor(
        [
            [-0.0, 0.25819889, 0.25819889, -0.25819889, -0.25819889, -0.25819889],
            [
                0.31622777,
                -0.18257419,
                -0.18257419,
                -0.18257419,
                -0.18257419,
                -0.18257419,
            ],
            [0.02810913, -0.09128826, 0.10345754, 0.13997471, -0.05477109, -0.10345754],
        ]
    ).unsqueeze(0)
    b = torch.Tensor([0.25819889, 0.18257419, 0.00608464]).unsqueeze(0)

    solver = ScipyLsqSolver.from_mat(A, b)

    assert torch.allclose(
        solver(A, b, min_bound=-10.0, max_bound=1e3, init=0.1),
        torch.zeros_like(b),
        atol=1e-4,
    )


if __name__ == "__main__":
    pytest.main([__file__, "-s"])
