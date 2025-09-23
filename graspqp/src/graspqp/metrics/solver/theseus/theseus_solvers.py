import torch
from theseus.optimizer.linear.dense_solver import (Any, DenseLinearization,
                                                   DenseSolver, Dict,
                                                   Linearization, Objective,
                                                   Optional, Type)


class CholeskyRegularizedDenseSolver(DenseSolver):
    def __init__(
        self,
        objective: Objective,
        linearization_cls: Optional[Type[Linearization]] = DenseLinearization,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        check_singular: bool = False,
    ):
        super().__init__(
            objective,
            linearization_cls,
            linearization_kwargs,
            check_singular=check_singular,
        )

    def _solve_sytem(self, Atb: torch.Tensor, AtA: torch.Tensor) -> torch.Tensor:
        ATA_hat = AtA + torch.eye(AtA.shape[-1], device=AtA.device) * 1e-4
        L, info = torch.linalg.cholesky_ex(ATA_hat)
        valid_cholesky = info == 0
        solution = torch.cholesky_solve(Atb, L).squeeze(2)
        # patch invalid
        # solution[~valid_cholesky] = torch.linalg.lstsq(ATA_hat[~valid_cholesky], Atb[~valid_cholesky]).solution.squeeze(2)
        return solution


class PinvDenseSolver(DenseSolver):
    """Dense solver using pseudo-inverse to solve the system of equations."""

    def __init__(
        self,
        objective: Objective,
        linearization_cls: Optional[Type[Linearization]] = DenseLinearization,
        linearization_kwargs: Optional[Dict[str, Any]] = None,
        check_singular: bool = True,
    ):
        super().__init__(
            objective,
            linearization_cls,
            linearization_kwargs,
            check_singular=check_singular,
        )

    def _solve_sytem(self, Atb: torch.Tensor, AtA: torch.Tensor) -> torch.Tensor:
        # sol = torch.linalg.lstsq(AtA, Atb)
        # if torch.isnan(sol.solution).any():
        #     print("Solution contains nan")
        #     import pdb; pdb.set_trace()

        return torch.linalg.lstsq(AtA, Atb).solution.squeeze(2)
