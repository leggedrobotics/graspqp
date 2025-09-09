import torch
from graspqp.metrics.solver.theseus_solvers import PinvDenseSolver, CholeskyRegularizedDenseSolver


from graspqp.metrics.solver.costs import quad_error_torch
import theseus as th
import numpy as np
    
    
class LsqSolver:
    
    def __init__(self):
        self._theseus_optim = None
        
    @classmethod
    def from_mat(cls, A, b, step_size = 0.3, solver_kwargs = {}):
        solver = cls()
        solver.build_solver_from_mat(A, b, step_size=step_size, solver_kwargs = solver_kwargs)
        return solver
    
    def quad_error_fn(self, optim_vars, aux_vars):
        """Wrappper to call the pytorch error function"""
        a,  = optim_vars 
        x, y, w = aux_vars
        err = quad_error_torch(a.tensor, x.tensor, y.tensor) * 10
        # err[:, -1:] += w.tensor * (a.tensor).pow(2).sum(-1, keepdim=True) # disbale reguralization for now
        # indexes
        return err

    def build_solver_from_mat(self, A, b, step_size = 0.15, solver_kwargs = {}):
        if A.ndim == 2:
            A = A.unsqueeze(0)
        if b.ndim == 1:
            b = b.unsqueeze(0)
            
        if A.ndim == 4:
            # two batch dimensions. Lets flatten them
            batch_size = A.shape[0] * A.shape[1]
        else:
            batch_size = A.shape[0]
            
        self.build_solver(A.shape[-1], b.shape[-1], batch_size, device=A.device, step_size=step_size, solver_kwargs = solver_kwargs)
        
    def to(self, device):
        self._device = device
        self._theseus_optim.to(device)
        
    def build_solver(self, num_wrenches: int, wrench_dim:int, batch_size:int = 1, step_size = 0.15, device = "cuda", solver_kwargs = {}):
        step_size = solver_kwargs.get("step_size", step_size)
        # raise ValueError("Solver not initialized. Make sure to call .build_solver before using the solver.")
        if solver_kwargs is None:
            solver_kwargs = {}
            
        x = th.Variable(torch.zeros((batch_size, wrench_dim, num_wrenches), device=device), name="x")
        y = th.Variable(torch.zeros((batch_size, wrench_dim), device=device), name="y")
        
        # l = th.Variable(torch.zeros((batch_size, 1), device=device), name = "l")
        # u = th.Variable(torch.zeros((batch_size, 1), device=device), name = "u")
        w = th.Variable(torch.zeros((batch_size, 1), device=device), name = "w")
        
        
        # optimization variables are of type Vector with 1 degree of freedom (dof)
        a = th.Vector(num_wrenches, name="a")
        
        optim_vars = a,
        # aux_vars = x, y, l, u, w
        aux_vars = x, y, w
        
        cost_function = th.AutoDiffCostFunction(
            optim_vars, self.quad_error_fn, wrench_dim, aux_vars=aux_vars, name="quadratic_cost_fn"
        )
        
        objective = th.Objective()
        objective.add(cost_function)
        objective.to(device)
        
        solver_type = solver_kwargs.get("solver_type", "LevenbergMarquardt")
        # print("USing solver type", solver_type)
        # print("Solver kwargs", solver_kwargs)
        # print("Solver step size", step_size)
        # print("Solver device", device)
        print("Using solver type", solver_type, "with step size", step_size, "on device", device)
        if solver_type == "Dogleg":
            optimizer = th.Dogleg(
                objective,
                linear_solver_cls = CholeskyRegularizedDenseSolver,
                linear_solver_kwargs={"check_singular": True},
                max_iterations=solver_kwargs.get("max_iterations", 8),
                step_size=step_size,
                vectorize=True
            )
        elif solver_type == "LevenbergMarquardt":
            optimizer = th.LevenbergMarquardt(
                objective,
                linear_solver_cls = CholeskyRegularizedDenseSolver,
                linear_solver_kwargs={"check_singular": True},
                max_iterations=solver_kwargs.get("max_iterations", 8),
                step_size=step_size,
                vectorize=True
            )
        elif solver_type == "GaussNewton":
            optimizer = th.GaussNewton(
                objective,
                linear_solver_cls = CholeskyRegularizedDenseSolver,
                linear_solver_kwargs={"check_singular": True},
                max_iterations=solver_kwargs.get("max_iterations", 8),
                step_size=step_size,
                vectorize=True
            )
        elif solver_type == "DCEM":
            optimizer = th.DCEM(
                objective,
                linear_solver_cls = CholeskyRegularizedDenseSolver,
                linear_solver_kwargs={"check_singular": True},
                max_iterations=solver_kwargs.get("max_iterations", 4),
                step_size=step_size,
                vectorize=True
            )
        else:
            raise ValueError("Unknown solver type" + solver_type)
        
        self._theseus_optim = th.TheseusLayer(optimizer)
        self._theseus_optim.to(device)
        
        self._num_wrenches = num_wrenches
        self._wrench_dim = wrench_dim
        self._batch_size = batch_size
        self._device = device
        self._step_size = step_size 
        
    @property
    def theseus_optim(self):
        if self._theseus_optim is None:
            raise ValueError("Solver not initialized. Make sure to call .build_solver before using the solver.")
        return self._theseus_optim
    
    def solve(self, A, b, reg = 0.0, init=None, min_bound = -1e4, max_bound = 1e4, return_solution = False, verbose = True):
        # verbose = Trues
        

        batch_shape = (A.shape[0],)
        if A.ndim == 4:
            
            # two batch dimensions. Lets flatten them
            batch_shape = A.shape[0], A.shape[1]
            A = A.flatten(0, 1)
            b = b.flatten(0, 1)


        n_batches = np.prod(batch_shape)

        if init is None:
            init = torch.ones((n_batches, self._num_wrenches))
        else:
            if isinstance(init, (int, float)):
                init =  torch.tensor(init, device=self._device, dtype=A.dtype).view(1,1).expand(n_batches,  self._num_wrenches).clone()
            if init.ndim == 1:
                init = init.unsqueeze(0).expand(n_batches, self._num_wrenches).clone()
            elif init.ndim == 2:
                init = init.clone().to(self._device)
            elif init.ndim == 3:
                init = init.flatten(0, 1).clone().to(self._device)
        
        regularizer = torch.tensor(reg, device=self._device, dtype=A.dtype).view(1, 1).expand(n_batches, 1).clone()
        
        # u = torch.ones((n_batches, 1), device=self._device, dtype=A.dtype) * max_bound
        # l = torch.ones((n_batches, 1), device=self._device, dtype=A.dtype) * min_bound
        
        # u = torch.ones((A.shape[0], self._num_wrenches), device=self._device, dtype=A.dtype) * max_bound
        # l = torch.ones((A.shape[0], self._num_wrenches), device=self._device, dtype=A.dtype) * min_bound
        
        # clamp init to bounds
        init = init.clamp(min_bound + 1e-6, max_bound - 1e-6)
        init[torch.isnan(init)] = 1.0
        # init = torch.tensor([[1.0, 2.765, 1.0, 1.0]]).to(init.device)
        theseus_inputs = {
            "x": A,
            "y": b,
            "a": init,
            # "u": u,
            # "l": l,
            "w": regularizer
        }
        
        
        updated_inputs, info = self.theseus_optim.forward(
            theseus_inputs, optimizer_kwargs={"track_best_solution": True, "verbose": verbose, 
            "backward_mode": "truncated" if not isinstance(self.theseus_optim.optimizer, th.DCEM) else "unroll",
            "backward_num_iterations": 5,
            })
        
        best_solution = info.best_solution["a"]
        
        if verbose:
            print("Solution", best_solution, "with error", info.last_err)
            quad_error_torch(best_solution.to(self._device), A, b, verbose=True)
            
        last_err = (quad_error_torch(best_solution.to(self._device), A, b, verbose=False)[..., :-1]**2).sum(-1)
        best_solution = best_solution.view(*batch_shape, self._num_wrenches)
        # last_err = info.last_err.to(A.device)
        last_err = last_err.view(*batch_shape).to(A.device).clamp(max=100)
        last_err[torch.isnan(last_err)] = 100
        
        if return_solution:
            return last_err, best_solution
        return last_err
