# Copyright (c) 2025 ETH Zurich, René Zurbrügg
# SPDX-License-Identifier: MIT

from dataclasses import dataclass

# Third Party
import numpy as np
import torch


@dataclass(frozen=True)
class TensorDeviceType:
    device: torch.device = torch.device("cuda", 0)
    dtype: torch.dtype = torch.float32
    collision_geometry_dtype: torch.dtype = torch.float32
    collision_gradient_dtype: torch.dtype = torch.float32
    collision_distance_dtype: torch.dtype = torch.float32

    @staticmethod
    def from_basic(device: str, dev_id: int):
        return TensorDeviceType(torch.device(device, dev_id))

    def to_device(self, data_tensor):
        if isinstance(data_tensor, torch.Tensor):
            return data_tensor.to(device=self.device, dtype=self.dtype)
        else:
            return torch.as_tensor(np.array(data_tensor), device=self.device, dtype=self.dtype)

    def to_int8_device(self, data_tensor):
        return data_tensor.to(device=self.device, dtype=torch.int8)

    def cpu(self):
        return TensorDeviceType(device=torch.device("cpu"), dtype=self.dtype)

    def as_torch_dict(self):
        return {"device": self.device, "dtype": self.dtype}
    

@torch.jit.script
def _RELU(input, W, l, u, rho_ind, help_ind, nx: int, nc: int):
    torch.bmm(W[rho_ind, help_ind], input.unsqueeze(-1), out=input.unsqueeze(-1))
    input[:, nx:nx+nc].clamp_(l, u)
    return input

@torch.jit.script
def compute_residuals(H, A, x, z, lam, rho, rho_min: float, rho_max: float):
    t1 = torch.matmul(A, x.unsqueeze(-1)).squeeze(-1)
    t2 = torch.matmul(H, x.unsqueeze(-1)).squeeze(-1)
    t3 = torch.matmul(A.transpose(-1, -2), lam.unsqueeze(-1)).squeeze(-1)
    primal_res = torch.linalg.vector_norm(t1 - z, dim=-1, ord=torch.inf)
    dual_res = torch.linalg.vector_norm(t2 + t3, dim=-1, ord=torch.inf)
    numerator = torch.div(primal_res, torch.max(torch.linalg.vector_norm(t1, dim=-1, ord=torch.inf), torch.linalg.vector_norm(z, dim=-1, ord=torch.inf)))
    denom = torch.div(dual_res, torch.max(torch.linalg.vector_norm(t2, dim=-1, ord=torch.inf), torch.linalg.vector_norm(t3, dim=-1, ord=torch.inf)).clamp(min=0))
    rho = torch.clamp(rho * torch.sqrt(numerator / denom), rho_min, rho_max)
    return primal_res, dual_res, rho


class _RELUQP:
    
    def __init__(self, batch, nx, nc, tensor_args: TensorDeviceType = TensorDeviceType()):
        self.tensor_args = tensor_args
        self.batch = batch
        self.nx = nx 
        self.nc = nc
        
        self.rho=0.1
        self.rho_min=1e-3
        self.rho_max=1e3
        self.adaptive_rho_tolerance=5
        self.setup_rhos(nc, batch)
        
        self.max_iter=1000
        self.eps_abs=1e-3
        self.check_interval=25
        self.sigma= 1e-6 * self.tensor_args.to_device(torch.eye(nx)).unsqueeze(0).unsqueeze(1)
        # self.Ic = tensor_args.to_device(torch.eye(nc)).unsqueeze(0).unsqueeze(0).repeat(self.rhos_matrix.shape[0],batch,1,1)
        # self.IW = tensor_args.to_device(torch.eye(nx+nc+nc)).unsqueeze(0).unsqueeze(0).repeat(1, batch, 1, 1)
        self.W_ks = tensor_args.to_device(torch.zeros(len(self.rhos)+1, batch, nx+nc+nc, nx+nc+nc))
        self.W_ks[0, ...] = torch.eye(nx+nc+nc)
        self.W_ks[1:, :, -nc:, -nc:] = torch.eye(nc)
        self.W_ks[1:, :, -nc:, nx:-nc] = - self.rhos_matrix
        self.help_arange = tensor_args.to_device(torch.arange(batch)).long()
        self.output = tensor_args.to_device(torch.zeros((batch, nx+nc+nc)))
        return 

    def setup_rhos(self, nc, batch):
        """
        Setup rho values for ADMM
        """
        rhos = [self.rho]
        rho = self.rho/self.adaptive_rho_tolerance
        while rho >= self.rho_min:
            rhos.append(rho)
            rho = rho/self.adaptive_rho_tolerance
        rho = self.rho*self.adaptive_rho_tolerance
        while rho <= self.rho_max:
            rhos.append(rho)
            rho = rho*self.adaptive_rho_tolerance
        rhos.sort()

        # conver to torch tensor
        self.rhos = self.tensor_args.to_device(rhos)
        self.rho_ind = torch.argmin(torch.abs(self.rhos - self.rho)).repeat(self.batch) # [b]
        self.rhos_matrix = self.rhos.view(-1, 1, 1, 1) * self.tensor_args.to_device(torch.eye(nc)).unsqueeze(0).unsqueeze(1)    # [r, 1, nc, nc]
        self.rhos_inv_matrix =  (1 / self.rhos.view(-1, 1, 1, 1)) * self.tensor_args.to_device(torch.eye(nc)).unsqueeze(0).unsqueeze(1) # [r, 1, nc, nc]
        return 
    
    @torch.no_grad()
    def solve(self, H, A, l, u):
        H = H.unsqueeze(0)
        A = A.unsqueeze(0)
        
        self.output[:] = 0    # initialize with 0. If disabled, use the last solution as initialization.
        
        # # # NOTE torch.inverse is very bad and will lead to poor converge behavior
        # # K = torch.inverse(H + self.sigma + A.transpose(-1,-2) @ (self.rhos_matrix @ A))
        # K_inv = H + self.sigma + A.transpose(-1,-2) @ (self.rhos_matrix @ A)
        # K_ArA = torch.linalg.solve(K_inv, (self.sigma - A.transpose(-1,-2) @ (self.rhos_matrix @ A)))
        # K_AT = torch.linalg.solve(K_inv, A.transpose(-1, -2))

        # W_ks = torch.cat([
        #     torch.cat([ K_ArA ,           2 * K_AT @ self.rhos_matrix,            - K_AT], dim=-1),
        #     torch.cat([ A @ K_ArA + A,   2 * A @ K_AT @ self.rhos_matrix - self.Ic,  -A @ K_AT + self.rhos_inv_matrix], dim=-1),
        #     torch.cat([ self.rhos_matrix @ A,               -self.rhos_matrix,      self.Ic], dim=-1),
        # ], dim=-2)
        # W_ks = torch.cat([self.IW, W_ks], dim=0)

        rhosA = torch.matmul(self.rhos_matrix, A, out=self.W_ks[1:, :, -self.nc:, :self.nx]) # [r, b, nc, nx]
        ArA = torch.matmul(A.transpose(-1,-2), rhosA, out=self.W_ks[1:, :, self.nx:2*self.nx, self.nx:2*self.nx]) # TMP! [r, b, nx, nx]
        assert self.nx < self.nc
        nK_inv = torch.add(self.sigma + H, ArA, out=self.W_ks[1:, :, self.nx:2*self.nx, 2*self.nx:3*self.nx])   # TMP! [r, b, nx, nx]
        sig_ArA = torch.sub(self.sigma, ArA, out=self.W_ks[1:, :, self.nx:2*self.nx, self.nx:2*self.nx]) # TMP! overlap ArA [r, b, nx, nx]
        nK_ArA = torch.linalg.solve(nK_inv, sig_ArA, out=self.W_ks[1:, :, :self.nx, :self.nx])   # [r, b, nx, nx]
        nK_AT = torch.linalg.solve(nK_inv, A.transpose(-1, -2), out=self.W_ks[1:, :, :self.nx, -self.nc:])    # [r, b, nx, nc]
        K_AT_rhos = torch.matmul(nK_AT, self.rhos_matrix, out=self.W_ks[1:, :, :self.nx, self.nx:-self.nc])   # [r, b, nx, nc]
        K_AT_rhos.mul_(2.0)
        torch.sub(0.0, nK_AT, out=self.W_ks[1:, :, :self.nx, -self.nc:])
        torch.matmul(A, self.W_ks[1:, :, :self.nx, :], out=self.W_ks[1:, :, self.nx:-self.nc, :])
        self.W_ks[1:, :, self.nx:-self.nc, :self.nx].add_(A)
        self.W_ks[1:, :, self.nx:-self.nc, self.nx:-self.nc].sub_(self.W_ks[1:, :, -self.nc:, -self.nc:])
        self.W_ks[1:, :, self.nx:-self.nc, -self.nc:].add_(self.rhos_inv_matrix)
        # assert (W_ks - self.W_ks).abs().max() < 1e-5
        # import pdb;pdb.set_trace()
        
        rho_ind = self.rho_ind
        rho = self.rhos[rho_ind]

        for k in range(1, self.max_iter + 1):
            self.output = _RELU(input=self.output, W=self.W_ks, l=l, u=u, rho_ind=(rho_ind+1), help_ind=self.help_arange, nx=self.nx, nc=self.nc)
            # rho update
            if k % self.check_interval == 0:
                x, z, lam = self.output[:, :self.nx], self.output[:, self.nx:self.nx+self.nc], self.output[:, self.nx+self.nc:self.nx+2*self.nc]
                primal_res, dual_res, rho = compute_residuals(H.squeeze(0), A.squeeze(0), x, z, lam, rho, self.rho_min, self.rho_max)

                rho_larger = (rho > self.rhos[rho_ind] * self.adaptive_rho_tolerance) & (rho_ind < len(self.rhos) - 1) & (rho_ind > -1)
                rho_smaller = (rho < self.rhos[rho_ind] / self.adaptive_rho_tolerance) & (rho_ind > 0)
                rho_ind = rho_ind + rho_larger.int() - rho_smaller.int()

                converge_flag = (primal_res < self.eps_abs * (self.nc ** 0.5)) & (dual_res < self.eps_abs * (self.nx ** 0.5))
                rho_ind = torch.where(converge_flag, -1, rho_ind)
                if torch.all(converge_flag):
                    break

        return x
    
    
    
if __name__ == "__main__":
    # test on simple QP
    # min 1/2 x' * H * x + g' * x
    # s.t. l <= A * x <= u
    tensor_args = TensorDeviceType(dtype=torch.float32, device=torch.device("cuda", 0))
    H = tensor_args.to_device([[6, 2, 1], [2, 5, 2], [1, 2, 4.0]])
    # g = torch.tensor([-8.0, -3, -3], dtype=torch.double)
    A = tensor_args.to_device([[1, 0, 1], [0, 1, 1], [1, 0, 0], [0, 1, 0], [0, 0, 1]])
    l = torch.tensor([3.0, 0, -10.0, -10, -10], dtype=torch.double).to(device=tensor_args.device)
    u = tensor_args.to_device([3.0, 0, torch.inf, torch.inf, torch.inf])

    batch = 1000
    H = H.unsqueeze(0).repeat(batch, 1, 1)
    A = A.unsqueeze(0).repeat(batch, 1, 1)
    u = u.unsqueeze(0).repeat(batch, 1)
    import time 
    
    qp = _RELUQP(batch, H.shape[1], A.shape[1], tensor_args)
    
    for i in range(10):
        s = time.time()
        x  = qp.solve(H=H, A=A, u=u, l=l)
        print(x)
        print(time.time() - s)