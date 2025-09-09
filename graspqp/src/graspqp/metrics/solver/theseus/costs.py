import torch

def log_barrier(x:torch.Tensor, limit:torch.Tensor, t:float = 1e2):
    # try to fullfill constraint h(x) <= 0
    h = (x-limit)
    # use log barrier and -h to have gradient for negative values
    value = (-h).relu()  + (- 1/t * torch.log(h.relu() + 1e-6)).relu()
    return value


def quad_error_torch(a:torch.Tensor, x:torch.Tensor, y:torch.Tensor, verbose = False):
    # least squares problem
    
    
    alpha = torch.exp(a) / torch.exp(a).sum(-1, keepdim=True)
    est = torch.bmm(x, alpha.unsqueeze(-1)).squeeze(-1)
    err = 0.707 * (y - est)*10

    # # barrier functions
    #lower_barrier = log_barrier(a, limit=l)
    #upper_barrier = log_barrier(-a, limit=-u)

    #barrier = (lower_barrier.sum(-1, True) + upper_barrier.sum(-1, True))
    
    if verbose:
        print(f"Final Solution: {alpha}, a {a}")
        print(f"Residual: {(err**2).sum()}")
    
    return err