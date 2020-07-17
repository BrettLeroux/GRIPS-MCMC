import torch
import numpy as np
from hypothesis import given
from hypothesis.extra.numpy import arrays, array_shapes
from hypothesis.strategies import floats

from qmc.tracehess import autograd_trace_hessian, gradient_f, hessian_f

# Gradient Function

#def func_cubic(x):
#    return 10.0 * torch.sum(x * x * x * x, dim=-1)
#print(autograd_trace_hessian(func_cubic, y))
self_alpha=1
y = torch.ones(1,1)
def func_exp(x):
    return torch.exp(x).prod(dim=-1)
print(autograd_trace_hessian(func_exp, y))
def hydro_ansatz_sup(x):
        return self_alpha*x*torch.exp(-self_alpha*x)

print(autograd_trace_hessian(hydro_ansatz_sup, y)/hydro_ansatz_sup(y))
