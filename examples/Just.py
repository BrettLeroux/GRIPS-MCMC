import torch
import numpy as np
#from qmc.autograd import autograd_trace_hessian
from torch import nn, optim
from torch.distributions import Normal, Bernoulli
LOGPI = np.log(np.pi)

def local_energy(x):
    return -(1**2)+(1)/x[:,0] +(1)/x[:,1] -(2)/x[:,0] -(2)/x[:,1] + (1)/(x[:,0]**2 +x[:,1]**2+x[:,0]*x[:,1]*torch.cos(x[:,2]) )

y = torch.rand(3)

A = local_energy(y)
print(A)
