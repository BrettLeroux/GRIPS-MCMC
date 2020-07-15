import torch
import numpy as np
from torch import nn

                                                              

class dim2Rosenbrock(nn.Module):
    def __init__(self, alpha, beta):
        super(dim2Rosenbrock, self).__init__()
        self.alpha = nn.Parameter(alpha)
        self.beta = nn.Parameter(beta)

    def forward(self, x):
        # output logprob
        dim2 = x.ndimension() > 2
        dim1 = x.ndimension() > 1 
        if dim2:
            result = - self.alpha * (x[:, :, 0] -1)**2 - self.beta * (x[:, :, 0]- x[:, :, 1]**2)**2
        else:
            x = x if dim1 else x.unsqueeze(0)
            result = - self.alpha * (x[:, 0] -1)**2 - self.beta * (x[:, 0]- x[:, 1]**2)**2
        return result if dim1 else result.squeeze(0)
        

    # gives the normalization constant
    def normalization(self):
        return torch.sqrt(self.alpha * self.beta) / np.pi




class RandomHybridRosenbrock(nn.Module):
    # RandomHybridRosenbrock(m,n) creates the "Hybrid Rosenbrock" from https://arxiv.org/pdf/1903.09556.pdf with random parameters
    # The dimension is (m-1)n + 1
    def __init__(self, n1, n2):
        super(RandomHybridRosenbrock, self).__init__()
        self.a = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(n2, n1 - 1))
        self.mu = nn.Parameter(torch.rand(1))
        self.n1 = n1
        self.n2 = n2

    def forward(self, x):
        # output logprob
        y = x[np.r_[1:(self.n1 - 1) * self.n2 + 1]].reshape((self.n2, self.n1 - 1))
        return (-self.a) * (x[0] - self.mu) ** 2 - sum(
            self.b[j, i] * (y[j, i] - y[j, i - 1] ** 2) ** 2 for i in range(self.n1 - 2) for j in range(self.n2 - 1))

    # gives the normalization constant
    def normalization(self):
        return (torch.sqrt(self.a) * torch.prod(torch.sqrt(self.b))) / np.power(np.pi, ((self.n1 - 1) * self.n2 + 1) / 2)
