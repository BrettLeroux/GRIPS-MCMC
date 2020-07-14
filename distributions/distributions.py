import torch
import numpy as np
from torch import nn


class RandomHybridRosenbrock(nn.Module):
    # RandomHybridRosenbrock(m,n) creates the "Hybrid Rosenbrock" from https://arxiv.org/pdf/1903.09556.pdf with random parameters
    # The dimension is (m-1)n + 1
    def __init__(self, n1, n2):
        super(RandomHybridRosenbrock, self).__init__()
        self.a = nn.Parameter(torch.rand(1))
        self.b = nn.Parameter(torch.rand(n2, n1 - 1))
        self.mu = nn.Parameter(torch.rand(1))

    def forward(self, x):
        # output logprob
        y = x[np.r_[1:(n1 - 1) * n2 + 1]].reshape((n2, n1 - 1))
        return (-self.a) * (x[0] - self.mu) ** 2 - sum(
            self.b[j, i] * (y[j, i] - y[j, i - 1] ** 2) ** 2 for i in range(n1 - 2) for j in range(n2 - 1))

    # gives the normalization constant
    def normalization(self):
        return (torch.sqrt(self.a) * torch.prod(torch.sqrt(self.b))) / np.power(np.pi, ((n1 - 1) * n2 + 1) / 2)
