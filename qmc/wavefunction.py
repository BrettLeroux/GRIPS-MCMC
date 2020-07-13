  
import torch
import numpy as np
from torch import nn, optim
from torch.distributions import Normal, Bernoulli

LOGPI = np.log(np.pi)

class ParticleBoxFunction(nn.Module):
    def __init__(self, alpha):
        super(ParticleBoxFunction, self).__init__()
        self.alpha = nn.Parameter(alpha)

    def forward(self, x):
        #
        prod_sin_alpha_pi_x = torch.sin(np.pi*self.alpha*x).prod(dim=-1)
        abs_psi_squared = (np.sqrt(8.0)*prod_sin_alpha_pi_x)**2.0
        return torch.log(abs_psi_squared)

    def local_energy(self, x):
        return ((self.alpha*np.pi)**2.0).sum(dim=-1)

class HarmonicTrialFunction(nn.Module):
    def __init__(self, alpha):
        super(HarmonicTrialFunction, self).__init__()
        self.alpha = nn.Parameter(alpha)

    def forward(self, x):
        # outputs logprob
        # 2.0 * because it's |\Psi|^2
        # squeeze last dim bc it's 1D and output here is a scalar logprob per point
        return 2.0 * (0.5 * torch.log(self.alpha) - 0.25 * LOGPI - 0.5 * x * x * self.alpha * self.alpha).squeeze(dim=-1)

    def local_energy(self, x):
        # squeeze last dim bc it's 1D and output here is a scalar energy per point
        return (self.alpha * self.alpha + (x * x) * (1.0 - self.alpha ** 4.0)).squeeze(dim=-1)

def harmonic_true_mean_energy(alpha):
    return ((alpha**2)/2) + (1.0/(2*(alpha**2)))
def harmonic_true_variance(alpha):
    return ((alpha**4 - 1)**2)/(2*alpha**4)


class HydrogenTrialWavefunction(nn.Module):
    def __init__(self, alpha):
        super(HydrogenTrialWavefunction, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(alpha))

    def forward(self, x):
        # outputs logprob
        # 2.0 * because it's |\Psi|^2
        return 2.0 * (torch.log(self.alpha) + torch.log(x) - self.alpha * x)

    def local_energy(self, x):
        return -(1.0 / x) - (self.alpha / 2) * (self.alpha - (2.0 / x))
