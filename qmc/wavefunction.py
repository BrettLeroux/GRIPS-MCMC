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
        return 2.0 * (0.5 * torch.log(self.alpha) - 0.25 * LOGPI - 0.5 * x * x * self.alpha * self.alpha)

    def local_energy(self, x):
        return self.alpha * self.alpha + (x * x) * (1.0 - self.alpha ** 4.0)

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


class HeliumTrialWavefunction(nn.Module):
    def __init__(self, alpha):
        super(HeliumTrialWavefunction, self).__init__()
        self.alpha = nn.Parameter(alpha.clone().detach())

    def forward(self, x):
        # outputs logprob
        # 2.0 * because it's |\Psi|^2

        return 2.0*(3*torch.log((2-self.alpha)+torch.log(torch.tensor(np.pi))-(2-self.alpha)*(x[:, 0]+x[:, 1])))
    
    def local_energy(self, x):
        return -(2-self.alpha)**2-self.alpha*(1/x[0]+1/x[1])+1/(torch.sqrt(x[0]**2+x[1]**2+x[1]*x[0]*torch.cos(x[2])))
