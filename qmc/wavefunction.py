import torch
import numpy as np

from qmc.local_energy import auto_hamiltonian_generator_atoms
from qmc.tracehess import autograd_trace_hessian
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
    def harmoni_ansatz_sup(self, x):
        # output dimensions should be either num_walkers x num_samples or just num_samples
        return torch.sqrt(self.alpha)*torch.exp(-0.5*self.alpha*x**2).squeeze(dim=-1)
    def local_energy(self, x):
        return ((x**2).squeeze(dim=-1)-(autograd_trace_hessian(self.harmoni_ansatz_sup,x)/(self.harmoni_ansatz_sup(x))))


    # def local_energy(self, x):
    #     squeeze last dim bc it's 1D and output here is a scalar energy per point
        # return (self.alpha * self.alpha + (x * x) * (1.0 - self.alpha ** 4.0)).squeeze(dim=-1)

def harmonic_true_mean_energy(alpha):
    return ((alpha**2)/2) + (1.0/(2*(alpha**2)))

def harmonic_true_variance(alpha):
    return ((alpha**4 - 1)**2)/(2*alpha**4)


class HydrogenTrialWavefunction(nn.Module):
    def __init__(self, alpha):
        super(HydrogenTrialWavefunction, self).__init__()
        self.alpha = nn.Parameter(alpha)
    

    def forward(self, x):
         #outputs logprob
         #2.0 * because it's |\Psi|^2
        return 2.0 * (torch.log(self.alpha) + torch.log(x) - self.alpha * x).squeeze(dim=-1)

    def hydro_ansatz_sup(self, x):
        x = x.squeeze(dim=-1)
        return self.alpha*x*torch.exp(-self.alpha*x)

    def local_energy(self, x):
        return (-(1.0/x.squeeze(dim=-1))-(0.5*autograd_trace_hessian(self.hydro_ansatz_sup,x)/(self.hydro_ansatz_sup(x))))
    #def local_energy(self, x):
      #  return (-(1.0 / x) - (self.alpha / 2) * (self.alpha - (2.0 / x))).squeeze(dim=-1)


class NElectronSlater(nn.Module):
    def __init__(self, alpha):
        super(NElectronSlater, self).__init__()
        # alpha is of size N, it's the number of basis functions
        self.alpha = nn.Parameter(alpha)

    def _slater_mat(self, x):
        x = x.unsqueeze(-1)
        slater_mat = torch.exp(-x * self.alpha)
        return slater_mat

    def forward(self, x):
        slater_mat = self._slater_mat(x)
        logdet, signs = slater_mat.slogdet()
        return 2.0*logdet

    def wave(self, x):
        slater_mat = self._slater_mat(x)

        det = slater_mat.det()
        return det

class HeliumTrialWavefunction(nn.Module):
    def __init__(self, alpha):
        super(HeliumTrialWavefunction, self).__init__()
        self.alpha = nn.Parameter(alpha.clone().detach())

    def forward(self, x):
        # outputs logprob
        # 2.0 * because it's |\Psi|^2

        return 2.0*(-self.alpha*(x[...,0]+x[...,1])-1/(x[...,0])-1/(x[...,1]))#+2*torch.log(self.alpha) + 2*torch.log(x[...,0]+x[...,1])
    #def helium_ansatz_sup_simple(self,x):
       # x = x.squeeze(dim=-1)
      #  return torch.exp(-self.alpha*(x[...,0]+x[...,1]))
    #def helium_ansatz_sup(self, x):
       # return (((2-self.alpha)**3)/np.pi)*torch.exp(-(2-self.alpha)*(x[:, 0]+x[:, 1]))
    #def helium_ansatz_sup1(self, x):
    #    return (((2-self.alpha)**3)/np.pi)*torch.exp(-(2-self.alpha)*(x[:, 0]))
    #def helium_ansatz_sup2(self, x):
    #    return (((2-self.alpha)**3)/np.pi)*torch.exp(-(2-self.alpha)*(x[:, 1]))
    #def local_energy(self, x):
     #   return autograd_trace_hessian(self.helium_ansatz_sup1,x)*self.helium_ansatz_sup2(x)+autograd_trace_hessian(self.helium_ansatz_sup2,x)*self.helium_ansatz_sup1(x)+2*(1/x[:,0]+1/x[:,1])+1/(torch.sqrt(x[:,0]**2+x[:,1]**2+torch.abs(x[:,1])*torch.abs(x[:,0])*torch.cos(x[:,2])))

    def wave(self, x):
        return torch.exp(self.forward(x) / 2.0)

    def existing_local_energy(self, x):
        return -(self.alpha) ** 2 + (self.alpha / (x[..., 0]) + self.alpha / (x[..., 1]) - 2 * (
                    1 / (x[..., 0]) + 1 / (x[..., 1])) + 1 / (torch.sqrt(
            x[..., 0] ** 2 + x[..., 1] ** 2 + torch.abs(x[..., 1]) * torch.abs(x[..., 0]) * torch.cos(x[..., 2]))))

    def local_energy(self, x):
        return auto_hamiltonian_generator_atoms(self, 2, x) / self.wave(x)


class NelectronVander(nn.Module):
    #ansatz given by the Vandermonde determinant of the one electron wavefunctions e^(-alpha*r_i)
    #input is 1-element tensor alpha and 1D tensor dim which determines the number of particles (i.e. the dimension)
    def __init__(self, alpha, dim):
        super(NelectronVander, self).__init__()
        self.alpha = nn.Parameter(alpha)
        self.dim = dim
            
    def forward(self, x):
        #returns the log prob. of the wavefunction
        #input is tensor of size m x alpha.size or m x n x alpha.size
        a = torch.exp(-self.alpha*x.unsqueeze(-1)) - torch.exp(-self.alpha*x.unsqueeze(-2))
        return 2 * torch.sum(torch.log(torch.abs(a[...,torch.triu(torch.ones(self.dim,self.dim), diagonal=1).nonzero(as_tuple = True)[0],torch.triu(torch.ones(self.dim,self.dim), diagonal=1).nonzero(as_tuple = True)[1] ])),-1)
    
    
    def wave(self,x):
        # Returns the value of the wavefunction
        #input is tensor of size m x alpha.size or m x n x alpha.size
        a = torch.exp(-self.alpha*x.unsqueeze(-1)) - torch.exp(-self.alpha*x.unsqueeze(-2))
        return torch.prod(a[...,torch.triu(torch.ones(self.dim,self.dim), diagonal=1).nonzero(as_tuple = True)[0],torch.triu(torch.ones(self.dim,self.dim), diagonal=1).nonzero(as_tuple = True)[1] ],-1)
    
    
        


class NelectronVanderCusp(nn.Module):
    #ansatz given by the Vandermonde determinant of the one electron wavefunctions e^(-alpha*r_i) multiplied by product of e^(-1/r_i) for all i
    #input is 1-element tensor alpha and 1D tensor dim which determines the number of particles (i.e. the dimension)
    def __init__(self, alpha, dim):
        super(NelectronVanderCusp, self).__init__()
        self.alpha = nn.Parameter(alpha)
        self.dim = dim
            
    def forward(self, x):
        #returns the log prob. of the wavefunction
        #input is tensor of size m x alpha.size or m x n x alpha.size
        a = torch.exp(-self.alpha*x.unsqueeze(-1)) - torch.exp(-self.alpha*x.unsqueeze(-2))
        return 2 * (-torch.sum(1/x, -1) + torch.sum(torch.log(torch.abs(a[...,torch.triu(torch.ones(self.dim,self.dim), diagonal=1).nonzero(as_tuple = True)[0],torch.triu(torch.ones(self.dim,self.dim), diagonal=1).nonzero(as_tuple = True)[1] ])),-1))
    
    def wave(self,x):
        # Returns the value of the wavefunction
        #input is tensor of size m x alpha.size or m x n x alpha.size
        a = torch.exp(-self.alpha*x.unsqueeze(-1)) - torch.exp(-self.alpha*x.unsqueeze(-2))
        return torch.exp(-torch.sum(1/x, -1)) * torch.prod(a[...,torch.triu(torch.ones(self.dim,self.dim), diagonal=1).nonzero(as_tuple = True)[0],torch.triu(torch.ones(self.dim,self.dim), diagonal=1).nonzero(as_tuple = True)[1] ],-1)
    
    
        
    
    
  
 



   
class NelectronVanderWithMult(nn.Module):
    #ansatz given by the Vandermonde determinant of the one electron wavefunctions e^(-alpha * r_i) 
    #multiplied by e^(-beta * (r_1 + r_2 + ... r_N))
    #input is 1-element tensors alpha, beta and 1D tensor dim which determines the number of particles (i.e. the dimension)
    def __init__(self, alpha, beta, dim):
        super(NelectronVanderWithMult, self).__init__()
        self.alpha = nn.Parameter(alpha)
        self.beta = nn.Parameter(beta)
        self.dim = dim

    
    def forward(self, x):
        #returns the log prob. of the wavefunction
        #input is tensor of size m x alpha.size or m x n x alpha.size
        a = torch.exp(-self.alpha*x.unsqueeze(-1)) - torch.exp(-self.alpha*x.unsqueeze(-2))
        return 2 * ( -self.beta * torch.sum(x, -1)
            + torch.sum(torch.log(torch.abs(a[...,torch.triu(torch.ones(self.dim,self.dim), diagonal=1).nonzero(as_tuple = True)[0],torch.triu(torch.ones(self.dim,self.dim), diagonal=1).nonzero(as_tuple = True)[1] ])),-1) )
    
    
    def wave(self, x):
        # Returns the value of the wavefunction
        #input is tensor of size m x alpha.size or m x n x alpha.size
        a = torch.exp(-self.alpha*x.unsqueeze(-1)) - torch.exp(-self.alpha*x.unsqueeze(-2))
        return torch.exp(-self.beta * torch.sum(x, -1)) * torch.prod(a[...,torch.triu(torch.ones(self.dim,self.dim), diagonal=1).nonzero(as_tuple = True)[0],torch.triu(torch.ones(self.dim,self.dim), diagonal=1).nonzero(as_tuple = True)[1] ],-1)
    
    




 

                    

class NelectronVanderCuspWithMult(nn.Module):
    #ansatz given by the Vandermonde determinant of the one electron wavefunctions e^(-alpha * r_i)
    #multiplied by e^(-beta * (r_1 + r_2 + ... r_N)) and multiplied by product of e^(-1/r_i) for all i
    #input is 1-element tensors alpha, beta and 1D tensor dim which determines the number of particles (i.e. the dimension)
    def __init__(self, alpha, beta, dim):
        super(NelectronVanderCuspWithMult, self).__init__()
        self.alpha = nn.Parameter(alpha)
        self.beta = nn.Parameter(beta)
        self.dim = dim


    def forward(self, x):
        #returns the log prob. of the wavefunction
        #input is tensor of size m x alpha.size or m x n x alpha.size
        a = torch.exp(-self.alpha*x.unsqueeze(-1)) - torch.exp(-self.alpha*x.unsqueeze(-2))
        return 2 * ( -self.beta * torch.sum(x, -1) - torch.sum(1/x, -1)
            + torch.sum(torch.log(torch.abs(a[...,torch.triu(torch.ones(self.dim,self.dim), diagonal=1).nonzero(as_tuple = True)[0],torch.triu(torch.ones(self.dim,self.dim), diagonal=1).nonzero(as_tuple = True)[1] ])),-1) )


    def wave(self,x):
        # Returns the value of the wavefunction
        #input is tensor of size m x alpha.size or m x n x alpha.size
        a = torch.exp(-self.alpha*x.unsqueeze(-1)) - torch.exp(-self.alpha*x.unsqueeze(-2))
        return torch.exp(-torch.sum(1/x, -1)) * torch.exp(-self.beta * torch.sum(x, -1)) * torch.prod(a[...,torch.triu(torch.ones(self.dim,self.dim), diagonal=1).nonzero(as_tuple = True)[0],torch.triu(torch.ones(self.dim,self.dim), diagonal=1).nonzero(as_tuple = True)[1] ],-1)
 
