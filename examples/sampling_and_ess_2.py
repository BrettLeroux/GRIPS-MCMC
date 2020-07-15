import torch
import matplotlib.pyplot as plt
import numpy
from torch.distributions import Normal, Bernoulli, MultivariateNormal
from qmc.mcmc import metropolis_symmetric, clip_mvnormal_proposal, NormalProposal
from qmc.wavefunction import HarmonicTrialFunction
import arviz as az

#First we begin by sampling from a 1D scalar field.
# We will  use a simple gaussian with one parameter.
# Infact, we will just the harmonic oscillator ansatz.
#We also compute the effective sample size using az.ess() from arviz package. 

for sigma in numpy.linspace(0.01, 3, 30):
    def normal_proposal(old_point):
     symmetric
        return Normal(old_point, sigma*torch.ones_like(old_point)).sample()
    tf= HarmonicTrialFunction(torch.ones(1))
    n_walkers = 2
    init_config = torch.ones(n_walkers, 1)
    results = metropolis_symmetric(tf, init_config, normal_proposal, num_walkers=n_walkers, num_steps=100000)
    dataset1 = az.convert_to_dataset(results.numpy())
    dataset2 = az.convert_to_inference_data(results.numpy())


    az.plot_ess(dataset2, kind = "local")
    plt.savefig("Local")
    az.plot_ess(dataset2, kind = "quantile")
    plt.savefig("quantile")
    az.plot_ess(dataset2, kind = "evolution")
    plt.savefig("Evolution")
    print( az.ess(dataset1).data_vars)
# In the Output_of_run array we are using units of 1000.    
#Output_of_run = numpy.array([0.02366, 1.087, 3.579, 7.21, 11.32, 15.9, 20.19, 25.2, 29.98, 32.94, 36.67, 39.41, 38.68, 42.96, 44.4, 45.35, 44.83, 45.94, 43.73, 46.34, 44.69, 45.15, 41.88,41.41, 41.33, 41, 38.46, 38.3, 37.49, 36.02]) 
#y_data = Output_of_run
#x_data = numpy.linspace(0.01, 3, 30)
#plt.scatter(x_data, y_data, c='r', label='ess scatter')
##plt.plot(x_data, y_data, label='ess fit')
#plt.xlabel('x')
#plt.ylabel('y')
#plt.title('ess vs sigma ')
#plt.legend()
#plt.show()
#plt.savefig('ess vs sigma')
