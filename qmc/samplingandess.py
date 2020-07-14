import torch
import matplotlib.pyplot as plt
from mcmc import metropolis_symmetric, clip_mvnormal_proposal
from wavefunction import HarmonicTrialFunction
import arviz as az

#First we begin by sampling from a 1D scalar field.
# We will  use a simple gaussian with one parameter.
# Infact, we will just the harmonic oscillator ansatz.
#We also compute the effective sample size using az.ess() from arviz package. 

tf= HarmonicTrialFunction(torch.ones(1))
n_walkers=1
init_config = torch.ones(n_walkers,1)
results = metropolis_symmetric(tf, init_config, clip_mvnormal_proposal, num_walkers=n_walkers, num_steps=10000)
data = az.convert_to_dataset(results.numpy())
ESS = az.ess(data)

