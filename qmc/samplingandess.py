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
print(ESS)

tf= HarmonicTrialFunction(torch.ones(1))
n_walkers=1
init_config = torch.ones(n_walkers,1)
results2 = def unadjusted_langevin(trialfunc, init_config, num_walkers=2, num_steps=10000, eta=0.01):
    # seems hard to get this to converge
    config = init_config.clone(requires_grad=True)
    grad_out = torch.ones_like(config, requires_grad=False)
    all_configs = []
    for step in range(num_steps):
        # next config is from Langevin proposal (grad of logprob + gaussian noise)
        curr_config_logprobs = trialfunc(config)
        grads, = torch.autograd.grad(curr_config_logprobs,
                                    config,
                                    grad_outputs=grad_out,
                                    retain_graph=False)
        with torch.no_grad():
            propdist = Normal(config + eta*grads, np.sqrt(2.0*eta))
            next_config = propdist.sample()
        # then just append
        next_config.requires_grad_(True)
        all_configs.append(next_config)
        config = next_config
    return torch.stack(all_configs, dim=1)
data = az.convert_to_dataset(results2.numpy())
ESS2 = az.ess(data)
print(ESS2)

