import torch
from torch import optim
import numpy as np

from qmc.mcmc import metropolis_symmetric, normal_proposal, clip_normal_proposal, NormalProposal, ClipNormalProposal
from qmc.wavefunction import HarmonicTrialFunction, HydrogenTrialWavefunction


def energy_minimize_step(trialfunc, samples, optimizer):
    with torch.no_grad():
        local_energies = trialfunc.local_energy(samples)
    sample_logprobs = trialfunc(samples)
    loss = (local_energies * sample_logprobs).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def vmc_iterate(tf, init_config, num_iters=100):
    opt = optim.SGD(tf.parameters(), lr=1e-2,momentum=0.9)
    propdist = NormalProposal(0.3)
    for i in range(num_iters):
        results=metropolis_symmetric(tf, init_config, propdist, num_walkers=1000, num_steps=5000)
        energy_minimize_step(tf, results, opt)
        print(tf.alpha)

def harmonic_energy_alpha_values():
    vals = np.arange(0.2,1.5,0.1)
    means = []
    for alpha_val in vals:
        print(alpha_val)
        tf = HarmonicTrialFunction(torch.tensor(alpha_val))
        init_config = 0.5*torch.ones(100,1)
        samples = metropolis_symmetric(tf, init_config, normal_proposal, num_walkers=100, num_steps=20000)
        means.append(torch.mean(tf.local_energy(samples)).item())
    return vals, means

def hydrogen_energy_alpha_values():
    vals = np.arange(0.2,1.5,0.1)
    means = []
    propdist = ClipNormalProposal(0.3, min_val=0.0)
    for alpha_val in vals:
        print(alpha_val)
        tf = HydrogenTrialWavefunction(torch.tensor(alpha_val))
        init_config = 0.5*torch.ones(100, 1)
        samples = metropolis_symmetric(tf, init_config, propdist, num_walkers=100, num_steps=20000)
        means.append(torch.mean(tf.local_energy(samples)).item())
    return vals, means


if __name__ == '__main__':
    tf = HarmonicTrialFunction(torch.tensor(1.2))
    init_config = 0.5*torch.ones(1000,1)
    vmc_iterate(tf, init_config)

