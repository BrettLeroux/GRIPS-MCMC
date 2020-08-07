import torch
from torch import optim
import numpy as np

from qmc.mcmc import metropolis_symmetric, normal_proposal, clip_normal_proposal, NormalProposal, ClipNormalProposal
from qmc.wavefunction import HarmonicTrialFunction, HydrogenTrialWavefunction, HeliumTrialWavefunction
from qmc.wavefunction import TwoParticlesInOneDimBox as twoinone
from qmc.wavefunction import OneParticlesInOneDimBox as oneinone
from qmc.wavefunction import HydrogenTrialWavefunction as htf
def energy_minimize_step(trialfunc, samples, optimizer):
    local_energies = trialfunc.local_energy(samples).detach()
    mean_local_energy = local_energies.mean()
    print('energy is', mean_local_energy)
    sample_logprobs = trialfunc(samples)
    loss = ((local_energies - mean_local_energy) * sample_logprobs).mean()
    optimizer.zero_grad()
    loss.backward()
    print('grad is', trialfunc.alpha.grad)
    optimizer.step()
   


def vmc_iterate(tf, init_config, num_iters=100):
    opt = optim.SGD(tf.parameters(), lr=5e-2,momentum=0.3)
    # propdist = NormalProposal(0.3)
    propdist = ClipNormalProposal(0.1)
    for i in range(num_iters):
        print(i)
        results=metropolis_symmetric(tf, init_config, propdist, num_steps=1000)
        energy_minimize_step(tf, results, opt)
        print(tf.alpha)
        print(results)
    

def harmonic_energy_alpha_values():
    vals = np.arange(0.2,1.5,0.1)
    means = []
    for alpha_val in vals:
        print(alpha_val)
        tf = HarmonicTrialFunction(torch.tensor(alpha_val))
        init_config = 0.5*torch.ones(100,1)
        samples = metropolis_symmetric(tf, init_config, normal_proposal, num_steps=20000)
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
        samples = metropolis_symmetric(tf, init_config, propdist, num_steps=20000)
        means.append(torch.mean(tf.local_energy(samples)).item())
    return vals, means

def helium_energy_alpha_values():
    vals = np.arange(1.2,2.5,0.1)
    means = []
    propdist = ClipNormalProposal(0.05, min_val=0.0)
    for alpha_val in vals:
        print(alpha_val)
        tf = HeliumTrialWavefunction(torch.ones(1)*alpha_val)
        init_config = 0.5*torch.ones(100, 3)
        samples = metropolis_symmetric(tf, init_config, propdist, num_steps=20000)
        means.append(torch.mean(tf.local_energy(samples)).item())
        print(means[-1])
    return vals, means

    #Oneparticle in box
#if __name__ == '__main__':
 #   tf = oneinone(torch.tensor(0.86))
 #   init_config = 0.9*torch.ones(1)
 #   vmc_iterate(tf, init_config)
    #Twoparticles in box
if __name__ == '__main__':
    tf = twoinone(torch.tensor([3.0,2.5]))
    init_config = 0.5*torch.rand(1,2)
    vmc_iterate(tf, init_config)
    # helium_energy_alpha_values()
#if __name__ == '__main__':
 #   tf = HeliumTrialWavefunction(torch.ones(1))
 #   init_config = 0.5*torch.ones(1000,3)
 #   vmc_iterate(tf, init_config)
    # helium_energy_alpha_values()
# Hydrogen

    #Oneparticle in box
#if __name__ == '__main__':
#    tf = htf(torch.ones(1))
#    init_config = torch.ones(1)
#    vmc_iterate(tf, init_config)
