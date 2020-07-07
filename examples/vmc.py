import torch
from torch import optim

from qmc.mcmc import metropolis_symmetric, normal_proposal
from qmc.wavefunction import HarmonicTrialFunction


def energy_minimize_step(trialfunc, samples, optimizer):
    with torch.no_grad():
        local_energies = trialfunc.local_energy(samples)
    sample_logprobs = trialfunc(samples)
    loss = (local_energies * sample_logprobs).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()



def vmc_iterate(tf, num_iters=100):
    opt = optim.SGD(tf.parameters(), lr=1e-2,momentum=0.9)
    for i in range(num_iters):
        results=metropolis_symmetric(tf, normal_proposal, num_walkers=1000, num_steps=5000)
        energy_minimize_step(tf, results, opt)
        print(tf.alpha)

if __name__ == '__main__':
    tf = HarmonicTrialFunction(torch.tensor(1.2))
    vmc_iterate(tf)

