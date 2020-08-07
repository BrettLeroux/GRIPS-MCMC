import torch
from torch import optim
import numpy as np
import matplotlib.pyplot as plt

from qmc.mcmc import metropolis_symmetric, normal_proposal, clip_normal_proposal, NormalProposal, ClipNormalProposal
from qmc.wavefunction import HarmonicTrialFunction, HydrogenTrialWavefunction, HeliumTrialWavefunction
from qmc.wavefunction import TwoParticlesInOneDimBox as twoinone

def local_energy_est_grad(trialfunc, samples):
    local_energies = trialfunc.local_energy(samples).detach()
    mean_local_energy = local_energies.mean()
    sample_logprobs = trialfunc(samples)
    loss = ((local_energies - mean_local_energy) * sample_logprobs).mean()
    loss.backward()
    return trialfunc.alpha.grad

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
    print(init_config)

def energy_minimize_explicit_grads(trialfunc, samples, step_size=5e-2):
    local_energies = trialfunc.local_energy(samples).detach()
    mean_local_energy = local_energies.mean()
    sample_logprobs = trialfunc(samples)
    loss = ((local_energies - mean_local_energy) * sample_logprobs).mean()
    grads_wrt_parameters = torch.autograd.grad(loss, trialfunc.parameters())
    with torch.no_grad():
        for (param, grad) in zip(trialfunc.parameters(), grads_wrt_parameters):
            param -= step_size*grad

def full_hessian(loss_grad_vector, parameters):
    all_rows = []
    for i in range(loss_grad_vector.shape[-1]):
        hessian_row, = torch.autograd.grad(loss_grad_vector[..., i], parameters, retain_graph=True)
        all_rows.append(hessian_row)
    return torch.stack(all_rows)

def slow_empirical_fisher(tf, samples):
    flat_samples = samples.view(-1, samples.shape[-1])
    param = list(tf.parameters())[0]
    final_hessian = torch.zeros(param.shape[0], param.shape[0])
    for i in range(flat_samples.shape[0]):
        print(i)
        sample = flat_samples[i]
        logprob = tf(sample)
        logprob_grad, = torch.autograd.grad(logprob, param, create_graph=True)
        logprob_hess = full_hessian(logprob_grad, param)
        final_hessian += logprob_hess
    final_hessian /= flat_samples.shape[0]
    return -final_hessian

def energy_minimize_natural(trialfunc, samples, optimizer, step_size=5e-2):
    # computes natural gradient
    local_energies = trialfunc.local_energy(samples).detach()
    mean_local_energy = local_energies.mean()
    print('energy is ', mean_local_energy)
    fisher_info_mat = slow_empirical_fisher(trialfunc, samples)
    sample_logprobs = trialfunc(samples)
    loss = ((local_energies - mean_local_energy) * sample_logprobs).mean()
    grad, = torch.autograd.grad(loss, trialfunc.parameters(), create_graph=True)
    param = list(trialfunc.parameters())[0]
    with torch.no_grad():
        param -= step_size*(torch.inverse(fisher_info_mat) @ grad)


def energy_minimize_newton(trialfunc, samples, optimizer, step_size=5e-2):
    local_energies = trialfunc.local_energy(samples).detach()
    mean_local_energy = local_energies.mean()
    print('energy is ', mean_local_energy)
    sample_logprobs = trialfunc(samples)
    loss = ((local_energies - mean_local_energy) * sample_logprobs).mean()
    grads_wrt_parameters = torch.autograd.grad(loss, trialfunc.parameters(), create_graph=True)
    hessians_wrt_parameters = [full_hessian(grad, param) for grad, param in zip(grads_wrt_parameters, trialfunc.parameters())]
    with torch.no_grad():
        for (param, grad, hessian) in zip(trialfunc.parameters(), grads_wrt_parameters, hessians_wrt_parameters):
            print('update', step_size * (torch.inverse(hessian) @ grad))
            print('inverse hessian', torch.inverse(hessian))
            print('grad', grad)
            param -= step_size*(torch.inverse(hessian) @ grad)




def vmc_iterate(tf, init_config, num_iters=100):
    opt = optim.SGD(tf.parameters(), lr=5e-2,momentum=0.3)
    # propdist = NormalProposal(0.3)
    propdist = ClipNormalProposal(0.1)
    for i in range(num_iters):
        results=metropolis_symmetric(tf, init_config, propdist, num_walkers=100, num_steps=10000)
        energy_minimize_step(tf, results, opt)
        print(tf.alpha)
        print(tf.alpha[...,0])

def harmonic_energy_alpha_values():
    vals = np.arange(0.5,1.5,0.1)
    means = []
    grads = []
    for alpha_val in vals:
        print(alpha_val)
        tf = HarmonicTrialFunction(torch.tensor(alpha_val))
        init_config = 0.5*torch.ones(100,1)
        samples = metropolis_symmetric(tf, init_config, normal_proposal, num_walkers=100, num_steps=20000)
        means.append(torch.mean(tf.local_energy(samples)).item())
        est_grad = local_energy_est_grad(tf, samples)
        grads.append(est_grad)
    return vals, means, grads

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

def helium_energy_alpha_values():
    vals = np.arange(1.2,2.5,0.1)
    means = []
    propdist = ClipNormalProposal(0.05, min_val=0.0)
    for alpha_val in vals:
        print(alpha_val)
        tf = HeliumTrialWavefunction(torch.ones(1)*alpha_val)
        init_config = 0.5*torch.ones(100, 3)
        samples = metropolis_symmetric(tf, init_config, propdist, num_walkers=100, num_steps=20000)
        means.append(torch.mean(tf.local_energy(samples)).item())
        print(means[-1])
    return vals, means

if __name__ == '__main__':
    # tf = twoinone(torch.tensor([5.0,6.0]))
    # init_config = 0.5*torch.rand(2)
    # vmc_iterate(tf, init_config)
    # helium_energy_alpha_values()
#if __name__ == '__main__':
    tf = HarmonicTrialFunction(torch.tensor([1.3]))
    init_config = 0.5*torch.ones(1000, 1)
    # vmc_iterate(tf, init_config)
    vals, means, grads = harmonic_energy_alpha_values()
    plt.plot(vals, means)
    plt.plot(vals, [(a**2)/2.0 + 1.0/(2*a**2) for a in vals])
    plt.show()
    plt.plot(vals, grads)
    plt.plot(vals, [-1.0/(a**3) + a for a in vals])
    plt.show()
    print(tf.alpha)
    # helium_energy_alpha_values()

