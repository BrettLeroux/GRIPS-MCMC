import torch
from torch.distributions import Normal, Bernoulli
import numpy as np

def normal_proposal(old_point):
    # symmetric
    return Normal(old_point, 0.3*torch.ones_like(old_point)).sample()

def clip_normal_proposal(old_point):
    samp = Normal(old_point, 0.3*torch.ones_like(old_point)).sample()
    samp.clamp_min_(0.0)
    return samp

def symmetric_mh_acceptance_ratio(logprob, old_config, new_config):
    logacc = torch.min(torch.tensor(0.0), logprob(new_config) - logprob(old_config))
    return torch.exp(logacc)

def asymmetric_mh_acceptance_ratio(logprob, new_old_logprob, old_new_logprob, old_config, new_config):
    logacc = torch.min(torch.tensor(0.0), logprob(new_config) +
                       old_new_logprob
                       - logprob(old_config)
                      - new_old_logprob)
    return torch.exp(logacc)

def metropolis_symmetric(trialfunc, proposal, num_walkers=2,num_steps=100):
    # with more walkers
    # design choice: walkers are always the batch dim
    config = torch.zeros(num_walkers, 1)
    all_configs = []
    for step in range(num_steps):
        next_config = proposal(config)
        with torch.no_grad():
            acc = symmetric_mh_acceptance_ratio(trialfunc, config, next_config)
        # acc shape should be (num_walkers, 1)
            accept_or_reject = Bernoulli(acc).sample() # accept is 1, reject is 0
            config = accept_or_reject*next_config + (1.0 - accept_or_reject)*config
            all_configs.append(config.clone()) # can we skip clone here?
    return torch.stack(all_configs, dim=1) # dim=1 to make walkers be the batch dim

def metropolis_asymmetric(trialfunc, proposal, num_walkers=2,num_steps=100):
    # with more walkers
    # design choice: walkers are always the batch dim
    config = torch.zeros(num_walkers, 1)
    all_configs = []
    for step in range(num_steps):
        next_config, next_current_logprob, current_next_logprob = proposal(config)
        with torch.no_grad():
            acc = asymmetric_mh_acceptance_ratio(trialfunc,
                                                 next_current_logprob,
                                                 current_next_logprob,
                                                 config, next_config)
            # acc shape should be (num_walkers, 1)
            accept_or_reject = Bernoulli(acc).sample() # accept is 1, reject is 0
            config = accept_or_reject*next_config + (1.0 - accept_or_reject)*config
            all_configs.append(config.clone()) # can we skip clone here?
    return torch.stack(all_configs, dim=1) # dim=1 to make walkers be the batch dim

def unadjusted_langevin(trialfunc, num_walkers=2, num_steps=100, eta=0.01):
    # seems hard to get this to converge
    config = torch.zeros(num_walkers, 1, requires_grad=True)
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


def mala(trialfunc, num_walkers=2, num_steps=100, eta=0.01):
    # this isn't right -- we need different MH filter for asymmetric proposal
    config = torch.zeros(num_walkers, 1)
    grad_out = torch.ones_like(config, requires_grad=False)
    brownian_dist = Normal(torch.zeros_like(config, requires_grad=False),
                           torch.ones_like(config, requires_grad=False))

    def mala_proposal(old_point):
        #
        old_point.requires_grad_(True)
        curr_config_logprobs = trialfunc(old_point)
        grads, = torch.autograd.grad(curr_config_logprobs,
                                     old_point,
                                     grad_outputs=grad_out,
                                     retain_graph=False)
        with torch.no_grad():
            propdist = Normal(old_point + eta * grads, np.sqrt(2.0 * eta))
            next_config = propdist.sample()
            new_old_logprob = propdist.log_prob(next_config)
        next_config.requires_grad_(True)
        next_config_logprobs = trialfunc(next_config)
        next_grads, = torch.autograd.grad(next_config_logprobs,
                                          next_config,
                                          grad_outputs=grad_out,
                                          retain_graph=False)
        with torch.no_grad():
            reverse_propdist = Normal(next_config + eta * next_grads, np.sqrt(2.0 * eta))
            old_new_logprob = reverse_propdist.log_prob(old_point)
        return next_config, new_old_logprob, old_new_logprob

    return metropolis_asymmetric(trialfunc, mala_proposal, num_walkers=num_walkers, num_steps=num_steps)
