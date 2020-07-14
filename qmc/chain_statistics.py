import torch
import numpy as np
import arviz

def compute_ess(chains):
    # shape N_walkers, N_samples
    return arviz.ess(chains.cpu().detach().numpy())
