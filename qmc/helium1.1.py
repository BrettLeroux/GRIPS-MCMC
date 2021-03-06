# %%
import torch
import matplotlib.pyplot as plt
from qmc.mcmc import metropolis_symmetric, ClipNormalProposal
from qmc.wavefunction import HeliumTrialWavefunction

# %%
d=3
tf = HeliumTrialWavefunction(torch.ones(1))

# 
n_walkers=10
init_config = torch.ones(n_walkers,3)
results = metropolis_symmetric(tf, init_config, ClipNormalProposal(sigma=0.01, min_val=0.0), num_walkers=n_walkers, num_steps=10000)
print(torch.min(results))
# %%
#results_numpy  = results.view(-1,3).numpy()

# %%
#plt.scatter(results_numpy[:,0],results_numpy[:,1], s=1)
#plt.savefig("box.png")samples[:, 1000:, :]
print(torch.mean(tf.local_energy(results[:,1000,:])))
