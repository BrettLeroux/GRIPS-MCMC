# %%
import torch
import matplotlib.pyplot as plt
from qmc.mcmc import metropolis_symmetric, clip_mvnormal_proposal
from qmc.wavefunction import HarmonicTrialFunction

# %%
d=3
tf = HarmonicTrialFunction(torch.ones(1))

# 
n_walkers=10
init_config = torch.rand(n_walkers,1)
results = metropolis_symmetric(tf, init_config, clip_mvnormal_proposal, num_walkers=n_walkers, num_steps=10000)

# %%
#results_numpy  = results.view(-1,3).numpy()

# %%
#plt.scatter(results_numpy[:,0],results_numpy[:,1], s=1)
#plt.savefig("box.png")
print(torch.mean(tf.local_energy(results)))
