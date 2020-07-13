# %%
import torch
import matplotlib.pyplot as plt
from mcmc import metropolis_symmetric, clip_mvnormal_proposal
from wavefunction import ParticleBoxFunction

# %%
d=2
tf = ParticleBoxFunction2(torch.ones(2))

# %%
n_walkers=10
init_config = torch.ones(n_walkers,2)
results = metropolis_symmetric(tf, init_config, clip_mvnormal_proposal, num_walkers=n_walkers, num_steps=10000)

# %%
results_numpy  = results.view(-1,2).numpy()

# %%
plt.scatter(results_numpy[:,0],results_numpy[:,1], s=1)
plt.savefig("/Users/curry/Downloads/particle_box_2d.png")

# %%
torch.mean(tf.local_energy(results))
