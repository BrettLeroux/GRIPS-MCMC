# %%
import torch
import matplotlib.pyplot as plt
from qmc.mcmc import metropolis_symmetric, ClipNormalProposal
from qmc.wavefunction import OneParticlesInOneDimBox as oneinone

# %%
Tdat = 0.86
a = torch.tensor(Tdat)
tf = oneinone(a)


# 
n_walkers=10
init_config = 0.5*torch.rand(n_walkers,1)
results = metropolis_symmetric(tf, init_config, ClipNormalProposal(sigma=0.01, min_val=-0.99, max_val=0.99), num_walkers=n_walkers, num_steps=10000)
# %%
#results_numpy  = results.view(-1,3).numpy()
# %%

# %%
#plt.scatter(results_numpy[:,0],results_numpy[:,1], s=1)
#plt.savefig("box.png")samples[:, 1000:, :]
print(results)
print(torch.mean(tf.local_energy(results)))


      

