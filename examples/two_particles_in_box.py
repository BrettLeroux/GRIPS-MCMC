# %%
import torch
import matplotlib.pyplot as plt
from qmc.mcmc import metropolis_symmetric, ClipNormalProposal
from qmc.wavefunction import TwoParticlesInOneDimBox as twoinone

# %%
Tdat = [3.0, 4.0]
a = torch.tensor(Tdat)
# a = torch.ones(2)
tf = twoinone(a)

#
n_walkers=20

#init_config = torch.ones(n_walkers,2)*0.5
#results = metropolis_symmetric(tf, init_config, ClipNormalProposal(sigma=0.001, min_val=0.1, max_val=0.45), num_walkers=n_walkers, num_steps=10000)

init_config = torch.rand(n_walkers,2)
results = metropolis_symmetric(tf, init_config, ClipNormalProposal(sigma=0.05), num_walkers=n_walkers, num_steps=10000)

# %%
#results_numpy  = results.view(-1,3).numpy()
# %%

# %%
#plt.scatter(results_numpy[:,0],results_numpy[:,1], s=1)
#plt.savefig("box.png")samples[:, 1000:, :]
print('configurations producing nan local energies are', results[torch.isnan(tf.local_energy(results))])
print(torch.mean(tf.local_energy(results)))

      
