# %%
import torch
import matplotlib.pyplot as plt
from qmc.mcmc import metropolis_symmetric, ClipNormalProposal
from qmc.wavefunction import TwoParticlesInOneDimBox as twoinone

# %%
Tdat = [0.5, 0.8]
a = torch.tensor(Tdat)
tf = twoinone(a)

#
n_walkers=10
<<<<<<< HEAD
init_config = torch.ones(n_walkers,2)
results = metropolis_symmetric(tf, init_config, ClipNormalProposal(sigma=0.001, min_val=0.1, max_val=0.45), num_walkers=n_walkers, num_steps=10000)
=======
init_config = torch.rand(n_walkers,2)
results = metropolis_symmetric(tf, init_config, ClipNormalProposal(sigma=0.05, min_val=0.0, max_val=1.0), num_walkers=n_walkers, num_steps=10000)
>>>>>>> 38129d5fc01f77195627e4a122e31514102df434
# %%
#results_numpy  = results.view(-1,3).numpy()
# %%

# %%
#plt.scatter(results_numpy[:,0],results_numpy[:,1], s=1)
#plt.savefig("box.png")samples[:, 1000:, :]
print(torch.mean(tf.local_energy(results)))

      
