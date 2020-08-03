import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import torch
from qmc.wavefunction import HydrogenTrialWavefunction
from qmc.wavefunction import HeliumTrialWavefunction
from qmc.tracehess import autograd_trace_hessian, gradient_f, hessian_f

def auto_hamiltonian_generator_atoms(ansatz, N_bodies, config):  #Unitless for now.
    Kinetic_source = autograd_trace_hessian( ansatz,config, return_grad =True)
    Kinetic_energy_0 = -0.5*Kinetic_source[0]
    Kinetic_energy_1 = 0
    for i in range(N_bodies):
        r = (Kinetic_source[1][..., i])/config[...,i]
        Kinetic_energy_1 = Kinetic_energy_1-r
    Kinetic_total = Kinetic_energy_0 + Kinetic_energy_1

    Potential_nucleous_electron = 0
    for i in range(N_bodies):
        Potential_nucleous_electron = Potential_nucleous_electron-(N_bodies/(config[...,i].squeeze(dim=-1)))*ansatz(config)
    Potential_electron_electron = 0
    k = 1
    for i in range(N_bodies):
        for j in range(i+1,N_bodies):
            Potential_electron_electron = Potential_electron_electron + (ansatz(config))/(torch.sqrt(config[...,i]**2+config[...,j]**2+torch.abs(config[...,i])*torch.abs(config[...,j])*torch.cos(config[...,N_bodies -1+k])))
            k+=1
    Energy_total = Kinetic_total+Potential_nucleous_electron+Potential_electron_electron
    return Energy_total
   
tf_0=HeliumTrialWavefunction(torch.ones(1)*1.8)
tf_1=HydrogenTrialWavefunction(torch.ones(2))

def Local_energy_of_psi_at_x( ansatz, N_bodies, config):
    Local_energy_at_x = auto_hamiltonian_generator_atoms(ansatz, N_bodies, config)/ansatz(config)
    
    
    return Local_energy_at_x

print(Local_energy_of_psi_at_x(tf_0, 2, torch.rand(10,4, 3)))

