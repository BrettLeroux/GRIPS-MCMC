import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import torch
from qmc.tracehess import autograd_trace_hessian, gradient_f, hessian_f


def auto_hamiltonian_generator_atoms(ansatz, N_bodies, config):  #Unitless for now.
    kinetic_source = autograd_trace_hessian( ansatz.wave,config, return_grad =True)
    kinetic_energy_0 = -0.5*kinetic_source[0]
    kinetic_energy_1 = 0
    for i in range(N_bodies):
        r = (kinetic_source[1][..., i])/config[...,i]
        kinetic_energy_1 = kinetic_energy_1-r
    kinetic_total = kinetic_energy_0 + kinetic_energy_1

    potential_nucleous_electron = 0
    for i in range(N_bodies):
        potential_nucleous_electron = potential_nucleous_electron-(N_bodies/(config[...,i].squeeze(dim=-1)))*ansatz.wave(config)
    potential_electron_electron = 0
    k = 1
    for i in range(N_bodies):
        for j in range(i+1,N_bodies):
            potential_electron_electron = potential_electron_electron + (ansatz.wave(config))/(torch.sqrt(config[...,i]**2+config[...,j]**2+torch.abs(config[...,i])*torch.abs(config[...,j])*torch.cos(config[...,N_bodies -1+k])))
            k+=1
    energy_total = kinetic_total+potential_nucleous_electron+potential_electron_electron
    return energy_total
   
