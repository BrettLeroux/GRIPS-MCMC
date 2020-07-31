import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import torch
from wavefunction import HydrogenTrialWavefunction as htwf
from wavefunction import HeliumTrialWavefunction as hetwf
from tracehess import autograd_trace_hessian, gradient_f, hessian_f


def Auto_Hamiltonian_Generator_atoms(ansatz, N_bodies, config):  #Unitless for now.
    Kinetic_energy = -0.5*autograd_trace_hessian( ansatz,config.unsqueeze(0))
    Kinetic_energy_1 = 0
    for i in range(N_bodies):
        r = (gradient_f(config, ansatz)[:, i])/config[...,i]
        Kinetic_energy_1 = r
        
        
    Potential_nucleous_electron = 0
    for i in range(N_bodies):
        Potential_nucleous_electron = Potential_nucleous_electron-(N_bodies/(config[...,i].squeeze(dim=-1)))*ansatz(config)
    #Potential_electron_electron = 0
    Potential_energy_Helium_e_e = 1/(torch.sqrt(config[...,0]**2+config[...,1]**2+torch.abs(config[...,1])*torch.abs(config[...,0])*torch.cos(config[...,2])))
    #for i in range(N_bodies):
     #   for j in range(N_bodies):
      #      Potential_electron_electron = Potential_electron_electron + 1/(torch.sqrt(x[...,i]**2+x[...,j]**2+torch.abs(x[...,i])*torch.abs(x[...,j])*torch.cos(x[...,2])))
            
    Energy_total = Kinetic_energy+Potential_nucleous_electron+Potential_energy_Helium_e_e 
    return Energy_total
tf_0=hetwf(torch.ones(1)*1.8)
tf_1=htwf(torch.ones(1))
def Local_energy_of_psi_at_x( ansatz, N_bodies, config):
    Local_energy_at_x = Auto_Hamiltonian_Generator_atoms(ansatz, N_bodies, config)/ansatz(config)
    
    
    return Local_energy_at_x 
#print(Local_energy_of_psi_at_x(tf_1.hydro_ansatz_sup,1,torch.ones(1)))
print(Local_energy_of_psi_at_x(tf_0.helium_ansatz_sup_simple,2,torch.ones(3)))
