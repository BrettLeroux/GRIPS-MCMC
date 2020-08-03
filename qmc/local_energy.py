import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import torch
from wavefunction import HydrogenTrialWavefunction as htwf
from wavefunction import HeliumTrialWavefunction as hetwf
from wavefunction import ThreeElectronBasicAnsatzNoVandermonde as three_no_van
from tracehess import autograd_trace_hessian, gradient_f, hessian_f
from math import comb

def Auto_Hamiltonian_Generator_atoms(ansatz, N_bodies, config):  #Unitless for now.
    Kinetic_source = autograd_trace_hessian( ansatz,config.unsqueeze(0), return_grad =True)
    Kinetic_energy_0 = -0.5*Kinetic_source[0]
    Kinetic_energy_1 = 0 
    for i in range(N_bodies):
        r = (Kinetic_source[1][:, i])/config[...,i]
        Kinetic_energy_1 = Kinetic_energy_1-r
    Kinetic_total = Kinetic_energy_0 + Kinetic_energy_1
    #print(Kinetic_total)
        
    Potential_nucleous_electron = 0
    for i in range(N_bodies):
        Potential_nucleous_electron = Potential_nucleous_electron-(N_bodies/(config[...,i].squeeze(dim=-1)))*ansatz(config)
    #print(Potential_nucleous_electron)
    Potential_electron_electron = 0
    Potential_energy_Helium_e_e = 1/(torch.sqrt(config[...,0]**2+config[...,1]**2+torch.abs(config[...,1])*torch.abs(config[...,0])*torch.cos(config[...,2])))
    k = 1 
    for i in range(N_bodies):
        for j in range(i+1,N_bodies):
            Potential_electron_electron = Potential_electron_electron + (ansatz(config))/(torch.sqrt(config[...,i]**2+config[...,j]**2+torch.abs(config[...,i])*torch.abs(config[...,j])*torch.cos(config[...,N_bodies -1+k])))
            k+=1
    #print(Potential_electron_electron)
    Energy_total = Kinetic_total+Potential_nucleous_electron+Potential_electron_electron
    #print(Energy_total)
    print(ansatz(config))
    return Energy_total
   
tf_0=hetwf(torch.ones(1)*1.8)
tf_1=htwf(torch.ones(2))
tf_2=three_no_van(torch.ones(1))
def Local_energy_of_psi_at_x( ansatz, N_bodies, config):
    Local_energy_at_x = Auto_Hamiltonian_Generator_atoms(ansatz, N_bodies, config)/ansatz(config)
    
    
    return Local_energy_at_x
#print(Local_energy_of_psi_at_x(tf_1.hydro_ansatz_sup,1,torch.ones(1)))
#print(Local_energy_of_psi_at_x(tf_0.helium_ansatz_sup_simple,2,torch.ones(3)))
print(Local_energy_of_psi_at_x(tf_2.three_elec_ansatz_sup_simple,3,torch.ones(6)))     
