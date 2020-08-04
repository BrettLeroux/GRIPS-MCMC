import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import torch

N =2



alpha = torch.rand(1,N)
print(alpha)
Config = torch.rand(1,N)
print(Config)
ansatz_hydrogenic = []
slater_det_matrix = torch.zeros(N,N)
for i in range(N):
    ansatz_hydrogenic.append(torch.exp(-alpha[:,i]*Config[:,i]))
    print(ansatz_hydrogenic)


a = 1

for i in range(N-1):
    for j in range(i+1,N):
        a=a*(ansatz_hydrogenic[i]-ansatz_hydrogenic[j])
        print(a)
for i in range(N):
    for j in range(N):
       slater_det_matrix[i][j]=torch.exp(-alpha[:,i]*Config[:,i])
       D = torch.det(slater_det_matrix)
       print(D)
print(slater_det_matrix)       
print(a)
print(D)
