import numpy as np
import math
#Gradient Function
x= np.array([[2],[2],[2]])
def f(x):
    return x[0]**2*x[1]**2*x[2]**2
def gradient_f(x, f):
  assert (x.shape[0] >= x.shape[1]), "the vector should be a column vector"
  x = x.astype(float)
  N = x.shape[0]
  gradient = []
  for i in range(N):
    eps = 0.00001#abs(x[i]) *  np.finfo(np.float32).eps 
    xx0 = 1. * x[i]
    f0 = f(x)
    x[i] = x[i] + eps
    f1 = f(x)
    gradient.append(np.array([f1 - f0]).astype(np.float)/eps)
    x[i] = xx0
  return np.array(gradient).reshape(x.shape)
#Laplacian. This returns only zero values. debugging continues. 

#def laplacian_f (x, the_anzats):
#  N = x.shape[0]
#  gd_0 = gradient_f(x, the_anzats)
#  eps = 0.1
#  for i in range(N):
#    xx0 = 1.*x[i]
#    x[i] = x[i] + eps 
#    gd_1 =  gradient_f(x, the_anzats)[i]
#    lapnot = ((gd_1[i] - gd_0[i])/eps) #.reshape(x.shape[0])
#    x[i] = xx0
#  return (sum(lapnot))  
      
#def hessian_f (x, the_func):
 # N = x.shape[0]
#  hessian = np.zeros((N,N)) 
#  gd_0 = gradient_f( x, the_func)
#  eps = 1#np.linalg.norm(gd_0) * np.finfo(np.float32).eps 
#  for i in range(N):
#    xx0 = 1.*x[i]
#    x[i] = xx0 + eps
#    gd_1 =  gradient_f(x, the_func)
#    hessian[:,i] = ((gd_1 - gd_0)/eps).reshape(x.shape[0])
 #   x[i] =xx0
  #return hessian



