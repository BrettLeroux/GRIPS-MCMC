import torch
import numpy as np
from torch import nn

                                                              

class dim2Rosenbrock(nn.Module):
    def __init__(self, alpha, beta):
        super(dim2Rosenbrock, self).__init__()
        self.alpha = nn.Parameter(alpha)
        self.beta = nn.Parameter(beta)

    def forward(self, x):
        # output logprob
        dim2 = x.ndimension() > 2
        dim1 = x.ndimension() > 1 
        if dim2:
            result = - self.alpha * (x[:, :, 0] -1)**2 - self.beta * (x[:, :, 0]- x[:, :, 1]**2)**2
        else:
            x = x if dim1 else x.unsqueeze(0)
            result = - self.alpha * (x[:, 0] -1)**2 - self.beta * (x[:, 0]- x[:, 1]**2)**2
        return result if dim1 else result.squeeze(0)
        

    # gives the normalization constant
    def normalization(self):
        return torch.sqrt(self.alpha * self.beta) / np.pi





class Rosenbrock(nn.Module):
    # Rosenbrock(n1,n2) creates the "Hybrid Rosenbrock" from https://arxiv.org/pdf/1903.09556.pdf with parameters a = 1/20, b = 5
    # The dimension is (n1-1)n2 + 1
    def __init__(self, n1, n2):
        super(Rosenbrock, self).__init__()
        self.n1 = n1
        self.n2 = n2
        
    def forward(self, x):
        # output logprob
        dim2 = x.ndimension() > 2
        dim1 = x.ndimension() > 1
        if dim2:
            y = x[:, :, 0]
            x = torch.reshape(x[:, :, 1:], (x.size()[0], x.size()[1], self.n2, self.n1-1))
            xx = x[:, :, :, 1:]
            xxx = x[:, :, :, 0:-1]
            result = - (1/20) * (y -1)**2 
            - 5 * torch.sum(torch.sum((xx - xxx**2)**2, -1), -1)
            
        else:
            x = x if dim1 else x.unsqueeze(0)
            y = x[:, 0]
            x = torch.reshape(x[:, 1:], (x.size()[0], self.n2, self.n1-1))
            xx = x[:, :, 1:]
            xxx = x[:, :, 0:-1]
            result = - (1/20) * (y -1)**2 - 5 * torch.sum(torch.sum((xx - xxx**2)**2,-1), -1)
        return result if dim1 else result.squeeze(0)
      
      
      
    # gives the normalization constant
    def normalization(self):
    return ((1/20)**(1/2) * 5**((self.n2 * (self.n1-1)) / 2) ) / (np.pi)**((self.n2 * (self.n1 - 1) +1)/2)
    
    
    
    #Returns a N x d array of iid samples from the Hybrid rosenbrock above
    def Iid(self, N):
        a = 1/20, b = 5, mu = 1
        S = np.zeros((1,self.n2*(self.n1-1)+1))
        for k in range(N):
            s = np.array([[]])
            y = np.random.normal(mu, 1/(2*a), size = (1,1))
            s = np.concatenate((s,y),1)
            for j in range(1, self.n2 + 1):
                z = y
                for i in range(2, self.n1 + 1):
                    x = np.random.normal(z**2, 1/(2*b), size = (1,1))
                    s = np.concatenate((s,x),1)
                    z = x
            S = np.concatenate((S,s))
        return S[1:,:] 
    
    
    
    
  

class MixtureOfGaussians(nn.Module):

    def __init__(self, mean_list, covmat_list):
        super(MixtureOfGaussians, self).__init__()

        self.gaussians_list = [torch.distributions.MultivariateNormal(m, c) for m, c in zip(mean_list, covmat_list)]

    def forward(self, x):
        result = 0.0
        for gaussian in self.gaussians_list:
            result = result + gaussian.log_prob(x)
        return result

