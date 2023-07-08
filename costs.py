import numpy as np
import torch


class QuadraticCost:
    '''
    Quadratic cost
    '''
    def __init__(self,R):
        '''
        Inputs:
            R - cost matrix
        '''
        self.R = R

    def get_cost(self,x,u,h,N=-1):
        '''
        Inputs:
            x - state
            u - input
            h - current step
            N - if N > 0, then state (resp. input) is understood to be N states (resp. inputs) concatenated, and cost is computed for 
                    each state-input (used when evaluating many trajectories in parallel)

        Outputs: 
            If N==-1, then scalar cost for (x,u,h). If N > 0, then list of length N with cost of each individual state-input
        '''
        if N > 0:
            return self.get_cost_N(x,u,h,N)
        z = torch.concat((x,u))
        if len(x.shape) > 1:
            return torch.trace(z.T @ self.R @ z)
        else:
            return z @ self.R @ z

    def get_cost_N(self,x,u,h,N):
        d1,T = x.shape
        d2,T = u.shape
        dimx = torch.tensor(d1/N, dtype=torch.int)
        dimu = torch.tensor(d2/N, dtype=torch.int)

        cost_vals = torch.zeros(N)
        for i in range(N):
            cost_vals[i] = (self.get_cost(x[dimx*i:dimx*(i+1),:],u[dimu*i:dimu*(i+1),:],h=h,N=-1))
        return cost_vals
    
    def get_cost_params(self):
        return self.R



class SpecialCost:
    '''
    Cost for motivating example
    '''
    def __init__(self,center=None):
        if center is None:
            self.center = 10
        else:
            self.center = center
        return

    def get_cost(self,x,u,h,N=-1):
        if len(x.shape) > 1:
            return torch.trace((x - self.center*torch.ones(x.shape)).T @ (x - self.center*torch.ones(x.shape))) + 0.01*torch.trace(u.T @ u)
        else:
            return (x - self.center*torch.ones(x.shape))**2 + 0.01*(u**2)

    def get_center(self):
        return self.center

    def get_cost_old(self,x,u,h,N=-1):
        z = torch.concat((x,u))
        offset = 10*torch.ones(z.shape)
        if len(x.shape) > 1:
            return torch.trace((z - offset).T @ (z - offset))
        else:
            return (z - offset) @ (z - offset)

