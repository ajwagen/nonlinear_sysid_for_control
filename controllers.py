import numpy as np
import torch



class AffineFeedbackController:
    '''
    Controller of form u_h = K_h * x_h + util_h
    '''
    def __init__(self,K,util,K0=None,time_invariant=False):
        '''
        Inputs:
            K - linear feedback component of controller
            util - affine component of controller
            K0 - baseline linear feedback controller (for example, baseline stabilizing controller)
            time_invariate - if False, controller varies with h, otherwise controller is the same for each h
        '''
        self.K = K
        self.util = util
        self.K0 = K0
        self.time_invariant = time_invariant
    
    def get_input(self,x,h):
        '''
        Inputs:
            x - current state
            h - current step

        Outputs: 
            u - input to play
        '''
        dimx = len(x)
        if self.K0 is None:
            if self.time_invariant:
                Kh = self.K
            else:
                Kh = self.K[:,dimx*h:dimx*(h+1)]
        else:
            if self.time_invariant:
                Kh = self.K0 + self.K
            else:
                Kh = self.K0[:,dimx*h:dimx*(h+1)] + self.K[:,dimx*h:dimx*(h+1)]    
        if self.time_invariant:
            utilh = self.util
        else:
            utilh = self.util[:,h]
        if len(x.shape) > 1:
            utilh = torch.outer(utilh,torch.ones(x.shape[1]))
        return Kh @ x + utilh

    def init_params(self,requires_grad=False,random=False):
        if random:
            Kinit = 0.005*(torch.rand(self.K.shape) - 0.5*torch.ones(self.K.shape))
            utilinit = 0.005*(torch.rand(self.util.shape) - 0.5*torch.ones(self.util.shape))
            self.K = torch.autograd.Variable(Kinit, requires_grad=requires_grad)
            self.util = torch.autograd.Variable(utilinit, requires_grad=requires_grad)
        else:
            self.K = torch.autograd.Variable(self.K, requires_grad=requires_grad)
            self.util = torch.autograd.Variable(self.util, requires_grad=requires_grad)

    def zero_params(self):
        self.K = torch.zeros(self.K.shape)
        self.util = torch.zeros(self.util.shape)

    def update_params(self,new_params):
        self.K = new_params[0]
        self.util = new_params[1]

    def detach(self):
        self.K = self.K.detach()
        self.util = self.util.detach()

    def get_controller_params(self):
        return [self.K,self.util]

    def generate_new(self,params):
        return AffineFeedbackController(params[0],params[1],K0=self.K0,time_invariant=self.time_invariant)

    def randomize_params(self):
        self.zero_params()

    def generate_duplicate(self):
        return AffineFeedbackController(self.K.clone().detach(),self.util.clone().detach(),K0=self.K0,time_invariant=self.time_invariant)





class CarController:
    '''
    Controller used for car experiment. 4-parameter hierarchical controller class.
    '''
    def __init__(self,K=None,K0=None):
        '''
        Inputs:
            K - controller parameters
            K0 - baseline controller parameters (for example, baseline stabilizing controller)
        '''
        if K is None:
            self.K = torch.zeros(4)
        else:
            self.K = K
        self.K0 = K0
    
    def get_input(self,x,h):
        '''
        Inputs:
            x - current state
            h - current step

        Outputs: 
            u - input to play
        '''
        if len(x.shape) > 1:
            N = x.shape[1]
            u = torch.zeros(2,N)
            ud = -self.K[0]*x[0:2,:] - self.K[1]*x[2:4,:]
            a = torch.stack((torch.cos(x[4,:]),torch.sin(x[4,:])),dim=1)
            a = torch.multiply(ud,a.T)
            a = torch.sum(a,dim=0)
            u[0,:] = a
            thetap = torch.atan(ud[1,:] / ud[0,:])
            u[1,:] = -self.K[2]*(x[4,:] - thetap) - self.K[3]*x[5,:]
        else:
            u = torch.zeros(2)
            ud = -self.K[0]*x[0:2] - self.K[1]*x[2:4]
            u[0] = ud @ torch.tensor([torch.cos(x[4]),torch.sin(x[4])])
            thetap = torch.atan(ud[1] / ud[0])
            u[1] = -self.K[2]*(x[4] - thetap) - self.K[3]*x[5]
        if self.K0 is not None:
            if len(x.shape) > 1:
                N = x.shape[1]
                ud = -self.K0[0]*x[0:2,:] - self.K0[1]*x[2:4,:]
                a = torch.stack((torch.cos(x[4,:]),torch.sin(x[4,:])),dim=1)
                a = torch.multiply(ud,a.T)
                a = torch.sum(a,dim=0)
                u[0,:] += a
                thetap = torch.atan(ud[1,:] / ud[0,:])
                u[1,:] += -self.K0[2]*(x[4,:] - thetap) - self.K0[3]*x[5,:]
            else:
                ud = -self.K0[0]*x[0:2] - self.K0[1]*x[2:4]
                u[0] += ud @ torch.tensor([torch.cos(x[4]),torch.sin(x[4])])
                thetap = torch.atan(ud[1] / ud[0])
                u[1] += -self.K0[2]*(x[4] - thetap) - self.K0[3]*x[5]
        return u

    def init_params(self,requires_grad=False,random=False):
        if random:
            Kinit = 0.005*(torch.rand(self.K.shape) - 0.5*torch.ones(self.K.shape))
            self.K = torch.autograd.Variable(Kinit, requires_grad=requires_grad)
        else:
            self.K = torch.autograd.Variable(self.K, requires_grad=requires_grad)

    def zero_params(self):
        self.K = 0.1*torch.ones(self.K.shape)

    def update_params(self,new_params):
        self.K = new_params[0]

    def set_K0(self,K0):
        self.K0 = K0.detach()

    def detach(self):
        self.K = self.K.detach()

    def get_controller_params(self):
        return [self.K]

    def generate_new(self,params):
        return CarController(K=params[0],K0=self.K0)

    def randomize_params(self):
        self.zero_params()

    def generate_duplicate(self):
        return CarController(K=self.K.clone().detach(),K0=self.K0)




class KernelLinearController:
    '''
    Nonlinear feedback controller. Plays input u_h = K * x_h + K_kernel * phi(x) + v, for phi(x) a nonlinear transform of
    x. Time-invariant by default.
    '''
    def __init__(self,K,K_kernel,kernel,K0=None,v=None):
        '''
        Inputs:
            K - linear feedback component of controller
            K_kernel - nonlinear feedback component of controller
            kernel - kernel object to compute nonlinear transform of x
            K0 - baseline linear feedback controller (for example, baseline stabilizing controller)
            v - affine component of controller
        '''
        self.K = K
        self.K_kernel = K_kernel
        self.K0 = K0
        self.kernel = kernel
        self.v = v
    
    def get_input(self,x,h):
        '''
        Inputs:
            x - current state
            h - current step

        Outputs: 
            u - input to play
        '''
        u = self.K @ x + self.K_kernel @ self.kernel.phi(x,0)
        if self.K0 is not None:
            u = u + self.K0 @ x
        if self.v is not None:
            if len(x.shape) > 1:
                u = u + torch.outer(self.v,torch.ones(x.shape[1]))
            else:
                u = u + self.v
        return u

    def init_params(self,requires_grad=False,random=False):
        if random:
            Kinit = 0.005*(torch.rand(self.K.shape) - 0.5*torch.ones(self.K.shape))
            self.K = torch.autograd.Variable(Kinit, requires_grad=requires_grad)
            K_kernelinit = 0.005*(torch.rand(self.K_kernel.shape) - 0.5*torch.ones(self.K_kernel.shape))
            self.K_kernel = torch.autograd.Variable(K_kernelinit, requires_grad=requires_grad)
            if self.v is not None:
                vinit = 0.005*(torch.rand(self.v.shape) - 0.5*torch.ones(self.v.shape))
                self.v = torch.autograd.Variable(vinit, requires_grad=requires_grad)
        else:
            self.K = torch.autograd.Variable(self.K, requires_grad=requires_grad)
            self.K_kernel = torch.autograd.Variable(self.K_kernel, requires_grad=requires_grad)
            if self.v is not None:
                self.v = torch.autograd.Variable(self.v, requires_grad=requires_grad)

    def randomize_params(self):
        self.zero_params()

    def zero_params(self):
        self.K = torch.zeros(self.K.shape)
        self.K_kernel = torch.zeros(self.K_kernel.shape)
        if self.v is not None:
            self.v = torch.zeros(self.v.shape)

    def update_params(self,new_params):
        self.K = new_params[0]
        self.K_kernel = new_params[1]
        if self.v is not None:
            self.v = new_params[2]

    def detach(self):
        self.K = self.K.detach()
        self.K_kernel = self.K_kernel.detach()
        if self.v is not None:
            self.v = self.v.detach()

    def get_controller_params(self):
        if self.v is not None:
            return [self.K, self.K_kernel, self.v]
        else:
            return [self.K, self.K_kernel]

    def generate_parallel(self,params,N):
        if self.K0 is not None:
            K0_parallel = torch.kron(torch.eye(N),self.K0)
        else:
            K0_parallel = None
        return KernelLinearController(params[0],params[1],self.kernel,K0=K0_parallel)

    def generate_new(self,params):
        if len(params) > 2:
            return KernelLinearController(params[0],params[1],self.kernel,K0=self.K0,v=params[2])
        else:
            return KernelLinearController(params[0],params[1],self.kernel,K0=self.K0)

    def generate_duplicate(self):
        if self.v is not None:
            return KernelLinearController(self.K.clone().detach(),self.K_kernel.clone().detach(),self.kernel,K0=self.K0,v=self.v.clone().detach())
        else:
            return KernelLinearController(self.K.clone().detach(),self.K_kernel.clone().detach(),self.kernel,K0=self.K0)









    