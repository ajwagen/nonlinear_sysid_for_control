import numpy as np
import numpy.random as rand
import torch
import matplotlib.pyplot as plt
import time
import copy
from controllers import AffineFeedbackController




class AffineOptimalCEController:
	'''
	Computes optimal controller on affine dynamical system.
	'''
	def __init__(self):
		return

	def optimize(self,environment,compute_hessian=None):
		'''
		Dynamic programming-based computation of optimal controller for affine dynamical system.

		Inputs: 
			environment - AffineDynamicalSystem object
			compute_hessian - inactive input

		Outputs:
			controller - AffineFeedbackController object giving optimal controller for environment
		'''
		dimx,dimu,dimz,H = environment.get_dim()
		A,B,v = environment.get_affine_dynamics()
		R = environment.get_cost_params()
		Rx = R[:dimx,:dimx]
		Ru = R[dimx:,dimx:]

		Kopt = torch.zeros(dimu,dimx*H)
		utilopt = torch.zeros(dimu,H)
		P = copy.copy(Rx)
		L = torch.zeros(dimx)
		c = torch.tensor(0)
		for h in range(H):
			Kh = torch.linalg.inv(Ru + B.T @ P @ B) @ B.T @ P @ A
			utilh = torch.linalg.inv(Ru + B.T @ P @ B) @ (B.T @ P @ v + 0.5 * B.T @ L)
			Kopt[:,dimx*(H-h-1):dimx*(H-h)] = Kh
			utilopt[:,H-h-1] = utilh

			P = (A - B @ Kh).T @ P @ (A - B @ Kh) + Kh.T @ Ru @ Kh + Rx
			L = 2*(v - B @ utilh) @ P @ (A - B @ Kh) + L @ (A - B @ Kh) + 2 * utilh @ Ru @ Kh
			c = (v - B @ utilh) @ P @ (v - B @ utilh) + L @ (v - B @ utilh) + utilh @ Ru @ utilh
		return AffineFeedbackController(-Kopt,-utilopt)

	def randomized(self):
		return False

	def randomize_controller_params(self):
		return





class PolicySearch:
	'''
	Random policy search.
	'''
	def __init__(self,N,N_hess,T,controller,temp,T_eval=None,debug_path=None):
		'''
		Inputs:
			N - number of controllers to search over
			N_hess - number of perturbations of base controller to use in hessian computation
			T - number of rollouts to collect to evaluate sampled controller at first round of filtering
			controller - initial controller to start search at
			temp - temperature of softmin
			T_eval - number of rollouts to collect to evaluate sampled controller at second round of filtering
			debug_path - path to directory to save logging information
		'''
		self.N = N
		self.N_hess = N_hess
		self.T = T
		if T_eval is None:
			self.T_eval = T
		else:
			self.T_eval = T_eval
		self.temp = temp
		self.controller = controller
		self.debug_path = debug_path

	def optimize(self,environment,compute_hessian=False,N=None,scaling=1):
		'''
		Compute optimal controller on environment.

		Inputs:
			environment - environment object to compute optimal controller for
			compute_hessian - if True, run differentiable variant of search procedure 
			N - number of controllers to search over
			scaling - weighting of softmin

		Outputs:
			controller - optimal controller found on environment in search procedure
		'''
		if compute_hessian:
			return self.optimize_hessian(environment,scaling=scaling)
		if N is None:
			N = self.N
		control_params = self.controller.get_controller_params()
		num_params = len(control_params)
		best_params = control_params
		best_val = environment.controller_cost(self.controller,T=self.T_eval)
		loss_vals = []
		for i in range(N):
			new_params_i = []
			for j in range(num_params):
				# sample controllers from noise distributions of different variances
				if np.mod(i,3) == 0 or best_params is None:
					v = 5*rand.rand() 
					Xj = best_params[j] + v*torch.randn(control_params[j].shape)
					new_params_i.append(Xj)
				elif np.mod(i,3) == 1:
					Xj = best_params[j] + 0.1*torch.randn(control_params[j].shape)
					new_params_i.append(Xj)
				elif np.mod(i,3) == 2:
					Xj = best_params[j] + 0.01*torch.randn(control_params[j].shape)
					new_params_i.append(Xj)
			controller_i = self.controller.generate_new(new_params_i)
			# evaluate cost of sampled controller
			cost_i = environment.controller_cost(controller_i,T=self.T)
			
			# check if cost is better than current best
			if cost_i < best_val:
				# if so, evaluate controller using more samples to ensure it is still best
				cost_i2 = environment.controller_cost(controller_i,T=10*self.T)
				if cost_i2 < best_val:
					# if so, evaluate controller using T_eval samples to ensure it is best
					cost_i_full = environment.controller_cost(controller_i,T=self.T_eval)
					if cost_i_full < best_val:
						# update best controller
						best_val = cost_i_full
						best_params = new_params_i
						loss_vals.append(cost_i_full.detach().numpy())
		if self.debug_path is not None:
			plt.plot(loss_vals)
			plt.savefig(self.debug_path + '/PolicySearch_loss_' + str(time.time()) + '.png')
			plt.close()

		self.controller.update_params(best_params)
		self.controller.detach()
		return self.controller.generate_duplicate()

	def optimize_hessian(self,environment,scaling=1):
		'''
		Differentiable search procedure, used to compute hessian. 

		Inputs:
			environment - environment object to compute optimal controller for
			scaling - weighting of softmin

		Outputs:
			weights - softmin weights of controllers
			controllers - list of controllers searched over
		'''
		control_params = self.controller.get_controller_params()
		num_params = len(control_params)
		new_params = []
		controller_costs = torch.zeros(self.N_hess+1)
		new_params.append(control_params)
		controller_costs[0] = environment.controller_cost(self.controller,T=2500)
		for i in range(self.N_hess):
			new_params_i = []
			for j in range(num_params):
				# perturb current controller parameters
				Xj = control_params[j] + (0.1/scaling)*torch.randn(control_params[j].shape) 
				new_params_i.append(Xj)
			controller_i = self.controller.generate_new(new_params_i)
			cost_i = environment.controller_cost(controller_i,T=2500)
			controller_costs[i+1] = self.temp/scaling * cost_i
			new_params.append(new_params_i)
		# compute softmin distribution over costs of perturbed controllers
		weights = torch.nn.functional.softmin(controller_costs,dim=0)
		controllers = []
		for i in range(len(weights)):
			controllers.append(self.controller.generate_new(new_params[i]))
		return weights, controllers

	def zero_controller_params(self):
		self.controller.zero_params()

	def randomize_controller_params(self):
		self.controller.randomize_params()

	def randomized(self):
		return True
		


class PolicySpecial:
	'''
	Policy optimization for motivating example. Computes controller that cancels out unwanted part of estimated dynamics.
	'''
	def __init__(self,controller,N_hess,temp,debug_path=None):
		'''
		Inputs:
			controller - initial controller
			N_hess - number of perturbations of base controller to use in hessian computation
			temp - temperature of softmin
			debug_path - path to directory to save logging information
		'''
		self.N_hess = N_hess
		self.temp = temp
		self.controller = controller
		self.debug_path = debug_path

	def optimize(self,environment,compute_hessian=False,scaling=1,linear=False):
		'''
		Computes controller on environment which cancels out estimated nonlinearities.

		Inputs:
			environment - environment object to compute optimal controller for
			compute_hessian - if True, run differentiable variant of search procedure 
			scaling - weighting of softmin
			linear - if True, dynamics are taken to be linear

		Outputs:
			controller - optimal controller found on environment in search procedure
		'''
		if compute_hessian:
			return self.optimize_hessian(environment,scaling=scaling)
		if linear:
			A,B = environment.get_dynamics()
			cost = environment.get_cost()
			center = cost.get_center()
			control_params = self.controller.get_controller_params()
			self.controller.update_params([-A,torch.zeros(control_params[1].shape),center*torch.ones(1)])
		else:
			A,B,C = environment.get_dynamics()
			cost = environment.get_cost()
			center = cost.get_center()
			self.controller.update_params([-A,-C,center*torch.ones(1)])
		self.controller.detach()
		return self.controller.generate_duplicate()

	def optimize_hessian(self,environment,scaling=1):
		'''
		Differentiable search procedure, used to compute hessian. 

		Inputs:
			environment - environment object to compute optimal controller for
			scaling - weighting of softmin

		Outputs:
			weights - softmin weights of controllers
			controllers - list of controllers searched over
		'''
		control_params = self.controller.get_controller_params()
		num_params = len(control_params)
		new_params = []
		controller_costs = torch.zeros(self.N_hess+1)
		new_params.append(control_params)
		controller_costs[0] = environment.controller_cost(self.controller,T=2500)
		for i in range(self.N_hess):
			new_params_i = []
			for j in range(num_params):
				Xj = control_params[j] + (0.1/scaling)*torch.randn(control_params[j].shape) 
				new_params_i.append(Xj)
			controller_i = self.controller.generate_new(new_params_i)
			cost_i = environment.controller_cost(controller_i,T=2500)
			controller_costs[i+1] = self.temp/scaling * cost_i 
			new_params.append(new_params_i)
		weights = torch.nn.functional.softmin(controller_costs,dim=0)
		controllers = []
		for i in range(len(weights)):
			controllers.append(self.controller.generate_new(new_params[i]))
		return weights, controllers

	def zero_controller_params(self):
		self.controller.zero_params()

	def randomize_controller_params(self):
		self.controller.randomize_params()

	def randomized(self):
		return True



def compute_hessian_jacobian(environment,policy_opt,num_grads=40):
	'''
	Computes Jacobian approximation to Hessian.

	Inputs:
		environment - environment to compute Hessian approximation for
		policy_opt - policy optimization routine
		num_grads - number of gradients to approximate Hessian with

	Output:
		grads - gradients computed
		hess - estimate of Hessian
	'''
	environment2 = copy.deepcopy(environment)
	scaling = 1
	def total_loss(theta):
		environment2.set_theta(theta)
		if policy_opt.randomized():
			weights, controllers = policy_opt.optimize(environment2,compute_hessian=True,scaling=scaling)
			loss = torch.tensor(0, dtype=torch.float32)
			for i in range(len(weights)):
				loss_i = environment.controller_cost(controllers[i],T=2500)
				loss = loss + weights[i] * loss_i.detach()
		else:
			controller = policy_opt.optimize(environment2,compute_hessian=True)
			loss = environment.controller_cost(controller,T=1,noiseless=True)
		return loss

	thetahat2 = torch.autograd.Variable(environment.get_theta(), requires_grad=True)
	N_grad = thetahat2.shape[0]
	grads = torch.zeros(num_grads,N_grad)
	hess = torch.zeros(N_grad,N_grad)
	for i in range(num_grads):
		for j in range(10):
			grad_i = torch.autograd.functional.jacobian(total_loss,thetahat2).detach()
			grad_i = grad_i[0]
			if torch.norm(grad_i) > torch.tensor(0.00000001):
				break
			else:
				scaling = 2*scaling 
		grad_nan = torch.isnan(grad_i)
		grad_inf = torch.isinf(grad_i)
		grad_i[grad_nan] = 0
		grad_i[grad_inf] = 0
		grads[i,:] = grad_i
		hess = hess + torch.outer(grad_i,grad_i) / num_grads

	hess_inf = torch.isinf(hess)
	hess_nan = torch.isnan(hess)
	hess[hess_inf] = 0
	hess[hess_nan] = 0
	hess = (hess + hess.T) / 2
	return grads, hess



def compute_hessian(environment,policy_opt,make_psd=True,num_hess=1):
	'''
	Computes full Hessian via the torch.autograd.functional.hessian() function.

	Inputs:
		environment - environment to compute Hessian approximation for
		policy_opt - policy optimization routine
		make_psd - if True, forces returned matrix to be PSD
		num_hess - number of Hessian computations to run (average Hessian is returned)

	Output:
		hess_all - estimate of Hessian
	'''
	environment2 = copy.deepcopy(environment)
	scaling = 1
	def total_loss(theta):
		environment2.set_theta(theta)
		if policy_opt.randomized():
			weights, controllers = policy_opt.optimize(environment2,compute_hessian=True,scaling=scaling)
			loss = torch.tensor(0, dtype=torch.float32)
			for i in range(len(weights)):
				loss_i = environment.controller_cost(controllers[i],T=2500)
				loss = loss + weights[i] * loss_i.detach()
		else:
			controller = policy_opt.optimize(environment2,compute_hessian=True)
			loss = environment.controller_cost(controller,T=1,noiseless=True)
		return loss
	thetahat2 = torch.autograd.Variable(environment.get_theta(), requires_grad=True)
	dim = thetahat2.shape[0]
	hess_all = torch.zeros(dim,dim)

	for i in range(num_hess):
		for j in range(10):
			hess = torch.autograd.functional.hessian(total_loss,thetahat2)
			if torch.norm(hess) > torch.tensor(0.00000001):
				break
			else:
				scaling = 2*scaling
		hess_inf = torch.isinf(hess)
		hess_nan = torch.isnan(hess)
		hess[hess_inf] = 0
		hess[hess_nan] = 0
		hess = (hess + hess.T) / 2
		if make_psd:
			hess = hess.detach()
			hess = hess.T @ hess
			U,S,V = torch.svd(hess)
			hess = U @ torch.diag(torch.sqrt(S)) @ U.T
			if torch.norm(hess) == 0:
				hess = torch.eye(hess.shape[0])
		hess_all = hess_all + hess.detach() / num_hess
	return hess_all





