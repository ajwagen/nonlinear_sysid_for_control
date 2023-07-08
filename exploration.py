import numpy as np 
import torch
from policy_optimizers import compute_hessian, compute_hessian_jacobian
import copy


class DynamicOED:
	'''
	MPC-based implementation of DynamicOED.
	'''
	def __init__(self,N,max_power,stab_controller=None,hessian_type=None,unif_exp_frac=0.1,lammin=False,planning_horizon=None,update_estimates=True):
		'''
		Inputs:
			N - number of input trajectories to sample at each step
			max_power - maximum average power of inputs
			stab_controller - controller object that will be played during exploration (for example, to stabilize system while exploring)
			hessian_type - method to compute model-task hessian. If "jacobian" uses jacobian approximation, if "full" or None computes full hessian
			unif_exp_frac - fraction of episodes to explore uniformly
			lammin - if True, in uniform exploration phase explores to maximize the minimum eigenvalue of covariates (otherwise explores randomly)
			planning_horizon - length of MPC rollouts
			update_estimates - if True, updates estimate of system dynamics after each episode
		'''
		self.N = N
		self.fraction_noise = unif_exp_frac
		self.power = torch.tensor(max_power)
		self.stab_controller = stab_controller
		self.hessian_type = hessian_type
		self.lammin = lammin
		self.planning_horizon = planning_horizon
		self.update_estimates = update_estimates

	def explore(self,true_instance,est_instance,policy_opt,epoch_len,epoch_idx,past_cov=None):
		'''
		Main exploration routine. Interacts with true_instance to collect data.

		Inputs:
			true_instance - environment object of true dynamics
			est_instance - environment object of estimated dynamics (for planning)
			policy_opt - policy optimization object
			epoch_len - number of episodes to explore for
			epoch_idx - if epoch_idx == 0, explores randomly, otherwise runs task-driven exploration
			past_cov - previously collected covariates

		Outputs:
			states - list of length epoch_len, each element is a list of length H+1 containing states encountered for each episode
			inputs - list of length epoch_len, each element is a list of length H containt actions played in each episode
			input_power - total power of inputs played
			hessian2 - hessian computed from est_instance and used to direct exploration
			cov - collected covariates (if past_cov is not None, cov = past_cov + new covariates collected)
		'''
		dimx,dimu,dimz,H = true_instance.get_dim()
		est_instance2 = copy.deepcopy(est_instance)

		if epoch_idx == 0:
			uniform = RandomExploration(self.power.detach().numpy(),stab_controller=self.stab_controller)
			return uniform.explore(true_instance,est_instance,policy_opt,epoch_len,epoch_idx)
		else:
			# compute hessian on estimated system to direct exploration
			hess_instance = copy.deepcopy(est_instance)
			if self.hessian_type is None:
				hessian = compute_hessian(hess_instance,policy_opt)
			elif self.hessian_type == "jacobian":
				_, hessian = compute_hessian_jacobian(hess_instance,policy_opt)
			elif self.hessian_type == "full":
				hessian = compute_hessian(hess_instance,policy_opt)
			else:
				_, hessian = compute_hessian_jacobian(hess_instance,policy_opt)
			hessian2 = hessian.clone().detach()
			hessian = hessian / torch.max(hessian)
			hessian = hessian / torch.linalg.matrix_norm(hessian)
	
			epoch_len_noise = int(epoch_len * self.fraction_noise)
			epoch_len_task = epoch_len - epoch_len_noise
			# run epoch_len * self.fraction_noise episodes of uniform (task-agnostic) exploration
			if self.lammin:
				lammin = CEEDExploration(self.N,self.power.detach().numpy(),unif_exp_frac=self.fraction_noise,stab_controller=self.stab_controller,planning_horizon=self.planning_horizon,objective='lammin')
				states, inputs, input_power, _, cov = lammin.explore(true_instance,est_instance,policy_opt,epoch_len_noise,epoch_idx,past_cov=past_cov)
				if self.update_estimates:
					est_instance2.update_parameter_estimates(states,inputs)
			else:
				uniform = RandomExploration(self.power.detach().numpy(),stab_controller=self.stab_controller)
				states, inputs, input_power, _, cov = uniform.explore(true_instance,est_instance,policy_opt,epoch_len_noise,epoch_idx,past_cov=past_cov)
				if self.update_estimates:
					est_instance2.update_parameter_estimates(states,inputs)

			N = np.max([int(epoch_len_task**(1/3)),dimz])
			Tvals = np.zeros(N).astype(int)
			idx = 0
			while np.sum(Tvals) < epoch_len_task:
				Tvals[idx] += 1
				idx += 1
				if idx > N - 1:
					idx = 0

			# run task-motivated exploration
			for n in range(N):
				# compute quadratic cost to optimize at round n of online FW procedure
				Rn = self.hess_to_cost(hessian,cov,dimx)
				for t in range(Tvals[n]):
					x = true_instance.get_init_state()
					epoch_power = 0
					U_init = torch.zeros(dimu,H)
					epoch_states = [x]
					epoch_inputs = []
					est_instance_h = copy.deepcopy(est_instance2)
					theta_h = est_instance_h.get_theta()
					U,S,V = torch.linalg.svd(torch.linalg.inv(cov + torch.eye(dimz)))
					cov_inv_half = U @ torch.diag(torch.sqrt(S)) @ U.T
					theta_h = theta_h + torch.kron(torch.eye(dimx),cov_inv_half) @ torch.randn(theta_h.shape)
					est_instance_h.set_theta(theta_h)
					for h in range(H):
						# run MPC procedure to compute best input to play
						next_input, U_init = self.compute_next_input(est_instance_h,Rn,x,h,U_init,epoch_power)
						u = next_input
						input_power = input_power + u @ u
						epoch_power = epoch_power + u @ u
						if self.stab_controller is not None:
							u = u + self.stab_controller.get_input(x,h)
						u = u.clone().detach()
						cov = cov + true_instance.get_cov(x,u)
						x = true_instance.dynamics(x,u)
						epoch_states.append(x)
						epoch_inputs.append(u)
					if self.update_estimates:
						est_instance2.update_parameter_estimates([epoch_states],[epoch_inputs])
					states.append(epoch_states)
					inputs.append(epoch_inputs)
		return states, inputs, input_power, hessian2, cov

	def compute_next_input(self,est_instance,R,x,h0,U_init,epoch_power):
		dimx,dimu,dimz,H = est_instance.get_dim()
		power_to_go = H*self.power - epoch_power
		if self.planning_horizon is None:
			unroll_steps = H - h0
		else:
			unroll_steps = self.planning_horizon
			if unroll_steps > H - h0:
				unroll_steps = H - h0
			else:
				power_to_go = power_to_go*unroll_steps / (H - h0)
		if power_to_go < 0:
			return torch.zeros(dimu), torch.zeros(dimu,unroll_steps)

		# generate a normalize random inputs
		U_init2 = torch.zeros(dimu,unroll_steps)
		U_init2[:,0:unroll_steps-1] = U_init[:,0:unroll_steps-1]
		U_init = U_init2[:,:,None]
		U = U_init.repeat(1,1,self.N) 
		U = U + torch.randn(dimu,unroll_steps,self.N)
		for i in range(self.N):
			U[:,:,i] = torch.sqrt(power_to_go) * U[:,:,i] / torch.norm(U[:,:,i])
		x = x[:,None]
		x = x.repeat(1,self.N)
		
		# roll out trajectories from random inputs on est_instance
		R2 = R.clone().detach()
		R2 = R2[None,:,:]
		costs = torch.zeros(self.N)
		for h in range(unroll_steps):
			if self.stab_controller is None:
				u = U[:,h,:]
			else:
				u = U[:,h,:] + self.stab_controller.get_input(x,h+h0)
			z = est_instance.get_cov(x,u,parallel=True)
			costs += torch.einsum('iik->k',R2.mT @ z)
			x = est_instance.dynamics(x,u,noiseless=True)

		# choose trajectory with minimal cost
		min_idx = torch.argmax(costs)
		return U[:,0,min_idx], U[:,1:,min_idx]

	def hess_to_cost(self,Hess,cov,dimx):
		# compute quadratic cost to use at step n of online FW procedure
		dimz = cov.shape[0]
		R = torch.zeros(dimz,dimz)
		cov_inv = torch.linalg.inv(cov + 0.0001*torch.eye(cov.shape[0]))
		for i in range(dimz):
			for j in range(dimz):
				eij = torch.zeros(dimz,dimz)
				eij[i,j] = 1
				temp = Hess @ torch.kron(torch.eye(dimx), cov_inv @ eij @ cov_inv)
				R[i,j] = torch.trace(temp)
		e,_ = torch.linalg.eig(R)

		if torch.min(torch.real(e)) < 0:
			R = R.T @ R
			U,S,V = torch.svd(R)
			R = U @ torch.diag(torch.sqrt(S)) @ U.T

		R = R / torch.max(R)
		R = R / torch.linalg.matrix_norm(R,2)
		return R.detach()




class CEEDExploration:
	'''
	Certainty-Equivalent Experiment Design Exploration
	This exploration strategy differs from DynamicOED in that, instead of running the online FW procedure, it attempts to directly minimize
	the exploration objective (for example, if the objective is tr(H * cov_t^{-1}), it chooses the inputs that will approximately minimize 
	tr(H * cov_{t+1}^{-1})). Relies on an MPC-based implementation to optimize next input. With objective='lammin', corresponds to the
	UniformExploration strategy inspired by (Mania et al., 2022).
	'''
	def __init__(self,N,max_power,stab_controller=None,hessian_type=None,unif_exp_frac=0.1,lammin=False,planning_horizon=None,objective='hessian',update_estimates=True):
		'''
		Inputs:
			N - number of input trajectories to sample at each step
			max_power - maximum average power of inputs
			stab_controller - controller object that will be played during exploration (for example, to stabilize system while exploring)
			hessian_type - method to compute model-task hessian. If "jacobian" uses jacobian approximation, if "full" or None computes full hessian
			unif_exp_frac - fraction of episodes to explore uniformly
			lammin - if True, in uniform exploration phase explores to maximize the minimum eigenvalue of covariates (otherwise explores randomly)
			planning_horizon - length of MPC rollouts
			objective - if objective == 'lammin' explores so as to maximize minimum eigenvalue of covariates, similar to approach of (Mania et al., 2022),
							otherwise explores to minimize tr(H * cov^{-1})
			update_estimates - if True, updates estimate of system dynamics after each episode
		'''
		self.N = N
		self.fraction_noise = unif_exp_frac
		self.power = torch.tensor(max_power)
		self.stab_controller = stab_controller
		self.hessian_type = hessian_type
		self.lammin = lammin
		self.planning_horizon = planning_horizon
		self.objective = objective
		self.update_estimates = update_estimates

	def explore(self,true_instance,est_instance,policy_opt,epoch_len,epoch_idx,past_cov=None):
		'''
		Main exploration routine. Interacts with true_instance to collect data.

		Inputs:
			true_instance - environment object of true dynamics
			est_instance - environment object of estimated dynamics (for planning)
			policy_opt - policy optimization object
			epoch_len - number of episodes to explore for
			epoch_idx - if epoch_idx == 0, explores randomly, otherwise runs task-driven exploration
			past_cov - previously collected covariates

		Outputs:
			states - list of length epoch_len, each element is a list of length H+1 containing states encountered for each episode
			inputs - list of length epoch_len, each element is a list of length H containt actions played in each episode
			input_power - total power of inputs played
			hessian2 - hessian computed from est_instance and used to direct exploration
			cov - collected covariates (if past_cov is not None, cov = past_cov + new covariates collected)
		'''
		dimx,dimu,dimz,H = true_instance.get_dim()
		est_instance2 = copy.deepcopy(est_instance)
		if epoch_idx == 0:
			uniform = RandomExploration(self.power.detach().numpy(),stab_controller=self.stab_controller)
			return uniform.explore(true_instance,est_instance,policy_opt,epoch_len,epoch_idx)
		else:
			hess_instance = copy.deepcopy(est_instance)
			if self.objective == "lammin":
				hessian = torch.eye(dimx*dimz)
				hessian2 = torch.eye(dimx*dimz)
			else:
				if self.hessian_type is None:
					hessian = compute_hessian(hess_instance,policy_opt)
				elif self.hessian_type == "jacobian":
					_, hessian = compute_hessian_jacobian(hess_instance,policy_opt)
				elif self.hessian_type == "full":
					hessian = compute_hessian(hess_instance,policy_opt)
				else:
					_, hessian = compute_hessian_jacobian(hess_instance,policy_opt)
				hessian2 = hessian.clone().detach()
				hessian = hessian / torch.max(hessian)
				hessian = hessian / torch.linalg.matrix_norm(hessian)

			epoch_len_noise = int(epoch_len * self.fraction_noise)
			epoch_len_task = epoch_len - epoch_len_noise
			if self.lammin:
				lammin = CEEDExploration(self.N,self.power.detach().numpy(),unif_exp_frac=self.fraction_noise,stab_controller=self.stab_controller,planning_horizon=self.planning_horizon,objective='lammin')
				states, inputs, input_power, _, cov = lammin.explore(true_instance,est_instance,policy_opt,epoch_len_noise,epoch_idx,past_cov=past_cov)
				if self.update_estimates:
					est_instance2.update_parameter_estimates(states,inputs)
			else:
				uniform = RandomExploration(self.power.detach().numpy(),stab_controller=self.stab_controller)
				states, inputs, input_power, _, cov = uniform.explore(true_instance,est_instance,policy_opt,epoch_len_noise,epoch_idx,past_cov=past_cov)
				if self.update_estimates:
					est_instance2.update_parameter_estimates(states,inputs)

			for t in range(epoch_len_task):
				x = true_instance.get_init_state()
				epoch_power = 0
				U_init = torch.zeros(dimu,H)
				epoch_states = [x]
				epoch_inputs = []
				for h in range(H):
					next_input, U_init = self.compute_next_input(est_instance2,hessian,cov,x,h,U_init,epoch_power)
					u = next_input
					input_power = input_power + u @ u
					epoch_power = epoch_power + u @ u
					if self.stab_controller is not None:
						u = u + self.stab_controller.get_input(x,h)
					u = u.clone().detach()
					cov = cov + true_instance.get_cov(x,u)
					x = true_instance.dynamics(x,u)
					epoch_states.append(x)
					epoch_inputs.append(u)
				if self.update_estimates:
					est_instance2.update_parameter_estimates([epoch_states],[epoch_inputs])
				states.append(epoch_states)
				inputs.append(epoch_inputs)
		return states, inputs, input_power, hessian2, cov

	def compute_next_input(self,est_instance,hessian,cov,x,h0,U_init,epoch_power):
		dimx,dimu,dimz,H = est_instance.get_dim()
		power_to_go = H*self.power - epoch_power
		if self.planning_horizon is None:
			unroll_steps = H - h0
		else:
			unroll_steps = self.planning_horizon
			if unroll_steps > H - h0:
				unroll_steps = H - h0
			else:
				power_to_go = power_to_go*unroll_steps / (H - h0)
		if power_to_go < 0:
			return torch.zeros(dimu), torch.zeros(dimu,unroll_steps)

		U_init2 = torch.zeros(dimu,unroll_steps)
		U_init2[:,0:unroll_steps-1] = U_init[:,0:unroll_steps-1]
		U_init = U_init2[:,:,None]
		U = U_init.repeat(1,1,self.N) 
		U = U + torch.randn(dimu,unroll_steps,self.N)
		for i in range(self.N):
			U[:,:,i] = torch.sqrt(power_to_go) * U[:,:,i] / torch.norm(U[:,:,i])
		x = x[:,None]
		x = x.repeat(1,self.N)
		cov = cov[:,:,None]
		cov = cov.repeat(1,1,self.N)
		
		for h in range(unroll_steps):
			if self.stab_controller is None:
				u = U[:,h,:]
			else:
				u = U[:,h,:] + self.stab_controller.get_input(x,h+h0)
			cov = cov + est_instance.get_cov(x,u,parallel=True)
			x = est_instance.dynamics(x,u,noiseless=True)

		if self.objective == 'lammin':
			e,_ = torch.linalg.eig(cov[:,:,0])
			min_loss = torch.min(torch.real(e))
		else:
			cov_inv = torch.linalg.inv(cov[:,:,0] + 0.001 * torch.eye(dimz)).contiguous()
			min_loss = torch.trace(hessian @ torch.kron(torch.eye(dimx),cov_inv))
		min_idx = 0
		for i in range(self.N):
			if self.objective == 'lammin':
				e,_ = torch.linalg.eig(cov[:,:,i])
				loss = torch.min(torch.real(e))
				if loss.detach().numpy() > min_loss.detach().numpy():
					min_loss = loss
					min_idx = i
			else:
				cov_inv = torch.linalg.inv(cov[:,:,i] + 0.001 * torch.eye(dimz)).contiguous()
				loss = torch.trace(hessian @ torch.kron(torch.eye(dimx),cov_inv))
				if loss.detach().numpy() < min_loss.detach().numpy():
					min_loss = loss
					min_idx = i
		return U[:,0,min_idx], U[:,1:,min_idx]






class RandomExploration:
	'''
	Random exploration---chooses input to be isotropic Gaussian.
	'''
	def __init__(self,max_power,stab_controller=None):
		'''
		Inputs:
			max_power - maximum average power of inputs
			stab_controller - controller object that will be played during exploration (for example, to stabilize system while exploring)
		'''
		self.max_power = torch.tensor(max_power, dtype=torch.float32)
		self.stab_controller = stab_controller

	def explore(self,true_instance,est_instance,policy_opt,epoch_len,epoch_idx,past_cov=None):
		'''
		Main exploration routine. Interacts with true_instance to collect data.

		Inputs:
			true_instance - environment object of true dynamics
			est_instance - environment object of estimated dynamics (for planning)
			policy_opt - policy optimization object
			epoch_len - number of episodes to explore for
			epoch_idx - index of current epoch
			past_cov - previously collected covariates

		Outputs:
			states - list of length epoch_len, each element is a list of length H+1 containing states encountered for each episode
			inputs - list of length epoch_len, each element is a list of length H containt actions played in each episode
			input_power - total power of inputs played
			torch.tensor(0)
			cov - collected covariates (if past_cov is not None, cov = past_cov + new covariates collected)
		'''
		dimx,dimu,dimz,H = true_instance.get_dim()
		input_power = torch.tensor(0, dtype=torch.float32)
		states = []
		inputs = []
		if past_cov is None:
			cov = torch.zeros(dimz,dimz)
		else:
			cov = past_cov
		for t in range(epoch_len):
			x = true_instance.get_init_state()
			epoch_states = [x]
			epoch_inputs = []
			for h in range(H):
				u = torch.sqrt(self.max_power / dimu) * torch.randn(dimu)
				input_power = input_power + u @ u
				if self.stab_controller is not None:
					u = u + self.stab_controller.get_input(x,h)
				u = u.detach()
				cov = cov + true_instance.get_cov(x,u)
				x = true_instance.dynamics(x,u)
				epoch_states.append(x)
				epoch_inputs.append(u)
			states.append(epoch_states)
			inputs.append(epoch_inputs)
		return states, inputs, input_power, torch.tensor(0), cov


