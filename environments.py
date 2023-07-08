import numpy as np
import torch
import copy


class LinearDynamicalSystem:
	'''
	Linear dynamical system. Implements dynamics of form x_{h+1} = A * x_h + B * u_h + w_u
	'''
	def __init__(self,A,B,H,noise_cov,cost,exact_cost=False):
		'''
		Inputs:
			A - dimx x dimx system matrix
			B - dimx x dimu control matrix
			H - horizon
			noise_cov - noise covariance. If scalar, noise covariance is sqrt{noise_cov} * I, otherwise
							noise covariance is noise_cov
			cost - cost object
			exact_cost - if True, cost is computed in closed-form when possible; otherwise cost is computed via sampling
		'''
		self.A = A
		self.B = B
		self.H = H
		self.exact_cost = exact_cost
		self.cost = cost
		dimx,dimu = B.shape
		self.dimx = dimx
		self.dimu = dimu
		self.init_state = torch.ones(self.dimx)
		self.hessian = None
		self.parallel_N = -1
		self.car = False

		self.data_cov = torch.zeros(self.dimx+self.dimu,self.dimx+self.dimu)
		self.data_process = torch.zeros(self.dimx+self.dimu,self.dimx)

		if torch.is_tensor(noise_cov) is False:
			noise_cov = torch.tensor(noise_cov, dtype=torch.float32)
		if len(noise_cov.shape) == 0:
			self.noise_std = torch.sqrt(noise_cov) * torch.eye(dimx)
		else:
			U,S,V = torch.svd(noise_cov)
			self.noise_std = U @ torch.diag(torch.sqrt(S)) @ V.T

	def dynamics(self,x,u,noiseless=False):
		'''
		Dynamics of system.

		Inputs: 
			x - matrix of size dimx x T denoting system state (if T > 1, this corresponds to
				running T trajectories in parallel)
			u - matrix of size dimu x T denoting system input

		Outputs:
			x_plus - matrix of size dimx x T denote next state
		'''
		if noiseless:
			return torch.matmul(self.A,x) + torch.matmul(self.B,u)
			
		if len(x.shape) > 1:
			noise = self.noise_std @ torch.randn(self.dimx,x.shape[1])
		else:
			noise = self.noise_std @ torch.randn(self.dimx)
		return torch.matmul(self.A,x) + torch.matmul(self.B,u) + noise

	def controller_cost(self,controller,T=50000,noiseless=False,exact_cost=False):
		'''
		Estimates cost of controller.

		Inputs: 
			controller - controller object
			T - number of trajectories to average costs across
			noiseless - if True, rollout trajectories on system with no noise
			exact_cost - if True, compute cost exactly (when this is possible)

		Outputs:
			estimated cost of controller
		'''
		if self.exact_cost or exact_cost:
			return self.controller_cost_exact_lqr(controller)
			
		if self.parallel_N > 0:
			loss = torch.zeros(self.parallel_N)
		else:
			loss = torch.zeros(1)
		if T > 1000:
			T0 = 0
			while T0 < T:
				x = self.get_init_state(N=1000)
				for h in range(self.H):
					u = controller.get_input(x,h)
					loss = loss + self.cost.get_cost(x,u,h,N=self.parallel_N)
					x = self.dynamics(x,u,noiseless=noiseless)
				T0 = T0 + 1000
		else:
			x = self.get_init_state(N=T)
			for h in range(self.H):
				u = controller.get_input(x,h)
				loss = loss + self.cost.get_cost(x,u,h,N=self.parallel_N)
				x = self.dynamics(x,u,noiseless=noiseless)
		return loss / T

	def controller_cost_exact_lqr(self,controller):
		# only works if the cost is quadratic
		R = self.cost.get_cost_params()
		K = controller.get_controller_params()[0]

		loss = 0
		Lam = torch.zeros(self.dimx+self.dimu,self.dimx+self.dimu)
		z = torch.zeros(self.dimx+self.dimu)
		z[0:self.dimx] = self.init_state
		z[self.dimx:] = K[:,0:self.dimx] @ self.init_state
		for h in range(self.H):
			Atil = torch.zeros(self.dimx+self.dimu,self.dimx+self.dimu)
			Atil[0:self.dimx,0:self.dimx] = self.A
			Atil[0:self.dimx,self.dimx:] = self.B 
			if h < self.H - 1:
				Atil[self.dimx:,0:self.dimx] = K[:,self.dimx*(h+1):self.dimx*(h+2)] @ self.A
				Atil[self.dimx:,self.dimx:] = K[:,self.dimx*(h+1):self.dimx*(h+2)] @ self.B

			Lam_half = torch.zeros(self.dimx+self.dimu,self.dimx)
			Lam_half[0:self.dimx,:] = torch.eye(self.dimx)
			if h < self.H - 1:
				Lam_half[self.dimx:,:] = K[:,self.dimx*(h+1):self.dimx*(h+2)]
			Lam_noise = Lam_half @ Lam_half.T

			loss = loss + z @ R @ z
			loss = loss + torch.trace(Lam @ R)
			z = Atil @ z
			Lam = Atil @ Lam @ Atil.T + Lam_noise
		if torch.isinf(loss):
			return torch.tensor(100000)
		return loss

	def get_cov(self,x,u,parallel=False):
		if parallel:
			z = torch.concat((x,u),dim=0)
			cov = torch.einsum('ik, jk -> ijk',z,z)
			return cov
		else:
			z = self.get_z(x,u)
			return torch.outer(z,z)

	def get_z(self,x,u):
		return torch.concat((x,u))

	def update_parameter_estimates(self,states,inputs):
		T = len(inputs)
		for t in range(T):
			for h in range(self.H):
				z = self.get_z(states[t][h],inputs[t][h])
				self.data_cov += torch.outer(z,z)
				self.data_process += torch.outer(z,states[t][h+1])

		thetahat = torch.linalg.inv(self.data_cov) @ self.data_process
		thetahat = thetahat.T
		self.A = thetahat[:,0:self.dimx]
		self.B = thetahat[:,self.dimx:]

	def compute_opt_val(self,policy_opt):
		ce_opt_controller = policy_opt.optimize(self)
		return self.controller_cost(ce_opt_controller).detach().numpy()

	def compute_est_error(self,est_instance,metrics):
		Ahat,Bhat = est_instance.get_dynamics()
		thetast = torch.flatten(torch.cat((self.A,self.B),dim=1))
		thetahat = torch.flatten(torch.cat((Ahat,Bhat),dim=1))

		if 'Ast_est_error' in metrics:
			metrics['Ast_est_error'].append((torch.linalg.norm(Ahat - self.A)).detach().numpy())
		else:
			metrics['Ast_est_error'] = [(torch.linalg.norm(Ahat - self.A)).detach().numpy()]
		if 'Bst_est_error' in metrics:
			metrics['Bst_est_error'].append((torch.linalg.norm(Bhat - self.B)).detach().numpy())
		else:
			metrics['Bst_est_error'] = [(torch.linalg.norm(Bhat - self.B)).detach().numpy()]
		if 'thetast_est_error' in metrics:
			metrics['thetast_est_error'].append(((thetast - thetahat) @ (thetast - thetahat)).detach().numpy())
		else:
			metrics['thetast_est_error'] = [((thetast - thetahat) @ (thetast - thetahat)).detach().numpy()]
		if 'hess_est_error' in metrics and self.hessian is not None:
			metrics['hess_est_error'].append(((thetast - thetahat) @ (self.hessian / torch.norm(self.hessian)) @ (thetast - thetahat)).detach().numpy())
		elif self.hessian is not None:
			metrics['hess_est_error'] = [((thetast - thetahat) @ (self.hessian / torch.norm(self.hessian)) @ (thetast - thetahat)).detach().numpy()]
		
		return metrics

	def reset_parameters(self):
		self.A = torch.zeros(self.dimx,self.dimx)
		self.B = torch.zeros(self.dimx,self.dimu)

	def get_dim(self):
		return self.dimx, self.dimu, self.dimx+self.dimu, self.H

	def set_dynamics(self,A,B):
		self.A = A
		self.B = B

	def get_dynamics(self):
		return self.A, self.B

	def get_init_state(self,N=1):
		if self.car:
			if N == 1:
				init_pos = torch.ones(2)
				init_pos = 100 * init_pos / torch.norm(init_pos)
				x_init = torch.zeros(6)
				x_init[0:2] = init_pos
				return x_init
			else:
				init_pos = torch.ones(2,N)
				init_pos = 100 * torch.divide(init_pos,torch.norm(init_pos,dim=0))
				x_init = torch.zeros(6,N)
				x_init[0:2,:] = init_pos
				return x_init
		else:
			if N == 1:
				return self.init_state.clone().detach()
			else: 
				return torch.outer(self.init_state.clone().detach(),torch.ones(N))

	def set_theta(self,theta):
		theta_rs = torch.reshape(theta,(self.dimx,self.dimx+self.dimu))
		self.A = theta_rs[:,:self.dimx]
		self.B = theta_rs[:,self.dimx:]

	def get_theta(self):
		return torch.flatten(torch.cat((self.A,self.B),dim=1))

	def get_cost_params(self):
		return self.cost.get_cost_params()

	def set_hessian(self,hess):
		self.hessian = hess

	def get_hessian(self):
		if self.hessian is None:
			return None
		else:
			return self.hessian.clone().detach()

	def set_init_state(self,new_init_state):
		self.init_state = new_init_state

	def get_data_cov(self):
		return self.data_cov

	def get_cost(self):
		return self.cost




class AffineDynamicalSystem(LinearDynamicalSystem):
	'''
	Affine dynamical system. Implements dynamics of form x_{h+1} = A * x_h + B * u_h + v + w_u
	'''
	def __init__(self,A,B,v,H,noise_cov,cost,exact_cost=False,unknown_v=False):
		'''
		Inputs:
			A - dimx x dimx system matrix
			B - dimx x dimu control matrix
			v - dimx x 1 affine component of dynamics
			H - horizon
			noise_cov - noise covariance. If scalar, noise covariance is sqrt{noise_cov} * I, otherwise
							noise covariance is noise_cov
			cost - cost object
			exact_cost - if True, cost is computed in closed-form when possible; otherwise cost is computed via sampling
			unknown_v - if True, v is estimated as well, otherwise it is assumed known
		'''
		self.A = A
		self.B = B
		self.v = v
		self.H = H
		self.exact_cost = exact_cost
		self.cost = cost
		dimx,dimu = B.shape
		self.dimx = dimx
		self.dimu = dimu
		self.init_state = torch.zeros(self.dimx)
		self.init_state[0] = 0
		self.init_state[2] = 0
		self.init_state[4] = 0
		self.hessian = None
		self.parallel_N = -1
		self.car = False
		self.unknown_v = unknown_v

		if unknown_v:
			self.data_cov = torch.zeros(self.dimx+self.dimu+1,self.dimx+self.dimu+1)
			self.data_process = torch.zeros(self.dimx+self.dimu+1,self.dimx)
		else:
			self.data_cov = torch.zeros(self.dimx+self.dimu,self.dimx+self.dimu)
			self.data_process = torch.zeros(self.dimx+self.dimu,self.dimx)

		if torch.is_tensor(noise_cov) is False:
			noise_cov = torch.tensor(noise_cov, dtype=torch.float32)
		if len(noise_cov.shape) == 0:
			self.noise_std = torch.sqrt(noise_cov) * torch.eye(dimx)
		else:
			U,S,V = torch.svd(noise_cov)
			self.noise_std = U @ torch.diag(torch.sqrt(S)) @ V.T

	def dynamics(self,x,u,noiseless=False):
		if noiseless:
			if len(x.shape) > 1:
				v = torch.outer(self.v, torch.ones(x.shape[1]))
			else:
				v = self.v
			return torch.matmul(self.A,x) + torch.matmul(self.B,u) + v
			
		if len(x.shape) > 1:
			noise = self.noise_std @ torch.randn(self.dimx,x.shape[1])
			v = torch.outer(self.v, torch.ones(x.shape[1]))
		else:
			noise = self.noise_std @ torch.randn(self.dimx)
			v = self.v
		return torch.matmul(self.A,x) + torch.matmul(self.B,u) + v + noise

	def controller_cost_exact_lqr(self,controller):
		R = self.cost.get_cost_params()
		K = controller.get_controller_params()[0]
		util = controller.get_controller_params()[1]

		loss = 0
		Lam = torch.zeros(self.dimx+self.dimu,self.dimx+self.dimu)
		z = torch.zeros(self.dimx+self.dimu)
		z[0:self.dimx] = self.init_state
		z[self.dimx:] = K[:,0:self.dimx] @ self.init_state + util[:,0]
		for h in range(self.H):
			Atil = torch.zeros(self.dimx+self.dimu,self.dimx+self.dimu)
			Atil[0:self.dimx,0:self.dimx] = self.A
			Atil[0:self.dimx,self.dimx:] = self.B 
			if h < self.H - 1:
				Atil[self.dimx:,0:self.dimx] = K[:,self.dimx*(h+1):self.dimx*(h+2)] @ self.A
				Atil[self.dimx:,self.dimx:] = K[:,self.dimx*(h+1):self.dimx*(h+2)] @ self.B

			Lam_half = torch.zeros(self.dimx+self.dimu,self.dimx)
			Lam_half[0:self.dimx,:] = torch.eye(self.dimx)
			if h < self.H - 1:
				Lam_half[self.dimx:,:] = K[:,self.dimx*(h+1):self.dimx*(h+2)]
			Lam_noise = Lam_half @ Lam_half.T

			loss = loss + z @ R @ z
			loss = loss + torch.trace(Lam @ R)
			v = torch.zeros(self.dimx+self.dimu)
			v[0:self.dimx] = self.v
			if h < self.H - 1:
				v[self.dimx:] = util[:,h+1]
			z = Atil @ z + v
			Lam = Atil @ Lam @ Atil.T + Lam_noise
		if torch.isinf(loss):
			return torch.tensor(100000)
		return loss

	def get_affine_dynamics(self):
		return self.A, self.B, self.v

	def get_z(self,x,u):
		if self.unknown_v:
			if len(x.shape) > 1:
				N = x.shape[1]
				return torch.concat((x,u,torch.ones(1,N)))
			else:
				return torch.concat((x,u,torch.ones(1)))
		else:
			return torch.concat((x,u))

	def update_parameter_estimates(self,states,inputs):
		T = len(inputs)
		for t in range(T):
			for h in range(self.H):
				z = self.get_z(states[t][h],inputs[t][h])
				self.data_cov += torch.outer(z,z)
				if self.unknown_v:
					self.data_process += torch.outer(z,states[t][h+1])
				else:
					self.data_process += torch.outer(z,states[t][h+1] - self.v)

		thetahat = torch.linalg.inv(self.data_cov) @ self.data_process
		thetahat = thetahat.T
		if self.unknown_v:
			self.A = thetahat[:,0:self.dimx]
			self.B = thetahat[:,self.dimx:self.dimx+self.dimu]
			self.v = thetahat[:,self.dimx+self.dimu:]
			self.v = self.v.flatten()
		else:
			self.A = thetahat[:,0:self.dimx]
			self.B = thetahat[:,self.dimx:]

	def reset_parameters(self):
		if self.unknown_v:
			self.v = torch.zeros(self.dimx)
		self.A = torch.zeros(self.dimx,self.dimx)
		self.B = torch.zeros(self.dimx,self.dimu)

	def set_dynamics(self,A,B):
		assert('trying to set A/B dynamics only')

	def set_theta(self,theta):
		if self.unknown_v:
			theta_rs = torch.reshape(theta,(self.dimx,self.dimx+self.dimu+1))
			self.A = theta_rs[:,:self.dimx]
			self.B = theta_rs[:,self.dimx:self.dimx+self.dimu]
			self.v = theta_rs[:,self.dimx+self.dimu:]
			self.v = self.v.flatten()
		else:
			theta_rs = torch.reshape(theta,(self.dimx,self.dimx+self.dimu))
			self.A = theta_rs[:,:self.dimx]
			self.B = theta_rs[:,self.dimx:]

	def get_theta(self):
		if self.unknown_v:
			if len(self.v.shape) > 1:
				v = self.v
			else:
				v = self.v[:,None]
			return torch.flatten(torch.cat((self.A,self.B,v),dim=1))
		else:
			return torch.flatten(torch.cat((self.A,self.B),dim=1))

	def compute_est_error(self,est_instance,metrics):
		Ahat,Bhat = est_instance.get_dynamics()
		if self.unknown_v:
			thetast = self.get_theta()
			thetahat = est_instance.get_theta()
		else:
			thetast = torch.flatten(torch.cat((self.A,self.B),dim=1))
			thetahat = torch.flatten(torch.cat((Ahat,Bhat),dim=1))

		if 'Ast_est_error' in metrics:
			metrics['Ast_est_error'].append((torch.linalg.norm(Ahat - self.A)).detach().numpy())
		else:
			metrics['Ast_est_error'] = [(torch.linalg.norm(Ahat - self.A)).detach().numpy()]
		if 'Bst_est_error' in metrics:
			metrics['Bst_est_error'].append((torch.linalg.norm(Bhat - self.B)).detach().numpy())
		else:
			metrics['Bst_est_error'] = [(torch.linalg.norm(Bhat - self.B)).detach().numpy()]
		if 'thetast_est_error' in metrics:
			metrics['thetast_est_error'].append(((thetast - thetahat) @ (thetast - thetahat)).detach().numpy())
		else:
			metrics['thetast_est_error'] = [((thetast - thetahat) @ (thetast - thetahat)).detach().numpy()]
		if 'hess_est_error' in metrics and self.hessian is not None:
			metrics['hess_est_error'].append(((thetast - thetahat) @ (self.hessian / torch.norm(self.hessian)) @ (thetast - thetahat)).detach().numpy())
		elif self.hessian is not None:
			metrics['hess_est_error'] = [((thetast - thetahat) @ (self.hessian / torch.norm(self.hessian)) @ (thetast - thetahat)).detach().numpy()]
		
		return metrics

	def get_dim(self):
		if self.unknown_v:
			return self.dimx, self.dimu, self.dimx+self.dimu+1, self.H
		else:
			return self.dimx, self.dimu, self.dimx+self.dimu, self.H

	def get_cov(self,x,u,parallel=False):
		if parallel:
			if self.unknown_v:
				if len(x.shape) > 1:
					N = x.shape[1]
					z = torch.concat((x,u,torch.ones(1,N)),dim=0)
				else:
					z = torch.concat((x,u,torch.ones(1)),dim=0)
			else:
				z = torch.concat((x,u),dim=0)
			cov = torch.einsum('ik, jk -> ijk',z,z)
			return cov
		else:
			z = self.get_z(x,u)
			return torch.outer(z,z)




class LDSwNonlinearity(LinearDynamicalSystem):
	'''
	Linear dynamical system with nonlinearity. Implements dynamics of form x_{h+1} = A * x_h + B * u_h + C * phi(x_h,u_h) + v + w_u
	'''
	def __init__(self,A,B,C,H,noise_cov,cost,phi,exact_cost=False,parallel_N=-1,car=False,v=None):
		'''
		Inputs:
			A - dimx x dimx system matrix
			B - dimx x dimu control matrix
			C - dimx x dimphi kernel matrix
			H - horizon
			noise_cov - noise covariance. If scalar, noise covariance is sqrt{noise_cov} * I, otherwise
							noise covariance is noise_cov
			cost - cost object
			phi - kernel nonlinearity
			exact_cost - if True, cost is computed in closed-form when possible; otherwise cost is computed via sampling
			unknown_v - if True, v is estimated as well, otherwise it is assumed known
			parallel_N - if > 0, parallels cost computation
			car - set to true for car experiments (adjusts initial condition)
			v - affine component
		'''
		self.A = A
		self.B = B
		self.C = C
		self.H = H
		self.exact_cost = False
		self.cost = cost
		self.phi = phi
		dimx,dimu = B.shape
		_,dimphi = C.shape
		self.dimx = dimx
		self.dimu = dimu
		self.dimphi = dimphi
		self.init_state = torch.ones(self.dimx)
		self.hessian = None
		self.parallel_N = parallel_N
		self.car = car
		self.v = v

		self.data_cov = torch.zeros(self.dimx+self.dimu+self.dimphi,self.dimx+self.dimu+self.dimphi)
		self.data_process = torch.zeros(self.dimx+self.dimu+self.dimphi,self.dimx)

		noise_cov = torch.tensor(noise_cov, dtype=torch.float32)
		if len(noise_cov.shape) == 0:
			self.noise_std = torch.sqrt(noise_cov) * torch.eye(dimx)
		else:
			U,S,V = torch.svd(noise_cov)
			self.noise_std = U @ torch.diag(torch.sqrt(S)) @ V.T    

	def dynamics(self,x,u,noiseless=False):
		if noiseless:
			if self.v is not None:
				if len(x.shape) > 1:
					v = torch.outer(self.v, torch.ones(x.shape[1]))
				else:
					v = self.v
			else:
				v = torch.zeros(x.shape)
			return torch.matmul(self.A,x) + torch.matmul(self.B,u) + torch.matmul(self.C,self.phi.phi(x,u)) + v
			
		if len(x.shape) > 1:
			noise = self.noise_std @ torch.randn(self.dimx,x.shape[1])
			if self.v is not None:
				v = torch.outer(self.v, torch.ones(x.shape[1]))
		else:
			noise = self.noise_std @ torch.randn(self.dimx)
			if self.v is not None:
				v = self.v
		if self.v is None:
			v = torch.zeros(x.shape)
		return torch.matmul(self.A,x) + torch.matmul(self.B,u) + torch.matmul(self.C,self.phi.phi(x,u)) + v + noise

	def controller_cost_exact_lqr(self,controller):
		return torch.tensor(-1)

	def get_cov(self,x,u,parallel=False):
		if parallel:
			phi_xu = self.phi.phi(x,u)
			z = torch.concat((x,u,phi_xu),dim=0)
			cov = torch.einsum('ik, jk -> ijk',z,z)
			return cov
		else:
			z = self.get_z(x,u)
			return torch.outer(z,z)

	def get_z(self,x,u):
		phi_xu = self.phi.phi(x,u)
		return torch.concat((x,u,phi_xu))

	def update_parameter_estimates(self,states,inputs):
		T = len(inputs)
		for t in range(T):
			for h in range(self.H):
				z = self.get_z(states[t][h],inputs[t][h])
				self.data_cov += torch.outer(z,z)
				if self.v is None:
					self.data_process += torch.outer(z,states[t][h+1])
				else:
					self.data_process += torch.outer(z,states[t][h+1] - self.v)

		thetahat = torch.linalg.inv(self.data_cov + 0.001*torch.eye(self.dimx+self.dimu+self.dimphi)) @ self.data_process
		thetahat = thetahat.T
		self.A = thetahat[:,0:self.dimx]
		self.B = thetahat[:,self.dimx:self.dimx+self.dimu]
		self.C = thetahat[:,self.dimx+self.dimu:]

	def compute_est_error(self,est_instance,metrics):
		Ahat,Bhat,Chat = est_instance.get_dynamics()
		thetast = torch.flatten(torch.cat((self.A,self.B,self.C),dim=1))
		thetahat = torch.flatten(torch.cat((Ahat,Bhat,Chat),dim=1))

		if 'Ast_est_error' in metrics:
			metrics['Ast_est_error'].append((torch.linalg.norm(Ahat - self.A)).detach().numpy())
		else:
			metrics['Ast_est_error'] = [(torch.linalg.norm(Ahat - self.A)).detach().numpy()]
		if 'Bst_est_error' in metrics:
			metrics['Bst_est_error'].append((torch.linalg.norm(Bhat - self.B)).detach().numpy())
		else:
			metrics['Bst_est_error'] = [(torch.linalg.norm(Bhat - self.B)).detach().numpy()]
		if 'Cst_est_error' in metrics:
			metrics['Cst_est_error'].append((torch.linalg.norm(Chat - self.C)).detach().numpy())
		else:
			metrics['Cst_est_error'] = [(torch.linalg.norm(Chat - self.C)).detach().numpy()]
		if 'thetast_est_error' in metrics:
			metrics['thetast_est_error'].append(((thetast - thetahat) @ (thetast - thetahat)).detach().numpy())
		else:
			metrics['thetast_est_error'] = [((thetast - thetahat) @ (thetast - thetahat)).detach().numpy()]
		if 'hess_est_error' in metrics and self.hessian is not None:
			metrics['hess_est_error'].append(((thetast - thetahat) @ (self.hessian / torch.norm(self.hessian)) @ (thetast - thetahat)).detach().numpy())
		elif self.hessian is not None:
			metrics['hess_est_error'] = [((thetast - thetahat) @ (self.hessian / torch.norm(self.hessian)) @ (thetast - thetahat)).detach().numpy()]
		
		return metrics

	def reset_parameters(self):
		self.A = torch.zeros(self.dimx,self.dimx)
		self.B = torch.zeros(self.dimx,self.dimu)
		self.C = torch.zeros(self.dimx,self.dimphi)

	def get_dim(self):
		return self.dimx, self.dimu, self.dimx+self.dimu+self.dimphi, self.H

	def set_dynamics(self,A,B,C):
		self.A = A
		self.B = B
		self.C = C

	def get_dynamics(self):
		return self.A, self.B, self.C

	def set_theta(self,theta):
		theta_rs = torch.reshape(theta,(self.dimx,self.dimx+self.dimu+self.dimphi))
		self.A = theta_rs[:,:self.dimx]
		self.B = theta_rs[:,self.dimx:self.dimx+self.dimu]
		self.C = theta_rs[:,self.dimx+self.dimu:]

	def get_theta(self):
		return torch.flatten(torch.cat((self.A,self.B,self.C),dim=1))

	def get_LDS(self):
		return LinearDynamicalSystem(self.A,self.B,self.H,self.noise_std @ self.noise_std.T,self.cost)

	def make_parallel(self,N):
		Ap = torch.kron(torch.eye(N),self.A.contiguous())
		Bp = torch.kron(torch.eye(N),self.B.contiguous())
		Cp = torch.kron(torch.eye(N),self.C.contiguous())
		noise_cov = torch.kron(torch.eye(N),self.noise_std @ self.noise_std.T).detach().numpy()
		return LDSwNonlinearity(Ap,Bp,Cp,self.H,noise_cov,self.cost,self.phi,exact_cost=self.exact_cost,parallel_N=N)





class CarKernel:
	def __init__(self):
		return

	def phi(self,x,u):
		if len(x.shape) > 1:
			N = x.shape[1]
			sin_x = torch.sin(x[4,:])
			cos_x = torch.cos(x[4,:])
			phi_out = torch.zeros(6,N)
			phi_out[0,:] = cos_x
			phi_out[1,:] = sin_x
			phi_out[2,:] = torch.multiply(cos_x,u[0,:])
			phi_out[3,:] = torch.multiply(sin_x,u[0,:])
			phi_out[4,:] = torch.multiply(cos_x,u[1,:])
			phi_out[5,:] = torch.multiply(sin_x,u[1,:])
		else:
			sin_x = torch.sin(x[4])
			cos_x = torch.cos(x[4])
			phi_out = torch.tensor([cos_x,sin_x,cos_x * u[0],sin_x * u[0],cos_x * u[1],sin_x * u[1]])
		return phi_out



class SpecialKernel:
	def __init__(self,centers=[torch.tensor(10)]):
		self.centers = centers
		self.dimphi = len(centers)
		return

	def phi(self,x,u):
		if len(x.shape) > 1:
			phi_out = torch.zeros(self.dimphi,x.shape[1])
			for i in range(self.dimphi):
				phi_out[i,:] = 1 - 100*(x[0,:]-self.centers[i])**2
			phi_out[phi_out < 0] = 0
			return phi_out
		else:
			phi_out = torch.zeros(self.dimphi)
			for i in range(self.dimphi):
				phi_out[i] = 1 - 100*(x[0]-self.centers[i])**2
			phi_out[phi_out < 0] = 0
			return phi_out





