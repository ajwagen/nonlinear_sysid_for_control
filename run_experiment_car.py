import numpy as np 
import matplotlib.pyplot as plt
import torch
import copy
import os
import pickle
import sys
import shutil
from controllers import CarController
from costs import QuadraticCost
from environments import LDSwNonlinearity, CarKernel
from exploration import RandomExploration, CEEDExploration, DynamicOED
from policy_optimizers import PolicySearch
from system_id import system_id


experiment_id = "car"
if not os.path.isdir("./results/" + experiment_id):
	os.makedirs("./results/" + experiment_id)


# Set hyperparameters
H = 50
dimx = 6
dimu = 2
dimphi = 6
input_power = 10
N_search_explore = 500
N_search = 10000
T_search = 100
N_hess = 20 
temp_search = 1
T_eval = 100000
sigma = 0.1
unif_exp_frac = 0.2


# Set system parameters
dt = torch.tensor(0.1)
Ast = torch.eye(dimx,dimx)
Ast[0,2] = dt
Ast[1,3] = dt
Ast[4,5] = dt

Bst = torch.zeros(dimx,dimu)
Bst[5,1] = dt

Cst = torch.zeros(dimx,dimphi)
Cst[2,2] = dt
Cst[3,3] = dt

R = torch.tensor([[ 0.5120477 , -0.12616527,  0.22211356,  0.10806924, -0.00257879, -0.21285707, -0.14114721, -0.2612673 ],
       	  [-0.12616527,  0.3320703 ,  0.07745402, -0.26458284,  0.00593216, -0.2640173 ,  0.11435509,  0.09762527],
		  [ 0.22211356,  0.07745401,  0.18799447, -0.06585785,  0.00138652, -0.2511247 , -0.02613468, -0.10369612],
		  [ 0.10806926, -0.26458284, -0.06585785,  0.24778903, -0.00506225,  0.22499618, -0.09767092, -0.083513  ],
		  [-0.00257879,  0.00593216,  0.00138652, -0.00506225,  0.02212525, -0.00493085,  0.00222346,  0.0019511 ],
		  [-0.21285708, -0.26401734, -0.2511247 ,  0.22499618, -0.00493085,  0.4806623 , -0.02829697,  0.07861674],
		  [-0.14114721,  0.11435509, -0.02613468, -0.09767091,  0.00222346, -0.02829697,  0.08459379,  0.08378664],
		  [-0.2612673 ,  0.09762527, -0.10369612, -0.083513  ,  0.0019511 ,  0.07861675,  0.08378664,  0.16462927]])	


# generate instance
cost = QuadraticCost(R)
nonlinearity = CarKernel()
true_instance = LDSwNonlinearity(Ast,Bst,Cst,H,sigma,cost,nonlinearity,exact_cost=False,car=True)
controller = CarController(K=0.1*torch.ones(4))
policy_opt = PolicySearch(N_search,N_hess,T_search,controller,temp_search,T_eval=10000)

opt_val = true_instance.compute_opt_val(policy_opt)
# _, hess = compute_hessian_jacobian(true_instance,policy_opt)
# hess = hess / torch.max(hess)
# hess = hess / torch.linalg.matrix_norm(hess,2)
# true_instance.set_hessian(hess)


# run experiment
epochs = [100,100,100,100,100,100]

print('Running Task-Driven Exploration')
policy_opt.randomize_controller_params()
explore_policy = DynamicOED(N_search_explore,input_power,unif_exp_frac=unif_exp_frac,hessian_type="jacobian",lammin=False,planning_horizon=10)
controller_task_driven, metrics_task_driven, loss_time_task_driven, in_pow_task_driven = system_id(true_instance,explore_policy,policy_opt,epochs,T_eval=T_eval)

print('Running Random Exploration')
policy_opt.randomize_controller_params()
explore_policy = RandomExploration(input_power)
controller_rand, metrics_rand, loss_time_rand, in_pow_rand = system_id(true_instance,explore_policy,policy_opt,epochs,T_eval=T_eval)

print('Running Uniform Exploration')
policy_opt.randomize_controller_params()
explore_policy = CEEDExploration(N_search_explore,input_power,unif_exp_frac=unif_exp_frac,planning_horizon=10,objective='lammin')
controller_uniform, metrics_uniform, loss_time_uniform, in_pow_uniform = system_id(true_instance,explore_policy,policy_opt,epochs,T_eval=T_eval)


# save data
params = {}
params['Ast'] = Ast.detach().numpy()
params['Bst'] = Bst.detach().numpy()
params['H'] = H
params['R'] = R.detach().numpy()
params['opt_val'] = opt_val
params['epochs'] = epochs
# params['hessian'] = hess.detach().numpy()
params['input_power'] = input_power
params['in_pow_rand'] = in_pow_rand.detach().numpy()
params['metrics_rand'] = metrics_rand
params['loss_time_rand'] = loss_time_rand
params['in_pow_task_driven'] = in_pow_task_driven.detach().numpy()
params['metrics_task_driven'] = metrics_task_driven
params['loss_time_task_driven'] = loss_time_task_driven
params['metrics_mpc_uniform'] = metrics_uniform
params['in_pow_mpc_uniform'] = in_pow_uniform.detach().numpy()
params['loss_time_mpc_uniform'] = loss_time_uniform

if experiment_id is not None:
	f = open("./results/" + experiment_id + "/data"  + "_" + sys.argv[1],"wb")
	pickle.dump(params,f)
	f.close()





