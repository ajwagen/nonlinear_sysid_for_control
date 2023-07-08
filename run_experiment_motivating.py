import numpy as np 
import torch
import copy
import os
import pickle
import sys
import shutil
from controllers import KernelLinearController
from costs import SpecialCost
from environments import LDSwNonlinearity, SpecialKernel
from exploration import RandomExploration, CEEDExploration, DynamicOED
from policy_optimizers import PolicySpecial
from system_id import system_id


experiment_id = "motivating_example"

if not os.path.isdir("./results/" + experiment_id):
	os.makedirs("./results/" + experiment_id)

# Set hyperparameters
H = 10
dimx = 1
dimu = 1
dimphi = 10
dimphi_u = 3
input_power = 5
N_search_explore = 500
N_hess = 20 
temp_search = 1
T_eval = 100000
sigma = 0.1
unif_exp_frac = 0.2

# Set system parameters
a = torch.tensor(0.8)
Ast = torch.zeros(dimx,dimx)
Ast[0,0] = a
Bst = torch.ones(dimx,dimu)
Cst = -3*torch.ones(dimx,dimphi)


# generate instance
cost = SpecialCost(center=10)
centers = []
for i in range(dimphi):
	centers.append(torch.tensor(3*i-14))
nonlinearity = SpecialKernel(centers=centers)
true_instance = LDSwNonlinearity(Ast,Bst,Cst,H,sigma,cost,nonlinearity,exact_cost=False,car=False)
nonlinearity_control = SpecialKernel(centers=centers)
controller = KernelLinearController(torch.zeros(dimx,dimu),torch.zeros(dimx,dimphi),nonlinearity_control,v=torch.zeros(dimu))
policy_opt = PolicySpecial(controller,N_hess,temp_search)
true_instance.set_init_state(torch.zeros(dimx))
opt_val = true_instance.compute_opt_val(policy_opt)
hess = torch.eye(dimx*(dimx+dimu+dimphi))
true_instance.set_hessian(hess)


# run trial
epochs = [30,20,20,20,20]

print('Running Task-Driven Exploration')
policy_opt.randomize_controller_params()
explore_policy = DynamicOED(N_search_explore,input_power,unif_exp_frac=unif_exp_frac,hessian_type="jacobian",lammin=True,planning_horizon=H)
controller_task_driven, metrics_task_driven, loss_time_task_driven, in_pow_task_driven = system_id(true_instance,explore_policy,policy_opt,epochs,T_eval=T_eval)

print('Running Random Exploration')
policy_opt.randomize_controller_params()
explore_policy = RandomExploration(input_power)
controller_rand, metrics_rand, loss_time_rand, in_pow_rand = system_id(true_instance,explore_policy,policy_opt,epochs,T_eval=T_eval)

print('Running Uniform Exploration')
policy_opt.randomize_controller_params()
explore_policy = CEEDExploration(N_search_explore,input_power,unif_exp_frac=unif_exp_frac,planning_horizon=H,objective='lammin')
controller_uniform, metrics_uniform, loss_time_uniform, in_pow_uniform = system_id(true_instance,explore_policy,policy_opt,epochs,T_eval=T_eval)

	
# save data
params = {}
params['Ast'] = Ast.detach().numpy()
params['Bst'] = Bst.detach().numpy()
params['H'] = H
params['opt_val'] = opt_val
params['epochs'] = epochs
params['hessian'] = hess.detach().numpy()
params['input_power'] = input_power
params['in_pow_rand'] = in_pow_rand.detach().numpy()
params['metrics_rand'] = metrics_rand
params['loss_time_rand'] = loss_time_rand
params['metrics_mpc_uniform'] = metrics_uniform
params['in_pow_mpc_uniform'] = in_pow_uniform.detach().numpy()
params['loss_time_mpc_uniform'] = loss_time_uniform
params['in_pow_task_driven'] = in_pow_task_driven.detach().numpy()
params['metrics_task_driven'] = metrics_task_driven
params['loss_time_task_driven'] = loss_time_task_driven

if experiment_id is not None:
	f = open("./results/" + experiment_id + "/data"  + "_" + sys.argv[1],"wb")
	pickle.dump(params,f)
	f.close()





