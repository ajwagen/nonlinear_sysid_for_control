import numpy as np 
import matplotlib.pyplot as plt
import torch
import copy
import os
import pickle
import sys
import shutil
from costs import QuadraticCost
from environments import AffineDynamicalSystem
from exploration import RandomExploration, CEEDExploration, DynamicOED
from policy_optimizers import AffineOptimalCEController, compute_hessian
from system_id import system_id


experiment_id = "drone2"
if not os.path.isdir("./results/" + experiment_id):
	os.makedirs("./results/" + experiment_id)


# Set hyperparameters
H = 50
dimx = 6
dimu = 3
input_power = 10
N_search_explore = 500
T_eval = 100000
sigma = 0.1
unif_exp_frac = 0.2

# Set system parameters
dt = torch.tensor(0.1)
Ast = torch.eye(dimx,dimx)
Ast[0,3] = dt
Ast[1,4] = dt
Ast[2,5] = dt

vst = torch.zeros(dimx)
vst[dimx-1] = -dt*9.8

Bst = torch.zeros(dimx,dimu)
Bst[3,0] = dt
Bst[4,1] = dt
Bst[5,2] = dt

R = 0.1*torch.eye(dimx+dimu)
R[dimx,dimx] = 1
R[dimx+1,dimx+1] = 5
R[dimx+2,dimx+2] = 5
R = R / 5



# generate instance
cost = QuadraticCost(R)
true_instance = AffineDynamicalSystem(Ast,Bst,vst,H,sigma,cost,exact_cost=True,unknown_v=True)
policy_opt = AffineOptimalCEController()

opt_val = true_instance.compute_opt_val(policy_opt)
hess = compute_hessian(true_instance,policy_opt)
true_instance.set_hessian(hess)


# run trial
epochs = [10,5,5,5,5,5,5,5,5]

print('Running Task-Driven Exploration')
policy_opt.randomize_controller_params()
explore_policy = DynamicOED(N_search_explore,input_power,unif_exp_frac=unif_exp_frac,hessian_type="full",lammin=True,planning_horizon=20)
controller_task_driven, metrics_task_driven, loss_time_task_driven, in_pow_task_driven = system_id(true_instance,explore_policy,policy_opt,epochs,T_eval=T_eval)

print('Running Random Exploration')
policy_opt.randomize_controller_params()
explore_policy = RandomExploration(input_power)
controller_rand, metrics_rand, loss_time_rand, in_pow_rand = system_id(true_instance,explore_policy,policy_opt,epochs,T_eval=T_eval)

print('Running Uniform Exploration')
policy_opt.randomize_controller_params()
explore_policy = CEEDExploration(N_search_explore,input_power,unif_exp_frac=unif_exp_frac,planning_horizon=20,objective='lammin')
controller_uniform, metrics_uniform, loss_time_uniform, in_pow_uniform = system_id(true_instance,explore_policy,policy_opt,epochs,T_eval=T_eval)



# save data
params = {}
params['Ast'] = Ast.detach().numpy()
params['Bst'] = Bst.detach().numpy()
params['H'] = H
params['R'] = R.detach().numpy()
params['opt_val'] = opt_val
params['epochs'] = epochs
params['hessian'] = hess.detach().numpy()
params['input_power'] = input_power
params['in_pow_rand'] = in_pow_rand.detach().numpy()
params['metrics_rand'] = metrics_rand
params['loss_time_rand'] = loss_time_rand
params['in_pow_task_driven'] = in_pow_task_driven.detach().numpy()
params['metrics_task_driven'] = metrics_task_driven
params['loss_time_task_driven'] = loss_time_task_driven
params['metrics_uniform'] = metrics_uniform
params['in_pow_uniform'] = in_pow_uniform.detach().numpy()
params['loss_time_uniform'] = loss_time_uniform


if experiment_id is not None:
	f = open("./results/" + experiment_id + "/data"  + "_" + sys.argv[1],"wb")
	pickle.dump(params,f)
	f.close()





