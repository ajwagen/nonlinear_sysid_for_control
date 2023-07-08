import numpy as np 
import matplotlib.pyplot as plt
import pickle
import os
import sys


start_idx = 0
directory = './results/' + sys.argv[1]
data_dir = os.listdir(directory)

first_file = True
count = 0
opt_val = None
stats = {}
hess_est = False

for d in data_dir:
    if d[0:4] == 'data':
        with open(directory + '/' + d, 'rb') as fp:
            data = pickle.load(fp)
            count += 1
            if first_file:
                first_file = False
                opt_val = data['opt_val']
                for key in data:
                    if key[0:6] == 'in_pow':
                        name = key[7:]
                        stats[name] = {}
                        stats[name]['loss_vals'] = np.array(data['metrics_' + name]['controller_loss'])
                        stats[name]['thetast_est'] = np.array(data['metrics_' + name]['thetast_est_error'])
                        if 'hess_est_error' in data['metrics_' + name]:
                            hess_est = True
                            stats[name]['hess_est'] = np.array(data['metrics_' + name]['hess_est_error'])
                            stats[name]['exp_loss'] = np.array(data['metrics_' + name]['exploration_loss'])
                        stats[name]['loss_time'] = np.array(data['loss_time_' + name])
            else:
                if data['opt_val'] < opt_val:
                    opt_val = data['opt_val']
                for key in data:
                    if key[0:6] == 'in_pow':
                        name = key[7:]
                        stats[name]['loss_vals'] += np.array(data['metrics_' + name]['controller_loss'])
                        stats[name]['thetast_est'] += np.array(data['metrics_' + name]['thetast_est_error'])
                        if hess_est:
                            stats[name]['hess_est'] += np.array(data['metrics_' + name]['hess_est_error'])
                            stats[name]['exp_loss'] += np.array(data['metrics_' + name]['exploration_loss'])
                        
for key in stats:
    for key2 in stats[key]:
        if key2 != 'loss_time':
            stats[key][key2] /= count

for key in stats:
    plt.plot(stats[key]['loss_time'][start_idx:],stats[key]['loss_vals'][start_idx:].flatten()-opt_val*np.ones(len(stats[key]['loss_time'][start_idx:])),label=key,linewidth=4)
plt.xlabel('Time')
plt.ylabel('Mean Controller Loss')
plt.legend()
plt.savefig(directory + '/mean_controller_loss.png')
plt.close()

for key in stats:
    plt.plot(stats[key]['loss_time'][start_idx:],stats[key]['thetast_est'][start_idx:],label=key,linewidth=4)
plt.xlabel('Time')
plt.ylabel('Parameter Estimation Error (Frobenius Norm)')
plt.legend()
plt.savefig(directory + '/parameter_estimation_error_fro.png')
plt.close()

if hess_est:
    for key in stats:
        plt.plot(stats[key]['loss_time'][start_idx:],stats[key]['hess_est'][start_idx:],label=key,linewidth=4)
    plt.xlabel('Time')
    plt.ylabel('Parameter Estimation Error (Hessian Norm)')
    plt.legend()
    plt.savefig(directory + '/parameter_estimation_error_hess.png')
    plt.close()

    for key in stats:
        plt.plot(stats[key]['loss_time'][start_idx:],stats[key]['exp_loss'][start_idx:],label=key,linewidth=4)
    plt.xlabel('Time')
    plt.ylabel('Exploration Loss')
    plt.legend()
    plt.savefig(directory + '/exploration_loss.png')
    plt.close()




