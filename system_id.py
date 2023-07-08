import numpy as np 
import torch
import copy


def system_id(true_instance,explore_policy,policy_opt,epochs,T_eval=None):
    '''
    Runs system identification procedure, computes optimal controller on estimated system.

    Inputs:
        true_instance - environment object of ground truth dynamics
        explore_policy - explore object providing exploration policy to run
        policy_opt - policy optimization object to compute controller
        epoch - list containing length of each epoch
        T_eval - when controller is evaluated by simply rolling out, T_eval denotes number of rollouts to average over

    Outputs:
        ce_opt_controller - controller returned by policy_opt on final estimated system
        metrics - dictionary containing recorded metrics
        loss_time - episode indices at which metrics were recorded
        in_pow - total power of inputs played
    '''
    metrics = {}
    metrics['controller_loss'] = []
    metrics['exploration_loss'] = []
    metrics['est_hessian'] = []
    total_elapsed = 0
    loss_time = []
    in_pow = torch.tensor(0, dtype=torch.float32)
    cov = None

    est_instance = copy.deepcopy(true_instance)
    est_instance.reset_parameters()

    for epoch_idx in range(len(epochs)):
        print('epoch: ' + str(epoch_idx+1) + "/" + str(len(epochs)))
        states, inputs, in_powi, est_hess, cov = explore_policy.explore(true_instance, est_instance, policy_opt, epochs[epoch_idx], epoch_idx, past_cov=cov)
        est_instance.update_parameter_estimates(states,inputs)
        in_pow = in_pow + in_powi
        total_elapsed += epochs[epoch_idx]
        loss_time.append(total_elapsed)
        metrics['est_hessian'].append(est_hess.detach().numpy())

        metrics = true_instance.compute_est_error(est_instance,metrics)
        ce_opt_controller = policy_opt.optimize(est_instance)
        if T_eval is None:
            controller_cost = true_instance.controller_cost(ce_opt_controller)
        else:
            controller_cost = true_instance.controller_cost(ce_opt_controller,T=T_eval)
        metrics['controller_loss'].append(controller_cost.detach().numpy())

        dimx,_,dimz,H = true_instance.get_dim()
        cov_inv = torch.linalg.inv(cov.clone().detach() + 0.00001*torch.eye(cov.shape[0])).contiguous()
        hessian = true_instance.get_hessian()
        if hessian is not None:
            hessian = hessian / torch.norm(hessian)
            exp_loss = torch.trace(hessian @ torch.kron(torch.eye(dimx),cov_inv))
            metrics['exploration_loss'].append(exp_loss.detach().numpy())

    return ce_opt_controller, metrics, loss_time, in_pow










