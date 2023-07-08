Repository for the paper:

> Wagenmaker, Andrew, Guanya Shi, and Kevin Jamieson. "Optimal Exploration for Model-Based RL in Nonlinear Systems." arXiv preprint arXiv:2306.09210 (2023).

To replicate experiments from paper, run the files `run_experiment_motivating.py`, `run_experiment_drone.py`, and `run_experiment_car.py`. Calling each file runs all algorithms for a single trial on the specified system. Each file must be run with a single argument, which denotes the experiment id. Results will then be saved to a directory `results/SYSTEM_NAME`. For example, running `python run_experiment_drone.py 1` will run a single trial of each algorithm on the drone system, and will save the results to `./results/drone/data_1`.

It is recommended that each file is run multiple times and the results averaged (in the paper, the plots correspond to 100 trials for the motivating example and car system, and 200 trials for the drone system). After running experiments, to plot the averaged results run the command `python make_plots.py SYSTEM_NAME`. Plots will be generated and saved in the folder `./results/SYSTEM_NAME`. 
