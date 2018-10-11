# Testing Hyperparameters
## Hyperparameters
If not specified in configuration file, [default value] is used.
* **gamma** [0.99] - discount factor, quantifies how much importance we give to the future rewards.
* **lam** [1.0] - advantage estimation
* **max_kl** [0.001] - max KL divergence between old and new policy
* **ent_coef** [0.0] - coefficient of policy entropy term in the optimization objective
* **cg_iters** [10] - number of iterations of conjugate gradient algorithm
* **cg_damping** [1e-2] - conjugate gradient damping
* **vf_stepsize** [3e-4] - learning rate for adam optimizer
* **vf_iters** [3] - number of iterations of value function optimization iterations per each policy optimization step

## Usage
1. Write TOML configuration file with lists of hyperparameter values you want to test.
2. Generate json files with all the possible hyperparameter combinations with 
    ```
    ./params_generator.py OUTDIR CONFIGURATION_FILE
    ```
3. Use bash script to train model with every configuration
    ```
    ./run_experiments.sh OUTDIR PARAM_FILES...
    ```
    or run the training script with one selected configuration
    ```
    ./train_cartpole.py OUTDIR PARAM_FILE
    ```