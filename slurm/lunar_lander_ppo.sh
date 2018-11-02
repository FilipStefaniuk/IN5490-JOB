#!/bin/bash

#SBATCH --job-name=LunarLander
#SBATCH --account=nn9447k
#SBATCH --time=01:0:00
#SBATCH --mem-per-cpu=1000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/dev/null

SEED=1337

module purge
module add /usit/abel/u1/filipste/IN5490/baselines.module

python ./run.py "$1" --config="$2" --env="LunarLander-v2" --alg="PPO" --total_timesteps=2000000 --seed="$SEED"
