#!/bin/bash

#SBATCH --job-name=BipedalWalker
#SBATCH --account=nn9447k
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=2000
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --output=/dev/null

SEED=1234

module purge
module add /usit/abel/u1/filipste/IN5490/baselines.module

python ./run.py "$1" --config="$2" --env="BipedalWalker-v2" --alg="TRPO" --total_timesteps=5000000 --seed="$SEED"
