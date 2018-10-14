#!/usr/bin/env python3

import os
import json
import argparse

from baselines import logger
from baselines.a2c import a2c
from baselines.common.cmd_util import make_vec_env
from baselines.run import get_env_type

GAME_ENVIRONMENT = "CartPole-v0"
NETWORK_ARCHITECTURE = "mlp"


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("""
    Train the A2C algorithm with 'CartPole' environment.
    """)

    parser.add_argument('outdir', help='output directory')
    parser.add_argument('config', help='configuration file')
    parser.add_argument('--seed', type=int, default=123, help='random seed')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='total number of time steps')
    parser.add_argument('--tensorboard', action='store_true', help='whether to use tensorboard')
    parser.add_argument('--save', action='store_true', help='whether to save the model')

    return parser.parse_args()


def parse_config(path):
    """Parse json with hyperparameters."""
    with open(path, "r") as config_file:
        return json.load(config_file)


def main():

    args = parse_args()

    format_strs = ['log', 'csv', 'stdout']

    if args.tensorboard:
        format_strs.append('tensorboard')

    config = parse_config(args.config)

    outdir = os.path.join(args.outdir, os.path.splitext(os.path.basename(args.config))[0])
    logger.configure(dir=outdir, format_strs=format_strs)

    env_type, env_id = get_env_type(GAME_ENVIRONMENT)
    env = make_vec_env(env_id, env_type, 1, args.seed)

    model = a2c.learn(
        env=env,
        network=NETWORK_ARCHITECTURE,
        total_timesteps=args.total_timesteps,
        **config
    )

    env.close()

    if args.save:
        model.save(os.path.join(outdir, 'model'))


if __name__ == '__main__':
    main()
