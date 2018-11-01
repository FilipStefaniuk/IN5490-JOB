import os
import json
import argparse

from baselines import logger
from baselines.run import get_env_type
from baselines.trpo_mpi import trpo_mpi
from baselines.ppo2 import ppo2
from baselines.common.cmd_util import make_vec_env


ENVS = [
    'CartPole-v0',
    'LunarLander-v2',
    'BipedalWalker-v2',
]

ALGS = {
    "TRPO": trpo_mpi,
    "PPO": ppo2
}


def get_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser("""
        Train trpo algorithm on different envs, with different sets of hyperparameters.

        This simple script allows to pass set of hyperparameters as a json configuration file,
        train the trpo algorithm and save the log from training, metrics from eatch iteration
        and trained model.
    """)

    parser.add_argument('outdir', help='output directory, where to save logs, data and trained model')
    parser.add_argument('--config', type=argparse.FileType(), help='configuration file with hyperparameters')
    parser.add_argument('--env', choices=ENVS, default='CartPole-v0', help="environment to use")
    parser.add_argument('--alg', choices=ALGS.keys(), default='TRPO', help="which algorithm to use")
    parser.add_argument('--network', choices=['mlp'], default='mlp', help="network policy")
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument('--total_timesteps', type=int, default=100000, help='total number of time steps')
    parser.add_argument('--tensorboard', action='store_true', help='wether to save tensorboard log')
    parser.add_argument('--save', action='store_true', help='wether to save the model after the training finishes')

    return parser.parse_args()


def get_default_hyp(alg_name):
    """Returns dictinary of default hyperparameters."""
    defaults = {
        'gamma': 0.99,
        'lam': 1.0,
        'ent_coef': 0.0,
    }

    if alg_name in ('TRPO'):
        defaults['max_kl'] = 0.001
        defaults['timesteps_per_batch'] = 4096
    elif alg_name in ('PPO'):
        defaults['cliprange'] = 0.2
        defaults['nsteps'] = 4096
        defaults['log_interval'] = 1
        defaults['value_network'] = 'copy'
    else:
        raise ValueError("Unknown algorithm")

    return defaults


def get_env(env_name, seed=123, num_envs=1):
    """builds environment"""
    env_type, env_id = get_env_type(env_name)
    env = make_vec_env(env_id, env_type, num_envs, seed)
    return env


def configure_logger(outdir, format_strs=['stdout', 'log', 'csv'], tensorboard=False):
    """logger.configure wrapper"""
    if tensorboard:
        format_strs.append('tensorboard')

    logger.configure(dir=outdir, format_strs=format_strs)


def main():

    args = get_args()
    config = json.load(args.config) if args.config else {}
    config = dict(get_default_hyp(args.alg), **config)

    if not os.path.exists(args.outdir):
        os.makedirs(args.outdir)

    configure_logger(args.outdir, tensorboard=args.tensorboard)
    env = get_env(args.env, seed=args.seed)

    logger.log("=" * 120)
    logger.log("=" * 120)
    logger.log("""
    Training {} algorithm
        environment: {}
        policy network: {}
        total timesteps: {}
        random seed: {}

        hyperparameters used: {}
    """.format(args.alg, args.env, args.network,
               args.total_timesteps, args.seed, config))
    logger.log("=" * 120)
    logger.log("=" * 120)

    model = ALGS[args.alg].learn(
        env=env,
        network=args.network,
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        **config
    )

    env.close()

    if args.save:
        model_path = os.path.join(args.outdir, 'model')
        logger.info("saving model to {}".format(model_path))
        model.save(model_path)


if __name__ == '__main__':
    main()
