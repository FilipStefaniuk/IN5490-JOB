import argparse
import itertools
import time

import numpy as np

from baselines.trpo_mpi import trpo_mpi
from baselines.ppo2 import ppo2
from baselines import logger

from run_trpo import ALGS, ENVS, get_env


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--load_path", help="file with saved weights")
    parser.add_argument("--network", choices=["mlp"], default="mlp")
    parser.add_argument("--alg", choices=ALGS.keys(), default="TRPO")
    parser.add_argument('--seed', type=int, default=1337, help='random seed')
    parser.add_argument("--env", choices=ENVS, default="CartPole-v0")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_episodes", type=int)

    return parser.parse_args()


def main():
    args = get_args()

    logger.configure()

    env = get_env(args.env, seed=args.seed)

    model = ALGS[args.alg].learn(
        env=env,
        network=args.network,
        total_timesteps=0,
        load_path=args.load_path,
    )

    rewards = []
    state, dones = np.zeros((1, 2*128)), np.zeros((1))

    try:
        for i in itertools.count(0):

            if args.max_episodes and i >= args.max_episodes:
                break

            done = False
            episode_rew = 0
            obs = env.reset()

            while not done:
                actions, _, state, _ = model.step(obs, S=state, M=dones)
                obs, rew, done, _ = env.step(actions)
                episode_rew += rew
                env.render()
                time.sleep(1/args.fps)

            logger.log("Episode reward: {}".format(*episode_rew))
            rewards.append(*episode_rew)

    except KeyboardInterrupt:
        pass

    logger.log("Mean reward in {} episodes: {}".format(i, np.mean(rewards)))

    env.close()


if __name__ == '__main__':
    main()
