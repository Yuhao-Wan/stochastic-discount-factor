import sys
import tensorflow as tf
import gym
import numpy as np
import gridworld
import matplotlib.pyplot as plt

from baselines import logger
from baselines import deepq
from baselines.common import models
from baselines.common import plot_util as pu


def main(seed, discount, path):
    logger.configure(dir='./logs/%s/%s/' % (path, seed), format_strs=['csv'])

    kwargs = dict(seed=seed,
        network=models.mlp(num_layers=2, num_hidden=128, activation=tf.nn.relu),
        lr=1e-4,
        total_timesteps=1500000,
        buffer_size=150000,
        exploration_fraction=0.2,
        exploration_final_eps=0.02,
        learning_starts=2000,
        target_network_update_freq=500,
        gamma=discount,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        print_freq=5)

    f = open('./logs/%s/%s/params.txt' % (path, seed), 'w')
    f.write(str(kwargs))
    f.close()

    env = gym.make("maze-v0")
    act = deepq.learn(
        env=env,
        **kwargs
    )
    print("Saving model to maze.pkl")
    act.save("./logs/%s/%s/maze.pkl" % (path, seed))
    save_plot(path, seed)


def save_plot(path, seed):
    results = pu.load_results('./logs/%s/%s/' % (path, seed))
    r = results[0]
    plt.plot(r.progress.steps, r.progress["mean 100 episode reward"])
    plt.savefig('./logs/%s/%s/plot.png' % (path, seed))

if __name__ == '__main__':
    seed = int(sys.argv[1])
    discount = float(sys.argv[2])
    main(seed, discount, sys.argv[3])

