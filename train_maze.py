import sys
import tensorflow as tf
import gym
import numpy as np
import gridworld
import matplotlib.pyplot as plt

import deepq
from baselines import logger
from baselines.common import models
from baselines.common import plot_util as pu


def main(fraction, gamma, path):
    logger.configure(dir='./logs/%s/' % path, format_strs=['csv'])

    kwargs = dict(network=models.mlp(num_layers=2, num_hidden=128, activation=tf.nn.relu),
        lr=1e-4,
        total_timesteps=2000000,
        buffer_size=200000,
        exploration_fraction=0.5,
        exploration_final_eps=0.02, 
        learning_starts=2000,
        target_network_update_freq=500,
        myopic_fraction=fraction,
        final_gamma=gamma,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        print_freq=5)

    f = open('./logs/%s/params.txt' % path, 'w')
    f.write(str(kwargs))
    f.close()

    env = gym.make("maze-v0")
    act = deepq.learn(
        env=env,
        seed=123,
        **kwargs
    )
    print("Saving model to maze.pkl")
    act.save("./logs/%s/maze.pkl" % path)
    save_plot(path)

def save_plot(path):
    results = pu.load_results('./logs/%s' % path)
    r = results[0]
    plt.plot(r.progress.steps, r.progress["mean 100 episode reward"])
    plt.savefig('./logs/%s/plot.png' % path)

if __name__ == '__main__':
    fraction = float(sys.argv[1])
    gamma = float(sys.argv[2])
    main(fraction, gamma, sys.argv[3])

