import tensorflow as tf
import gym
import numpy as np
import gridworld
import time
import datetime
import matplotlib.pyplot as plt

from baselines import logger
from baselines import deepq
from baselines.common import models
from baselines.common import plot_util as pu


def main():
    with tf.Graph().as_default():
        vals = [0.99]

        for val in vals:
            ts = time.time()
            st = datetime.datetime.fromtimestamp(ts).strftime('%m-%d-%H-%M')
            logger.configure(dir='./logs/%s/' % st, format_strs=['csv'])

            kwargs = dict(network=models.mlp(num_layers=2, num_hidden=128, activation=tf.nn.relu),
                lr=1e-4,
                total_timesteps=2000000,
                buffer_size=200000,
                exploration_fraction=1.0,
                exploration_final_eps=1.0,
                learning_starts=2000,
                target_network_update_freq=500,
                gamma=val,
                prioritized_replay=True,
                prioritized_replay_alpha=0.6,
                print_freq=5)

            f = open('./logs/%s/params.txt' % st, 'w')
            f.write(str(kwargs))
            f.close()

            env = gym.make("maze-v0")
            act = deepq.learn(
                env=env,
                seed=123,
                **kwargs
            )
            print("Saving model to maze.pkl")
            act.save("./logs/%s/maze.pkl" % st)
            save_plot(st)

def save_plot(path):
    results = pu.load_results('./logs/%s' % path)
    r = results[0]
    plt.plot(r.progress.steps, r.progress["mean 100 episode reward"])
    plt.savefig('./logs/%s/plot.png' % path)

if __name__ == '__main__':
    main()

