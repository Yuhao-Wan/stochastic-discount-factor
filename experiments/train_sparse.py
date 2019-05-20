import sys
import tensorflow as tf
import gym
import numpy as np
import gridworld
import matplotlib.pyplot as plt

from deepq import deepq
from baselines import logger
from baselines.common import models
from baselines.common import plot_util as pu



dirs = './logs/maze2/'

def main(seed, fraction, discount, path, gpu):
    with tf.device('/device:GPU:%s' % gpu):
      
        logger.configure(dir=dirs+'%s/%s/' % (path, seed), format_strs=['csv'])

        kwargs = dict(seed=seed,
            network=models.mlp(num_layers=2, num_hidden=128, activation=tf.nn.relu),
            lr=1e-4,
            total_timesteps=1500000,
            buffer_size=150000,
            #exploration_fraction=1.0, #random act
            #exploration_final_eps=1.0, #random act
            exploration_fraction=0.2,
            exploration_final_eps=0.02, 
            learning_starts=2000,
            target_network_update_freq=500,
            myopic_fraction=fraction,
            final_gamma=discount,
            gamma=discount,
            prioritized_replay=True,
            prioritized_replay_alpha=0.6,
            print_freq=5)


        f = open(dirs+'%s/%s/params.txt' % (path, seed), 'w')
        f.write(str(kwargs))
        f.close()

        env = gym.make("Sparse-v0")
        act = deepq.learn(
            env=env,
            **kwargs)
                      
        print("Saving model to maze.pkl")

        act.save(dirs+"%s/%s/maze.pkl" % (path, seed))
        save_plot(path, seed)


def save_plot(path, seed):
    results = pu.load_results(dirs+'%s/%s/' % (path, seed))
    r = results[0]
    plt.plot(r.progress.steps, r.progress["mean 100 episode reward"])
    plt.savefig(dirs+'%s/%s/plot.png' % (path, seed))


if __name__ == '__main__':
    seed = int(sys.argv[1])
    fraction = float(sys.argv[2])
    discount = float(sys.argv[3])
    path = sys.argv[4]
    gpu = sys.argv[5]
    main(seed, fraction, discount, path, gpu)
