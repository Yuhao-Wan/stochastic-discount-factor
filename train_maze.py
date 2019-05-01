import tensorflow as tf
import gym
import gridworld

from baselines import logger
from baselines import deepq
from baselines.common import models


# def callback(lcl, _glb):
#     # stop training if reward exceeds 1500
#     is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 1500
#     return is_solved


def main():
    logger.configure(dir='./logs', format_strs=['csv'])
    env = gym.make("maze-v0")
    act = deepq.learn(
        env,
        seed=123,
        network=models.mlp(num_layers=2, num_hidden=128, activation=tf.nn.relu),
        lr=1e-4,
        total_timesteps=20000,
        buffer_size=5000,
        exploration_fraction=0.9,
        exploration_final_eps=0.2,
        learning_starts=5000,
        target_network_update_freq=500,
        gamma=0.99,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        print_freq=5
        #callback=callback
    )
    print("Saving model to maze.pkl")
    act.save("maze.pkl")


if __name__ == '__main__':
    main()
