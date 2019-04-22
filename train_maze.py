import gym
import gridworld
from baselines import deepq

def callback(lcl, _glb):
    # stop training if reward exceeds 1900
    is_solved = lcl['t'] > 100 and sum(lcl['episode_rewards'][-101:-1]) / 100 >= 1900
    return is_solved


def main():
    env = gym.make("maze-v0")
    act = deepq.learn(
        env,
        network='mlp',
        lr=1e-4,
        total_timesteps=100000,
        buffer_size=50000,
        exploration_fraction=0.9,
        exploration_final_eps=0.7,
        learning_starts=10000,
        target_network_update_freq=1000,
        gamma=0.99,
        prioritized_replay=True,
        prioritized_replay_alpha=0.6,
        print_freq=10000,
        dueling=True,
        callback=callback
    )
    print("Saving model to maze.pkl")
    act.save("maze.pkl")


if __name__ == '__main__':
    main()
