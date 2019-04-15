import gridworld
import gym
from baselines import deepq

def main():
    env = gym.make("maze-v0")
    act = deepq.learn(env, network='mlp', total_timesteps=100, load_path="maze.pkl")

    while True:
        obs, done = env.reset(), False
        episode_rew = 0
        while not done:
            #env.render()
            obs, rew, done, _ = env.step(act(obs)[0])
            #nobs, rew, done, _ = env.step(env.action_space.sample())
            episode_rew += rew
        print("Episode reward", episode_rew)


if __name__ == '__main__':
    main()
