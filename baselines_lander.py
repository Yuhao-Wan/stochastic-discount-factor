import sys
import gym
import tensorflow as tf
from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.ppo2 import ppo2
from baselines.common.vec_env import DummyVecEnv


def main(gpu, seed, lam, path):
    
    with tf.device('/device:GPU:%s' % gpu):

        logger.configure(dir='./logs/lander-baselines/%s/%s/' % (path, seed))
        
        env = gym.make('LunarLanderContinuous-v2')
        env = bench.Monitor(env, logger.get_dir())
        env = DummyVecEnv([lambda: env]) 

        model = ppo2.learn(network='mlp', env=env, total_timesteps=2000000, lam=lam, seed=seed)

        model.save('./logs/lander-baselines/%s/%s/model.pkl' % (path, seed))
        env.close()

if __name__ == '__main__':
    gpu = sys.argv[1]
    seed = int(sys.argv[2])
    lam = float(sys.argv[3])
    path = sys.argv[4]
    
    main(gpu, seed, lam, path)
