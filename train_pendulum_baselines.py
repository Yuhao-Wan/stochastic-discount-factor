import sys
import gym
import tensorflow as tf
from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.ppo2 import ppo2
from baselines.common.vec_env import DummyVecEnv


dirs = './logs/ppo-pendulum/baselines/'

def main(gpu, seed, path):
    with tf.device('/device:GPU:%s' % gpu):

        logger.configure(dir=dirs+'%s/%s/' % (path, seed))

        env = gym.make("Pendulum-v0")
        env = bench.Monitor(env, logger.get_dir())
        env = DummyVecEnv([lambda: env]) 

        kwargs = dict(network='mlp', 
                  env=env, 
                  total_timesteps=2000000, 
                  seed=seed)

        f = open(dirs+'%s/%s/params.txt' % (path, seed), 'w')
        f.write(str(kwargs))
        f.close()

        model = ppo2.learn(**kwargs)
        model.save(dirs+'%s/%s/pendulum_model.pkl' % (path, seed))

        env.close()


if __name__ == '__main__':
    gpu = int(sys.argv[1])
    seed = int(sys.argv[2])
    path = sys.argv[3]

    main(gpu, seed, path)