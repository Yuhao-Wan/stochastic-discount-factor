import sys
import gym
import tensorflow as tf
from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.ppo2 import ppo2
from baselines.common.vec_env import DummyVecEnv


dirs = '../logs/mujoco/baselines/'


def main(gpu, envid, seed, path):

    with tf.device('/device:GPU:%s' % gpu):

        logger.configure(dir=dirs+'%s/%s/' % (path, seed))

        env = gym.make(envid)
        env = bench.Monitor(env, logger.get_dir())
        env = DummyVecEnv([lambda: env]) 

        kwargs = dict(network='mlp', 
                        env=env, 
                        total_timesteps=10000000, 
                        seed=seed,
                        nsteps=2048,
                        nminibatches=32,
                        lam=0.95,
                        gamma=0.99,
                        noptepochs=10,
                        log_interval=1,
                        ent_coef=0.0,
                        lr=lambda f: 3e-4 * f,
                        cliprange=0.2,
                        value_network='copy')

        f = open(dirs+'%s/%s/params.txt' % (path, seed), 'w')
        f.write(str(kwargs))
        f.close()

        model = ppo2.learn(**kwargs)
        model.save(dirs+'%s/%s/model.pkl' % (path, seed))

        env.close()

if __name__ == '__main__':
    gpu = int(sys.argv[1])
    envid = sys.argv[2]
    seed = int(sys.argv[3])
    path = sys.argv[4]

    main(gpu, envid, seed, path)