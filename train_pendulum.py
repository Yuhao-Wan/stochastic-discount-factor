import sys
import gym
import tensorflow as tf
from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.ppo2 import ppo2
from baselines.common.vec_env import DummyVecEnv
import ppo2


dirs = './logs/ppo-pendulum/'

def main(gpu, seed, lam_fraction, final_lam, path):
    with tf.device('/device:GPU:%s' % gpu):

        logger.configure(dir=dirs+'%s/%s/' % (path, seed))

        env = gym.make("Pendulum-v0")
        env = bench.Monitor(env, logger.get_dir())
        env = DummyVecEnv([lambda: env]) 

        kwargs = dict(network='mlp', env=env, total_timesteps=200000, 
            eval_env = None, seed=seed, nsteps=128, 
            nminibatches=4, lam=0.1, lam_fraction=lam_fraction, 
            final_lam=final_lam, gamma=0.99, noptepochs=4, 
            log_interval=1, ent_coef=.01, lr=lambda f : f * 2.5e-4, 
            cliprange=0.1, save_interval=0, load_path=None, model_fn=None)

        f = open(dirs+'%s/%s/params.txt' % (path, seed), 'w')
        f.write(str(kwargs))
        f.close()

        model = ppo2.learn(**kwargs)
        model.save(dirs+'%s/%s/pendulum_model.pkl' % (path, seed))

        env.close()


if __name__ == '__main__':
    gpu = int(sys.argv[1])
    seed = int(sys.argv[2])
    lam_fraction = float(sys.argv[3])
    final_lam = float(sys.argv[4])
    path = sys.argv[5]

    main(gpu, seed, lam_fraction, final_lam, path)