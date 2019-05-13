import sys
import gym
import tensorflow as tf
from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.ppo2 import ppo2
from baselines.common.vec_env import DummyVecEnv
import ppo2


dirs = './logs/mujoco/ascend/'


# def main(gpu, seed, lam_fraction, final_lam, path):
#     with tf.device('/device:GPU:%s' % gpu):

#         logger.configure(dir=dirs+'%s/%s/' % (path, seed))

#         env = gym.make("HalfCheetah-v2")
#         env = bench.Monitor(env, logger.get_dir())
#         env = DummyVecEnv([lambda: env]) 

#         kwargs = dict(network='mlp', 
#                       env=env, 
#                       total_timesteps=500000, 
#                       seed=seed,
#                       nsteps=2048,
#                       nminibatches=32,
#                       lam=0.1, 
#                       lam_fraction=lam_fraction, 
#                       final_lam=final_lam,
#                       gamma=0.99,
#                       noptepochs=10,
#                       log_interval=1,
#                       ent_coef=0.0,
#                       lr=lambda f: 3e-4 * f,
#                       cliprange=0.2,
#                       value_network='copy')

#         f = open(dirs+'%s/%s/params.txt' % (path, seed), 'w')
#         f.write(str(kwargs))
#         f.close()

#         model = ppo2.learn(**kwargs)
#         model.save(dirs+'%s/%s/model.pkl' % (path, seed))

#         env.close()


# if __name__ == '__main__':
#     gpu = int(sys.argv[1])
#     seed = int(sys.argv[2])
#     lam_fraction = float(sys.argv[3])
#     final_lam = float(sys.argv[4])
#     path = sys.argv[5]

#     main(gpu, seed, lam_fraction, final_lam, path)

def main(seed, lam_fraction, final_lam, path):

    logger.configure(dir=dirs+'%s/%s/' % (path, seed))

    env = gym.make("HalfCheetah-v2")
    env = bench.Monitor(env, logger.get_dir())
    env = DummyVecEnv([lambda: env]) 

    kwargs = dict(network='mlp', 
                env=env, 
                total_timesteps=500000, 
                seed=seed,
                nsteps=2048,
                nminibatches=32,
                lam=0.1, 
                lam_fraction=lam_fraction, 
                final_lam=final_lam,
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
    seed = int(sys.argv[1])
    lam_fraction = float(sys.argv[2])
    final_lam = float(sys.argv[3])
    path = sys.argv[4]

    main(seed, lam_fraction, final_lam, path)