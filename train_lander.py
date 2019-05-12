import sys
import tensorflow as tf
from baselines import deepq
from baselines import bench
from baselines import logger
from baselines.ppo2 import ppo2
from baselines.common.vec_env import DummyVecEnv
import ppo2


def main(gpu, seed, lam_fraction, final_lam, path):
    
    with tf.device('/device:GPU:%s' % gpu):

        logger.configure(dir='./logs/ppo-ascend/%s/%s/' % (path, seed))
        
        env = make_atari('LunarLanderContinuous-v2')
        env = bench.Monitor(env, logger.get_dir())
        env = DummyVecEnv([lambda: env]) 

        model = ppo2.learn(network='mlp', env=env, total_timesteps=2000000, 
                           seed=seed, lam=0.1, lam_fraction=lam_fraction, 
                           final_lam=final_lam)

        model.save('./logs/ppo-ascend/%s/%s/model.pkl' % (path, seed))
        env.close()

if __name__ == '__main__':
    gpu = sys.argv[1]
    seed = int(sys.argv[2])
    lam_fraction = float(sys.argv[3])
    final_lam = float(sys.argv[4])
    path = sys.argv[5]
    
    main(gpu, seed, lam_fraction, final_lam, path)
