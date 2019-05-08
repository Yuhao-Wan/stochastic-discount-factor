import matplotlib.pyplot as plt
import numpy as np
from baselines.common import plot_util as pu
from baselines import logger

#print(logger.get_dir())
results = pu.load_results('~/Desktop/Scholars/stochastic-discount-factor/logs/simple/explore05/')
names = []
for i in range(3):
    r = results[i]
    name = r.dirname.split('/')[-2]
    plt.plot(np.divide(r.progress.steps, 1000), r.progress["mean 100 episode reward"])
    #_, right = plt.xlim()
    #plt.xlim(right=1500)  
    names.append(name)

plt.legend(names)
plt.savefig('./logs/simple/explore05/plot.png')