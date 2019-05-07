import matplotlib.pyplot as plt
import numpy as np
from baselines.common import plot_util as pu
from baselines import logger

#print(logger.get_dir())
results = pu.load_results('~/Desktop/Scholars/stochastic-discount-factor/logs/exp1')
names = []
for i in range(11):
    r = results[i]
    name = r.dirname.split('/')[-1]
    plt.plot(np.divide(r.progress.steps, 1000), r.progress["mean 100 episode reward"])
    names.append(name)

plt.legend(names)
plt.savefig('./logs/exp1/plot3.png')