import matplotlib.pyplot as plt
import numpy as np
from baselines.common import plot_util as pu
from baselines import logger


#print(logger.get_dir())
results = pu.load_results('~/Desktop/Scholars/stochastic-discount-factor/logs')
r = results[0]
plt.plot(r.progress.episodes, r.progress["mean 100 episode reward"])
plt.show()