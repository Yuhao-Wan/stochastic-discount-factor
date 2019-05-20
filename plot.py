import matplotlib.pyplot as plt
import numpy as np

from baselines import logger
import plot_util as pu


dirs = './logs/'


results = pu.load_results(dirs, enable_progress=True, enable_monitor=False, verbose=True)

pu.plot_results(results,
                xy_fn=pu.xy_fn,
                average_group=True,
                split_fn=lambda _: '',
                group_fn=pu.split_fn,
                shaded_std=True,
                shaded_err=False)

#plt.xlim((0, 7200))
plt.tight_layout()
plt.savefig(dirs+'plot.png')