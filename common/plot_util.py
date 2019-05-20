import matplotlib.pyplot as plt
import os.path as osp
import json
import os
import numpy as np
import pandas
from collections import defaultdict, namedtuple
from baselines.bench import monitor
from baselines.logger import read_json, read_csv
from baselines.common.plot_util import smooth, one_sided_ema, symmetric_ema, load_results


Result = namedtuple('Result', 'monitor progress dirname metadata')
Result.__new__.__defaults__ = (None,) * len(Result._fields)

def load_results(root_dir_or_dirs, enable_progress=True, enable_monitor=True, verbose=False):
    import re
    if isinstance(root_dir_or_dirs, str):
        rootdirs = [osp.expanduser(root_dir_or_dirs)]
    else:
        rootdirs = [osp.expanduser(d) for d in root_dir_or_dirs]
    allresults = []
    for rootdir in rootdirs:
        assert osp.exists(rootdir), "%s doesn't exist"%rootdir
        for dirname, dirs, files in os.walk(rootdir):
            if '-proc' in dirname:
                files[:] = []
                continue
            monitor_re = re.compile(r'(\d+\.)?(\d+\.)?monitor\.csv')
            if set(['metadata.json', 'monitor.json', 'progress.json', 'progress.csv']).intersection(files) or \
               any([f for f in files if monitor_re.match(f)]): 
                result = {'dirname' : dirname}
                if "metadata.json" in files:
                    with open(osp.join(dirname, "metadata.json"), "r") as fh:
                        result['metadata'] = json.load(fh)
                progjson = osp.join(dirname, "progress.json")
                progcsv = osp.join(dirname, "progress.csv")
                if enable_progress:
                    if osp.exists(progjson):
                        result['progress'] = pandas.DataFrame(read_json(progjson))
                    elif osp.exists(progcsv):
                        try:
                            result['progress'] = read_csv(progcsv)
                        except pandas.errors.EmptyDataError:
                            print('skipping progress file in ', dirname, 'empty data')
                    else:
                        if verbose: print('skipping %s: no progress file'%dirname)

                if result.get('progress') is not None:
                    allresults.append(Result(**result))
                    if verbose:
                        print('successfully loaded %s'%dirname)

    if verbose: print('loaded %i results'%len(allresults))
    return allresults

COLORS = ['palevioletred', 'darkcyan', 'peru', 'seagreen', 'indianred', 'slategray', 'purple', 'pink',
        'brown', 'orange', 'teal',  'lightblue', 'lime', 'lavender', 'turquoise',
        'darkolivegreen', 'tan', 'salmon', 'gold',  'darkred', 'darkblue']

def xy_fn(r):
    x = np.divide(r.progress.steps, 1000)
    y = smooth(r.progress["mean 100 episode reward"], radius=1)    
    #x = np.divide(r.progress.total_timesteps, 1000)
    #y = smooth(r.progress.eprewmean, radius=1)
    return x,y

def split_fn(r):
    import re 
    match = re.search(r'[^/-]+(?=(-\d+)?\Z)', r.dirname.split('/')[-2])
    if match:
        return match.group(0)

def plot_results(
    allresults, *,
    xy_fn=xy_fn,
    split_fn=split_fn,
    group_fn=split_fn,
    average_group=False,
    shaded_std=True,
    shaded_err=True,
    figsize=None,
    legend_outside=False,
    resample=0,
    smooth_step=1.0,
    tiling='vertical',
    xlabel='steps (1000)',
    ylabel='mean reward (100 episodes)'
):

    if split_fn is None: split_fn = lambda _ : ''
    if group_fn is None: group_fn = lambda _ : ''
    sk2r = defaultdict(list) # splitkey2results
    for result in allresults:
        splitkey = split_fn(result)
        sk2r[splitkey].append(result)
    assert len(sk2r) > 0
    assert isinstance(resample, int), "0: don't resample. <integer>: that many samples"
    if tiling == 'vertical' or tiling is None:
        nrows = len(sk2r)
        ncols = 1

    figsize = figsize or (6 * ncols, 6 * nrows)

    f, axarr = plt.subplots(nrows, ncols, sharex=False, squeeze=False, figsize=figsize)

    groups = sorted(list(set(group_fn(result) for result in allresults)))

    default_samples = 512
    if average_group:
        resample = resample or default_samples

    for (isplit, sk) in enumerate(sorted(sk2r.keys())):
        g2l = {}
        g2c = defaultdict(int)
        sresults = sk2r[sk]
        gresults = defaultdict(list)
        idx_row = isplit // ncols
        idx_col = isplit % ncols
        ax = axarr[idx_row][idx_col]
        for result in sresults:
            group = group_fn(result)
            g2c[group] += 1
            x, y = xy_fn(result)
            if x is None: x = np.arange(len(y))
            x, y = map(np.asarray, (x, y))
            if average_group:
                gresults[group].append((x,y))
            else:
                if resample:
                    x, y, counts = symmetric_ema(x, y, x[0], x[-1], resample, decay_steps=smooth_step)
                l, = ax.plot(x, y, color=COLORS[groups.index(group) % len(COLORS)])
                g2l[group] = l
        if average_group:
            for group in sorted(groups):
                xys = gresults[group]
                if not any(xys):
                    continue
                color = COLORS[groups.index(group) % len(COLORS)]
                origxs = [xy[0] for xy in xys]
                minxlen = min(map(len, origxs))
                def allequal(qs):
                    return all((q==qs[0]).all() for q in qs[1:])
                if resample:
                    low  = max(x[0] for x in origxs)
                    high = min(x[-1] for x in origxs)
                    usex = np.linspace(low, high, resample)
                    ys = []
                    for (x, y) in xys:
                        ys.append(symmetric_ema(x, y, low, high, resample, decay_steps=smooth_step)[1])
                else:
                    assert allequal([x[:minxlen] for x in origxs]),\
                        'If you want to average unevenly sampled data, set resample=<number of samples you want>'
                    usex = origxs[0]
                    ys = [xy[1][:minxlen] for xy in xys]
                ymean = np.mean(ys, axis=0)
                ystd = np.std(ys, axis=0)
                ystderr = ystd / np.sqrt(len(ys))
                l, = axarr[idx_row][idx_col].plot(usex, ymean, color=color)
                g2l[group] = l
                if shaded_err:
                    ax.fill_between(usex, ymean - ystderr, ymean + ystderr, color=color, alpha=.4)
                if shaded_std:
                    ax.fill_between(usex, ymean - ystd,    ymean + ystd,    color=color, alpha=.3)


        # https://matplotlib.org/users/legend_guide.html
        plt.tight_layout()
        if any(g2l.keys()):
            ax.legend(
                g2l.values(),
                ['%s (%i)'%(g, g2c[g]) for g in g2l] if average_group else g2l.keys(),
                loc=2 if legend_outside else 4,
                prop={'size': 12},
                bbox_to_anchor=(1,1) if legend_outside else None)
        ax.set_title(sk)
        # add xlabels, but only to the bottom row
        if xlabel is not None:
            for ax in axarr[-1]:
                plt.sca(ax)
                plt.xlabel(xlabel)
        # add ylabels, but only to left column
        if ylabel is not None:
            for ax in axarr[:,0]:
                plt.sca(ax)
                plt.ylabel(ylabel)

    return f, axarr
