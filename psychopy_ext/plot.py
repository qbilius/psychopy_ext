#!/usr/bin/env python

# Part of the psychopy_ext library
# Copyright 2010-2016 Jonas Kubilius
# The program is distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.

"""
A wrapper of matplotlib for producing pretty plots by default. As `pandas`
evolves, some of these improvements will hopefully be merged into it.

Usage::

    import plot
    plt = plot.Plot(nrows_ncols=(1,2))
    plt.plot(data)  # plots data on the first subplot
    plt.plot(data2)  # plots data on the second subplot
    plt.show()

TO-DO:
- factorplot with:
    - predefined CIs
    - no points
    - fill_between
- tsplot improvements

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.stats
import pandas
import pandas.tools.plotting  # for rcParams

import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

try:
    import seaborn as sns   # hope you have it
    _has_seaborn = True
except:  # ok, stick to your ugly matplotlib then
    # but I'm still gonna improve it using the ggplot style
    # from https://gist.github.com/huyng/816622
    # inspiration from mpltools
    rc_params = pandas.tools.plotting.mpl_stylesheet
    rc_params['interactive'] = False  # doesn't display otherwise
    plt.rcParams.update(rc_params)

from psychopy_ext import stats, utils


def plot_ci(df, what=['Line2D'], hue=None, ax=None):
    """
    Add confidence intervals to a plot.

    .. note: Experimental

    :Args:
        df (pandas.DataFrame)
            Input data
    :Kwargs:
        - what ({'Line2D', anything else})
            Which plot elements should be enhanced with confidence intervals.
        - hue (str or None, default: None)
            Whether there is grouping by hue (seaborn's convention) in data or not.
    """

    # lines = sns.plt.gca().get_lines()
    children = sns.plt.gca().get_children() if ax is None else ax.get_children()

    colors = []
    x = []
    for child in children:
        spl = str(child).split('(')[0]
        if spl in what:
            if spl == 'Line2D':
                if child.get_color() not in colors:
                    colors.append(child.get_color())
                x.append(child.get_xdata())
            else:
                colors.append('.15')
                x.append((child.get_x(), child.get_x()))

    if hue is not None:
        for kind, color in zip(df[hue].unique(), colors):
            sel = df[df[hue]==kind]
            for r, (rowno, row) in enumerate(sel.iterrows()):
                sns.plt.plot([r,r], [row.ci_low, row.ci_high], color=color,
                             lw=sns.mpl.rcParams['lines.linewidth']*1.8)
    else:
        for r, (rowno, row) in enumerate(df.iterrows()):
            sns.plt.plot([x[r][0],x[r][1]], [row.ci_low, row.ci_high], color=colors[0],
                         lw=sns.mpl.rcParams['lines.linewidth']*1.8)


# def draw_sig(agg, ax, popmean=0, errkind='sem', within=None):
#     ax.text(ticks[rno], mn.max(), stats.get_star(p), ha='center')

def mdsplot(df, icons=None, zoom=None):
    """
    Plot multidimensional scaling results.

    :Args:
        df

    :Kwargs:
        - icons ()
        - zoom (int or None, default: None)
    """
    sns.set_style('white')
    g = sns.FacetGrid(df, col='layer', size=9, #col_wrap=4, size=2,
                      sharex=False, sharey=False, aspect=1)
    g.map(_mdsplot, 'x', 'y', color='white', icons=icons, zoom=zoom)

def _mdsplot(x, y, icons=None, zoom=None, **kwargs):
    ax = sns.plt.gca()
    x_inches = sns.plt.gcf().get_size_inches()[0]
    x_range = np.ptp(x)

    for imname, xi, yi in zip(icons, x, y):
        if isinstance(imname, (str, unicode)):
            im = utils.load_image(imname, flatten=False, keep_alpha=True)
        else:
            im = imname
        if zoom is None:
            zoom = max(im.shape[:2]) / 2000.
        # zoom = .1 #max(im.shape[:2]) / 3000. * x_inches/3.

        imagebox = OffsetImage(im, zoom=zoom)
        ab = AnnotationBbox(imagebox, (xi, yi),
                            xybox=(0,0), #(20., -20.),
                            xycoords='data',
                            boxcoords="offset points",
                            frameon=False
                            )
        ax.add_artist(ab)

    ax.scatter(x, y, **kwargs)
    ax.axis('off')

def tsplot(data, x=None, unit=None, hue=None, y=None, palette=None,
           err_style='ci_band', ci=95., interpolate=True, color=None,
           estimator=np.mean, n_boot=1000, err_palette=None,
           err_kws=None, legend=True, ax=None, **kwargs):

    """
    A poor-man's reimplementations of Seaborn's tsplot that is more reliable
    but does not have all options working yet.

    .. warning: Not fully working.
    """

    def bootstrap_resample(r):
        if n_boot == 0 or n_boot is None:
            return (np.nan, np.nan)
        else:
            return stats.bootstrap_resample(r, ci=ci, niter=n_boot)

    if isinstance(hue, (str, unicode)): hue = [hue]
    if unit is None:
        agg = data
    else:
        agg = data.groupby([x]+hue+[unit])[y].mean().reset_index()
    agg = agg.pivot_table(index=x, columns=hue, values=y,
                          aggfunc=[estimator, bootstrap_resample])
    if ax is None: ax = sns.plt.subplot(111)

    if 'lw' not in kwargs:
        kwargs['lw'] = sns.mpl.rcParams['lines.linewidth']*1.8

    if hue is None:
        ci_low = map(lambda x: x[0], agg['bootstrap_resample'])
        ci_high = map(lambda x: x[1], agg['bootstrap_resample'])
        ax.fill_between(agg.index, ci_low, ci_high, alpha=.5)
        ax.plot(agg.index, agg['mean'], **kwargs)
    else:
        if color is None: color = sns.color_palette(palette, n_colors=len(data.groupby(hue).groups))
        for n, col in enumerate(agg['mean']):
            c = color[n % len(color)]
            ci_low = map(lambda x: x[0], agg[('bootstrap_resample', col)])
            ci_high = map(lambda x: x[1], agg[('bootstrap_resample', col)])
            ax.fill_between(agg.index, ci_low, ci_high, alpha=.5, color=c)
            ax.plot(agg.index, agg[('mean', col)], c=c, label=col, **kwargs)
    if legend:
        handles, labels = ax.get_legend_handles_labels()
        lgd = ax.legend(handles, labels, loc='center left', bbox_to_anchor=(1.1, .5))

    ax.set_xlabel(x)
    ax.set_ylabel(y)
