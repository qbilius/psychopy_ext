#!/usr/bin/env python

# Part of the psychopy_ext library
# Copyright 2010-2013 Jonas Kubilius
# The program is distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.

"""
A wrapper of matplotlib for producing pretty plots by default. As `pandas`
evolves, some of these improvements will hopefully be merged into it.

Usage:
    import plot
    plt = plot.Plot(nrows_ncols=(1,2))
    plt.plot(data)  # plots data on the first subplot
    plt.plot(data2)  # plots data on the second subplot
    plt.show()

"""

import fractions

import numpy as np
import scipy.stats
import pandas

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Rectangle
from matplotlib.ticker import MultipleLocator

import stats


# parameters for pretty plots in the ggplot style
# from https://gist.github.com/huyng/816622
# inspiration from mpltools
# will soon be removed as pandas has this implemented in the dev versions
s = {'axes.facecolor': '#eeeeee',
     'axes.edgecolor': '#bcbcbc',
     'axes.linewidth': 1,
     'axes.grid': True,
     'axes.titlesize': 'x-large',
     'axes.labelsize': 'large',  # 'x-large'
     'axes.labelcolor': '#555555',
     'axes.axisbelow': True,
     'axes.color_cycle': ['#348ABD', # blue
                          '#7A68A6', # purple
                          '#A60628', # red
                          '#467821', # green
                          '#CF4457', # pink
                          '#188487', # turquoise
                          '#E24A33'], # orange

     'figure.facecolor': '0.85',
     'figure.edgecolor': '0.5',
     'figure.subplot.hspace': .5,

     'font.family': 'monospace',
     'font.size': 10,

     'xtick.color': '#555555',
     'xtick.direction': 'in',
     'xtick.major.pad': 6,
     'xtick.major.size': 0,
     'xtick.minor.pad': 6,
     'xtick.minor.size': 0,

     'ytick.color': '#555555',
     'ytick.direction': 'in',
     'ytick.major.pad': 6,
     'ytick.major.size': 0,
     'ytick.minor.pad': 6,
     'ytick.minor.size': 0,

     'legend.fancybox': True,

     'lines.antialiased': True,
     'lines.linewidth': 1.0,

     'patch.linewidth'        : .5,     # edge width in points
     'patch.facecolor'        : '#348ABD', # blue
     'patch.edgecolor'        : '#eeeeee',
     'patch.antialiased'      : True,    # render patches in antialised (no jaggies)

     }
plt.rcParams.update(s)


class Plot(object):

    def __init__(self, kind='', figsize=None, nrows=1, ncols=1, rect=111,
                 cbar_mode='single', squeeze=False, **kwargs):
       self._create_subplots(kind=kind, figsize=figsize, nrows=nrows,
            ncols=ncols, **kwargs)

    def _create_subplots(self, kind='', figsize=None, nrows=1, ncols=1, rect=111,
        cbar_mode='single', squeeze=False, **kwargs):
        """
        :Kwargs:
            - kind (str, default: '')
                The kind of plot. For plotting matrices or images
                (`matplotlib.pyplot.imshow`), choose `matrix`, otherwise leave
                blank.
            - figsize (tuple, defaut: None)
                Size of the figure.
            - nrows_ncols (tuple, default: (1, 1))
                Shape of subplot arrangement.
            - **kwargs
                A dictionary of keyword arguments that `matplotlib.ImageGrid`
                or `matplotlib.pyplot.suplots` accept. Differences:
                    - `rect` (`matplotlib.ImageGrid`) is a keyword argument here
                    - `cbar_mode = 'single'`
                    - `squeeze = False`
        :Returns:
            `matplotlib.pyplot.figure` and a grid of axes.
        """

        if 'nrows_ncols' not in kwargs:
            nrows_ncols = (nrows, ncols)
        else:
            nrows_ncols = kwargs['nrows_ncols']
            del kwargs['nrows_ncols']
        try:
            num = self.fig.number
            self.fig.clf()
        except:
            num = None

        if kind == 'matrix':
            if 'sharex' in kwargs and 'sharey' in kwargs:
                if kwargs['sharex'] and kwargs['sharey']:
                    kwargs['share_all'] = True
                    del kwargs['sharex']
                    del kwargs['sharey']
            self.fig = self.figure(figsize=figsize, num=num)
            if 'label_mode' not in kwargs:
                kwargs['label_mode'] = "L"
            if 'axes_pad' not in kwargs:
                kwargs['axes_pad'] = .5
            if 'share_all' not in kwargs:
                kwargs['share_all'] = True
            if kwargs['share_all']:
                if 'cbar_mode' not in kwargs:
                    kwargs['cbar_mode'] = "single"
            self.axes = ImageGrid(self.fig, rect,
                                  nrows_ncols=nrows_ncols,
                                  **kwargs
                                  )
            self.naxes = len(self.axes.axes_all)

        else:
            self.fig, self.axes = plt.subplots(
                nrows=nrows_ncols[0],
                ncols=nrows_ncols[1],
                figsize=figsize,
                squeeze=squeeze,
                num=num,
                **kwargs
                )
            self.axes = self.axes.ravel()  # turn axes into a list
            self.naxes = len(self.axes)
        self.kind = kind
        self.subplotno = -1  # will get +1 after the plot command
        self.nrows_ncols = nrows_ncols
        try:
            self.sharex = kwargs['sharex']
            self.sharey = kwargs['sharey']
        except:
            self.sharex = False
            self.sharey = False
        self.rcParams = plt.rcParams
        return (self.fig, self.axes)

    def __getattr__(self, name):
        """Pass on a `matplotlib` function that we haven't modified
        """
        def method(*args, **kwargs):
            return getattr(plt, name)(*args, **kwargs)

        try:
            return method  # is it a function?
        except TypeError:  # so maybe it's just a self variable
            return getattr(self, name)

    def __getitem__(self, key):
        """Allow to get axes as Plot()[key]
        """
        if key > self.naxes:
            raise IndexError
        if key < 0:
            key += self.naxes
        return self.axes[key]

    def get_ax(self, subplotno=None):
        """
        Returns the current or the requested axis from the current figure.

        .. note: The :class:`Plot()` is indexable so you should access axes as
                 `Plot()[key]` unless you want to pass a list like (row, col).

        :Kwargs:
            subplotno (int, default: None)
                Give subplot number explicitly if you want to get not the
                current axis

        :Returns:
            ax
        """
        if subplotno is None:
            no = self.subplotno
        else:
            no = subplotno

        if isinstance(no, int):
            try:
                ax = self.axes[no]
            except:  # a single subplot
                ax = self.axes
        else:
            if no[0] < 0: no += len(self.axes._nrows)
            if no[1] < 0: no += len(self.axes._ncols)

            if isinstance(self.axes, ImageGrid):  # axes are a list
                if self.axes._direction == 'row':
                    no = self.axes._ncols * no[0] + no[1]
                else:
                    no = self.axes._nrows * no[0] + no[1]
            else:  # axes are a grid
                no = self.axes._ncols * no[0] + no[1]
            ax = self.axes[no]

        return ax

    def next(self):
        """
        Returns the next axis.

        This is useful when a plotting function is not implemented by
        :mod:`plot` and you have to instead rely on matplotlib's plotting
        which does not advance axes automatically.
        """
        self.subplotno += 1
        return self.get_ax()

    def sample_paired(self, ncolors=2):
        """
        Returns colors for matplotlib.cm.Paired.
        """
        if ncolors <= 12:
            colors_full = [mpl.cm.Paired(i * 1. / 11) for i in range(1, 12, 2)]
            colors_pale = [mpl.cm.Paired(i * 1. / 11) for i in range(10, -1, -2)]
            colors = colors_full + colors_pale
            return colors[:ncolors]
        else:
            return [mpl.cm.Paired(c) for c in np.linspace(0,ncolors)]

    def get_colors(self, ncolors=2, cmap='Paired'):
        """
        Get a list of nice colors for plots.

        FIX: This function is happy to ignore the ugly settings you may have in
        your matplotlibrc settings.
        TODO: merge with mpltools.color

        :Kwargs:
            ncolors (int, default: 2)
                Number of colors required. Typically it should be the number of
                entries in the legend.
            cmap (str or matplotlib.cm, default: 'Paired')
                A colormap to sample from when ncolors > 12

        :Returns:
            a list of colors
        """
        colorc = plt.rcParams['axes.color_cycle']
        if ncolors < len(colorc):
            colors = colorc[:ncolors]
        elif ncolors <= 12:
            colors = self.sample_paired(ncolors=ncolors)
        else:
            thisCmap = mpl.cm.get_cmap(cmap)
            norm = mpl.colors.Normalize(0, 1)
            z = np.linspace(0, 1, ncolors + 2)
            z = z[1:-1]
            colors = thisCmap(norm(z))
        return colors

    def pivot_plot(self,df,rows=None,cols=None,values=None,yerr=None,
                   **kwargs):
        agg = self.aggregate(df, rows=rows, cols=cols,
                                 values=values, yerr=yerr)
        if yerr is None:
            no_yerr = True
        else:
            no_yerr = False
        return self._plot(agg, no_yerr=no_yerr,**kwargs)


    def _plot(self, agg, ax=None,
                   title='', kind='bar', xtickson=True, ytickson=True,
                   no_yerr=False, numb=False, autoscale=True, **kwargs):
        """DEPRECATED plotting function"""
        print "plot._plot() has been DEPRECATED; please don't use it anymore"
        self.plot(agg, ax=ax,
                   title=title, kind=kind, xtickson=xtickson, ytickson=ytickson,
                   no_yerr=no_yerr, numb=numb, autoscale=autoscale, **kwargs)

    def plot(self, agg, kind='bar', subplots=None, subplots_order=None,
            autoscale=True, **kwargs):
        """
        The main plotting function.

        :Args:
            agg (`pandas.DataFrame` or similar)
                A structured input, preferably a `pandas.DataFrame`, but in
                principle accepts anything that can be converted into it.

        :Kwargs:
            - subplots (None, True, or False; default=None)
                Whether you want to split data into subplots or not. If True,
                the top level is treated as a subplot. If None, detects
                automatically based on `agg.columns.names` -- the first entry
                to start with `subplots.` will be used. This is the default
                output from `stats.aggregate` and is recommended.
            - kwargs
                Keyword arguments for plotting

        :Returns:
            A list of axes of all plots.
        """
        #if isinstance(agg, (list, tuple)):
            #agg = np.array(agg)
        if len(self.get_fignums()) == 0:
            self.draw()
        if not isinstance(agg, pandas.DataFrame):
            agg = pandas.DataFrame(agg)
            if agg.shape[1] == 1:  # Series
                agg = pandas.DataFrame(agg).T
        else:
            agg = pandas.DataFrame(agg)
        axes = []

        if subplots_order is not None:
            sbp = subplots_order
        else:
            try:
                s_idx = [s for s,n in enumerate(agg.columns.names) if n.startswith('subplots.')]
            except:
                s_idx = None

            if s_idx is not None:  # subplots implicit in agg
                try:
                    sbp = agg.columns.levels[s_idx[0]]
                except:
                    if len(s_idx) > 0:
                        sbp = agg.columns
                    else:
                        sbp = None
            elif subplots:  # get subplots from the top level column
                sbp = agg.columns.levels[0]
            else:
                sbp = None

        if sbp is None:
            axes = self._plot_ax(agg, kind=kind, **kwargs)
            axes.agg = agg
            #axes, xmin, xmax, ymin, ymax = self._label_ax(agg, mean, p_yerr, axes, kind=kind,
                                    #autoscale=autoscale, **kwargs)
            axes = [axes]
            if 'title' in kwargs:
                if kwargs['title'] is not None:
                    axes[0].set_title(kwargs['title'])
            #if 'title' not in kwargs:
                #kwargs['title']
            #else:
                #title = ''
        else:
            # if we haven't made any plots yet...
            #import pdb; pdb.set_trace()
            if self.subplotno == -1:
                num_subplots = len(sbp)
                # ...can still adjust the number of subplots
                if num_subplots > self.naxes:
                    self._create_subplots(ncols=num_subplots, kind=kind)

            #if 'share_all' in kwargs:
                #if kwargs['share_all']:
                    #norm = (
            for no, subname in enumerate(sbp):

                if 'title' not in kwargs:
                    title = subname
                else:
                    title = kwargs['title']
                    if title is None:
                        title = subname

                ax = self._plot_ax(agg[subname], kind=kind, **kwargs)
                #ax, xmin, xmax, ymin, ymax = self._label_ax(agg[subname],
                                        #mean, p_yerr, ax, kind=kind,
                                        #legend=legend, autoscale=autoscale, **kwargs)
                ax.agg = agg[subname]
                ax.set_title(title)
                #ax.mean = mean

                #xmins.append(xmin)
                #xmaxs.append(xmax)
                #ymins.append(ymin)
                #ymaxs.append(ymax)
                axes.append(ax)

        self.decorate(axes, kind=kind, **kwargs)
        return axes

    def decorate(self, axes, kind='bar', **kwargs):
        lims = np.zeros((len(axes),3,2))

        for i, ax in enumerate(axes):
            #if kind in ['scatter', 'mds']:
                #lims[i,0,0] = ax.mean.min().min()
                #lims[i,0,1] = ax.mean.max().max()
                #lims[i,1,0] = ax.mean.min().min()
                #lims[i,1,1] = ax.mean.max().max()
            if kind in ['histogram']:
                lims[i,0] = ax.get_xlim()
                lims[i,1] = ax.get_ylim()
                #import pdb; pdb.set_trace()
            elif kind == 'matrix':
                lims[i,0] = ax.get_xlim()
                lims[i,1] = ax.get_xlim()
                lims[i,2,0] = ax.mean.min().min()
                lims[i,2,1] = ax.mean.max().max()
            else:
                lims[i,0] = ax.get_xlim()
                lims[i,1] = self._autoscale(ax.mean, ax.p_yerr, kind=kind)

        if 'xlim' in kwargs:
            for lim in lims:
                lim[0,:] = kwargs['xlim']
        if 'ylim' in kwargs:
            for lim in lims:
                lim[1,:] = kwargs['ylim']
        #if kind=='matrix': import pdb; pdb.set_trace()

        if kind == 'matrix':
            for ax in axes:
                for im in ax.images:
                    norm = mpl.colors.normalize(np.min(lims[:,2,0]),
                                                np.max(lims[:,2,1]))
                    im.set_norm(norm)

        elif kind not in ['scatter', 'mds']:
            if self.sharex:
                #if kind in ['scatter', 'mds']: # set y-axis limits globally
                #if 'xlim' in kwargs:
                    #axes[0].set_xlim(kwargs['xlim'])
                #else:
                #for ax in axes:
                    #ax.set_xlim(np.min(lims[:,0], 0))
                axes[0].set_xlim([np.min(lims[:,0,0]), np.max(lims[:,0,1])])
            else:
                for i, ax in enumerate(axes):
                    ax.set_xlim(lims[i,0])
                #self._label_x(mean, p_yerr, ax, kind=kind, **kwargs)

            if self.sharey: # set y-axis limits globally
                #if 'ylim' in kwargs:
                    #ax.set_ylim(kwargs['ylim'])
                #elif autoscale and kind in ['line', 'bar', 'scatter', 'mds']:
                    #ax.set_ylim([np.min(ymins), np.max(ymaxs)])
                axes[0].set_ylim([np.min(lims[:,1,0]), np.max(lims[:,1,1])])
            else:
                for i, ax in enumerate(axes):
                    ax.set_ylim(lims[i,1])


        for axno, ax in enumerate(axes):
            if kind not in ['scatter', 'mds']:
                #import pdb; pdb.set_trace()
                #print axno / self.nrows_ncols[1]
                #print self.nrows_ncols[1]
                if axno / self.nrows_ncols[1] == self.nrows_ncols[0]-1 \
                or not self.sharex:
                    self._label_x(ax.mean, ax.p_yerr, ax, kind=kind, **kwargs)
                if axno % self.nrows_ncols[1] == 0 or not self.sharey:
                    self._label_y(ax.mean, ax.p_yerr, ax, kind=kind, **kwargs)

            if kind == 'bar':
                self.draw_sig(ax.agg, ax, **kwargs)

            #if kind in ['matrix', 'scatter', 'mds']:
                #legend = False
            #else:
            # all plots are the same, onle legend will suffice
            if len(axes) > 1:
                if axno == 0:
                    legend = None
                else:
                    legend = False
            else:  # plots vary; each should get a legend
                legend = None

            if kind not in ['matrix', 'scatter', 'mds']:
                self._draw_legend(ax, visible=legend, data=ax.mean, **kwargs)
            if 'numb' in kwargs:
                if kwargs['numb'] == True:
                    self.add_inner_title(ax, title='%s' % self.subplotno, loc=2)

            #self._label_ax(ax.agg, ax.mean, ax.p_yerr, ax, kind=kind, legend=legend, **kwargs)
        if self.sharex or len(axes) == 1:
            try:
                self.sig_t = pandas.concat([ax.sig_t for ax in axes], axis=1)
                            #keys=[ax.get_title() for ax in axes])
                self.sig_p = pandas.concat([ax.sig_p for ax in axes], axis=1)
                            #keys=[ax.get_title() for ax in axes])
            except:
                pass
        return axes

    def printfig(self):
        try:
            #import pdb; pdb.set_trace()
            for ax in self.axes:
                print ax.get_title()
                print ax.mean
        except:
            pass

    def _plot_ax(self, agg, ax=None, kind='bar', order=None, **kwargs):
        if ax is None:
            ax = self.next()

        mean, p_yerr = stats.confidence(agg)

        if mean.index.nlevels == 1:  # oops, nothing to unstack
            mean = pandas.DataFrame(mean)  # assume rows are rows
            p_yerr = pandas.DataFrame(p_yerr)
        else:
            # make columns which will turn into legend entries
            mean = self._unstack_levels(mean, 'cols')
            p_yerr = self._unstack_levels(p_yerr, 'cols')
        if isinstance(agg, pandas.Series) and kind=='bean':
            kind = 'bar'
            print 'WARNING: Beanplot not available for a single measurement'

        if kind == 'bar':
            self.bar_plot(mean, yerr=p_yerr, ax=ax)
        elif kind == 'line':
            self.line_plot(mean, yerr=p_yerr, ax=ax)
        elif kind == 'bean':
            ax = self.bean_plot(agg, ax=ax, order=order, **kwargs)
        elif kind == 'histogram':
            ax = self.histogram(agg, ax=ax, **kwargs)
        elif kind == 'matrix':
            ax = self.matrix_plot(mean, ax=ax)
        elif kind == 'scatter':
            ax = self.scatter_plot(mean, ax=ax)
        elif kind == 'mds':
            ax, mean = self.mds_plot(mean, ax=ax, **kwargs)
        else:
            raise Exception('%s plot not recognized. Choose from '
                            '{bar, line, bean, matrix, scatter, mds}.' %kind)
        ax.mean = mean
        ax.p_yerr = p_yerr
        return ax

    def _label_ax_old(self, agg, mean, p_yerr, ax, title='', kind='bar', legend=None,
                   autoscale=True, **kwargs):
        if kind not in ['scatter', 'mds']:
            if self.subplotno / self.nrows_ncols[1] == self.nrows_ncols[1] \
            or not self.sharex:
                self._label_x(mean, p_yerr, ax, kind=kind, **kwargs)
            if self.subplotno % self.nrows_ncols[1] == 0 or not self.sharey:
                self._label_y(mean, p_yerr, ax, kind=kind, **kwargs)

        if kind == 'bar':
            self.draw_sig(agg, ax)

        #if not self.sharex and kind in ['scatter', 'mds']:
            ## set x-axis limits
            #if 'xlim' in kwargs:
                #ax.set_xlim(kwargs['xlim'])
            #else:
                #ax.set_xlim([xmin, xmax])

        #if not self.sharey:
            ## set y-axis limits
            #if 'ylim' in kwargs:
                #ax.set_ylim(kwargs['ylim'])
            #elif autoscale and kind in ['line', 'bar']:
                #ax.set_ylim([ymin, ymax])

        #if title is not None: ax.set_title(title)

        if kind not in ['matrix', 'scatter', 'mds']:
            self._draw_legend(ax, visible=legend, data=mean, **kwargs)
        if 'numb' in kwargs:
            if kwargs['numb'] == True:
                self.add_inner_title(ax, title='%s' % self.subplotno, loc=2)

        return ax

    def _label_x(self, mean, p_yerr, ax, kind='bar', xtickson=True,
                   rotate=True, **kwargs):

        if kind not in ['line', 'histogram']:
            ax.set_xticklabels(self._format_labels(labels=mean.index))
        labels = ax.get_xticklabels()
        if len(labels) > 0:
            max_len = max([len(label.get_text()) for label in labels])
            if max_len > 10 and rotate:  #FIX to this: http://stackoverflow.com/q/5320205
                for label in labels:
                    label.set_ha('right')
                    label.set_rotation(30)
            else:
                for label in labels:
                    label.set_rotation(0)

        if 'xlabel' in kwargs:
            xlabel = kwargs['xlabel']
        else:
            xlabel = None
        if xlabel is None:
            xlabel = self._get_title(mean, 'rows')
        ax.set_xlabel(xlabel)

        return ax

    def _label_y(self, mean, p_yerr, ax, kind='bar', ytickson=True,
                   **kwargs):

        if kind == 'matrix':
            ax.set_yticklabels(self._format_labels(labels=mean.columns))
        if not ytickson:
            ax.set_yticklabels(['']*len(ax.get_yticklabels()))

        if 'ylabel' in kwargs:
            ylabel = kwargs['ylabel']
        else:
            ylabel = None
        if ylabel is None:
            ylabel = ''
        ax.set_ylabel(ylabel)

        return ax

    def _autoscale(self, mean, p_yerr, kind='bar'):
        #mean_array = np.asarray(mean)
        r = mean.max().max() - mean.min().min()
        ebars = np.where(np.isnan(p_yerr), 0, p_yerr)
        if np.all(ebars == 0):  # basically no error bars
            ebars = r/3.  # so give so space above the bars
        ymin = (mean - ebars).min().min()
        ymax = (mean + ebars).max().max()
        if kind == 'bar':  # for barplots, 0 must be included
            if ymin > 0:
                ymin = 0
            if ymax < 0:
                ymax = 0
        xyrange = ymax - ymin
        if ymin != 0:
            ymin -= xyrange / 3.
        if ymax != 0:
            ymax += xyrange / 3.
        return ymin, ymax

    def _get_title(self, data, pref):
        if pref == 'cols':
            dnames = data.columns.names
            try:
                dlevs = data.columns.levels
            except:
                dlevs = [data.columns]
        else:
            dnames = data.index.names
            try:
                dlevs = data.index.levels
            except:
                dlevs = [data.index]
        if len(dnames) == 0 or dnames[0] == None: dnames = ['']
        title = [n.split('.',1)[1] for n in dnames if n.startswith(pref+'.')]
        levels = [l for l,n in zip(dlevs,dnames) if n.startswith(pref+'.')]
        title = [n for n,l in zip(title,levels) if len(l) > 1]

        title = ', '.join(title)
        return title

    def draw_sig(self, agg, ax, popmean=0, **kwargs):
        if isinstance(agg, pandas.DataFrame):
            mean, p_yerr = stats.confidence(agg)
        else:
            mean = agg
        #import pdb; pdb.set_trace()
        r = mean.max().max() - mean.min().min()
        ebars = np.where(np.isnan(p_yerr), r/3., p_yerr)
        eb = np.max([r/6., 1.5*np.max(ebars)/2])

        try:  # not guaranteed that columns have names
            inds = [i for i,n in enumerate(agg.columns.names) if n.startswith('rows.')]
        except:
            rlabels = agg.columns
        else:
            #rlevel = inds[-1] + 1
            #try:
                #rlabels = agg.columns.levels[inds[-1]]
            #except:
                #if len(inds) == 1:
                    #rlabels = agg.columns
                #else:
                    #rlabels = 1
            #import pdb; pdb.set_trace()
            # all columns names start with 'row.'
            if len(inds) == len(agg.columns.names):
                rlabels = agg.columns
            elif len(inds) == 0:  # no rows at all
                rlabels = [None]
            else:
                rlabels = agg.mean().unstack(inds).columns
        ticks = ax.get_xticks()

        sig_t = []
        sig_p = []

        for rno, rlab in enumerate(rlabels):
            #import pdb; pdb.set_trace()
            #rlab = rlabel[:rlevel]
            if len(inds) == 0:  # no rows at all
                d = agg
            else:
                d = agg[rlab]

            if d.ndim == 1:
                d = d.dropna() #d[pandas.notnull(d)]
                t, p = scipy.stats.ttest_1samp(d, popmean=popmean)
                ax.text(ticks[rno], mean[rlab] + np.sign(mean[rlab])*eb,
                        stats.get_star(p), ha='center')
            elif d.ndim == 2 and d.shape[1] == 2:
                d1 = d.iloc[:,0].dropna()
                d2 = d.iloc[:,1].dropna()
                #import pdb; pdb.set_trace()
                t, p = scipy.stats.ttest_rel(d1, d2)
                mn = mean[rlab] + np.sign(mean[rlab])*eb
                ax.text(ticks[rno], mn.max(), stats.get_star(p), ha='center')
            try:
                sig_t.append((rlab, t))
                sig_p.append((rlab, p))
            except:
                pass
        if len(sig_t) > 0:
            ax.sig_t = pandas.DataFrame(sig_t)
            ax.sig_p = pandas.DataFrame(sig_p)

    def _draw_legend(self, ax, visible=None, data=None, **kwargs):
        leg = ax.get_legend()  # get an existing legend
        if leg is None:  # create a new legend
            leg = ax.legend(loc='center left')
        if leg is not None:
            leg.legendPatch.set_alpha(0.5)

            try:  # may or may not have any columns
                leg.set_title(self._get_title(data, 'cols'))
            except:
                pass
            new_texts = self._format_labels(data.columns)
            texts = leg.get_texts()
            for text, new_text in zip(texts, new_texts):
                text.set_text(new_text)

            if 'legend_visible' in kwargs:
                leg.set_visible(kwargs['legend_visible'])
            elif visible is not None:
                leg.set_visible(visible)
            else:  #decide automatically
                if len(leg.texts) == 1:  # showing a single legend entry is useless
                    leg.set_visible(False)
                else:
                    leg.set_visible(True)

    def set_legend_pos(self, subplot=1, loc=6,#'center left',
                        bbox_to_anchor=(1.1, 0.5)):
        #for ax in self.axes:
            ##import pdb; pdb.set_trace()
            #leg = ax.get_legend()
            #if leg is not None: break
        leg = self.axes[subplot-1].get_legend()
        if leg is not None:
            #leg.set_axes(self.axes[subplot-1])
            leg._set_loc(loc)
            leg.set_bbox_to_anchor(bbox_to_anchor)
            leg.set_visible(True)
        # frameon=False

    def _format_labels(self, labels='', names=''):
        """Formats labels to avoid uninformative (singular) entries
        """
        if len(labels) > 1:
            try:
                labels.levels
            except:
                new_labs = [str(l) for l in labels]
            else:
                sel = [i for i,l in enumerate(labels.levels) if len(l) > 1]
                new_labs = []
                for r in labels:
                    label = [l for i,l in enumerate(r) if i in sel]
                    if len(label) == 1:
                        label = label[0]
                    else:
                        label = ', '.join([str(lab) for lab in label])
                    new_labs.append(label)
        else:
            new_labs = ''
        return new_labs

    def hide_plots(self, nums):
        """
        Hides an axis.

        :Args:
            nums (int, tuple or list of ints)
                Which axes to hide.
        """
        if isinstance(nums, int) or isinstance(nums, tuple):
            nums = [nums]
        for num in nums:
            ax = self.get_ax(num)
            ax.axis('off')

    def bar_plot(self, data, yerr=None, ax=None):
        """
        Plots a bar plot.

        :Args:
            data (`pandas.DataFrame` or any other array accepted by it)
                A data frame where rows go to the x-axis and columns go to the
                legend.

        """
        data = pandas.DataFrame(data)
        if yerr is None:
            yerr = np.empty(data.shape)
            yerr = yerr.reshape(data.shape)  # force this shape
            yerr = np.nan
        if ax is None:
            self.subplotno += 1
            ax = self.get_ax()

        colors = self.get_colors(len(data.columns))

        n = len(data.columns)
        idx = np.arange(len(data))
        width = .75 / n
        rects = []
        for i, (label, column) in enumerate(data.iteritems()):
            rect = ax.bar(idx + i*width - .75/2, column, width, label=str(label),
                yerr=yerr[label].tolist(), color = colors[i], ecolor='black')
            # TODO: yerr indexing might need fixing
            rects.append(rect)
        ax.set_xticks(idx)# + width*n/2 + width/2)
        ax.legend(rects, data.columns.tolist())

        return ax

    def line_plot(self, data, yerr=None, ax=None):
        """
        Plots a bar plot.

        :Args:
            data (`pandas.DataFrame` or any other array accepted by it)
                A data frame where rows go to the x-axis and columns go to the
                legend.

        """
        data = pandas.DataFrame(data)
        if yerr is None:
            yerr = np.empty(data.shape)
            yerr = yerr.reshape(data.shape)  # force this shape
            yerr = np.nan
        if ax is None:
            self.subplotno += 1
            ax = self.get_ax()

        #colors = self.get_colors(len(data.columns))

        x = range(len(data))
        lines = []
        for i, (label, column) in enumerate(data.iteritems()):
            line = ax.plot(x, column, label=str(label))
            lines.append(line)
            ax.errorbar(x, column, yerr=yerr[label].tolist(), fmt=None,
                ecolor='black')
        step = np.ptp(x) / (len(x) - 1.)
        xlim = ax.get_xlim()
        if xlim[0] != np.min(x) - step/2:  # if sharex, this might have been set
            ax.set_xlim((xlim[0] - step/2, xlim[1] + step/2))
            ax.set_xticks(x)

        # nicely space tick labels
        if len(x) <= 5:
            largest = len(x)
        else:
            largest = [fractions.gcd(len(x),i+1) for i in range(5)]
            largest = np.argsort(largest)[-1] + 1
        tickpos = len(x) / largest
        majorLocator = MultipleLocator(tickpos)
        ax.xaxis.set_major_locator(majorLocator)
        ax.set_xticklabels(self._format_labels(labels=data.index)[0:len(x):tickpos])

        #minorLocator = MultipleLocator(1)
        #ax.xaxis.set_minor_locator(minorLocator)
        return ax

    def scatter_plot(self, data, ax=None, labels=None, **kwargs):
        return self._scatter(data.iloc[:,0], data.iloc[:,1], labels=data.index,
                             ax=ax, **kwargs)

    def _scatter(self, x, y, labels=None, ax=None, **kwargs):
        """
        Draws a scatter plot.

        This is very similar to `matplotlib.pyplot.scatter` but additionally
        accepts labels (for labeling points on the plot), plot title, and an
        axis where the plot should be drawn.

        :Args:
            - x (an iterable object)
                An x-coordinate of data
            - y (an iterable object)
                A y-coordinate of data

        :Kwargs:
            - ax (default: None)
                An axis to plot in.
            #- labels (list of str, default: None)
                #A list of labels for each plotted point
            #- title (str, default: '')
                #Plot title
            - kwargs
                Additional keyword arguments for `matplotlib.pyplot.scatter`

        :Return:
            Current axis for further manipulation.

        """
        if ax is None:
            self.subplotno += 1
            ax = self.get_ax()
        plt.rcParams['axes.color_cycle']
        if labels is not None:
            #for label, (pointx, pointy) in data.iterrows():# enumerate(zip(x,y)):
                #ax.text(pointx, pointy, label, backgroundcolor=(1,1,1,.5))
            for i, (pointx, pointy) in enumerate(zip(x,y)):
                ax.text(pointx, pointy, labels[i], backgroundcolor=(1,1,1,.5))
        ax.scatter(x, y, marker='o', color=self.get_colors()[0], **kwargs)
        return ax

    def histogram(self, data, ax=None, bins=100, **kwargs):
        data = pandas.DataFrame(data)
        #import pdb; pdb.set_trace()
        if ax is None:
            self.subplotno += 1
            ax = self.get_ax()
        #import pdb; pdb.set_trace()
        #data.ndim == 1
        ax.hist(np.array(data), bins=bins)
        return ax

    def matrix_plot(self, data, ax=None, normalize='auto', **kwargs):
        """
        Plots a matrix.

        .. warning:: Not tested yet

        :Args:
            matrix

        :Kwargs:
            - ax (default: None)
                An axis to plot on.
            - title (str, default: '')
                Plot title
            - kwargs
                Keyword arguments to pass to :func:`matplotlib.pyplot.imshow`

        """
        #if ax is None: ax = self.next()
        #from mpl_toolkits.axes_grid1 import make_axes_locatable
        #import pdb; pdb.set_trace()
        import matplotlib.colors
        if normalize != 'auto':
            norm = matplotlib.colors.normalize(vmin=data.min().min(),
                                               vmax=data.max().max())
        else:
            norm = None
        data = self._unstack_levels(data, 'cols')
        im = ax.imshow(data, norm=norm, interpolation='none', **kwargs)
        ax.set_xticks(range(data.shape[1]))
        ax.set_yticks(range(data.shape[0]))
        self.axes[self.subplotno].cax.colorbar(im)

        #divider = make_axes_locatable(ax)
        #cax = divider.append_axes("right", size="5%", pad=0.05)
        #self.colorbar(im, cax=cax)
        #self.colorbar(im, cax = self.axes.cbar_axes[0])
        return ax

    def add_inner_title(self, ax, title, loc=2, size=None, **kwargs):
        from matplotlib.offsetbox import AnchoredText
        from matplotlib.patheffects import withStroke
        if size is None:
            size = dict(size=plt.rcParams['legend.fontsize'])
        at = AnchoredText(title, loc=loc, prop=size,
                          pad=0., borderpad=0.5,
                          frameon=False, **kwargs)
        ax.add_artist(at)
        at.txt._text.set_path_effects([withStroke(foreground="w", linewidth=3)])
        return at

    def mds_plot(self, mean, fonts='freesansbold.ttf', ax=None, ndim=2, **kwargs):
        """Plots Multidimensional scaling results"""
        # plot each point with a name
        #dims = results.ndim
        #try:
            #if results.shape[1] == 1:
                #dims = 1
        #except:
            #pass
        #import pdb; pdb.set_trace()
        res = stats.mds(mean, ndim=ndim)

        if ndim == 1:
            ax = self.bar_plot(res, ax=ax)
            #df = pandas.DataFrame(results, index=labels, columns=['data'])
            #df = df.sort(columns='data')
            #self._plot(df)
        elif ndim == 2:
            #x, y = results.T
            self.scatter_plot(res, ax=ax)
            ##for c, coord in enumerate(results):
                ##ax.plot(coord[0], coord[1], 'o', color=mpl.cm.Paired(.5))
                ##ax.text(coord[0], coord[1], labels[c], fontproperties=fonts[c])
        else:
            print 'Cannot plot more than 2 dims'
        return ax, res

    def _violin_plot(self, data, pos, rlabels, ax=None, bp=False, cut=None, **kwargs):
        """
        Make a violin plot of each dataset in the `data` sequence.

        Based on `code by Teemu Ikonen
        <http://matplotlib.1069221.n5.nabble.com/Violin-and-bean-plots-tt27791.html>`_
        which was based on `code by Flavio Codeco Coelho
        <http://pyinsci.blogspot.com/2009/09/violin-plot-with-matplotlib.html>`)
        """
        def draw_density(p, low, high, k1, k2, ncols=2):
            m = low #lower bound of violin
            M = high #upper bound of violin
            x = np.linspace(m, M, 100) # support for violin
            if k1 is not None:
                v1 = k1.evaluate(x) # violin profile (density curve)
                v1 = w*v1/v1.max() # scaling the violin to the available space
            if k2 is not None:
                v2 = k2.evaluate(x) # violin profile (density curve)
                v2 = w*v2/v2.max() # scaling the violin to the available space

            if ncols == 2:
                if k1 is not None:
                    ax.fill_betweenx(x, -v1 + p, p, facecolor='black', edgecolor='black')
                if k2 is not None:
                    ax.fill_betweenx(x, p, p + v2, facecolor='grey', edgecolor='gray')
            else:
                #if k1 is not None and k2 is not None:
                ax.fill_betweenx(x, -v1 + p, p + v2, facecolor='black',
                                     edgecolor='black')

        if pos is None:
            pos = [0,1]
        dist = np.max(pos)-np.min(pos)
        w = min(0.15*max(dist,1.0),0.5) * .5

        #for major_xs in range(data.shape[1]):
        for rno, rlabel in enumerate(rlabels):
            p = pos[rno]
            #d1 = data.iloc[rlabel].icol(0)
            ##s1 = sel.iloc[rlabel].icol(0)
            ##if s1:
            #d1 = d1[pandas.notnull(d1)]
            #k1 = scipy.stats.gaussian_kde(d1)  # calculates kernel density
            #else:
                #k1 = None
            #import pdb; pdb.set_trace()
            d = data.loc[:,rlabel]

            if d.ndim == 1:
                d1 = d
                d1 = d1[pandas.notnull(d1)]
                d2 = d1
                if len(d1) > 1:
                    k1 = scipy.stats.gaussian_kde(d1)  # calculates kernel density
                else:
                    k1 = None
                k2 = k1
            elif d.ndim == 2:
                d1 = d.iloc[:,0]
                d1 = d1[pandas.notnull(d1)]
                if len(d1) > 1:
                    k1 = scipy.stats.gaussian_kde(d1)  # calculates kernel density
                else:
                    k1 = None

                d2 = d.iloc[:,1]
                d2 = d2[pandas.notnull(d2)]
                #s2 = sel.ix[rlabel].icol(1)
                if len(d2) > 1:
                    k2 = scipy.stats.gaussian_kde(d2)  # calculates kernel density
                else:
                    k2 = None
            else:
                raise Exception('beanplots are only available for one or two '
                    'columns, but we detected %d columns' % data.ndim)

            if k1 is not None and k2 is not None:
                cutoff = .001
                if cut is None:
                    #if s1 and s2:
                    high = max(d1.max(),d2.max())
                    low = min(d1.min(),d2.min())
                    #elif s1:
                        #high = d1.max()
                        #low = d1.min()
                    #elif s2:
                        #high = d2.max()
                        #low = d2.min()
                    stepsize = (high - low) / 100
                    area_low1 = 1  # max cdf value
                    area_low2 = 1  # max cdf value
                    while area_low1 > cutoff or area_low2 > cutoff:
                        area_low1 = k1.integrate_box_1d(-np.inf, low)
                        area_low2 = k2.integrate_box_1d(-np.inf, low)
                        low -= stepsize
                    area_high1 = 1  # max cdf value
                    area_high2 = 1  # max cdf value
                    while area_high1 > cutoff or area_high2 > cutoff:
                        area_high1 = k1.integrate_box_1d(high, np.inf)
                        area_high2 = k2.integrate_box_1d(high, np.inf)
                        high += stepsize
                else:
                    low, high = cut

                draw_density(p, low, high, k1, k2, ncols=d.ndim)

        # a work-around for generating a legend for the PolyCollection
        # from http://matplotlib.org/users/legend_guide.html#using-proxy-artist
        left = Rectangle((0, 0), 1, 1, fc="black", ec='black')
        right = Rectangle((0, 0), 1, 1, fc="gray", ec='gray')

        ax.legend((left, right), data.columns.tolist())
        #ax.set_xlim(pos[0]-3*w, pos[-1]+3*w)
        #if bp:
            #ax.boxplot(data,notch=1,positions=pos,vert=1)
        return ax

    def _stripchart(self, data, pos, rlabels, ax=None,
        mean=False, median=False, width=None, discrete=True, bins=30):
        """Plot samples given in `data` as horizontal lines.

        :Kwargs:
            mean: plot mean of each dataset as a thicker line if True
            median: plot median of each dataset as a dot if True.
            width: Horizontal width of a single dataset plot.
        """
        def draw_lines(p, d, maxcount, hist, bin_edges, sides=None):
            d = d[pandas.notnull(d)]
            if discrete:
                bin_edges = bin_edges[:-1]  # upper edges not needed
                hw = hist * w / (2.*maxcount)
            else:
                bin_edges = d
                hw = w / 2.
            if mean or len(d) < 2:  # draws a longer black line
                ax.hlines(np.mean(d), sides[0]*2*w + p, sides[1]*2*w + p,
                    lw=2, color='black')
            #if sel:
            #import pdb; pdb.set_trace()
            if len(d) > 1:
                ax.hlines(bin_edges, sides[0]*hw + p, sides[1]*hw + p, color='white')
            if median and len(d) > 1:  # puts a white dot
                ax.plot(p, np.median(d), 'x', color='white', mew=2)

        if width:
            w = width
        else:
            dist = np.max(pos)-np.min(pos)
            w = min(0.15*max(dist,1.0),0.5) * .5

        ## put rows and cols in cols, yerr in rows (original format)
        #data = self._stack_levels(data, 'cols')
        #data = self._unstack_levels(data, 'yerr').T
        #sel = self._stack_levels(sel, 'cols')
        #sel = self._unstack_levels(sel, 'yerr').T
        # apply along cols
        #import pdb; pdb.set_trace()
        rng = (data.min().min(), data.max().max())
        hists = []#data.max()
        for dno, d in data.iteritems():
            #d = data.iloc[:,p]
            d = d[pandas.notnull(d)]
            hist, bin_ed = np.histogram(d, bins=bins, range=rng)
            #import pdb; pdb.set_trace()
            hists.extend(hist.tolist())
            #hists.iloc[p] = hist  # hists is Series
        maxcount = np.max(hists)
        #import pdb; pdb.set_trace()

        for rno, rlab in enumerate(rlabels):
            d = data.loc[:,rlab]
            # awful repetition of hist
            # until I figure out something better
            if d.ndim == 1:
                d = d[pandas.notnull(d)]
                hist, bin_edges = np.histogram(pos[rno], d, bins=bins, range=rng)
                draw_lines(d, maxcount, hist, bin_edges, sides=[-1,1])
            elif d.ndim == 2:
                #import pdb; pdb.set_trace()
                d1 = d.iloc[pandas.notnull(d.iloc[:,0]),0]
                hist, bin_edges = np.histogram(d1, bins=bins, range=rng)
                draw_lines(pos[rno], d1, maxcount, hist, bin_edges, sides=[-1,0])
                d2 = d.iloc[pandas.notnull(d.iloc[:,1]),1]
                hist, bin_edges = np.histogram(d2, bins=bins, range=rng)
                draw_lines(pos[rno], d2, maxcount, hist, bin_edges, sides=[ 0,1])
            else:
                raise Exception('beanplots are only available for one or two '
                    'columns, but we detected %d columns' % d.ndim)


        #hist, bin_edges = np.apply_along_axis(np.histogram, 0, data, bins)
        ## it return arrays of object type, so we got to correct that
        #hist = np.array(hist.tolist())
        #bin_edges = np.array(bin_edges.tolist())
        #maxcount = np.max(hist)

        #for n, rlabel in enumerate(rlabels):
            #p = pos[n]
            #d = data.ix[:, rlabel]
            #s = sel.ix[:, rlabel]

            #if len(d.columns) == 2:
                #draw_lines(d.ix[:,0], s.ix[:,0], maxcount, hist[0],
                    #bin_edges[0], sides=[-1,0])
                #draw_lines(d.ix[:,1], s.ix[:,0], maxcount, hist[1],
                    #bin_edges[1], sides=[ 0,1])
            #else:
                #draw_lines(d.ix[:,0], s.ix[:,0], maxcount, hist[n],
                            #bin_edges[n], sides=[-1,1])

        ax.set_xlim(min(pos)-3*w, max(pos)+3*w)
        ax.set_xticks(pos)
        return ax

    def bean_plot(self, data, ax=None, pos=None, mean=True, median=True, cut=None,
        order=None, discrete=True, **kwargs):
        """Make a bean plot of each dataset in the `data` sequence.

        Reference: `<http://www.jstatsoft.org/v28/c01/paper>`_
        """
        #data_tr, pos, rlabels, sel = self._beanlike_setup(data, ax, order)
        #data_mean = self._stack_levels(data_tr, 'cols')
        #data_mean = self._unstack_levels(data_mean, 'yerr')
        #data_mean = data_mean.mean(1)

        try:  # not guaranteed that columns have names
            len(data.columns.names)
        except:
            rlabels = data.columns
        else:
            inds = [i for i,n in enumerate(agg.columns.names) if n.startswith('rows.')]
            if len(inds) == 0:
                rlabels = agg.columns.levels
            elif len(inds) == 1:
                rlabels = agg.columns.levels[inds[-1]]
            else:
                rlabels = 1
        pos = range(len(rlabels))

        dist = np.max(pos) - np.min(pos)
        w = min(0.15*max(dist,1.0),0.5) * .5
        ax = self._stripchart(data, pos, rlabels, ax=ax, mean=mean, median=median,
            width=0.8*w, discrete=discrete)
        ax = self._violin_plot(data, pos, rlabels, ax=ax, bp=False, cut=cut)
        #ax = self._stripchart(data_tr, pos, rlabels, sel, ax=ax, mean=mean, median=median,
            #width=0.8*w, discrete=discrete)
        #ax = self._violinplot(data_tr, pos, rlabels, sel, ax=ax, bp=False, cut=cut)

        return ax

    def _unstack_levels(self, data, pref):
        try:
            levels = [n for n in data.index.names if n.startswith(pref+'.')]
        except:
            unstacked = data
        else:
            if len(levels) == 0:
                unstacked = pandas.DataFrame(data)
            else:
                try:
                    levs = data.columns.names + levels
                except:
                    levs = levels
                if len(levels) == 1:
                    levels = levels[0]
                unstacked = data.unstack(levels)
                unstacked.columns.names = levs
        return unstacked

    def _stack_levels(self, data, pref):
        try:
            levels = [n for n in data.columns.names if n.startswith(pref+'.')]
        except:
            stacked = data
        else:
            if len(levels) == 1:
                levels = levels[0]
            stacked = data.stack(levels)
        return stacked

    def _beanlike_setup(self, data, ax, order=None):
        if self._stack_levels(data, 'rows').shape[1] > 2:  # more than 2 columns
            new_levs = []
            for n in data.columns.names:
                if n.startswith('cols.'):
                    new_lev = 'rows.' + n.split('cols.')[1]
                else:
                    new_lev = n
                new_levs.append(new_lev)
            data.columns.names = new_levs

            idx = [d + (0,) for d in data.columns]
            data_new = pandas.DataFrame(data, columns=idx)
            data_new.columns.names = data.columns.names + ['cols.fakecol']
            data_new.ix[:] = np.array(data)
            data = data_new


        def mask(data):
            ptp = np.ptp(np.array(data), axis=0)
            sel = np.logical_or(ptp == 0, np.isnan(ptp))
            sel = np.logical_not(sel)
            return sel

        sel = data.apply(mask)
        sel = pandas.DataFrame(sel).T
        sel.index.names = ['yerr.mask']
        sel = self._unstack_levels(sel, 'yerr')
        sel = self._unstack_levels(sel, 'rows')
        sel = self._unstack_levels(sel, 'yerr')
        sel = sel.T  # now rows and yerr are in rows, cols in cols

        data = pandas.DataFrame(data)  # Series will be forced into a DataFrame
        data = self._unstack_levels(data, 'yerr')
        data = self._unstack_levels(data, 'rows')
        rlabels = data.columns
        data = self._unstack_levels(data, 'yerr')
        data = data.T  # now rows and yerr are in rows, cols in cols

        if len(data.index.levels[-1]) <= 1:
            raise Exception('Cannot make a beanplot for a single observation')

        if ax is None:
            ax = self.next()
        #if order is None:
        pos = range(len(rlabels))

        return data, pos, rlabels, sel


if __name__ == '__main__':
    n = 8
    nsampl = 10
    k = n * nsampl
    data = {
        'subplots': ['session1']*k*18 + ['session2']*k*18,
        'cond': [1]*k*9 + [2]*k*9 + [1]*k*9 + [2]*k*9,
        'name': (['one', 'one', 'one']*k + ['two', 'two', 'two']*k +
                ['three', 'three', 'three']*k) * 4,
        'levels': (['small']*k + ['medium']*k + ['large']*k)*12,
        'subjID': ['subj%d' % (i+1) for i in np.repeat(range(n),nsampl)] * 36,
        'RT': range(k)*36,
        'accuracy': np.random.randn(36*k)
        }
    df = pandas.DataFrame(data, columns = ['subplots','cond','name','levels','subjID','RT',
        'accuracy'])
    #df = df.reindex_axis(['subplots','cond','name','levels','subjID','RT',
        #'accuracy'], axis=1)
    agg = stats.aggregate(df, subplots='subplots', rows=['cond', 'name'],
        col='levels', yerr='subjID', values='RT')
    fig = Plot(ncols=2)
    fig.plot(agg, subplots=True, yerr=True)
    fig.show()
