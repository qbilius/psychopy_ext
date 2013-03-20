#!/usr/bin/env python

# Part of the psychopy_ext library
# Copyright 2010-2013 Jonas Kubilius
# The program is distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.

"""A wrapper of matplotlib for producing pretty plots by default"""

import sys

import numpy as np
import scipy.stats
import pandas

import matplotlib.pyplot as plt
import matplotlib as mpl
from mpl_toolkits.axes_grid1 import ImageGrid
from matplotlib.patches import Rectangle


# from https://gist.github.com/huyng/816622
# inspiration from mpltools
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

    def __init__(self, kind='', figsize=None, nrows_ncols=(1, 1)):
        self.subplots(kind=kind, figsize=figsize, nrows_ncols=nrows_ncols)

    def subplots(self, kind='', figsize=None, nrows_ncols=(1, 1),direction="row",
                 axes_pad = 0.05, add_all=True, label_mode="L", share_all=True,
                 cbar_location="right", cbar_mode="single", cbar_size="10%",
                 cbar_pad=0.05, **kwargs):
        if kind == 'matrix':
            self.fig = self.figure(figsize=figsize)
            self.axes = self.ImageGrid(self.fig, 111,
                                  nrows_ncols=nrows_ncols,
                                  direction=direction,
                                  axes_pad=axes_pad,
                                  add_all=add_all,
                                  label_mode=label_mode,
                                  share_all=share_all,
                                  cbar_location=cbar_location,
                                  cbar_mode=cbar_mode,
                                  cbar_size=cbar_size,
                                  cbar_pad=cbar_pad,
                                  )
        else:
            self.fig, self.axes = plt.subplots(
                nrows=nrows_ncols[0],
                ncols=nrows_ncols[1],
                sharex=True,
                sharey=False,
                squeeze=True,
                figsize=figsize,
                **kwargs
                )
            try:
                self.axes[0]
            except:
                self.axes = [self.axes]
        self.subplotno = 0
        self.nrows_ncols = nrows_ncols
        return (self.fig, self.axes)

    def __getattr__(self, name):
        """Pass on a matplotlib function that we haven't modified
        """
        def method(*args, **kwargs):
            getattr(plt, name)(*args, **kwargs)
        return method

    def scatter(self, x, y, labels=None, title='', *args, **kwargs):
        try:
            row = self.subplotno / self.axes[0][0].numCols
            col = self.subplotno % self.axes[0][0].numCols
            ax = self.axes[row][col]
        except:
            ax = self.axes[self.subplotno]

        ax.scatter(x, y, marker='o', color=mpl.cm.Paired(.5))
        for c, (pointx, pointy) in enumerate(zip(x,y)):
            ax.text(pointx, pointy, labels[c])
        ax.set_title(title)
        self.subplotno += 1
        return ax

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

    def nan_outliers(self, df, values=None, group=None):
        """
        Remove outliers 3 standard deviations away from the mean

        Args:
            df (pandas.DataFrame): A DataFrame with your data
        Kwargs:
            values (str): Name of the column that needs to have outliers removed
            group (str): Name of the column to group reponses by. Typically, this
            is the column with subject IDs, so that you remove outliers for each
            participant separately (based on their mean and std)
        Returns:
            df (pandas.DataFrame): A DataFrame without outliers
        """
        zscore = lambda x: (x - x.mean()) / x.std()
        tdf = df.groupby(group)[values].transform(zscore)
        df[values] = np.select([tdf<-3, tdf>3],[np.nan, np.nan],
                               default=df[values])
        return df

    def aggregate(self, df, rows=None, cols=None, values=None,
        value_filter=None, yerr=None, func='mean'):
        """
        Aggregates data over specified dimensions

        :Args:
            df (pandas.DataFrame): A DataFrame with your data

        :Kwargs:
            - rows (str or list of str): Name(s) of column(s) that will be
                aggregated and plotted on the x-axis
            - columns (str or list of str): ame(s) of column(s) that will be
                aggregated and plotted in the legend
            - values (str): Name of the column that are aggregated
            - yerr (str): Name of the column to group reponses by. Typically,
                this is the column with subject IDs to remove outliers for each
                participant separately (based on their mean and std)

        :Returns:
            df (pandas.DataFrame): A DataFrame without outliers
        """
        if type(rows) != list: rows = [rows]
        #if yerr is None:
            #yerr = []
        if yerr is not None:
            if type(yerr) in [list, tuple]:
                if len(yerr) > 1:
                    raise ValueError('Only one field can be used for calculating'
                        'error bars.')
                else:
                    yerr = yerr[0]

        if cols is None:
            if yerr is None:
                allconds = rows
            else:
                allconds = rows + [yerr]
        else:
            if type(cols) != list: cols = [cols]
            if yerr is None:
                allconds = rows + cols
            else:
                allconds = rows + cols + [yerr]

        if yerr is None:
            panel = self._agg_df(df, rows=rows, cols=cols, values=values,
                value_filter=value_filter, func=func)
        else:

            if df[values].dtype in [str, object]:  # calculate accuracy
                size = df.groupby(allconds)[values].size()
                if value_filter is not None:
                    dff = df[df[values] == value_filter]
                else:
                    raise Exception('value_filter must be defined when aggregating '
                        'over str or object types')
                size_filter = dff.groupby(allconds)[values].size()
                agg = size_filter / size.astype(float)
            else:
                if func == 'mean':
                    agg = df.groupby(allconds)[values].mean()
                elif func == 'median':
                    agg = df.groupby(allconds)[values].median()

            agg = agg.unstack(yerr)
            columns = agg.columns
            panel = {}

            for col in columns:
                if cols is None:
                    #if yerr is None:
                        #df_col = pandas.DataFrame({'data': agg})
                    #else:
                    df_col = pandas.DataFrame({'data': agg[col]})
                    panel[col] = df_col
                else:
                    df_col = agg[col].reset_index()
                    #import pdb; pdb.set_trace()
                    panel[col] = pandas.pivot_table(df_col, rows=rows, cols=cols,
                                                values=col)

        return pandas.Panel(panel)

    def _agg_df(self, df, rows=None, cols=None, values=None,
                value_filter=None, func='mean'):
        if df[values].dtype in [str, object]:  # calculate accuracy
            size = pandas.pivot_table(df, rows=rows, cols=cols, values=values,
                aggfunc=np.size)
                #df.groupby(allconds)[values].size()
            if value_filter is not None:
                dff = df[df[values] == value_filter]
            else:
                raise Exception('value_filter must be defined when aggregating '
                    'over str or object types')
            size_filter = pandas.pivot_table(dff, rows=rows, cols=cols, values=values,
                aggfunc=np.size)
            agg = size_filter / size.astype(float)
        else:
            if func == 'mean':
                aggfunc = np.mean
            elif func == 'median':
                aggfunc = np.median
            agg = pandas.pivot_table(df, rows=rows, cols=cols, values=values,
                aggfunc=aggfunc)
        return {'column': agg}

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
        self.plot(agg, ax=ax,
                   title=title, kind=kind, xtickson=xtickson, ytickson=ytickson,
                   no_yerr=no_yerr, numb=numb, autoscale=autoscale, **kwargs)

    def plot(self, agg, ax=None,
                   title='', kind='bar', xtickson=True, ytickson=True,
                   no_yerr=False, numb=False, autoscale=True, order=None,
                   **kwargs):

        if ax is None:
            try:
                row = self.subplotno / self.axes[0][0].numCols
                col = self.subplotno % self.axes[0][0].numCols
                ax = self.axes[row][col]
            except:
                ax = self.axes[self.subplotno]

        if type(agg) == pandas.Panel:
            mean, p_yerr = self.errorbars(agg)
            p_yerr = np.array(p_yerr)
        else:
            mean = agg
            p_yerr = np.zeros((len(agg), 1))

        colors = self.get_colors(len(mean.columns))

        if len(agg.items) == 1 and kind=='bean':
            kind = 'bar'
            print 'WARNING: Beanplot not available for a single measurement'

        if kind == 'bar':
            n = len(mean.columns)
            idx = np.arange(len(mean))
            width = .75 / n
            rects = []
            for i, (label, column) in enumerate(mean.iteritems()):
                rect = ax.bar(idx+i*width, column, width, label=label,
                    yerr = p_yerr[:,i], color = colors[i],
                    ecolor='black')
                rects.append(rect)
            ax.set_xticks(idx + width*n/2)
            #FIX: mean.index returns float even if it is int because dtype=object
            ax.set_xticklabels(mean.index.tolist())
            l = ax.legend(rects, mean.index.tolist())

        elif kind == 'line':
            x = range(len(mean))
            lines = []
            for i, (label, column) in enumerate(mean.iteritems()):
                line = ax.plot(x, column, label=label)
                lines.append(line)
                ax.errorbar(x, column, yerr=p_yerr[:, i], fmt=None,
                    ecolor='black')
            ticks = ax.get_xticks().astype(int)
            if ticks[-1] >= len(mean.index):
                labels = mean.index[ticks[:-1]]
            else:
                labels = mean.index[ticks]
            ax.set_xticklabels(labels)
            l = ax.legend()
            #loc='center left', bbox_to_anchor=(1.3, 0.5)
            #loc='upper right', frameon=False

        elif kind == 'bean':
            if len(mean.columns) <= 2:
                #kk = pandas.Series([float(agg[s]['morph']) for s in agg.items])
                #import pdb; pdb.set_trace()
                ax, l = self.beanplot(agg, ax=ax, order=order, **kwargs)#, pos=range(len(mean.index)))
            else:
                raise Exception('Beanplot is not available for more than two '
                                'classes.')

        else:
            sys.exit('%s plot not recognized' %kind)


        # TODO: xticklabel rotation business is too messy
        if 'xticklabels' in kwargs:
            ax.set_xticklabels(kwargs['xticklabels'], rotation=0)
        if not xtickson:
            ax.set_xticklabels(['']*len(ax.get_xticklabels()))

        labels = ax.get_xticklabels()
        max_len = max([len(label.get_text()) for label in labels])
        for label in labels:
            if max_len > 10:
                label.set_rotation(90)
            else:
                label.set_rotation(0)
            #label.set_size('x-large')
        #ax.set_xticklabels(labels, rotation=0, size='x-large')

        if not ytickson:
            ax.set_yticklabels(['']*len(ax.get_yticklabels()))
        ax.set_xlabel('')

        # set y-axis limits
        if 'ylim' in kwargs:
            ax.set_ylim(kwargs['ylim'])
        elif autoscale:
            mean_array = np.asarray(mean)
            r = np.max(mean_array) - np.min(mean_array)
            ebars = np.where(np.isnan(p_yerr), r/3., p_yerr)
            if kind == 'bar':
                ymin = np.min(np.asarray(mean) - ebars)
                if ymin > 0:
                    ymin = 0
                else:
                    ymin = np.min(np.asarray(mean) - 3*ebars)
            else:
                ymin = np.min(np.asarray(mean) - 3*ebars)
            if kind == 'bar':
                ymax = np.max(np.asarray(mean) + ebars)
                if ymax < 0:
                    ymax = 0
                else:
                    ymax = np.max(np.asarray(mean) + 3*ebars)
            else:
                ymax = np.max(np.asarray(mean) + 3*ebars)
            ax.set_ylim([ymin, ymax])
        #if 'xlabel' in kwargs:
            #ax.set_xlabel(kwargs['xlabel'], size='x-large')
        if 'ylabel' in kwargs:
            ax.set_ylabel(kwargs['ylabel'], size='x-large')
        # else:
        #     import pdb; pdb.set_trace()
        #     ax.set_ylabel(grouped.name)

        ax.set_title(title, size='x-large')

        l.legendPatch.set_alpha(0.5)
        if 'legend_visible' in kwargs:
            l.set_visible(kwargs['legend_visible'])
        elif len(l.texts) == 1:  # showing a single legend entry is useless
            l.set_visible(False)
        else:
            if self.subplotno == 0:
                l.set_visible(True)
            else:
                l.set_visible(False)
        if numb == True:
            self.add_inner_title(ax, title='%s' % self.subplotno, loc=2)
        self.subplotno += 1
        return ax

    def hide_plots(self, nums):
        # check if nums is iterable
        try:
            num_iter = iter(nums)
        except TypeError:
            num_iter = [nums]

        for num in num_iter:
            try:
                row = num / self.axes[0][0].numCols
                col = num % self.axes[0][0].numCols
                ax = self.axes[row][col]
            except:
                ax = self.axes[num]
            ax.axis('off')

    def matrix_plot(self, matrix, ax=None, title='', **kwargs):
        """
        Plots a matrix.
        """
        if ax is None:
            ax = plt.subplot(111)
        import matplotlib.colors
        norm = matplotlib.colors.normalize(vmax=1, vmin=0)
        mean, sem = self.errorbars(matrix)
        #matrix = pandas.pivot_table(mean.reset_index(), rows=)
        im = ax.imshow(mean, norm=norm, interpolation='none', **kwargs)
        # ax.set_title(title)

        ax.cax.colorbar(im)#, ax=ax, use_gridspec=True)
        # ax.cax.toggle_label(True)

        t = self.add_inner_title(ax, title, loc=2)
        t.patch.set_ec("none")
        t.patch.set_alpha(0.8)
        xnames = ['|'.join(map(str,label)) for label in matrix.minor_axis]
        ax.set_xticks(range(len(xnames)))
        ax.set_xticklabels(xnames)
        # rotate long labels
        if max([len(n) for n in xnames]) > 20:
            ax.axis['bottom'].major_ticklabels.set_rotation(90)
        ynames = ['|'.join(map(str,label)) for label in matrix.major_axis]
        ax.set_yticks(range(len(ynames)))
        ax.set_yticklabels(ynames)
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

    def pivot_plot_old(self, pivot, persubj=None, kind='bar', ncols=2, ax=None, title='',
                xlabels=True, yerr_type='sem', legend=True, **kwargs):
        """
        Generates a bar plot from pandas pivot table.

        Not very stable due to a hack of ncols.
        """
        if ax is None:
            ax = plt.subplot(111)

        if persubj is not None:
            # calculate errorbars
            p_yerr = self.errorbars(persubj, ncols, yerr_type=yerr_type)
            # generate zero errorbars for the pandas plot
            p_yerr_zeros = np.zeros((ncols, p_yerr.shape[0]))

            # plot data with zero errorbars
            pplot = pivot.plot(kind=kind, ax=ax, legend=legend,
                            **{'yerr': p_yerr_zeros,
                            'color': self.sample_paired(ncols)})
            # put appropriate errorbars for each bar
            for i, col in enumerate(pivot.columns):
                x = pplot.get_lines()[i * ncols].get_xdata()
                y = pivot[col]
                pplot.errorbar(x, y, yerr=p_yerr[:, i], fmt=None,
                    ecolor='b')
        else:
            # plot data without errorbars
            pplot = pivot.plot(kind=kind, ax=ax, legend=legend)#,
                            #**{'color': sample_paired(ncols)})
        ax.legend(loc=8)
        ax.set_title(title)
        if not xlabels:
            ax.set_xticklabels([''] * pivot.shape[0])
        plt.tight_layout()

        return pplot

    def errorbars(self, panel, yerr_type='sem'):
        # Set up error bar information
        if yerr_type == 'sem':
            mean = panel.mean(0)  # mean across items
            # std already has ddof=1
            sem = panel.std(0) / np.sqrt(len(panel.items))
        elif yerr_type == 'binomial':
            pass
            # alpha = .05
            # z = stats.norm.ppf(1-alpha/2.)
            # count = np.mean(persubj, axis=1, ddof=1)
            # p_yerr = z*np.sqrt(mean*(1-mean)/persubj.shape[1])

        return mean, sem

    def stats_test(self, agg, test='ttest'):
        d = agg.shape[0]

        if test == 'ttest':
            # 2-tail T-Test
            ttest = (np.zeros((agg.shape[1]*(agg.shape[1]-1)/2, agg.shape[2])),
                     np.zeros((agg.shape[1]*(agg.shape[1]-1)/2, agg.shape[2])))
            ii = 0
            for c1 in range(agg.shape[1]):
                for c2 in range(c1+1,agg.shape[1]):
                    thisTtest = stats.ttest_rel(agg[:,c1,:], agg[:,c2,:], axis = 0)
                    ttest[0][ii,:] = thisTtest[0]
                    ttest[1][ii,:] = thisTtest[1]
                    ii += 1
            ttestPrint(title = '**** 2-tail T-Test of related samples ****',
                values = ttest, plotOpt = plotOpt,
                type = 2)

        elif test == 'ttest_1samp':
            # One-sample t-test
            m = .5
            oneSample = stats.ttest_1samp(agg, m, axis = 0)
            ttestPrint(title = '**** One-sample t-test: difference from %.2f ****' %m,
                values = oneSample, plotOpt = plotOpt, type = 1)

        elif test == 'binomial':
            # Binomial test
            binom = np.apply_along_axis(stats.binom_test,0,agg)
            print binom
            return binom

    @classmethod
    def oneway_anova(cls, data):
        """
        Calculates one-way ANOVA on a pandas.DataFrame.

        Args:
            data (pandas.DataFrame): rows contain groups (e.g., different
            conditions), while columns have samples (e.g., participants)

        Returns:
            F (float): F-value
            p (float): p-value
            k-1 (int): Between Group degrees of freedom
            N-k (int): Within Group degrees of freedom

        """
        F, p = scipy.stats.f_oneway(*[d[1] for d in data.iterrows()])
        k = len(data)  # number of conditions
        N = k*len(data.columns)  # conditions times participants
        return F, p, k-1, N-k

    @classmethod
    def p_corr(cls, df1, df2):
        """
        Computes Pearson correlation and its significance (using a t
        distribution) on a pandas.DataFrame.

        Ignores null values when computing significance. Based on
        http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Testing_using_Student.27s_t-distribution

        Args:
            df1 (pandas.DataFrame): one dataset
            df2 (pandas.DataFrame): another dataset

        Returns:
            corr (float): correlation between the two datasets
            t (float): an associated t-value
            p (float): one-tailed p-value that the two datasets differ
        """
        corr = df1.corr(df2)
        N = np.sum(df1.notnull())
        t = corr*np.sqrt((N-2)/(1-corr**2))
        p = 1-scipy.stats.t.cdf(abs(t),N-2)  # one-tailed
        return corr, t, p

    @classmethod
    def reliability(cls, panel, level=1, niter=100):
        subjIDs = panel.items.tolist()
        corrs = []
        for n in range(niter):
            np.random.shuffle(subjIDs)
            split1inds = subjIDs[:len(subjIDs)/2]
            split2inds = subjIDs[len(subjIDs)/2:]
            split1 = pandas.concat([panel[item] for item in panel.items
                                    if item in split1inds])
            split2 = pandas.concat([panel[item] for item in panel.items
                                    if item in split2inds])
            split1 = split1.mean(0)
            split2 = split2.mean(0)
            for lev in range(level):
                split1 = split1.stack()
                split2 = split2.stack()
            corrs.append(split1.corr(split2))

        N = np.sum(split1.notnull())
        corr = np.mean(corrs)
        t = corr*np.sqrt((N-2)/(1-corr**2))
        p = 1-scipy.stats.t.cdf(abs(t),N-2)  # one-tailed
        p = 2*p/(1.+p) # Spearman-Brown prediction for twice
                       # the amount of data
                       # we need this because we use only half
                       # of the data here
        return corr, t, p

    def ttestPrint(self, title = '****', values = None, xticklabels = None, legend = None, bon = None):

        d = 8
        # check if there are any negative t values (for formatting purposes)
        if np.any([np.any(val < 0) for val in values]): neg = True
        else: neg = False

        print '\n' + title
        for xi, xticklabel in enumerate(xticklabels):
            print xticklabel

            maxleg = max([len(leg) for leg in legend])
#            if type == 1: legendnames = ['%*s' %(maxleg,p) for p in plotOpt['subplot']['legend.names']]
#            elif type == 2:
            pairs = q.combinations(legend,2)
            legendnames = ['%*s' %(maxleg,p[0]) + ' vs ' + '%*s' %(maxleg,p[1]) for p in pairs]
            #print legendnames
            for yi, legendname in enumerate(legendnames):
                if values[0].ndim == 1:
                    t = values[0][xi]
                    p = values[1][xi]
                else:
                    t = values[0][yi,xi]
                    p = values[1][yi,xi]
                if p < .001/bon: star = '***'
                elif p < .01/bon: star = '**'
                elif p < .05/bon: star = '*'
                else: star = ''

                if neg and t > 0:
                    outputStr = '    %(s)s: t(%(d)d) =  %(t).3f, p = %(p).3f %(star)s'
                else:
                    outputStr = '    %(s)s: t(%(d)d) = %(t).3f, p = %(p).3f %(star)s'

                print outputStr \
                    %{'s': legendname, 'd':(d-1), 't': t,
                    'p': p, 'star': star}


    def plotrc(self, theme = 'default'):
        """
        Reads in the rc file
        Returns: plotOpt
        """

        if theme == None or theme == 'default': themeFile = 'default'
        else: themeFile = theme

        rcFile = open(os.path.dirname(__file__) + '/' + themeFile + '.py')

        plotOpt = {}
        for line in rcFile.readlines():
            line = line.strip(' \n\t')
            line = line.split('#')[0] # getting rid of comments
            if line != '': # skipping blank and commented lines

                linesplit = line.split(':',1)
                (key, value) = linesplit
                key = key.strip()
                value = value.strip()

                # try recognizing numbers, lists etc.
                try: value = eval(value)
                except: pass

                linesplit = key.rsplit('.',1)
                if len(linesplit) == 2:
                    (plotOptKey, optKey) = linesplit
                    if not plotOptKey in plotOpt: plotOpt[plotOptKey] = {}
                    plotOpt[plotOptKey][optKey] = value
                elif key in ['cmap', 'simpleCmap']: plotOpt[key] = value
                else: print 'Option %s not recognized and will be ignored' %line

        rcFile.close
        return plotOpt


    def prettyPlot(self,
        x,
        y = None,
        yerr = None,
        plotType = 'errorbar',
        fig = None,
        subplotno = 111,
        max_y = None,
        theme = 'default',
        userOpt = None
        ):

        """
        Automatically makes beautiful plots and with error bars.

        **Parameters**

            x: numpy.ndarray
                Either x or y values
            y: **None** or numpy.ndarray
                + if **None**, then x value is assumed to be y, while x is generated automatically
                + else it is simpy a y value
            yerr: **None** a 2D numpy.ndarray or list
                Specifies y error bars. If **None**, then no error bars are plotted
            plotType: 'errorbar' or 'bar'
                + 'errorbar' plots lines with error bars
                + 'bar' plots bars with error bars
            subplotno: int
                Use 111 for a single plot; otherwise specify subplot position
            max_y: **None** or int
                Use to indicate the maximum allowed ytick value.
                Useful when plotting accuracy since yticks should not go above 1
            theme: str
                Specifies the file name where all the custom plotting parameters are stores
            userOpt: dict
                Any plotting options that should override default options

        """

        if y == None: # meaning: x not specified
            y = x
            if y.ndim == 1:
                x = np.arange(y.shape[0])+1
                numCat = 1
            else:
                x = np.arange(y.shape[1])+1
                numCat = y.shape[0]
            # this is because if y.ndim == 1, y.shape = (numb,) instead of (1,num)

        numXTicks = x.shape[0]

        plotOpt = plotrc(theme)

        if not 'cmap' in plotOpt: plotOpt['cmap'] = 'Paired'
        if not 'hatch' in plotOpt: plotOpt['hatch'] = ['']*numCat

        if not 'subplot' in plotOpt: plotOpt['subplot'] = {}
        if not 'xticks' in plotOpt['subplot']:
            if numXTicks > 1:
                plotOpt['subplot']['xticks'] = x
            else:
                plotOpt['subplot']['xticklabels'] = ['']
        if not 'legend' in plotOpt: plotOpt['legend'] = ['']

        # x limits
        xMin = x.min()
        xMax = x.max()
        if xMax == xMin: plotOpt['subplot']['xlim'] = (-.5,2.5)
        elif plotType == 'errorbar':
            plotOpt['subplot']['xlim'] = (xMin - (xMax - xMin)/(2.*(len(x)-1)),
                xMax + (xMax - xMin)/(2.*(len(x)-1)))
        elif plotType == 'bar':
            plotOpt['subplot']['xlim'] = (xMin - (xMax - xMin)/(1.5*(len(x)-1)),
                xMax + (xMax - xMin)/(1.5*(len(x)-1)))

        # y limits
        if yerr == None or not np.isfinite(yerr.max()):
            yMin = y.min()
            yMax = y.max()
        else:
            yMin = (y - yerr).min()
            yMax = (y + yerr).max()
        if yMax == yMin: plotOpt['subplot']['ylim'] = (0, yMax + yMax/4.)
        else: plotOpt['subplot']['ylim'] = (yMin - (yMax - yMin)/4., yMax + (yMax - yMin)/2.)


        # overwrite plotOpt by userOpt
        if userOpt != None:
            for (key, value) in userOpt.items():
                if type(value)==dict:
                    if not key in plotOpt: plotOpt[key] = {}
                    for (key2, value2) in value.items():
                        plotOpt[key][key2] = value2
                else: plotOpt[key] = value


        # set all values recognized by mpl
        # this is stupid but not all values are recognized by mpl
        # thus, we later set the unrecognized ones manually
        for key, value in plotOpt.items():
            if not key in ['cmap', 'colors', 'hatch', 'legend.names', 'other']:
                for k, v in value.items():
                    fullKey = key + '.' + k
                    if fullKey in mpl.rcParams.keys():
                        try: mpl.rcParams[fullKey] = v
                        except: mpl.rcParams[fullKey] = str(v) # one more stupidity in mpl
                            # for ytick, a string has to be passed, even if it is a number
        plotOpt['colors'] = getColors(numCat, plotOpt, userOpt)


        # make a new figure
        if fig == None:
            fig = plt.figure()
            fig.canvas.set_window_title('Results')
        ax = fig.add_subplot(subplotno)

        output = []

        # generate plots
        if plotType == 'errorbar':

            if yerr != None:
                for i in range(numCat):
                    output.append(ax.errorbar(
                        x,
                        y[i,:],
                        yerr = yerr[i,:],
                        color = plotOpt['colors'][i],
                        markerfacecolor = plotOpt['colors'][i],
                        **plotOpt[plotType]
                        )
                    )
            else:
                if numCat == 1:
                    output.append(ax.plot(
                        x,
                        y[:],
                        color = plotOpt['colors'][0],
                        markerfacecolor = plotOpt['colors'][0])
                        )
                else:
                    for i in range(numCat):
                        output.append(ax.plot(
                            x,
                            y[i,:],
                            color = plotOpt['colors'][i],
                            markerfacecolor = plotOpt['colors'][i]
                            )
                        )

        elif plotType == 'bar':

            barWidth = (1-.2)/numCat       # the width of the bars
            middle = x - numCat*barWidth/2.
            if numCat == 1:
                output.append(ax.bar(
                    middle,
                    y[:],
                    barWidth,
                    facecolor = plotOpt['colors'][0],
                    yerr = None,#yerr[:],
                    hatch = plotOpt['hatch'][0],
                    **plotOpt[plotType]
                    )
                )
            else:
                for i in range(numCat):
                    output.append(ax.bar(
                        middle+i*barWidth,
                        y[i,:],
                        barWidth,
                        facecolor = plotOpt['colors'][i],
                        yerr = yerr[i,:],
                        hatch = plotOpt['hatch'][i],
                        **plotOpt[plotType]
                        )
                    )


        # set up legend
        if plotOpt['subplot'].get('legend.names'):
            if len(plotOpt['subplot']['legend.names']) > 1:
                outLeg = [i[0] for i in output]
                leg = ax.legend(outLeg, plotOpt['subplot']['legend.names'])
                leg.draw_frame(False)
            del plotOpt['subplot']['legend.names']

        # draw other things
        if 'other' in plotOpt['subplot']:
            for item in q.listify(plotOpt['subplot']['other']): eval(item)
            del plotOpt['subplot']['other']

        # set the remaining subplot options
        ax.set(**plotOpt['subplot'])

        if not 'yticks' in plotOpt['subplot'] and not 'yticklabels' in plotOpt['subplot']:
            import decimal
            yticks = ax.get_yticks()
            if max_y != None: yticks = yticks[yticks <= max_y]
            else: yticks = yticks[yticks <= yticks[-2]]
            # now some fancy way to make pretty labels
            kk = min([decimal.Decimal(str(t)).adjusted() for t in yticks])
            lens2 = np.array([t for t in yticks if len(('%g' %(t/10**kk)).split('.')) == 1])
            spacing = len(lens2)/6 + 1
            if max_y != None: use = lens2[np.arange(len(lens2)-1,-1,-spacing)]
            else: use = lens2[np.arange(0,len(lens2),spacing)]

            ax.set_yticks(use)

        return fig

    def mds(self, d, ndim=2):
        """
        Metric Unweighted Classical Multidimensional Scaling

        Based on Forrest W. Young's notes on Torgerson's (1952) algorithm as
        presented in http://forrest.psych.unc.edu/teaching/p230/Torgerson.pdf:
        Step 0: Make data matrix symmetric with zeros on the diagonal
        Step 1: Double center the data matrix (d) to obtain B by removing row
        and column means and adding the grand mean of the squared data
        Step 2: Solve B = U * L * U.T for U and L
        Step 3: Calculate X = U * L**(-.5)

        **Parameters**
            d: numpy.ndarray
                A symmetric dissimilarity matrix
            ndim: int
                The number of dimensions to project to
        **Returns**
            X[:,:ndim]: numpy.ndarray
                The projection of d into ndim dimensions
        """
        # Step 0
        # make distances symmetric
        d = (d + d.T) / 2
        # diagonal must be 0
        np.fill_diagonal(d, 0)
        # Step 1: Double centering
        # remove row and column mean and add grand mean of d**2
        oo = d**2
        rowmean = np.tile(np.mean(oo, 1), (oo.shape[1],1)).T
        colmean = np.mean(oo, 0)
        B = -.5 * (oo - rowmean - colmean + np.mean(oo))
        # Step2: do singular value decomposition
        # find U (eigenvectors) and L (eigenvalues)
        [U, L, V] = np.linalg.svd(B)  # L is already sorted (desceding)
        # Step 3: X = U*L**(-.5)
        X = U * np.sqrt(L)
        return X[:,:ndim]

    def plot_mds(self, results, labels, fonts='freesansbold.ttf', title='',
        ax = None):
        """Plots Multidimensional scaling results"""
        if ax is None:
            try:
                row = self.subplotno / self.axes[0][0].numCols
                col = self.subplotno % self.axes[0][0].numCols
                ax = self.axes[row][col]
            except:
                ax = self.axes[self.subplotno]
        ax.set_title(title)
        # plot each point with a name
        dims = results.ndim
        try:
            if results.shape[1] == 1:
                dims = 1
        except:
            pass
        if dims == 1:
            df = pandas.DataFrame(results, index=labels, columns=['data'])
            df = df.sort(columns='data')
            self._plot(df)
        elif dims == 2:
            for c, coord in enumerate(results):
                ax.plot(coord[0], coord[1], 'o', color=mpl.cm.Paired(.5))
                ax.text(coord[0], coord[1], labels[c], fontproperties=fonts[c])
        else:
            print 'Cannot plot more than 2 dims'


    # Based on code by Teemu Ikonen <tpikonen@gmail.com>,
    # http://matplotlib.1069221.n5.nabble.com/Violin-and-bean-plots-tt27791.html
    # which was based on code by Flavio Codeco Coelho,
    # http://pyinsci.blogspot.com/2009/09/violin-plot-with-matplotlib.html

    def violinplot(self, data, ax=None, pos=None, bp=False, cut=None):
        """Make a violin plot of each dataset in the `data` sequence.
        """
        def draw_density(p, low, high, k1, k2, ncols=2):
            m = low #lower bound of violin
            M = high #upper bound of violin
            x = np.linspace(m, M, 100) # support for violin
            v1 = k1.evaluate(x) # violin profile (density curve)
            v1 = w*v1/v1.max() # scaling the violin to the available space
            v2 = k2.evaluate(x) # violin profile (density curve)
            v2 = w*v2/v2.max() # scaling the violin to the available space

            if ncols == 2:
                ax.fill_betweenx(x, -v1 + p, p, facecolor='black', edgecolor='black')
                ax.fill_betweenx(x, p, p + v2, facecolor='grey', edgecolor='gray')
            else:
                ax.fill_betweenx(x, -v1 + p, p + v2, facecolor='black', edgecolor='black')


        if pos is None:
            pos = [0,1]
        dist = np.max(pos)-np.min(pos)
        w = min(0.15*max(dist,1.0),0.5) * .5

        for major_xs in range(data.shape[1]):
            p = pos[major_xs]
            d1 = data.ix[:,major_xs,0]
            k1 = scipy.stats.gaussian_kde(d1) #calculates the kernel density
            if data.shape[2] == 1:
                d2 = d1
                k2 = k1
            else:
                d2 = data.ix[:,major_xs,1]
                k2 = scipy.stats.gaussian_kde(d2) #calculates the kernel density
            cutoff = .001
            if cut is None:
                upper = max(d1.max(),d2.max())
                lower = min(d1.min(),d2.min())
                stepsize = (upper - lower) / 100
                area_low1 = 1  # max cdf value
                area_low2 = 1  # max cdf value
                low = min(d1.min(), d2.min())
                while area_low1 > cutoff or area_low2 > cutoff:
                    area_low1 = k1.integrate_box_1d(-np.inf, low)
                    area_low2 = k2.integrate_box_1d(-np.inf, low)
                    low -= stepsize
                    #print area_low, low, '.'
                area_high1 = 1  # max cdf value
                area_high2 = 1  # max cdf value
                high = max(d1.max(), d2.max())
                while area_high1 > cutoff or area_high2 > cutoff:
                    area_high1 = k1.integrate_box_1d(high, np.inf)
                    area_high2 = k2.integrate_box_1d(high, np.inf)
                    high += stepsize
            else:
                low, high = cut

            draw_density(p, low, high, k1, k2, ncols=data.shape[2])


        # a work-around for generating a legend for the PolyCollection
        # from http://matplotlib.org/users/legend_guide.html#using-proxy-artist
        left = Rectangle((0, 0), 1, 1, fc="black", ec='black')
        right = Rectangle((0, 0), 1, 1, fc="gray", ec='gray')
        l = ax.legend((left, right), data.minor_axis.tolist(), frameon=False)
        #import pdb; pdb.set_trace()
        #ax.set_xlim(pos[0]-3*w, pos[-1]+3*w)
        #if bp:
            #ax.boxplot(data,notch=1,positions=pos,vert=1)
        return ax, l


    def stripchart(self, data, ax=None, pos=None, mean=False, median=False,
        width=None, discrete=True, bins=50):
        """Plot samples given in `data` as horizontal lines.

        :Kwargs:
            mean: plot mean of each dataset as a thicker line if True
            median: plot median of each dataset as a dot if True.
            width: Horizontal width of a single dataset plot.
        """
        def get_hist(d, bins):
            hists = []
            bin_edges_all = []
            for rowno, row in d.iterrows():
                hist, bin_edges = np.histogram(row, bins=bins)
                hists.append(hist)
                bin_edges_all.append(bin_edges)
            maxcount = np.max(hists)
            return maxcount, hists, bin_edges_all

        def draw_lines(d, maxcount, hist, bin_edges, sides=None):
            if discrete:
                bin_edges = bin_edges[:-1]  # upper edges not needed
                hw = hist * w / (2.*maxcount)
            else:
                bin_edges = d
                hw = w / 2.

            ax.hlines(bin_edges, sides[0]*hw + p, sides[1]*hw + p, color='white')
            if mean:  # draws a longer black line
                ax.hlines(np.mean(d), sides[0]*2*w + p, sides[1]*2*w + p,
                    lw=2, color='black')
            if median:  # puts a white dot
                ax.plot(p, np.median(d), 'o', color='white', markeredgewidth=0)

        if width:
            w = width
        else:
            if pos is None:
                pos = [0,1]
            dist = np.max(pos)-np.min(pos)
            w = min(0.15*max(dist,1.0),0.5) * .5
        for major_xs in range(data.shape[1]):  # go over 'rows'
            p = pos[major_xs]
            maxcount, hists, bin_edges_all = get_hist(data.ix[:,major_xs], bins)
            if data.shape[2] == 1:
                draw_lines(data.ix[:, major_xs, 0], maxcount, hists[0],
                    bin_edges_all[0], sides=[-1,1])
            else:
                draw_lines(data.ix[:, major_xs, 0], maxcount, hists[0],
                    bin_edges_all[0], sides=[-1,0])
                draw_lines(data.ix[:, major_xs, 1], maxcount, hists[1],
                    bin_edges_all[1], sides=[0, 1])

        ax.set_xlim(min(pos)-3*w, max(pos)+3*w)
        #ax.set_xticks([-1]+pos+[1])
        ax.set_xticks(pos)
        #import pdb; pdb.set_trace()
        #ax.set_xticklabels(['-1']+np.array(data.major_axis).tolist()+['1'])
        ax.set_xticklabels(data.major_axis)

        #return ax


    def beanplot(self, data, ax, pos=None, mean=True, median=True, cut=None,
        order=None, discrete=True, **kwargs):
        """Make a bean plot of each dataset in the `data` sequence.

        Reference: http://www.jstatsoft.org/v28/c01/paper
        """

        #if pos is None:
            #pos = range(len(data.major_axis))
        if order is None:
            pos = range(len(data.major_axis))
        else:
            pos = np.lexsort((np.array(data.major_axis).tolist(),order))

        dist = np.max(pos)-np.min(pos)
        w = min(0.15*max(dist,1.0),0.5) * .5
        self.stripchart(data=data, ax=ax, pos=pos, mean=mean, median=median,
            width=0.8*w, discrete=discrete)
        ax,l = self.violinplot(data=data, ax=ax, pos=pos, bp=False, cut=cut)

        return ax,l