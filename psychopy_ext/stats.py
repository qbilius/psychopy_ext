#!/usr/bin/env python

# Part of the psychopy_ext library
# Copyright 2010-2013 Jonas Kubilius
# The program is distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.

"""
A collection of useful descriptive and basic statistical functions for
psychology research. Some functions are meant to improve `scipy.stats`
functionality and integrate seamlessly with `pandas`.
"""

import numpy as np
import scipy.stats
import pandas

import plot

try:
    import OrderedDict
except:
    from exp import OrderedDict


def aggregate(df, rows=None, cols=None, values=None,
    subplots=None, yerr=None, aggfunc='mean', unstack=False,
    order='natural', add_names=True):
    """
    Aggregates data over specified columns.

    :Args:
        df (pandas.DataFrame)
            A DataFrame with your data

    :Kwargs:
        - rows (str or list of str, default: None)
            Name(s) of column(s) that will be aggregated and plotted on the x-axis
        - cols (str or list of str, default: None)
            Name(s) of column(s) that will be shown in the legend
        - values (str or list of str, default: None)
            Name(s) of the column(s) that is aggregated
        - yerr (str, default: None)
            Name of the column for the y-errorbar calculation. Typically,
            this is the column with participant IDs.
        - aggfunc (str or a 1d function)
            A function to use for aggregation. If a string, it is interpreted
            as a `numpy` function.
        - unstack (bool, default: False)
            If True, returns an unstacked version of aggregated data (i.e.,
            rows in rows and columns in columns)). Useful for printing and
            other non-plotting tasks.
        - order (str, {'natural', 'sorted'}, default: 'natural')
            If order is 'natural', attempts to keep the original order of the
            data (as in the file). Works only sometimes though. If 'sorted',
            then will come out sorted as is by default in pandas.

    :Returns:
        A `pandas.DataFrame` where data has been aggregated in the following
        MultiIndex format:
        - columns:
        
          - level 0: subplots
          - level 1 to n-2: rows
          - level n-1: column
          
        - rows:
            yerr

        This format makes it easy to do further computations such as mean over
        `yerr`: a simple `df.mean()` will do the trick.

    :See also:
        :func:`accuracy`
    """
    df = pandas.DataFrame(df)  # make sure it's a DataFrame
    if isinstance(rows, str) or rows is None:
        rows = [rows]
    if isinstance(cols, str) or cols is None:
        cols = [cols]
    if isinstance(yerr, str) or yerr is None:
        yerr = [yerr]
    if values is None:
        raise Exception('You must provide the name(s) of the column(s) that '
                        'is/are aggregated.')
    allconds = [subplots] + rows + cols + yerr
    allconds = [c for c in allconds if c is not None]

    if isinstance(aggfunc, str):
        try:
            aggfunc = getattr(np, aggfunc)
        except:
            raise
    agg = df.groupby(allconds)[values].aggregate(aggfunc)

    groups = [('subplots', [subplots]), ('rows', rows), ('cols', cols),
              ('yerr', yerr)]
    index_names = agg.index.names
    g = 0
    for group in groups:
        for item in group[1]:
            if item is not None:
                index_names[g] = group[0] + '.' + item
                g += 1
    agg.index.names = index_names

    if yerr[0] is not None:  # if yerr present, yerr is in rows, the rest in cols
        for yr in yerr:
            agg = agg.unstack(level='yerr.'+yr)
        # seems like a pandas bug here for not naming levels properly
        agg.columns.names = ['yerr.'+yr for yr in yerr]
        agg = agg.T
    else:
        agg = pandas.DataFrame(agg).T
        
    if order != 'sorted':
        names = agg.columns.names  # store it; will need it later
        if isinstance(order, dict):
            items = order
        else:
            items = dict([(col, order) for col in agg.columns.names])
        for level, level_ord in items.items():
            if isinstance(level_ord, str):
                if level_ord == 'natural':
                    col = '.'.join(level.split('.')[1:])
                    thisord = df[col].unique()
            else:
                thisord = level_ord
            agg = reorder(agg, level=level, order=thisord)
            agg.columns.names = names  # buggy pandas        
        
    # rows should become rows, and cols should be cols if so desired
    if yerr[0] is None and unstack:
        agg = plot._stack_levels(agg, 'rows.')
            
    if not add_names:
        names = []
        for name in agg.columns.names:
            spl = name.split('.')
            if spl[0] in ['cols','rows','subplots']:
                names.append('.'.join(spl[1:]))
        agg.columns.names = names

    agg.names = values
    return agg

def accuracy(df, values=None, correct='correct', incorrect='incorrect', **kwargs):
    """
    Computes accuracy given correct and incorrect data labels.

    :Args:
        df (pandas.DataFrame)
            Your data

    :Kwargs:
        - values (str or list of str, default: None)
            Name(s) of the column(s) that is aggregated
        - correct (str or a number or list of str or numbers, default: None)
            Labels that are treated as correct responses.
        - incorrect (str or a number or list of str or numbers, default: None)
            Labels that are treated as incorrect responses.
        - kwargs
            Anything else you want to pass to :func:`aggregate`. Note that
            ``aggfunc`` is set to ``np.size`` and you cannot change that.

    :Returns:
        A pandas.DataFrame in the format of :func:`accuracy` where the
        reported values are a fraction correct / (correct+incorrect).

    :See also:
        :func:`accuracy`
    """
    if isinstance(correct, str):
        correct = [correct]
    if isinstance(incorrect, str):
        incorrect = [incorrect]
    corr = df[df[values].isin(correct)]
    if len(corr) == 0:
        raise Exception('There are no %s responses' % correct[0])
    agg_corr = aggregate(corr, aggfunc=np.size, values=values, **kwargs)
    df_all = df[df[values].isin(correct + incorrect)]
    agg_all = aggregate(df_all, aggfunc=np.size, values=values, **kwargs)
    agg = agg_corr.astype(float) / agg_all
    agg.names = values
    return agg

def confidence(agg, kind='sem', nsamples=None, skipna=True):
    if isinstance(agg, pandas.DataFrame):
        mean = agg.mean(skipna=skipna)  # mean across items
        if kind == 'sem':
            p_yerr = agg.std(skipna=skipna) / np.sqrt(len(agg))  # std already has ddof=1

        # compute binomial distribution error bars if there is a single sample
        # only (presumably the number of correct responses, i.e., successes)
        # from http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Normal_approximation_interval
        elif kind == 'binomial':
            alpha = .05
            z = scipy.stats.norm.ppf(1-alpha/2.)
            p_yerr = z * np.sqrt(mean * (1-mean) / nsamples)
    else:
        #mean, p_yerr = self.errorbars(agg, kind='binomial')
        mean = agg
        p_yerr = np.zeros((len(agg), 1))

    return mean, p_yerr

def get_star(p):
    if p < .001:
        star = '***'
    elif p < .01:
        star = '**'
    elif p < .05:
        star = '*'
    else:
        star = ''
    return star

def reorder(agg, order=None, level=0, dim='columns'):
    """
    Reorders rows or columns in a pandas.DataFrame.

    If hierarchical indexing is used, level must be specified.
    It relies on for loops, so it will be slow for large data frames.

    :Args:
        agg (pandas.DataFrame)
            Your (usually aggregated) data
    :Kwargs:
        - order (list-like, default: None)
            Order of entries
        - level (int or str, default: 0)
            Which level needs to be reordered
        - dim (str, {'rows', 'index', 'columns'}, default: 'columns')
            Whether to reorder rows (or index) or columns.
    :Returns:
        Reordered pandas.DataFrame
    """
    if dim in ['index', 'rows']:
        orig_idx = agg.index
    else:
        orig_idx = agg.columns

    if not hasattr(orig_idx, 'levels'):
        multidx = order
    else:
        n = len(orig_idx.levels)
        count = [len(lev) for lev in orig_idx.levels]
        multidx = []
        for i in range(n):
            if orig_idx.names[i] == level or i == level:
                vals = order
            else:
                vals = orig_idx.get_level_values(i).unique()
            rep = np.repeat(vals, np.product(count[i+1:]))
            tile = np.tile(rep, np.product(count[:i]))
            multidx.append(tile)
        multidx = pandas.MultiIndex.from_tuples(list(zip(*multidx)),
                                                names=orig_idx.names)

    if dim in ['index', 'rows']:
        agg_new = agg.reindex(index=multidx)
    else:
        agg_new = agg.reindex(columns=multidx)
    return agg_new

def df_fromdict(data, repeat=1):
    """
    Produces a factorial DataFrame from a dict or list of tuples.

    For example, suppose you want to generate a DataFrame like this::

           a    b
        0  one  0
        1  one  1
        2  two  0
        3  two  1

    This function generates such output simply by providing the following:
    df_fromdict([('a', ['one', 'two']), ('b', [0, 1])])

    :Args:
        data: dict or a list of tuples
            Data used to produce a DataFrame. Keys specify column names, and
            values specify possible (unique) values.
    :Kwargs:
        repeat: int (default: 1)
            How many times everything should be repeated. Useful if you want to
            simulate multiple samples of each condition, for example.
    :Returns:
        pandas.DataFrame with data.items() column names
    """
    data = OrderedDict(data)
    count = map(len, data.values())
    df = {}
    for i, (key, vals) in enumerate(data.items()):
        rep = np.repeat(vals, np.product(count[i+1:]))
        tile = np.tile(rep, np.product(count[:i]))
        df[key] = np.repeat(tile, repeat)
    df = pandas.DataFrame(df, columns=data.keys())
    return df


def _aggregate_panel(df, rows=None, cols=None, values=None,
    value_filter=None, yerr=None, func='mean'):
    """
    Aggregates `pandas.DataFrame` over specified dimensions into `pandas.Panel`.

    Forces output to always be a Panel.

    Shape:
        - items: `yerr`
        - major_xs: `rows`
        - minor_xs: `cols`

    TODO: convert between DataFrame/Panel upon request
    TODO: use `reindexing <reindex <http://pandas.pydata.org/pandas-docs/dev/indexing.html#advanced-reindexing-and-alignment-with-hierarchical-index>`_
    to keep a stable order

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
        A `pandas.Panel`

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
        panel = _agg_df(df, rows=rows, cols=cols, values=values,
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

def _agg_df(df, rows=None, cols=None, values=None,
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

def nan_outliers(df, values=None, group=None):
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

def stats_test(agg, test='ttest'):
    #import pdb; pdb.set_trace()
    array1 = agg.ix[:, 0:len(agg.columns):2]
    array2 = agg.ix[:, 1:len(agg.columns):2]
    t, p = scipy.stats.ttest_rel(array1, array2, axis=0)
    #import pdb; pdb.set_trace()
    print agg.columns.levels[0].values.tolist()
    print 't = %.2f, p = %.5f, df = %d' % (t[0], p[0], len(agg)-1)
    return

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

def ttestPrint(title = '****', values = None, xticklabels = None, legend = None, bon = None):

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


def oneway_anova(data):
    """
    Calculates one-way ANOVA on a pandas.DataFrame.

    :Args:
        data: `pandas.DataFrame`:
            rows contain groups (e.g., different conditions),
            columns have samples (e.g., participants)

    :Returns:
        F: float
            F-value
        p: float
            p-value
        k-1: int
            Between Group degrees of freedom
        N-k: int
            Within Group degrees of freedom

    """
    F, p = scipy.stats.f_oneway(*[d[1] for d in data.iterrows()])
    k = len(data)  # number of conditions
    N = k*len(data.columns)  # conditions times participants
    return F, p, k-1, N-k

def p_corr(df1, df2=None):
    """
    Computes Pearson correlation and its significance (using a t
    distribution) on a pandas.DataFrame.

    Ignores null values when computing significance. Based on
    `this Wikipedia entry <http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Testing_using_Student.27s_t-distribution>`_

    :Args:
        df1: `pandas.DataFrame`
            one dataset
        df2: `pandas.DataFrame`
            another dataset

    :Returns:
        corr: float
            correlation between the two datasets
        t: float
            an associated t-value
        p: float
            one-tailed p-value that the two datasets differ
    """
    if df2 is not None:
        if isinstance(df2, pandas.Series):
            df2 = pandas.DataFrame(df2)
        elif df2.shape[1] > 1:
            raise Exception('Cannot correlate two DataFrames')
        corrs = df1.corrwith(df2)
    else:
        corrs = df1.corr()
    Ns = corrs.copy()
    ts = corrs.copy()
    ps = corrs.copy()
    for colname, col in corrs.iteritems():
        for rowname, corr in col.iteritems():
            N = min(df1[colname].count(), df1[rowname].count())
            Ns.loc[rowname, colname] = N
            t = corr * np.sqrt((N-2) / (1-corr**2))
            ts.loc[rowname, colname] = t
            ps.loc[rowname, colname] = scipy.stats.t.sf(abs(t), N-2)  # one-tailed
    #import pdb; pdb.set_trace()
    #N = np.sum(df1.notnull())
    #t = corr * (N-2).applymap(np.sqrt) / (1-corr**2))
    #p = 1-scipy.stats.t.cdf(t.abs(), N-2)  # one-tailed
    return corrs, ts, ps


def reliability(panel, level=1, niter=100, func='mean'):
    """
    Computes data reliability by splitting it a data set into two random
    subsets of half the set size
    """
    if func == 'corr':
        pass
    else:
        try:
            getattr(np, func)
        except:
            raise
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

def mds(mean, ndim=2):
    if mean.ndim != 2:
        raise Exception('Similarity matrix for MDS must have exactly 2 '
                        'dimensions but only %d were found.' % mean.ndim)
    res = classical_mds(np.array(mean), ndim=ndim)
    if ndim <= 3:
        columns = ['x', 'y', 'z']
    else:
        columns = ['x%d' % i for i in range(ndim)]
    res = pandas.DataFrame(res, index=mean.index, columns=columns[:ndim])
    res.columns.names = ['cols.mds']
    return res

def classical_mds(d, ndim=2):
    """
    Metric Unweighted Classical Multidimensional Scaling

    Based on Forrest W. Young's notes on Torgerson's (1952) algorithm as
    presented in http://forrest.psych.unc.edu/teaching/p230/Torgerson.pdf:
    Step 0: Make data matrix symmetric with zeros on the diagonal
    Step 1: Double center the data matrix (d) to obtain B by removing row
    and column means and adding the grand mean of the squared data
    Step 2: Solve B = U * L * U.T for U and L
    Step 3: Calculate X = U * L**(-.5)

    :Args:
        - d: `numpy.ndarray`
            A symmetric dissimilarity matrix
        - ndim (int, default: 2)
            The number of dimensions to project to
    :Kwargs:
        X[:, :ndim]: `numpy.ndarray`
            The projection of d into ndim dimensions

    :Returns:
        A `numpy.array` with `ndim` columns representing the multidimensionally
        scaled data.

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


if __name__ == '__main__':
    pass
