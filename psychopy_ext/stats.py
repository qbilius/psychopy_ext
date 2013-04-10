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


def aggregate(df, rows=None, cols=None, values=None,
    value_filter=None, subplots=None, yerr=None, func='mean', unstack=False):
    """
    Aggregates data over specified columns.

    :Args:
        df (pandas.DataFrame)
            A DataFrame with your data

    :Kwargs:
        - rows (str or list of str, default: None)
            Name(s) of column(s) that will be aggregated and plotted on the x-axis
        - cols (str, default: None)
            Name(s) of column(s) that will be shown in the legend
        - values (str, default: None)
            Name of the column that is aggregated
        - yerr (str, default: None)
            Name of the column for the y-errorbar calculation. Typically,
            this is the column with participant IDs.

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
    """
    df = pandas.DataFrame(df)  # make sure it's a DataFrame
    if isinstance(rows, str) or rows is None:
        rows = [rows]
    if isinstance(cols, str) or cols is None:
        cols = [cols]
    allconds = [subplots] + rows + cols + [yerr]
    allconds = [c for c in allconds if c is not None]

    if df[values].dtype in (str, object):  # calculate accuracy
        size = df.groupby(allconds)[values].size()
        if value_filter is not None:
            dff = df[df[values] == value_filter]
        else:
            raise Exception('value_filter must be defined when aggregating '
                'over str or object types')
        size_filter = dff.groupby(allconds)[values].size()
        agg = size_filter / size.astype(float)
    else:
        if isinstance(func, str):
            try:
                func = getattr(np, func)
            except:
                raise
        agg = df.groupby(allconds)[values].aggregate(func)

    g = 0
    groups = [('subplots',[subplots]), ('rows',rows), ('cols', cols),
              ('yerr',[yerr])]
    for group in groups:
        for item in group[1]:
            if item is not None:
                agg.index.names[g] = group[0] + '.' + item
                g += 1

    if yerr is not None:
        agg = agg.unstack().T
    else:  # then rows should become rows, and cols should be cols :)
        if unstack:
            for name in agg.index.names:
                if name.startswith('cols.'):
                    agg = agg.unstack(level=name)
        else:
            agg = pandas.DataFrame(agg).T

    return agg


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

def p_corr(df1, df2):
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
    corr = df1.corr(df2)
    N = np.sum(df1.notnull())
    t = corr*np.sqrt((N-2)/(1-corr**2))
    p = 1-scipy.stats.t.cdf(abs(t),N-2)  # one-tailed
    return corr, t, p


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