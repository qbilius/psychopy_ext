#!/usr/bin/env python

# Part of the psychopy_ext library
# Copyright 2010-2014 Jonas Kubilius
# The program is distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.

"""
A collection of useful descriptive and basic statistical functions for
psychology research. Some functions are meant to improve `scipy.stats`
functionality and integrate seamlessly with `pandas`.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import scipy.stats
import pandas

from collections import OrderedDict


def aggregate(df, groupby=None, agg_out=None, aggfunc=None, reset_index=True):
    if aggfunc is None:
        aggfunc = np.mean

    df = pandas.DataFrame(df)  # make sure it's a DataFrame
    # import ipdb; ipdb.set_trace()
    df = factorize(df)

    if groupby is not None:
        if not isinstance(groupby, (list, tuple)):
            groupby = [groupby]
        skip = []
    elif agg_out is not None:
        if not isinstance(agg_out, (list, tuple)):
            agg_out = [agg_out]

        if isinstance(aggfunc, dict):
            skip = agg_out + aggfunc.keys()
        else:
            skip = agg_out

        groupby = [c for c in df if c not in skip and df[c].dtype.name=='category']
    else:
        raise 'You must provide either columns to group by or to aggregate out.'

    groups = df.groupby(groupby)
    agg = groups.aggregate(aggfunc)
    agg = agg[[c for c in agg if c not in skip]]
    if reset_index:
        agg = agg.reset_index()

    return agg

def factorize(df, order={}):
    for col in df:
        if df[col].dtype.name == 'object' or col in order:
            set_categories(df, col, order=order.get(col))
    return df

def set_categories(df, col, order=None):
    df.loc[:,col] = df.loc[:,col].astype('category', ordered=True)
    if order is None:
        order = df[col].unique()
    order = np.array(order)
    if df[col].dtype.name == 'category':
        try:
            df[col].cat.reorder_categories(order, ordered=True, inplace=True)
        except:
            print('WARNING: Could not reorder ' + col)
            pass


def _signal_detection(df, values, signal_noise_col, groupby=None,
                      signal='signal', noise='noise', hits='correct',
                      fas='incorrect'):

    if len(df[signal_noise_col].unique()) != 2:
        raise ValueError('There must be two values in the %s column.' %
                         signal_noise_col)

    if groupby is None:
        groupby = [signal_noise_col]
    else:
        if isinstance(groupby, (str, unicode)):
            groupby = [groupby, signal_noise_col]
        else:
            groupby += [signal_noise_col]

    rate_rows = df[values].isin([hits, fas])
    df[values] = 0
    df[values] = df[values].astype(float)
    df.ix[rate_rows, values] = 1
    agg = aggregate(df, groupby, aggfunc=np.mean, reset_index=False)

    agg = agg.unstack(signal_noise_col)
    return agg

def d_prime(df, values, signal_noise_col, groupby=None, signal='signal',
            noise='noise', hits='correct', fas='incorrect'):
    """
    Computes d' sensitivity measure.

    From: Stanislaw & Todorov (1999). doi: 10.3758/BF03207704

    .. warning:: This feature has not been tested yet!

    :Args:
        - df (pandas.DataFrame)
            Your data
        - value (str or list of str)
            Name of the column where d' is computed.

    :Kwargs:
        - groupby (str or list of str, default: None)
            Name(s) of the column(s) that is aggregated
        - hits (str or a number or list of str or numbers, default: None)
            Labels that are treated as hits.
        - fas (str or a number or list of str or numbers, default: None)
            Labels that are treated as false alarms.
        - kwargs
            Anything else you want to pass to :func:`aggregate`. Note that
            ``aggfunc`` is set to ``np.size`` and you cannot change that.

      :Returns:
          A pandas.DataFrame.
    """
    agg = _signal_detection(df, values, signal_noise_col, groupby=groupby,
                            signal=signal, noise=noise, hits=hits, fas=fas)
    agg[signal] = scipy.stats.zscore(agg[signal], ddof=1)
    agg[noise] = scipy.stats.zscore(agg[noise], ddof=1)
    dp = agg[signal] - agg[noise]
    return dp

def a_prime(df, values=None, hits='correct', fas='incorrect', **kwargs):
    """
    Computes A' sensitivity measure.

    From: Stanislaw & Todorov (1999). doi: 10.3758/BF03207704

    .. warning:: This feature has not been tested yet!

    :Args:
        df (pandas.DataFrame)
            Your data

    :Kwargs:
        - values (str or list of str, default: None)
            Name(s) of the column(s) that is aggregated
        - hits (str or a number or list of str or numbers, default: None)
            Labels that are treated as hits.
        - fas (str or a number or list of str or numbers, default: None)
            Labels that are treated as false alarms.
        - kwargs
            Anything else you want to pass to :func:`aggregate`. Note that
            ``aggfunc`` is set to ``np.size`` and you cannot change that.

      :Returns:
          A pandas.DataFrame.
    """
    agg = _signal_detection(df, values, signal_noise_col, groupby=groupby,
                            signal=signal, noise=noise, hits=hits, fas=fas)
    d = agg[signal] - agg[noise]
    ap = .5 + (np.sign(d) * (d**2 + np.abs(d)) / (4 * np.max(agg[signal],agg[noise]) - 4 * agg[signal] * agg[noise]))
    return ap

def confidence(agg, kind='sem', within=None, alpha=.05, nsamples=None,
               skipna=True):
    """
    Compute confidence of measurements, such as standard error of means (SEM)
    or confidence intervals (CI).

    :Args:
        agg
    :Kwargs:
        - kind ('sem', 'ci', or 'binomial', default: 'sem')
            .. warning:: Binomial not tested throroughly
        - within (str or list, default: None)
            For repeated measures designs, error bars are too large.
            Specify which dimensions come from repeated measures
            (rows, cols, and/or subplots). It computes
            within-subject confidence intervals using a method by
            Loftus & Masson (1994) simplified by Cousinaueu (2005)
            with Morey's (2008) correction.
            Based on `Denis A. Engemann's gist <https://gist.github.com/dengemann/6030453>`_
        - alpha (float, default: .05)
            For CI and binomial. Computed for single-tail, so effectively
            alpha/2.
        - nsamples
            For binomial distribution confidence intervals if there is
            a single sample only (which presumably relfects the number
            of correct responses, i.e., successes). See `Wikipedia
            <http://en.wikipedia.org/wiki/Binomial_proportion_confidence_interval#Normal_approximation_interval>`_
        - skipna (default: True)
            Whether to skip NA / null values or not.

            .. warning:: Not tested thoroughly
    :Returns:
        mean, p_yerr
    """
    if isinstance(agg, pandas.DataFrame):
        mean = agg.mean(skipna=skipna)  # mean across participants

        if within is not None:
            if not isinstance(within, list):
                within = [within]

            levels = []
            for dim in within:
                tmp = [r for r in agg.columns.names if not r.startswith(dim + '.')]
                levels += tmp
            if len(levels) > 0:
                #raise Exception('Could not find levels that start with any of the following: %s' % within)
            # center data
            #try:
                try:
                    subj_mean = agg.mean(level=levels, skipna=skipna, axis=1)  # mean per subject
                except:
                    subj_mean = agg.mean(skipna=skipna)
            else:
                subj_mean = agg.mean(skipna=skipna)
            grand_mean = mean.mean(skipna=skipna)
            center = subj_mean - grand_mean

            aggc = pandas.DataFrame(index=agg.index, columns=agg.columns)
            for colname, value in center.iteritems():
                try:
                    aggc[colname]  # what a cool bug -- crashes without this line!
                except:
                    import pdb; pdb.set_trace()
                aggc[colname] = (agg[colname].T - value.T).T
            ddof = 0  # Morey's correction
        else:
            aggc = agg.copy()
            ddof = 1

        if skipna:
            count = agg.count()  # skips NA by default
        else:
            count = len(agg)

        confint = 1 - alpha/2.
        if kind == 'sem':
            p_yerr = aggc.std(skipna=skipna, ddof=ddof) / np.sqrt(count)
        elif kind == 'ci':
            p_yerr = aggc.std(skipna=skipna, ddof=ddof) / np.sqrt(count)
            p_yerr *= scipy.stats.t.ppf(confint, count-1)
        elif kind == 'binomial':
            z = scipy.stats.norm.ppf(confint)
            p_yerr = z * np.sqrt(mean * (1-mean) / nsamples)

    else:  # if there's only one condition? Not sure #FIXME
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
    print(agg.columns.levels[0].values.tolist())
    print('t = %.2f, p = %.5f, df = %d' % (t[0], p[0], len(agg)-1))
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
        print(binom)
        return binom

def ttestPrint(title = '****', values = None, xticklabels = None, legend = None, bon = None):

    d = 8
    # check if there are any negative t values (for formatting purposes)
    if np.any([np.any(val < 0) for val in values]): neg = True
    else: neg = False

    print('\n', title)
    for xi, xticklabel in enumerate(xticklabels):
        print(xticklabel)

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

            print(outputStr %
                {'s': legendname, 'd':(d-1), 't': t,
                'p': p, 'star': star})


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

def pearson_corr(df1, df2=None):
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


def reliability_splithalf_orig(df, level=1, niter=100, func='mean'):
    """
    Computes data reliability.

    Works by splitting it a data set into two random subsets of half the set
    size. A Spearman-Brown correction is applied (because by splitting data
    in half we use only have of the data).

    .. warning: Doesn't work yet.
    """
    # zscore = lambda x: scipy.stats.zscore(x, ddof=1)
    # # aggz = aggregate(df, agg_out=agg_out, aggfunc=zscore)
    # agg_zm = scipy.stats.zscore(agg, ddof=1).mean()
    # agg_zm = np.tile(agg_zm, len(agg))
    # rel = agg.corrwith(aggzm, axis=1).mean()
    #
    # return rel

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
    p = 2*p/(1.+p)
    return corr, t, p

def reliability_splithalf(data, nsplits=100):
    inds = range(len(data))
    h = len(data)/2
    corr = []
    for i in range(nsplits):
        np.random.shuffle(inds)
        d1 = np.mean(data[inds[:h]], 0)
        d2 = np.mean(data[inds[h:]], 0)
        c = np.corrcoef(d1, d2)[0,1]
        corr.append(c)
    r = np.mean(corr)
    r = 2*r/(1.+r)
    return r

def reliability(data):
    """
    Computes upper and lower boundaries of data reliability

    :Args:
        data (np.ndarray)
            N samples x M features
    :Returns:
        (floor, ceiling)
    """
    zdata = scipy.stats.zscore(data, axis=1)
    # remove data with no variance in it
    # sel = np.apply_along_axis(lambda x: ~np.all(np.isnan(x)), 1, zdata)
    # if not np.all(sel):
    #     print('WARNING: only {} measurements out of {} will be used.'.format(np.sum(sel), len(sel)))
    # zdata = zdata[sel]
    for i,z in enumerate(zdata):
        if np.all(np.isnan(z)):
            zdata[i] = np.zeros_like(z)
            zdata[i,0] = np.finfo(float).eps
            print('WARNING: some measurements are all equal')
    zmn = np.mean(zdata, axis=0)
    ceil = np.mean([np.corrcoef(subj,zmn)[0,1] for subj in zdata])
    rng = np.arange(zdata.shape[0])

    floor = []
    for s, subj in enumerate(zdata):
        mn = np.mean(zdata[rng!=s], axis=0)
        floor.append(np.corrcoef(subj,mn)[0,1])
    floor = np.mean(floor)
    return floor, ceil

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

def corr(data1, data2, sel=None, axis=1, drop=True):
    assert data1.ndim <= 2, 'must have at most 2 dimensions'
    assert data1.shape == data2.shape, 'input dimensions must match'
    if sel is 'upper':
        assert data1.shape[0] == data2.shape[1], 'must be square matrices'
        inds = np.triu_indices(data1.shape[0], k=1)
        data1 = np.mat(data1[inds])
        data2 = np.mat(data2[inds])
    else:
        data1 = np.mat(data1.ravel())
        data2 = np.mat(data2.ravel())

    # d1 = d1[~np.isnan(d1)]
    # d2 = d2[~np.isnan(d2)]
    df1 = pandas.DataFrame(data1)
    df2 = pandas.DataFrame(data2)
    c = df1.corrwith(df2, axis=axis, drop=drop)
    if c.values.size == 1:
        return c.values[0]
    else:
        return c.values

def partial_corr(x, y, z):
    """
    Partial correlation.

    .. warning:: Not tested.
    """
    x = np.mat(x)
    y = np.mat(y)
    z = np.mat(z)

    beta_x = scipy.linalg.lstsq(z.T, x.T)[0]
    beta_y = scipy.linalg.lstsq(z.T, y.T)[0]

    res_x = x.T - z.T.dot(beta_x)
    res_y = y.T - z.T.dot(beta_y)

    pcorr = np.corrcoef(res_x.T, res_y.T)[0,1]
    return pcorr

def bootstrap_permutation(m1, m2=None, func=np.mean, niter=10000, ci=95,
                          *func_args, **func_kwargs):

    m1 = np.array(m1)
    if m2 is not None:
        m2 = np.array(m2)

    df = []
    for i in range(niter):
        mp1 = np.random.permutation(m1)
        if m1.ndim == 2:
            mp1 = np.random.permutation(mp1.T).T
        if m2 is not None:
            mp2 = np.random.permutation(m2)
            if m2.ndim == 2:
                mp2 = np.random.permutation(mp2.T).T
            out = func(mp1, mp2, *func_args, **func_kwargs)
        else:
            out = func(mp1, *func_args, **func_kwargs)
        df.append(out)

    if ci is not None:
        return (np.percentile(df, 50-ci/2.), np.percentile(df, 50+ci/2.))
    else:
        return df

def bootstrap_resample(data1, data2=None, func=np.mean, niter=1000, ci=95,
                       struct=None, seed=None, *func_args, **func_kwargs):

    np.random.seed(seed)  # useful for paired resampling
    # if func is None:
    #     func = lambda x,y: np.corrcoef(x,y)[0,1]
    #     import pdb; pdb.set_trace()

    d1 = np.squeeze(np.array(data1))
    if data2 is not None:
        d2 = np.squeeze(np.array(data2))
    # if d1.ndim == 2:
    #     d1 = (data1 + data1.T) / 2.
    #     d2 = (data2 + data2.T) / 2.
    df = []
    for n in range(niter):
        if struct is not None:
            inds = np.arange(len(data1)).astype(int)
            for s in np.unique(struct):
                inds[struct==s] = np.random.choice(inds[struct==s], size=sum(struct==s))
        else:
            inds = np.random.randint(len(data1), size=(len(data1),))
        d1s = d1[inds]
        if data2 is not None:
            d2s = d2[inds]
        # df.append(func(d1s[~np.isnan(d1s)], d2s[~np.isnan(d2s)]))
        # import pdb; pdb.set_trace()
            out = func(d1s, d2s, *func_args, **func_kwargs)
        else:
            out = func(d1s, *func_args, **func_kwargs)
        df.append(out)

    if ci is not None:
        return np.percentile(df, 50-ci/2.), np.percentile(df, 50+ci/2.)
    else:
        return df
    #return pct

def bootstrap_ttest(data1, data2=None, kind='ind', tails='one', struct=None,
                    *func_args, **func_kwargs):
    if data2 is None:
        return bootstrap_permutation(data1, func=np.mean, niter=niter, ci=None,
                                     *func_args, **func_kwargs)
    b1 = bootstrap_resample(data1, func=np.mean, ci=None, struct=struct,
                            *func_args, **func_kwargs)
    b2 = bootstrap_resample(data2, func=np.mean, ci=None, struct=struct,
                            *func_args, **func_kwargs)

    pct = scipy.stats.percentileofscore(b1 - b2, 0, kind='mean') / 100.
    p = min(pct, 1-pct)
    if tails == 'one':
        p *= 2

    return np.mean(b1-b2), p

def bootstrap_ttest_multi(bfg, tails='two'):
    st = []
    for d1 in bfg:
        for d2 in bfg:
            diff = bfg[d1] - bfg[d2]
            pct = scipy.stats.percentileofscore(diff, 0, kind='mean') / 100.
            p = min(pct, 1-pct)
            if tails == 'two': p *= 2
            star = stats.get_star(p)
            st.append([d1, d2, np.mean(diff), p, star])
    st = pandas.DataFrame(st, columns=['var1', 'var2', 'mean', 'p', 'sig'])
    return st

if __name__ == '__main__':
    pass
