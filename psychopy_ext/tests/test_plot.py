import numpy as np
from .. import stats, plot

import unittest

N = 8
NSAMPL = 10

class TestPlot(unittest.TestCase):

    def get_df(self, n=N, nsampl=NSAMPL):
        k = n * nsampl
        df = stats.df_fromdict(
            [('subplots', ['session1', 'session2']),
            ('cond', [1, 2]),
            ('name', ['one', 'two', 'three']),
            ('levels', ['small', 'medium', 'large']),
            ('subjID', ['subj%d' % (i+1) for i in range(8)])],
            repeat=10
            )
        df['rt'] = range(k)*36
        df['accuracy'] = ['correct','correct','incorrect','incorrect']*k*9
        return df

    def test_plot(self):
        """
        Only tests if plotting works at all. Appearance not tested.
        """
        df = self.get_df()
        df.rt = df.rt.astype(float)
        df['rt'] = np.genfromtxt('psychopy_ext/tests/rt.csv', delimiter=',')

        agg = stats.aggregate(df, subplots='subplots', rows=['cond', 'name'],
            cols='levels', yerr='subjID', values='rt')
        plt = plot.Plot()
        plt.plot(agg)

        agg = stats.aggregate(df, rows=['cond', 'name'],
            cols='levels', yerr='subjID', values='rt')
        plt = plot.Plot()
        plt.plot(agg)

        agg = stats.aggregate(df, subplots='subplots', rows='cond',
            cols='levels', yerr='subjID', values='rt')
        plt = plot.Plot()
        plt.plot(agg)

        agg = stats.aggregate(df, subplots='subplots',
            cols='levels', yerr='subjID', values='rt')
        plt = plot.Plot()
        plt.plot(agg)

        agg = stats.aggregate(df, subplots='subplots', rows=['cond', 'name'],
            yerr='subjID', values='rt')
        plt = plot.Plot()
        plt.plot(agg)

        agg = stats.aggregate(df, subplots='subplots', rows=['cond', 'name'],
            cols='levels', values='rt')
        plt = plot.Plot()
        plt.plot(agg)

if __name__ == '__main__':
    unittest.main()
