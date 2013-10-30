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
            ('name', ['condition one', 'condition two']),
            #('name', ['one', 'two', 'three']),
            ('levels', ['small', 'medium', 'large']),
            ('subjid', ['subj%d' % (i+1) for i in range(n)])],
            repeat=nsampl
            )
        df['rt'] = 0.5
        df.rt[df.cond==1] = np.random.random(12*k) * 1.2
        df.rt[df.cond==2] = np.random.random(12*k)
        #df['rt'] = range(k)*36
        df['accuracy'] = ['correct','correct','incorrect','incorrect']*k*6
        return df
        
    def get_df_line(self, n=N, nsampl=NSAMPL):
        k = n * nsampl
        df = stats.df_fromdict(
            [('subplots', ['session1', 'session2']),
            ('cond', range(10,46,2)),
            ('name', ['condition one', 'condition two']),
            #('name', ['one', 'two', 'three']),
            ('levels', ['small', 'medium', 'large']),
            ('subjid', ['subj%d' % (i+1) for i in range(n)])],
            repeat=nsampl
            )
        df['rt'] = 0.5
        for cond in df.cond.unique():
            scale = 1 + np.random.random()
            df.rt[df.cond==cond] = np.random.random(12*k) * scale
        return df

    def get_mtx(self, n=N, nsampl=NSAMPL):
        mtx = stats.df_fromdict(
            [('subplots', ['session1', 'session2']),
             ('cond1', ['cond%d' % i for i in range(6)]),
             ('cond2', ['cond%d' % i for i in range(6)]),
             ('subjid', ['subj%d' % (i+1) for i in range(n)])],
             repeat=nsampl
            )
        mtx['corr'] = np.random.random(len(mtx))
        return mtx

    def test_plot(self):
        """
        Only tests if plotting works at all. Appearance not tested.
        """
        df = self.get_df()
        #df.rt = df.rt.astype(float)
        #df['rt'] = np.genfromtxt('psychopy_ext/tests/rt.csv', delimiter=',')

        agg = range(6)
        agg[0] = stats.aggregate(df, subplots='subplots', rows=['cond', 'name'],
            cols='levels', yerr='subjid', values='rt')
        agg[1] = stats.aggregate(df, rows=['cond', 'name'],
            cols='levels', yerr='subjid', values='rt')
        agg[2] = stats.aggregate(df, subplots='subplots', rows='cond',
            cols='levels', yerr='subjid', values='rt')
        agg[3] = stats.aggregate(df, subplots='subplots',
            cols='levels', yerr='subjid', values='rt')
        agg[4] = stats.aggregate(df, subplots='subplots', rows=['cond', 'name'],
            yerr='subjid', values='rt')
        agg[5] = stats.aggregate(df, subplots='subplots', rows=['cond', 'name'],
            cols='levels', values='rt')

        kinds = ['line', 'bar', 'bean']
        for a in agg:
            for kind in kinds:
                plot.Plot().plot(a, kind=kind)
                
        df_line = self.get_df_line()
        agg = stats.aggregate(df_line, subplots='subplots', rows='cond',
            cols='levels', yerr='subjid', values='rt')
        plot.Plot().plot(agg, kind='line')

        sct = stats.aggregate(df, subplots='subplots', rows='subjid',
            cols='name', values='rt')
        plot.Plot().plot(sct, kind='scatter')

        df_mtx = self.get_mtx()
        mtx = stats.aggregate(df_mtx, subplots='subplots', rows='cond1',
            cols='cond2', values='corr')
        plot.Plot().plot(mtx, kind='matrix')
        plot.Plot().plot(mtx, kind='mds')

        plot.Plot().show()

if __name__ == '__main__':
    unittest.main()
