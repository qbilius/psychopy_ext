import numpy as np
import pandas
from .. import stats, plot

import unittest

N = 8
NSAMPL = 10

class TestAgg(unittest.TestCase):

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

    def test_df_fromdict(self):
        n = 8
        nsampl = 10
        df = self.get_df(n=n, nsampl=nsampl)
        k = n * nsampl
        data = {
            'subplots': ['session1']*k*18 + ['session2']*k*18,
            'cond': [1]*k*9 + [2]*k*9 + [1]*k*9 + [2]*k*9,
            'name': (['one', 'one', 'one']*k + ['two', 'two', 'two']*k +
                    ['three', 'three', 'three']*k) * 4,
            'levels': (['small']*k + ['medium']*k + ['large']*k)*12,
            'subjID': ['subj%d' % (i+1) for i in np.repeat(range(n),nsampl)] * 36,
            'rt': range(k)*36,
            'accuracy': ['correct','correct','incorrect','incorrect']*k*9
            }
        df_manual = pandas.DataFrame(data, columns = ['subplots','cond','name',
            'levels','subjID','rt','accuracy'])
        self.assertEqual(df.to_string(), df_manual.to_string())

    # def test_aggregate_random(self):
    #     df = self.get_df()
    #     df.rt = df.rt.astype(float)
    #     df['rt'] = np.genfromtxt('psychopy_ext/tests/rt.csv', delimiter=',')
    #     agg = stats.aggregate(df, subplots='subplots', rows=['cond', 'name'],
    #         cols='levels', yerr='subjID', values='rt')
    #
    #     # construct the correct answer
    #     tuples = zip(*[
    #         ['session1']*18 + ['session2']*18,
    #         [1]*9 + [2]*9 + [1]*9 + [2]*9,
    #         ['one', 'one', 'one', 'two', 'two', 'two','three', 'three', 'three'] * 4,
    #         ['small','medium','large']*12
    #         ])
    #     index = pandas.MultiIndex.from_tuples(tuples,
    #         names=['subplots','cond','name','levels'])
    #     cols = pandas.Index(['subj%d' % (i+1) for i in range(N)], names='subjID')
    #     data = np.genfromtxt('psychopy_ext/tests/agg.csv', delimiter=',')
    #     test_agg = pandas.DataFrame(data, index=index, columns=cols).T
    #     test_agg.index.names = ['subjID']  # yeah...
    #     self.assertEqual(test_agg.to_string(), agg.to_string())
    #
    # def test_aggregate(self):
    #     df = self.get_df()
    #     agg = stats.aggregate(df, subplots='subplots', rows=['cond', 'name'],
    #         cols='levels', yerr='subjID', values='rt')
    #
    #     # construct the correct answer
    #     col = [4.5, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 74.5]
    #     tuples = zip(*[
    #         ['session1']*18 + ['session2']*18,
    #         [1]*9 + [2]*9 + [1]*9 + [2]*9,
    #         ['one', 'one', 'one', 'two', 'two', 'two','three', 'three', 'three'] * 4,
    #         ['small','medium','large']*12
    #         ])
    #     index = pandas.MultiIndex.from_tuples(tuples,
    #         names=['subplots','cond','name','levels'])
    #     cols = pandas.Index(['subj%d' % (i+1) for i in range(N)], names='subjID')
    #     test_agg = pandas.DataFrame([col]*36, index=index, columns=cols).T
    #     test_agg.index.names = ['subjID']  # yeah...
    #     self.assertEqual(test_agg.to_string(), agg.to_string())
    #
    # def test_aggregate_nosubplots(self):
    #     df = self.get_df()
    #     agg = stats.aggregate(df, rows=['cond', 'name'],
    #         cols='levels', yerr='subjID', values='rt')
    #
    #     # construct the correct answer
    #     col = [4.5, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 74.5]
    #     tuples = zip(*[
    #         [1]*9 + [2]*9,
    #         ['one', 'one', 'one', 'two', 'two', 'two','three', 'three', 'three'] * 2,
    #         ['small','medium','large']*6
    #         ])
    #     index = pandas.MultiIndex.from_tuples(tuples,
    #         names=['cond','name','levels'])
    #     cols = pandas.Index(['subj%d' % (i+1) for i in range(N)], names='subjID')
    #     test_agg = pandas.DataFrame([col]*18, index=index, columns=cols).T
    #     test_agg.index.names = ['subjID']  # yeah...
    #     self.assertEqual(test_agg.to_string(), agg.to_string())
    #
    # def test_aggregate_onerow(self):
    #     df = self.get_df()
    #     agg = stats.aggregate(df, subplots='subplots', rows='cond',
    #         cols='levels', yerr='subjID', values='rt')
    #
    #     # construct the correct answer
    #     col = [4.5, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 74.5]
    #     tuples = zip(*[
    #         ['session1']*6 + ['session2']*6,
    #         [1]*3 + [2]*3 + [1]*3 + [2]*3,
    #         ['small','medium','large']*6
    #         ])
    #     index = pandas.MultiIndex.from_tuples(tuples,
    #         names=['subplots','cond','levels'])
    #     cols = pandas.Index(['subj%d' % (i+1) for i in range(N)], names='subjID')
    #     test_agg = pandas.DataFrame([col]*12, index=index, columns=cols).T
    #     test_agg.index.names = ['subjID']  # yeah...
    #     self.assertEqual(test_agg.to_string(), agg.to_string())
    #
    # def test_aggregate_nocols(self):
    #     df = self.get_df()
    #     agg = stats.aggregate(df, subplots='subplots', rows=['cond', 'name'],
    #         yerr='subjID', values='rt')
    #
    #     # construct the correct answer
    #     col = [4.5, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 74.5]
    #     tuples = zip(*[
    #         ['session1']*6 + ['session2']*6,
    #         [1]*3 + [2]*3 + [1]*3 + [2]*3,
    #         ['one', 'two', 'three'] * 4
    #         ])
    #     index = pandas.MultiIndex.from_tuples(tuples,
    #         names=['subplots','cond','name'])
    #     cols = pandas.Index(['subj%d' % (i+1) for i in range(N)], names='subjID')
    #     test_agg = pandas.DataFrame([col]*12, index=index, columns=cols).T
    #     test_agg.index.names = ['subjID']  # yeah...
    #     self.assertEqual(test_agg.to_string(), agg.to_string())
    #
    # def test_aggregate_norows(self):
    #     df = self.get_df()
    #     agg = stats.aggregate(df, subplots='subplots',
    #         cols='levels', yerr='subjID', values='rt')
    #
    #     # construct the correct answer
    #     col = [4.5, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 74.5]
    #     tuples = zip(*[
    #         ['session1']*3 + ['session2']*3,
    #         ['small','medium','large']*2
    #         ])
    #     index = pandas.MultiIndex.from_tuples(tuples,
    #         names=['subplots','levels'])
    #     cols = pandas.Index(['subj%d' % (i+1) for i in range(N)], names='subjID')
    #     test_agg = pandas.DataFrame([col]*6, index=index, columns=cols).T
    #     test_agg.index.names = ['subjID']  # yeah...
    #     self.assertEqual(test_agg.to_string(), agg.to_string())
    #
    # def test_aggregate_noyerr(self):
    #     df = self.get_df()
    #     agg = stats.aggregate(df, subplots='subplots', rows=['cond', 'name'],
    #         cols='levels', values='rt')
    #
    #     # construct the correct answer
    #     col = [39.5]
    #     tuples = zip(*[
    #         ['session1']*18 + ['session2']*18,
    #         [1]*9 + [2]*9 + [1]*9 + [2]*9,
    #         ['one', 'one', 'one', 'two', 'two', 'two','three', 'three', 'three'] * 4,
    #         ['small','medium','large']*12
    #         ])
    #     index = pandas.MultiIndex.from_tuples(tuples,
    #         names=['subplots','cond','name','levels'])
    #     test_agg = pandas.DataFrame([col]*36, index=index).T
    #     test_agg = test_agg.rename(index={0: 'rt'})
    #     self.assertEqual(test_agg.to_string(), agg.to_string())
    #
    # def test_accuracy(self):
    #     df = self.get_df()
    #     agg = stats.accuracy(df, subplots='subplots', rows=['cond', 'name'],
    #         cols='levels', yerr='subjID', values='accuracy')
    #
    #     # construct the correct answer
    #     col = [.6, .4] * (N/2)
    #     tuples = zip(*[
    #         ['session1']*18 + ['session2']*18,
    #         [1]*9 + [2]*9 + [1]*9 + [2]*9,
    #         ['one', 'one', 'one', 'two', 'two', 'two','three', 'three', 'three'] * 4,
    #         ['small','medium','large']*12
    #         ])
    #     index = pandas.MultiIndex.from_tuples(tuples,
    #         names=['subplots','cond','name','levels'])
    #     cols = pandas.Index(['subj%d' % (i+1) for i in range(N)], names='subjID')
    #     test_agg = pandas.DataFrame([col]*36, index=index, columns=cols).T
    #     test_agg.index.names = ['subjID']  # yeah...
    #     self.assertEqual(test_agg.to_string(), agg.to_string())

if __name__ == '__main__':
    unittest.main()
