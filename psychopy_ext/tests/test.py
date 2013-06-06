import pandas
import numpy as np
import sys
from .. import stats, plot

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
df = df.reindex_axis(['subplots','cond','name','levels','subjID','RT',
    'accuracy'], axis=1)
agg = stats.aggregate(df, subplots='subplots', rows=['cond', 'name'],
    cols='levels', yerr='subjID', values='RT')
agg2 = stats.aggregate(df, rows='cond',
    cols='levels', yerr='subjID', values='RT')
#agg = stats.aggregate(df,
    #cols='levels', yerr='subjID', values='RT')
plt = plot.Plot()
plt.plot(agg)
#plt.show()

#plt2 = plot.Plot()
#plt2.plot(agg2)
#
#plt2.show()

col = [4.5, 14.5, 24.5, 34.5, 44.5, 54.5, 64.5, 74.5]
tuples = zip(*[
    ['session1']*18 + ['session2']*18,
    [1]*9 + [2]*9 + [1]*9 + [2]*9,
    ['one', 'one', 'one', 'two', 'two', 'two','three', 'three', 'three'] * 4,
    ['small','medium','large']*12
    ])
index = pandas.MultiIndex.from_tuples(tuples,
    names=['subplots','cond','name','levels'])
cols = ['subj%d' % (i+1) for i in range(n)]
test_agg = pandas.DataFrame([col]*36, index=index, columns=cols).T