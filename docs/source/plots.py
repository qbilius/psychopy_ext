import sys
import numpy as np

sys.path.insert(0, '../../')
from psychopy_ext import stats, plot
import seaborn
seaborn.rcmod.set()  # somehow it doesn't work without it here

n = 8
nsampl = 10
k = n*nsampl
df = stats.df_fromdict(
    [('subplots', ['session1', 'session2']),
     ('cond', [1, 2]),
     ('name', ['condition one', 'condition two']),
     ('levels', ['small', 'medium', 'large']),
     ('subjID', ['subj%d' % (i+1) for i in range(n)])],
     repeat=nsampl
    )
df['rt'] = 0.5
df.rt[df.cond==1] = np.random.random(12*k) * 1.2
df.rt[df.cond==2] = np.random.random(12*k)
#df['accuracy'] = ['correct','correct','incorrect','incorrect']*k*9

agg = stats.aggregate(df, subplots='subplots', rows='levels',
            cols='cond', yerr='subjID', values='rt')
plt1 = plot.Plot(figsize=(6,3))
plt1.plot(agg, kind='bar')
plt1.tight_layout()

plt2 = plot.Plot(figsize=(6,3))
plt2.plot(agg, kind='line')
plt2.tight_layout()

plt3 = plot.Plot(figsize=(8,4))
plt3.plot(agg, kind='bean')
plt3.tight_layout()

agg = stats.aggregate(df, subplots='subplots', rows='subjID',
            cols='name', values='rt')
plt4 = plot.Plot(figsize=(6,3))
plt4.plot(agg, kind='scatter')
plt4.tight_layout()

mtx = stats.df_fromdict(
    [('subplots', ['session1', 'session2']),
     ('cond1', ['cond%d' % i for i in range(6)]),
     ('cond2', ['cond%d' % i for i in range(6)]),
     ('subjID', ['subj%d' % (i+1) for i in range(n)])],
     repeat=nsampl
    )
mtx['corr'] = np.random.random(len(mtx))
agg = stats.aggregate(mtx, subplots='subplots', rows='cond1',
            cols='cond2', values='corr')
plt5 = plot.Plot(figsize=(8,4), kind='matrix')
plt5.plot(agg, kind='matrix')

#agg = stats.aggregate(df, subplots='subplots', rows='cond',
            #cols='levels', yerr='subjID', values='rt')
#agg = stats.aggregate(df, subplots='subplots', rows=['cond', 'name'],
            #cols='levels', yerr='subjID', values='rt')

plt5.show()
