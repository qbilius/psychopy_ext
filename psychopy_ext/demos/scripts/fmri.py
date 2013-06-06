import sys, os, shutil
import cPickle as pickle

import numpy as np
import mvpa2.suite
import nibabel as nb
# pandas does not come by default with PsychoPy but that should not prevent
# people from running the experiment
try:
    import pandas
except:
    pass

from psychopy_ext import fmri, exp, plot, stats

# some modules are only available in Python 2.6
try:
    from collections import OrderedDict
except:
    from exp import OrderedDict

import computer  # for monitor size, paths etc settings across computers
# set up where all data, logs etc are stored for this experiment
# for a single experiment, '.' is fine -- it means data is stored in the 'data'
# folder where the 'run.py' file is, for example
# if you have more than one experiment, 'confsup' would be better -- data for
# this experiment will be in the 'data' folder inside the 'confsup' folder
computer.root = '/home/qbilius/Atsiuntimai/tutorial_data/data/'
PATHS = exp.set_paths('.', computer, fmri_rel='%s/')

class Analysis(fmri.Analysis):
    def __init__(self,
                 name='analysis',
                 extraInfo=OrderedDict([
                    ('subjID', 'twolines_'),
                    ('runType', 'main'),
                    ('runNo', 1),
                    ]),
                 runParams=OrderedDict([
                    ('noOutput', False),
                    ('force', False),
                    ('all', True),
                    ('rois', ['vt', (['random1','random2'],'random')]),
                    ('method', 'corr'),
                    ('values', 'raw'),
                    ('set_size', 'subset'),
                    ('degree', 'normal'),
                    ('plot', False),
                    ('saveplot', False),
                    ('visualize', False),
                    ])
                 ):
        super(Analysis, self).__init__(PATHS, tr=2.5,
            extraInfo=extraInfo, runParams=runParams, fmri_prefix='a*',
            fix='rest', dur=None, offset=0)
        if self.runParams['values'] == 'raw':
            if self.runParams['method'] == 'timecourse':
                self.offset = 0
                self.dur = 9
            else:
                self.offset = 1
                self.dur = 4
        self.name = name
        if self.runParams['all']: self.set_all_subj()

    def set_all_subj(self):
        self.extraInfo['subjID'] = ['subj_01', 'subj_02']

    def prepare_data(self):
        """
        Renames data files from PyMVPA2 in the expected format.
        """
        print 'preparing tutorial data... this will take a while...'
        attr = mvpa2.suite.SampleAttributes(self.paths['root'] + \
                                            'attributes.txt')
        ds = mvpa2.suite.fmri_dataset(samples=self.paths['root'] + 'bold.nii.gz',
                          targets=attr.targets, chunks=attr.chunks)
        for subjID in self.extraInfo['subjID']:
            try:
                os.makedirs(self.paths['data_behav'] % subjID)
                os.makedirs(self.paths['data_fmri'] % subjID)
                os.makedirs(self.paths['rois'] % subjID)
            except:
                pass

            for chunk in ds.UC:
                saveds = ds[ds.sa.chunks == chunk]
                # add a bit of noise so that participants are not identical
                noise = np.random.random_sample(saveds.samples.shape) * \
                        np.mean(ds.samples) / 10
                saveds.samples += noise
                n = len(saveds)
                df = pandas.DataFrame(OrderedDict([
                        ('subjID', [subjID] * n),
                        ('runNo', saveds.sa.chunks.astype(int) + 1),
                        ('runType', [self.extraInfo['runType']] * n),
                        ('cond', saveds.sa.targets),
                        ('dur', [self.tr] * n)
                        ]))
                df.to_csv(self.paths['data_behav'] % subjID + \
                          'data_%02d_%s.csv' %
                          (chunk + 1, self.extraInfo['runType']), index=False)
                nb.save(mvpa2.suite.map2nifti(saveds),
                    self.paths['data_fmri'] % subjID + 'afunc_%02d_%s.nii.gz' %
                    (chunk + 1, self.extraInfo['runType']))

            shutil.copy(self.paths['root'] + 'mask_vt.nii.gz',
                self.paths['rois'] % subjID + 'vt.nii.gz')

            # make some random mask; code adjusted from mvpa2
            nimg = nb.load(self.paths['root'] + 'mask_vt.nii.gz')
            tmpmask1 = nimg.get_data() < 0
            tmpmask2 = nimg.get_data() < 0
            slices = np.array(tmpmask1.shape)/2
            slices1 = slices - 4
            slices2 = slices + 4
            tmpmask1[slices1[0]:slices[0], slices1[1]:slices[1],
                    slices1[2]:slices[2]] = True
            mask1 = nb.Nifti1Image(tmpmask1.astype(int), None, nimg.get_header())
            nb.save(mask1, self.paths['rois'] % subjID + 'random1.nii.gz')
            tmpmask2[slices[0]:slices2[0], slices[1]:slices2[1],
                    slices[2]:slices2[2]] = True
            mask2 = nb.Nifti1Image(tmpmask2.astype(int), None, nimg.get_header())
            nb.save(mask2, self.paths['rois'] % subjID + 'random2.nii.gz')

        sys.exit()

    def run(self):
        df, df_fname = self.get_df()
        if self.runParams['plot']:
            self.plot(df, cols='cond')
        return df, df_fname

    def get_data(self, df, kind='abs'):
        if kind == 'matrix':
            agg = stats.aggregate(df, values='subjResp', yerr='subjID',
                                  subplots='ROI', rows='stim1.cond',
                                  cols='stim2.cond')
        elif kind == 'corr':
            df['same_diff'] = 'between'
            df['same_diff'][df['stim1.cond'] == df['stim2.cond']] = 'within'
            agg = stats.aggregate(df, values='subjResp', yerr='subjID',
                                  rows=['ROI', 'stim1.cond'], cols='same_diff')
        else:
            agg = stats.aggregate(df, values='subjResp', yerr='subjID',
                                  rows='ROI', cols='stim1.cond')
        return agg
