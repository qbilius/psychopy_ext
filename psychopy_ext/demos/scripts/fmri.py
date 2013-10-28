import sys, os, shutil, urllib2, tarfile, glob
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

# store fmri data in the Download folder in your home folder
computer.root = os.path.join(os.path.expanduser('~'), 'Downloads', 'fmri_demo/')
PATHS = exp.set_paths('fmri', computer, fmri_rel='%s/')

class Analysis(fmri.Analysis):
    def __init__(self,
                 name='analysis',
                 info=OrderedDict([
                    ('subjid', 'subj_'),
                    ('runtype', 'main'),
                    ('runno', 1),
                    ]),
                 rp=OrderedDict([
                    ('no_output', False),
                    ('force', False),
                    ('all', True),
                    ('rois', [(['rh_V1d','rh_V1v'],'V1'), 'rh_LO']),
                    ('reuserois', False),
                    ('method', ('timecourse', 'corr', 'svm')),
                    ('values', 'raw'),
                    ('plot', True),
                    ]),
                 actions=['prepare_data', 'run']
                 ):
        super(Analysis, self).__init__(PATHS, tr=2,
            info=info, rp=rp, fmri_prefix='swa*',
            fix=0, dur=None, offset=0)
        if self.rp['values'] == 'raw':
            if self.rp['method'] == 'timecourse':
                self.offset = 0
                self.dur = 7
            else:
                self.offset = 3
                self.dur = 3
        self.name = name
        self.actions = actions
        if self.rp['all']: self._set_all_subj()

    def _set_all_subj(self):
        self.info['subjid'] = ['subj_01', 'subj_02']

    def _download_data(self):
        """From http://stackoverflow.com/a/22776
        """
        url = 'http://download.klab.lt/fmri-demo.php'
        file_name = url.split('/')[-1].split('.')[0]
        file_name = os.path.join(self.paths['root'], file_name) + '.tar.gz'
        u = urllib2.urlopen(url)
        try: # if this fails (e.g. permissions) we will get an error
            os.makedirs(self.paths['root'])
        except:
            raise Exception('Path for data (%s) already exists. So if the data '
                            'is already there, proceed to run the analysis.' % self.paths['root'])

        f = open(file_name, 'wb')
        meta = u.info()
        file_size = int(meta.getheaders("Content-Length")[0])
        print 'downloading fmri data (%s bytes) to %s' % (file_size, self.paths['root'])

        file_size_dl = 0
        block_sz = 32*1024
        while True:
            buffer = u.read(block_sz)
            if not buffer:
                break

            file_size_dl += len(buffer)
            f.write(buffer)
            status = r"%10d  [%3.2f%%]" % (file_size_dl, file_size_dl * 100. / file_size)
            status = status + chr(8)*(len(status)+1)
            print status,

        f.close()

        # untar data
        try:
            tfile = tarfile.open(file_name)
            print
            print 'extracting...'
            tfile.extractall(self.paths['root'])
        except:
            raise Exception('Cannot unpack tutorial data. Try downloading '
                            'it again using the same command.')

    def prepare_data(self):
        """
        Renames data files from PyMVPA2 in the expected format.
        """
        nchunks=2

        print
        self._download_data()
        print 'restructuring...'

        ds_path = os.path.join(self.paths['root'], 'bold.nii')
        ds = mvpa2.suite.fmri_dataset(samples=ds_path)

        for subjid in self.info['subjid']:
            print subjid,
            try:
                os.makedirs(self.paths['data_behav'] % subjid)
                os.makedirs(self.paths['data_fmri'] % subjid)
                os.makedirs(self.paths['rois'] % subjid)
            except:
                pass

            for chunk in range(nchunks):
                # add a bit of noise so that participants are not identical
                noise = (np.random.random_sample(ds.samples.shape) *
                         np.mean(ds.samples) / 10)

                fname = 'swafunc_%02d_%s.nii' % (chunk + 1, self.info['runtype'])
                ds.samples += noise
                nb.save(mvpa2.suite.map2nifti(ds),
                        self.paths['data_fmri'] % subjid + fname)
                df = pandas.read_csv(os.path.join(self.paths['root'], 'behav.csv'))
                df.subjid = subjid
                df.runtype = self.info['runtype']
                df.runno = chunk + 1
                df.rt += np.random.random_sample(len(df)) * np.mean(df.rt) / 10
                df.to_csv(self.paths['data_behav'] % subjid + 'data_%02d_%s.csv' %
                          (chunk + 1, self.info['runtype']), index=False,
                          float_format='%.3f')

            rois = glob.glob(self.paths['root'] + 'rh_*.nii')

            for roi in rois:
                shutil.copy(roi, self.paths['rois'] % subjid + os.path.basename(roi))

        # cleanup
        os.remove(self.paths['root'] + 'fmri-demo.tar.gz')
        os.remove(self.paths['root'] + 'behav.csv')
        os.remove(self.paths['root'] + 'bold.nii')
        for roi in rois:
            os.remove(roi)
        print
        print 'DONE'

    def run(self):
        if not os.path.isdir(self.paths['root']) or len(os.listdir(self.paths['root'])) == 0:
            raise Exception('You must get the data first using "prepare_data"')
        else:
            super(Analysis, self).run()

    def get_agg(self, df, kind=None):
        if kind == 'matrix':
            agg = stats.aggregate(df, values='subj_resp', yerr='subjid',
                                  subplots='roi', rows='stim1.cond',
                                  cols='stim2.cond')
        elif self.rp['method'] in ['corr', 'svm']:
            df['same_diff'] = 'between'
            df['same_diff'][df['stim1.cond'] == df['stim2.cond']] = 'within'
            agg = stats.aggregate(df, values='subj_resp', yerr='subjid',
                                  rows='roi', cols='same_diff')
        else:
            agg = stats.aggregate(df, values='subj_resp', yerr='subjid',
                                  rows='roi', cols='cond')
        return agg
