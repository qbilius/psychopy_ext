from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import unittest

import numpy as np
import skimage

from psychopy_ext import models, utils


class TestOutput(unittest.TestCase):

    def read_orig(self, m, flatten=False, suffix=''):
        pic = 'cat-gray' if flatten else 'cat'
        path = 'psychopy_ext/tests/{}_{}{}.txt'.format(pic, m.safename, suffix)
        resps_orig = np.loadtxt(path, delimiter=',').ravel()
        return resps_orig

    def run_model(self, name, im=None, flatten=False, size=None,
                  layers='output', suffix=''):
        if im is None:
            if flatten:
                im = 'cat-gray-32x32' if size == 32 else 'cat-gray'
            else:
                im = 'cat-32x32' if size == 32 else 'cat'
            im = 'psychopy_ext/tests/%s.png' % im

        if isinstance(name, (unicode, str)):
            m = models.Model(name)
        else:
            m = name
        resps_test = m.run(im, layers=layers, return_dict=False)
        resps_test = np.around(resps_test, decimals=5)
        resps_orig = self.read_orig(m, flatten=flatten, suffix=suffix)
        if name == 'hmax_pnas':
            rms = np.mean(np.sqrt((np.sort(resps_test) - np.sort(resps_orig))**2))
        else:
            rms = np.mean(np.sqrt((resps_test - resps_orig)**2))
        self.assertAlmostEqual(rms, 0)

    def test_px(self):
        self.run_model('px', flatten=False, size=32)

    def test_gaborjet_mag(self):
        stim = utils.load_image('psychopy_ext/tests/cat-gray.png')
        im = np.array([skimage.img_as_ubyte(stim)]).astype(float)
        self.run_model('gaborjet', im, flatten=True, layers='magnitudes', suffix='-mag-matlab')

    def test_gaborjet_mag(self):
        stim = utils.load_image('psychopy_ext/tests/cat-gray.png')
        im = np.array([skimage.img_as_ubyte(stim)]).astype(float)
        self.run_model('gaborjet', im, layers='phases', flatten=True, suffix='-phase-matlab')

    def test_hmax99_gabor(self):
        stim = utils.load_image('psychopy_ext/tests/cat-gray.png')
        im = np.array([skimage.img_as_ubyte(stim)]).astype(float)
        m = models.HMAX99(matlab=True, filter_type='gabor')
        self.run_model(m, im, flatten=True, suffix='-gabor-matlab')

    def test_hog(self):
        self.run_model('hog', flatten=True, size=32)

    def test_caffenet(self):
        self.run_model('caffenet', flatten=False, size=None)

    def test_hmax_hmin(self):
        self.run_model('hmax_hmin', flatten=True, size=None)

    def test_hmax_pnas(self):
        self.run_model('hmax_pnas', flatten=False, size=None)

    def test_phog(self):
        self.run_model('phog', flatten=False, size=None)

    def test_phow(self):
        self.run_model('phow', flatten=False, size=None)

    def test_randfilt(self):
        self.run_model('randfilt', flatten=True, size=None)


if __name__ == '__main__':
    unittest.main()
