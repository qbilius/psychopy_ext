import numpy as np
import scripts.core.models as models

import unittest

class TestHMAX(unittest.TestCase):
    def test_gaussian(self):
        m = models.HMAX(matlab=True, filt_type='gaussian')
        out = m.run()
        fid = open('scripts/tests/lena_gaussian_matlab.txt')
        c2_matlab = np.array([float(i.strip('\n')) for i in fid.readlines()])
        c2_python = np.around(out['C2'], decimals=5)  # matlab's output has 5 
                                                      # significant digits
        rsm = np.mean(np.sqrt((c2_matlab - c2_python)**2))
        self.assertEqual(rsm, 0)

    def test_gabor(self):
        m = models.HMAX(matlab=False, filt_type='gabor')
        out = m.run()
        fid = open('scripts/tests/lena_gabor_matlab.txt')
        c2_matlab = np.array([float(i.strip('\n')) for i in fid.readlines()])
        c2_python = np.around(out['C2'], decimals=5)  # matlab's output has 5 
                                                      # significant digits
        rsm = np.mean(np.sqrt((c2_matlab - c2_python)**2))
        self.assertEqual(rsm, 0)

if __name__ == '__main__':
    unittest.main()