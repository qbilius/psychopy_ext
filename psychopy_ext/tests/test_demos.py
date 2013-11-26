import sys, os, subprocess
import unittest

class TestExp(unittest.TestCase):
    def test_gui(self):
        subprocess.call(sys.executable + ' psychopy_ext/demos/run.py', shell=False)
        
    def test_exp(self):
        for name in ['main', 'twotasks', 'staircase', 'perclearn']:
            command = '%s psychopy_ext/demos/run.py %s exp run --n --debug --unittest' % (sys.executable, name)
            subprocess.call(command, shell=False)

if __name__ == '__main__': 
    unittest.main()
