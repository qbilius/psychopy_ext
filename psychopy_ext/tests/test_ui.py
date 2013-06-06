import numpy as np
from .. import ui

import unittest

class TestBase(unittest.TestCase):

    def test_input(self):
        sys.argv = ['run.py', '--commit']
        exp_choices = None
        control = ui.Control(exp_choices)


