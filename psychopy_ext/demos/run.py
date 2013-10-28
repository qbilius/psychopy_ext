#! /usr/bin/env python
import sys
try:
    import psychopy_ext
except:  # if psychopy_ext not in python path, e.g., you cloned the repo
    try:  # you want to start a new project based on demos
        sys.path.insert(0, '../psychopy_ext')
        import psychopy_ext
    except:  # you're just running demos
        sys.path.insert(0, '../../')
from psychopy_ext import ui

__author__ = "Jonas Kubilius"
__version__ = "0.1"
exp_choices = [
    ui.Choices('scripts.main', name='Simple exp.', alias='main', order=['exp','analysis']),
    ui.Choices('scripts.twotasks', name='Two tasks', order=['exp','analysis']),
    ui.Choices('scripts.staircase', name='Staircase', order=['exp','analysis']),
    ui.Choices('scripts.perclearn', name='Advanced', order=['exp','analysis']),
    ui.Choices('scripts.fmri', name='fMRI')
    ]

# bring up the graphic user interface or interpret command line inputs
# usually you can skip the size parameter
ui.Control(exp_choices, title='Demo Project', size=(580,530))
