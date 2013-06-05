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
from psychopy_ext.ui import Control

__author__ = "Jonas Kubilius"
__version__ = "0.1"
exp_choices = [
    ('Experiment1',  # experiment name, optional
     'scripts.main',  # path to the experimental script
     'main',  # alias for calling it via the command-line inteface, optional
     ['exp','analysis']),  # order of classes in the experimental script, optional
     ('fMRI', 'scripts.fmri', 'fmri')
    ]
# other exp_choices formats also accepted:
# exp_choices = 'scripts.main'  # if you've got a single experiment
# exp_choices = scripts.main  # but then `scripts` must be imported
# exp_choices = [('Exp', 'scripts.main')]  # don't care about alias (inferred as 'main')
# exp_choices = [('Exp', 'scripts.main', 'main')]  # don't care about class order (default: alphabetical)
# exp_choices = [('Localizer', 'scripts.loc'), ('scripts.main')]  # for 2 experiments

# bring up the graphic user interface or interpret command line inputs
Control(exp_choices, title='Configural superiority experiment')