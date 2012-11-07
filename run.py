from scripts.core.gui import Control

exp_choices = [
    ('Experiment 1 (fMRI)', 'scripts.main.fmri'),
    ('Experiment 2 (behavioral)', 'scripts.main.behav4q'),
    ('Experiment 2 (fMRI)', 'scripts.main.fmri2'),
    #('Experiment 2 (localizer)', 'scripts.loc.loc'),
    ('Experiment 2 (outside scanner behavioral)', 'scripts.main.fmri2_behav'),
    ]
ctrl = Control()
ctrl.app(exp_choices, title='Two lines experiment', exp_parent=__name__)


