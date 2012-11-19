from scripts.gui import Control

exp_choices = [
    ('Experiment 1', 'scripts.exp1'),
    ('Experiment 2', 'scripts.exp2'),
    ]
ctrl = Control()
ctrl.app(exp_choices, title='My experiment', exp_parent=__name__)


