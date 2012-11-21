from scripts.core.ui import Control

exp_choices = [
    ('Experiment 1', 'scripts.main'),
    ]
ctrl = Control()
ctrl.app(exp_choices, title='My experiment', exp_parent=__name__)


