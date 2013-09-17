"""
The configural superiority effect experiment

Minimal implementation just to show how easy it is to build a full experiment.
This demo is part of the psychopy_ext library.
"""

import numpy as np
import pandas

from psychopy_ext import exp, ui, stats, plot

# some modules are only available in Python 2.6
try:
    from collections import OrderedDict
except:
    from exp import OrderedDict

import computer  # for monitor size, paths etc settings across computers
PATHS = exp.set_paths('', computer)

class Confsup(exp.Experiment):
    """
    The configural superiority effect experiment

    Task:

    Indicate which shape is different. Use the numeric pad to respond:
        Top left: 4
        Top right: 5
        Bottom left: 1
        Bottom right: 2

    Please remember to fixate on the central dot.
    Please press spacebar to begin.
    (Use Left Shift + Esc to exit.)
    """
    def __init__(self, name='exp', info=('subjid', 'confsup_'), **kwargs):
        # initialize the default Experiment class with our parameters
        super(Confsup, self).__init__(name=name, info=info, method='random',
                                      computer=computer, **kwargs)
        self.paths = PATHS
        self.computer = computer

        self.computer.valid_responses = {'num_4': 0, 'num_5': 1, 'num_1': 2, 'num_2': 3}
        self.stim_size = 3.  # in deg
        self.stim_width = .3  # px; the weight of the line
        self.stim_dist = 4.  # from the fixation in x or y dir
        self.stim_color = 'black'
        self.nreps = 5  # number of trials per condition per position

        self.paratable = OrderedDict([
            # condition 0 is reserved for fixation
            (1, ['parts', 'top left']),
            (2, ['parts', 'top right']),
            (3, ['parts', 'bottom left']),
            (4, ['parts', 'bottom right']),
            (5, ['whole', 'top left']),
            (6, ['whole', 'top right']),
            (7, ['whole', 'bottom left']),
            (8, ['whole', 'bottom right'])
            ])
        sh = self.stim_dist
        self.pos = [(-sh, sh),  # top left
                    (sh, sh),  # top right
                    (-sh, -sh),  # bottom left
                    (sh, -sh)]  # bottom right

    def create_stimuli(self):
        """Define stimuli
        """
        self.create_fixation()
        sh = self.stim_size/2
        diag45 = exp.ThickShapeStim(
            self.win,
            lineColor = self.stim_color,
            lineWidth = self.stim_width,
            fillColor = self.stim_color,
            closeShape = False,
            vertices = [[-sh, -sh], [sh, sh]]
            )
        diag135 = exp.ThickShapeStim(
            self.win,
            lineColor = self.stim_color,
            lineWidth = self.stim_width,
            fillColor = self.stim_color,
            closeShape = False,
            vertices = [[-sh, sh], [sh, -sh]]
            )
        corner = exp.ThickShapeStim(
            self.win,
            lineColor = self.stim_color,
            lineWidth = self.stim_width,
            fillColor = None,
            closeShape = False,
            vertices = [[-sh, sh], [-sh, -sh], [sh, -sh]]
            )

        self.s = {
            'fix': self.fixation,
            'parts': exp.GroupStim(stimuli=diag45, name='parts'),
            'parts_odd': exp.GroupStim(stimuli=diag135, name='parts_odd'),
            'whole': exp.GroupStim(stimuli=[corner, diag45],
                                   name='whole'),  # arrow
            'whole_odd': exp.GroupStim(stimuli=[corner, diag135],
                                       name='whole_odd')  # triangle
            }

    def create_trial(self):
        """Create trial structure
        """
        self.trial = [{'dur': 0.300,  # in seconds
                       'display': self.s['fix'],
                       'func': self.idle_event},

                      {'dur': 0,  # this means present until response
                       'display': None,  # `set_stimuli` will select stimuli
                       'func': self.during_trial},

                      {'dur': .300,
                       'display': self.s['fix'],
                       'func': self.feedback}
                     ]

    def create_trial_list(self):
        """Define each trial's parameters
        """
        self.trial_dur = sum(event['dur'] for event in self.trial)
        exp_plan = []
        for rep in range(self.nreps):  # repeat the defined number of times
            for cond, (context, posname) in self.paratable.items():
                pos = (cond - 1) % 4
                exp_plan.append(OrderedDict([
                    ('cond', cond),
                    ('context', context),
                    ('posname', posname),
                    ('pos', pos),
                    ('onset', ''),
                    ('dur', self.trial_dur),
                    ('corr_resp', pos),
                    ('subj_resp', ''),
                    ('accuracy', ''),
                    ('rt', ''),
                    ]))
        self.trialList = exp_plan

    def draw_stimuli(self, this_trial, *args):
        """
        Fully prepare the display but don't flip yet:
            - Determine which context is shown (parts or whole)
            - Set positions of all stimuli.
        This will be invoked at the beginning of each trial for the stimulus
        presentation events (called by :func:`during_trial`).
        """
        # first draw regular stimuli
        odd_pos = this_trial['pos']
        stim = self.s[this_trial['context']]
        for pos in range(4):
            if pos != odd_pos:
                stim.setPos(self.pos[pos])
                stim.draw()  # draw now because in the next iteration we'll change pos

        # now draw the odd (target) stimulus
        stim = self.s[this_trial['context'] + '_odd']
        stim.setPos(self.pos[odd_pos])
        stim.draw()

        # finally, draw the fixation
        self.s['fix'].draw()

    def during_trial(self, trial_clock=None, this_trial=None, *args, **kwargs):
        self.draw_stimuli(this_trial)
        self.win.flip()
        if self.rp['autorun']:
            event_keys = self.wait_for_response(rt_clock = trial_clock,
                fake_key=[this_trial['autoresp'],this_trial['autort']])
        else:
            event_keys = self.wait_for_response(rt_clock = trial_clock)
        return event_keys


class Analysis(object):
    def __init__(self, name='analysis', info=('subjid', 'confsup_')):
        self.name = name
        self.info = info
        self.exp = exp.Experiment(info=self.info, rp=self.rp)

    def behav(self):
        pattern = PATHS['data'] + '%s.csv'
        df = self.exp.get_behav_df(pattern=pattern)
        agg_acc = stats.accuracy(df, cols='context', values='accuracy', yerr='subjid')
        agg_rt = stats.aggregate(df[df.accuracy=='correct'], cols='context',
                                 values='rt', yerr='subjid')

        plt = plot.Plot(ncols=2)
        plt.plot(agg_acc, kind='bar')
        plt.plot(agg_rt, kind='bar')
        plt.show()


if __name__ == '__main__':
    ui.Control(__name__)