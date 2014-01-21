"""
The configural superiority effect experiment

A full implementation with many option exposed. For a minimal implementation,
see minimal.py.

This demo is part of psychopy_ext library.
"""

#import sys
#import cPickle as pickle

#from psychopy import core
import numpy as np
# pandas does not come by default with PsychoPy but that should not prevent
# people from at least running the experiment
try:
    import pandas
except:
    pass

from psychopy import visual
from psychopy_ext import exp, stats, plot

# some modules are only available in Python 2.6
try:
    from collections import OrderedDict
except:
    from exp import OrderedDict

import computer  # for monitor size, paths etc settings across computers
# set up where all data, logs etc are stored for this experiment
# for a single experiment, '' is fine -- it means data is stored in the 'data'
# folder where the 'run.py' file is, for example
# if you have more than one experiment, 'confsup' would be better -- data for
# this experiment will be in the 'data' folder inside the 'confsup' folder
PATHS = exp.set_paths(exp_root='mouse_resp', computer=computer)

class Confsup(exp.Experiment):
    """
    The configural superiority effect experiment
    ============================================

    Task
    ----

    Indicate which shape is different by clicking on it!

    Please remember to fixate on the central dot.

    **Please press spacebar to begin.**

    *(Use Left Shift + Esc to exit.)*
    """
    def __init__(self,
            name='exp',
            info=OrderedDict([  # these get printed in the output data file
                ('subjid', 'confsup_')
                ]),
            rp=OrderedDict([  # these control how the experiment is run
                ('no_output', False),  # do you want output? or just playing around?
                ('debug', False),  # not fullscreen presentation etc
                ('autorun', 0),  # if >0, will autorun at the specified speed
                ('register', False),  # add and commit changes, like new data files?
                ('push', False),  # add, commit and push to a hg repo?
                ]),
            actions='run',
            ):
        # initialize the default Experiment class with our parameters
        super(Confsup, self).__init__(
            name=name,
            info=info,
            rp=rp,
            actions=actions,
            method='random',  # order of trials; check `psychopy.TrialHandler` for acceptable formats
            computer=computer,
            paths=PATHS,
            blockcol='rep',  # experiment will be divided into blocks
            #dataFilename=PATHS['data'] + info['subjid'] + '.csv'
            )
        self.computer.trigger = 'left-click'
        self.computer.default_keys['trigger'] = self.computer.trigger
        self.computer.valid_responses = {'left-click': 1}
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
        self.create_respmap()
    
    def create_respmap(self):
        self.respmap = []
        qbs = (self.stim_dist + self.stim_width / 2) * 1.5  # a bit larger than stimulus
        qverts = [ [[-qbs, qbs], [0, qbs], [0, 0], [-qbs, 0]],
                  [[qbs, qbs], [0, qbs], [0, 0], [qbs, 0]],
                  [[-qbs, -qbs], [0, -qbs], [0, 0], [-qbs, 0]],
                  [[qbs, -qbs], [0, -qbs], [0, 0], [qbs, 0]]
                ]
        for qvertno, qvert in enumerate(qverts):
            box = visual.ShapeStim(
                self.win,
                name=qvertno,
                lineColor = None,
                fillColor = None,
                vertices = qvert
                )
            self.respmap.append(box)
            
    def create_trial(self):
        """Create trial structure
        """
        self.trial = [exp.Event(self,
                                dur=0.300,  # in seconds
                                display=self.s['fix'],
                                func=self.idle_event),
                      exp.Event(self,
                                dur=0,  # this means present until response
                                display=None,  # we'll select which condition to
                                               # present during the runtime with
                                               # :func:`set_stimuli`
                                func=self.show_stim),
                      exp.Event(self,
                                dur=.300,
                                display=self.s['fix'],
                                func=self.feedback)
                     ]

    def create_exp_plan(self):
        """Define each trial's parameters
        """
        exp_plan = []
        for rep in range(self.nreps):  # repeat the defined number of times
            for cond, (context, posname) in self.paratable.items():
                pos = (cond - 1) % 4
                exp_plan.append(OrderedDict([
                    ('rep', rep),
                    ('cond', cond),
                    ('context', context),
                    ('posname', posname),
                    ('pos', pos),
                    ('onset', ''),
                    ('dur', ''),
                    ('corr_resp', pos),
                    ('subj_resp', ''),
                    ('accuracy', ''),
                    ('rt', ''),
                    ]))
        self.exp_plan = exp_plan

    def set_autorun(self, exp_plan):
        def rt(mean, trialno):
            add = np.random.normal(mean,scale=.2) / self.rp['autorun']
            return self.trial[0].dur + add

        invert_resp = exp.invert_dict(self.computer.valid_responses)

        for trialno, trial in enumerate(exp_plan):
            if trial['context'] == 'parts':
                acc = [.1,.1,.1,.1]
                acc[trial['pos']] = .7
                resp = exp.weighted_choice(choices=invert_resp, weights=acc)
                trial['autoresp'] = resp  # poor accuracy
                trial['autort'] = rt(1., trialno)  # slow responses
            elif trial['context'] == 'whole':  # lower accuracy for morphed
                acc = [.05,.05,.05,.05]
                acc[trial['pos']] = .85
                resp = exp.weighted_choice(choices=invert_resp, weights=acc)
                trial['autoresp'] = resp  # good accuracy
                trial['autort'] = rt(.8, trialno)  # fast responses
        return exp_plan

    def show_stim(self):
        """
        Fully prepare the display but don't flip yet:
            - Determine which context is shown (parts or whole)
            - Set positions of all stimuli.
        This will be invoked at the beginning of each trial for the stimulus
        presentation events (called by :func:`during_trial`).
        #"""
        ## draw response boxes
        #for qbox in self.s['quadboxes']:
            #qbox.draw()
        # first draw regular stimuli
        odd_pos = self.this_trial['pos']
        stim = self.s[self.this_trial['context']]
        for pos in range(4):
            if pos != odd_pos:
                stim.setPos(self.pos[pos])
                stim.draw()  # draw now because in the next iteration we'll change pos

        # now draw the odd (target) stimulus
        stim = self.s[self.this_trial['context'] + '_odd']
        stim.setPos(self.pos[odd_pos])
        stim.draw()

        # finally, draw the fixation
        self.s['fix'].draw()
        self.win.flip()
        event_keys = self.wait_until_response(draw_stim=False)
        return event_keys


class Analysis(object):
    def __init__(self,
                 name='analysis',
                 info=OrderedDict([('subjid', 'confsup_')]),
                 rp=OrderedDict([('no_output', False),
                                 ('all', False),
                                ])
                 ):
        self.name = name
        self.info = info
        self.rp = rp
        if self.rp['all']:
            self._set_all_subj()
        self.exp = exp.Experiment(info=self.info, rp=self.rp)

    def _set_all_subj(self):
        self.info['subjid'] = ['subj_%02d' % i for i in range(1,9)]

    def run(self):
        pattern = PATHS['data'] + '%s.csv'
        df = self.exp.get_behav_df(pattern=pattern)
        agg_acc = stats.accuracy(df, cols='context', values='accuracy',
                                 yerr='subjid', order='sorted')
        agg_rt = stats.aggregate(df[df.accuracy=='correct'], cols='context',
                                 values='rt', yerr='subjid', order='sorted')

        plt = plot.Plot(ncols=2)
        if len(df.subjid.unique()) == 1:
            kind = 'bar'
        else:
            kind = 'bean'
        plt.plot(agg_acc, kind=kind, title='accuracy', ylabel='% correct')
        plt.plot(agg_rt, kind=kind, title='response time', ylabel='seconds')

        print agg_acc
        print agg_rt
        plt.show()
