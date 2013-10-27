"""
An simple example of an experiment with two tasks.

This demo is part of psychopy_ext library.
"""

import sys
import cPickle as pickle

from psychopy import core, visual, data, event
import numpy as np
# pandas does not come by default with PsychoPy but that should not prevent
# people from running the experiment
try:
    import pandas
except:
    pass

from psychopy_ext import exp, ui, stats, plot

# some modules are only available in Python 2.6
try:
    from collections import OrderedDict
except:
    from exp import OrderedDict

import computer
PATHS = exp.set_paths('twotasks', computer)


class TwoTasks(exp.Experiment):
    """
    A perceptual learning experiment
    ================================

    This experiment is composed of four parts. Each part consists of
    **training** for 1 min (for this demo purposes only) and 
    **testing** for 5 min.

    **Press spacebar to continue.**
    
    *(Use 'Left Shift + Esc' to exit.)*
    """
    def __init__(self,
        name='exp',
        info=('subjID', 'twotasks_'),
        rp=('phase', 'both')
        ):
        super(TwoTasks, self).__init__(name=name, info=info, rp=rp,
                actions='run', paths=PATHS, computer=computer)
            
        self.nsessions = 3 # number of sessions
        self.tasks = [_Train, _Test]
        
        # stimuli orientation
        self.oris = {'attended': 60, 'unattended': 150}        
        self.stim_size = 3.  # in deg
        self.stim_dist = 4.  # from the fixation in polar coords
        
    def run(self):
        """
        This is a more complicated run method than usually because
        I wanted to demonstrate how to have a few sessions of the
        same few tasks. If you don't need this, you can simply comment
        out this method and it will still run one session (note that
        ``phase argument will no longer work``).
        """
        self.setup()
        self.before_exp()
        for expno in range(self.nsessions):
            self.show_text('Session %d' % (expno+1), auto=1)
            if self.rp['phase'] in ['train','both']:
                self.tasks[0](self, session=expno+1).run_task()
            if self.rp['phase'] in ['test', 'both']:
                self.tasks[1](self, session=expno+1).run_task()
            pause_instr = ('End of session %d.\n'
                           'When ready, press spacebar to continue.'
                            % (expno+1))
            if expno != self.nsessions - 1:
                self.show_text(pause_instr)
        self.after_exp()
        
        if self.rp['register']:
            self.register()
        elif self.rp['push']:
            self.commitpush()
        self.quit()

class _Train(exp.Task):
    """
    Training
    ========
    
    Your task
    ---------

    **Fixate** on the central fixation spot.
    Attend to stimulus on the **left**.
    When you notice a decrease in its contrast, **press 'j'**.
    Please remember to fixate throughout the experiment.

    **Press spacebar to begin.**

    *(Use 'Left Shift + Esc' to exit.)*
    """

    def __init__(self, parent, session=1):
            
        data_fname = parent.paths['data'] + parent.info['subjid'] + '_train.csv'        
        super(_Train, self).__init__(
            parent,
            method='random',
            data_fname=data_fname
            )
        self.session = session
        self.computer.valid_responses = {'j': 1}
        
        self.oris = parent.oris
        self.stim_size = parent.stim_size
        self.stim_dist = parent.stim_dist

        self.ntrials = 75  # number of trials per session
        self.nblocks = 2
        self.task_rate = .1  # how often lower contrast stimulus should come up
        self.low_contrast = .5

    def create_stimuli(self):
        # Define stimuli
        self.create_fixation()
        self.s = {'fix': self.fixation}
        self.s['attended'] = visual.GratingStim(self.win, name='attended', 
                                mask='circle', sf=2, size=self.stim_size,
                                pos=(-self.stim_dist,0),
                                ori=self.oris['attended'])
        self.s['unattended'] = visual.GratingStim(self.win, name='unattended', 
                                mask='circle', sf=2, size=self.stim_size,
                                pos=(self.stim_dist,0),
                                ori=self.oris['unattended'])

    def create_trial(self):
        """Create trial structure
        """
        allstim = [self.s['fix'], self.s['attended'], self.s['unattended']]

        self.trial = [exp.Event(self,
                                dur=.200,  # in seconds
                                display=allstim,
                                func=self.show_stim),
                      exp.Event(self,
                                dur=.200,
                                display=self.s['fix'],
                                func=self.idle_event)
                     ]

    def create_exp_plan(self):
        """Define each trial's parameters
        """
        exp_plan = []
        for block in range(self.nblocks):
            for trial in range(self.ntrials):  # repeat the defined number of times
                if trial < self.ntrials * self.task_rate:
                    cond = 'low'
                    corr_resp = 1
                else:
                    cond = 'high'
                    corr_resp = ''
                exp_plan.append(OrderedDict([
                    ('session', self.session),
                    ('block', block),
                    ('cond', cond),
                    ('onset', ''),
                    ('dur', ''),
                    ('corr_resp', corr_resp),
                    ('subj_resp', ''),
                    ('rt', ''),
                    ]))
        self.exp_plan = exp_plan

    def show_stim(self):
        if self.this_trial['cond'] == 'low':
            contrast = self.low_contrast
        else:
            contrast = 1
        for stim in self.this_event.display:
            if stim.name == 'attended':
                stim.setContrast(contrast)
            stim.draw()
        self.win.flip()

        event_keys = self.idle_event(draw_stim=False)
        
        return event_keys

    def post_trial(self, this_trial, all_keys):
        """ What to do after a trial is over.
        """
        if len(all_keys) > 0:
            if all_keys[0][0] != '':
                this_trial['subj_resp'] = len(all_keys)
            else:
                this_trial['subj_resp'] = ''
            this_trial['rt'] = all_keys[-1][1]
        else:
            this_trial['subj_resp'] = ''
            this_trial['rt'] = ''

        return this_trial
        
    def before_task(self):
        """We slightly redefine the default function so that full
        instructions are shown the first time round.
        """
        if self.session == 1:
            super(_Train, self).before_task()
        else:
            text = '''
            Training, session %d
            --------------------
            
            (decrease in its contrast: **press 'j'**)
            '''
            super(_Train, self).before_task(text=text % self.session)            
            
            
class _Test(exp.Task):
    """
    Testing
    =======
    
    Your task
    ---------

    **Fixate** on the central fixation spot. Two stimuli will briefly flash
    in a row.
    If the second stimulus is oriented to the left with respect to the
    first one, **hit 'f'**.
    If the second stimulus is oriented to the right with respect to the
    first one, **hit 'j'**.
    Please remember to fixate throughout the experiment!

    **Press spacebar to begin.**

    *(Use 'Left Shift + Esc' to exit.)*
    """
    def __init__(self, parent, session=1):
        data_fname = (parent.paths['data'] + parent.info['subjid'] + '_test.csv')
        super(_Test, self).__init__(
            parent,
            method='random',
            blockcol = 'pos',
            data_fname=data_fname 
            )        
        self.session = session
        self.oris = parent.oris
        self.stim_size = parent.stim_size
        self.stim_dist = parent.stim_dist
        
        self.ntrials = 32  # must be a multiple of 4; very short, just for demo
        self.oridiff = 13

        self.computer.valid_responses = {'f': 'same', 'j': 'diff'}
        
        self.anl = Analysis(info=self.parent.info)

    def create_stimuli(self):
        self.create_fixation()
        stim1 = visual.GratingStim(self.win, name='stim1', mask='circle',
                                        sf=2, size=self.stim_size)
        stim2 = visual.GratingStim(self.win, name='stim2', mask='circle',
                                        sf=2, size=self.stim_size)
        self.s = {'fix': self.fixation, 'stim1': stim1, 'stim2': stim2}

    def create_trial(self):
        """Create trial structure
        """
        self.trial = [exp.Event(self,
                                dur=.200,
                                display=self.s['fix'],
                                func=self.idle_event),
                      exp.Event(self,
                                dur=.300,  # in seconds
                                display=[self.s['fix'], self.s['stim1']],
                                func=self.show_stim),
                      exp.Event(self,
                                dur=.600,
                                display=self.s['fix'],
                                func=self.idle_event),
                      exp.Event(self,
                                dur=.300,  # in seconds
                                display=[self.s['fix'], self.s['stim2']],
                                func=self.show_stim),
                      exp.Event(self,
                                dur=0,
                                display=self.s['fix'],
                                func=self.wait_until_response)
                     ]
                
    def create_exp_plan(self):
        exp_plan = []
        for name in self.oris.keys():
            for ori_dir in [-1,1]:
                for trial in range(self.ntrials/2):
                    if trial < self.ntrials/4:
                        corr_resp = 'same'
                        oridiff = 0
                    else:
                        corr_resp = 'diff'
                        oridiff = self.oridiff
                    exp_plan.append(OrderedDict([
                        ('session', self.session),
                        ('pos', name),
                        ('dir', ori_dir),
                        ('oridiff', oridiff),
                        ('onset', ''),
                        ('dur', ''),
                        ('corr_resp', corr_resp),
                        ('subj_resp', ''),
                        ('accuracy', ''),
                        ('rt', ''),
                        ]))
        self.exp_plan = exp_plan

    def set_autorun(self, exp_plan):
        def rt(mean):
            add = np.random.normal(mean,scale=.2) / self.rp['autorun']
            return trial_dur + add

        invert_resp = exp.invert_dict(self.computer.valid_responses)
        trial_dur = sum([ev.dur for ev in self.trial])

        for trial in exp_plan:
            if trial['pos'] == 'attended':
                acc = .9
            elif trial['pos'] == 'unattended':
                acc = .6
                
            if trial['corr_resp'] == 'same':
                resp_ind = exp.weighted_choice(choices=invert_resp.keys(),
                                                weights=[1-acc, acc])
            else:
                resp_ind = exp.weighted_choice(choices=invert_resp.keys(),
                                                weights=[acc, 1-acc])
            trial['autoresp'] = invert_resp[resp_ind]
            trial['autort'] = rt(.8)
        return exp_plan

    def show_stim(self):
        if self.this_trial['pos'] == 'attended':
            pos = (-self.stim_dist, 0)
        else:
            pos = (self.stim_dist, 0)

        for stim in self.this_event.display:
            if stim.name == 'stim1':
                stim.setPos(pos)
                stim.setOri(self.oris[self.this_trial['pos']])
            if stim.name == 'stim2':
                stim.setPos(pos)
                stim.setOri(self.oris[self.this_trial['pos']] + 
                            self.this_trial['dir']*self.this_trial['oridiff'])
            stim.draw()
        self.win.flip()
        
        self.idle_event(draw_stim=False)
        
    def before_task(self):
        """We slightly redefine the default function so that full
        instructions are shown the first time round.
        """
        if self.session == 1:
            super(_Test, self).before_task()
        else:
            text = '''
            Testing, session %d
            -------------------
            
            (Second stimulus oriented to the:
            
            - left: **hit 'f'**
            - right: **hit 'j'**)
            '''
            super(_Test, self).before_task(text=text % self.session)
            
    def after_task(self):
        acc = self.anl.test_feedback(self.exp_plan)
        pause_instr = 'Your accuracy is %d%%.' % acc
        super(_Test, self).after_task(text=pause_instr)
            
            
class Analysis(object):
    def __init__(self,
                 name='analysis',
                 info={'subjid': 'twotasks_'},
                 rp={'all': False},
                 actions='test'
                 ):
        self.name = name
        self.info = info
        self.rp = rp
        self.paths = PATHS

        if self.rp['all']:
            self._set_all_subj()

    def _set_all_subj(self):
        self.info['subjid'] = ['twotasks_%02d' % i for i in range(1,11)]
        
    def test_feedback(self, trial_list):
        """Provides feedback during the test phase
        """
        df = pandas.DataFrame(trial_list)
        acc = float(np.sum(df.accuracy == 'correct'))
        acc /= (np.sum(df.accuracy == 'correct') + \
                np.sum(df.accuracy == 'incorrect'))
        return acc*100
        
    def test(self):
        """Analysis of the test phase data
        """
        pattern = self.paths['data'] + '%s_test.csv'
        df = exp.get_behav_df(self.info['subjid'], pattern=pattern)
        agg_acc = stats.accuracy(df, rows='session', cols='pos',
            values='accuracy', yerr='subjid')
        plt = plot.Plot()
        plt.plot(agg_acc)
        plt.show()
