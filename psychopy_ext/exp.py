#!/usr/bin/env python

# Part of the psychopy_ext library
# Copyright 2010-2013 Jonas Kubilius
# The program is distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.

"""
A library of helper functions for creating and running experiments.

All experiment-related methods are kept here.
"""

import sys, os, csv, glob, random, warnings
from UserDict import DictMixin

import numpy as np
import wx
import psychopy.info
from psychopy import visual, core, event, logging, misc, monitors
from psychopy.data import TrialHandler, ExperimentHandler

import ui

# pandas does not come by default with PsychoPy but that should not prevent
# people from running the experiment
try:
    import pandas
except:
    pass

class default_computer:

    """The default computer parameters. Hopefully will form a full class at
    some point.
    """
    recognized = False
    # computer defaults
    root = '.'  # means store output files here
    stereo = False  # not like in Psychopy; this merely creates two Windows
    default_keys = {'exit': ('lshift', 'escape'),
                    'trigger': 'space'}  # "special" keys
    valid_responses = {'f': 0, 'j': 1}  # organized as input value: output value
    # monitor defaults
    name = 'default'
    distance = 80
    width = 37.5
    # window defaults
    screen = 0  # default screen is 0
    view_scale = [1,1]

    def __init__(self):
        pass

def set_paths(exp_root='.', computer=default_computer, fmri_rel=''):
    """Set paths to data storage.

    :Args:
        exp_root (str)
            Path to where the main file that starts the program is.

    :Kwargs:
        - computer (Namespace, default: :class:`default_computer`)
            A class with a computer parameters defined, such as the default
            path for storing data, size of screen etc. See
            :class:`default_computer` for an example.
        - fmri_rel (str, default: '')
            A path to where fMRI data and related analyzes should be stored.
            This is useful because fMRI data takes a lot of space so you may
            want to keep it on an external hard drive rather than on Dropbox
            where your scripts might live, for example.

    :Returns:
        paths (dict):
            A dictionary of paths.
    """
    run_tests(computer)
    fmri_root = os.path.join(computer.root, fmri_rel)
    if exp_root != '':
        exp_root += '/'
    paths = {
        'root': computer.root,
        'exp_root': exp_root,
        'fmri_root': fmri_root,
        'analysis': os.path.join(exp_root, 'analysis/'),  # where analysis files are stored
        'logs': os.path.join(exp_root, 'logs/'),
        'data': os.path.join(exp_root, 'data/'),
        'report': 'report/',
        'data_behav': os.path.join(fmri_root, 'data_behav/'),  # for fMRI behav data
        'data_fmri': os.path.join(fmri_root,'data_fmri/'),
        'data_struct': os.path.join(fmri_root,'data_struct/'),  # anatomical data
        'spm_analysis': os.path.join(fmri_root, 'analysis/'),
        'rec': os.path.join(fmri_root,'reconstruction/'), # CARET reconstructions
        'rois': os.path.join(fmri_root,'rois/'),  # ROIs (no data, just masks)
        'data_rois': os.path.join(fmri_root,'data_rois/'), # preprocessed and masked data
        'sim': exp_root,  # path for storing simulations of models
        }
    return paths

def run_tests(computer):
    """Runs basic tests before starting the experiment.

    At the moment, it only checks if the computer is recognized and if not,
    it waits for a user confirmation to continue thus preventing from running
    an experiment with incorrect settings, such as stimuli size.

    :Kwargs:
        computer (Namespace)
            A class with a computer parameters defined, such as the default
            path for storing data, size of screen etc. See
            :class:`default_computer` for an example.

    """
    if not computer.recognized:
        resp = raw_input("WARNING: This computer is not recognized.\n"
                "To continue, simply hit Enter (default)\n"
                #"To memorize this computer and continue, enter 'm'\n"
                "To quit, enter 'q'\n"
                "Your choice [C,q]: ")
        while resp not in ['', 'c', 'q']:
            resp = raw_input("Choose between continue (c) and quit (q): ")
        if resp == 'q':
            sys.exit()
        #elif resp == 'm':
            #mac = uuid.getnode()
            #if os.path.isfile('computer.py'):
                #write_head = False
            #else:
                #write_head = True
            #try:
                #dataFile = open(datafile, 'ab')
            #print ("Computer %d is memorized. Remember to edit computer.py"
                   #"file to " % mac


class _Common(object):

    def __init__(self):
        self.can_exit = 0

    def show_instructions(self, text='', wait=0, wait_stim=None, auto=0):
        """
        Displays instructions on the screen.

        :Kwargs:
            - text (str, default: '')
                Text to be displayed
            - wait (int, default: 0)
                Seconds to wait after removing the text from the screen after
                hitting a spacebar (or a `computer.default_keys['trigger']`)
            - wait_stim (a psychopy stimuli object or their list, default: None)
                Stimuli to show while waiting after the trigger. This is used
                for showing a fixation spot for people to get used to it.
        """
        # for some graphics drivers (e.g., mine:)
        # draw() command needs to be invoked once
        # before it can draw properly
        visual.TextStim(self.win, text='').draw()
        self.win.flip()

        instructions = visual.TextStim(self.win, text=text,
            color='white', height=20, units='pix', pos=(0,0),
            wrapWidth=30*20)
        instructions.draw()
        self.win.flip()

        if auto > 0:  # show text and blank out
            if self.rp['autorun']:
                auto /= self.rp['autorun']
            core.wait(auto)
        elif not self.rp['autorun'] or not self.rp['unittest']:
            this_key = None
            while this_key != self.computer.default_keys['trigger']:
                this_key = self.last_keypress()
            if self.rp['autorun']:
                wait /= self.rp['autorun']
        self.win.flip()

        if wait_stim is not None:
            if not isinstance(wait_stim, tuple) and not isinstance(wait_stim, list):
                wait_stim = [wait_stim]
            for stim in wait_stim:
                stim.draw()
            self.win.flip()
        core.wait(wait)  # wait a little bit before starting the experiment

    def last_keypress(self, keylist=None):
        """
        Extract the last key pressed from the event list.

        If escape is pressed, quits.

        :Kwargs:
            keylist (list of str, default: `self.computer.default_keys`)
                A list of keys that are recognized. Any other keys pressed will
                not matter.

        :Returns:
            An str of a last pressed key or None if nothing has been pressed.
        """
        if keylist is None:
            keylist = self.computer.default_keys.values()
        else:
            keylist.extend(self.computer.default_keys.values())
        keys = []
        for key in keylist:
            if isinstance(key, (tuple, list)):
                keys.append(key)
            else:
                keys.append([key])
        # keylist might have key combinations; get rid of them for now
        keylist_flat = []
        for key in keys:
            keylist_flat.extend(key)

        this_keylist = event.getKeys(keyList=keylist_flat)
        if len(this_keylist) > 0:
            this_key = this_keylist.pop()
            exit_keys = self.computer.default_keys['exit']
            if this_key in exit_keys:
                if self.can_exit < len(exit_keys):
                    if exit_keys[self.can_exit] == this_key:
                        if self.can_exit == len(exit_keys) - 1:
                            print  # in case there was anything without \n
                            self.quit()
                        else:
                            self.can_exit += 1
                    else:
                        self.can_exit = 0
                else:
                    self.can_exit = 0
            else:
                self.can_exit = 0
                return this_key
        else:
            return None

    def quit(self):
        """What to do when exit is requested.
        """
        logging.warning('Premature exit requested by user.')
        self.win.close()
        core.quit()
        # redefine core.quit() so that an App window would not be killed
        #logging.flush()
        #for thisThread in threading.enumerate():
            #if hasattr(thisThread,'stop') and hasattr(thisThread,'running'):
                ##this is one of our event threads - kill it and wait for success
                #thisThread.stop()
                #while thisThread.running==0:
                    #pass#wait until it has properly finished polling
        #sys.exit(0)

class Task(TrialHandler, _Common):

    def __init__(self,
                 #win,
                 #log=None,
                 #computer=default_computer,
                 parent,
                 info=None,
                 #extraInfo=None,
                 rp=None,
                 instructions={'text': '', 'wait': 0},

                 #parent=None,
                 name='',
                 version='0.1',
                 nreps=1,
                 method='random',
                 ):
        _Common.__init__(self)
        self.parent = parent
        self.computer = self.parent.computer
        self.paths = self.parent.paths

        self.name = name
        self.version = version
        self.instructions = instructions
        self.nReps = nreps
        self.method = method

        self.info = parent.info
        #self.extraInfo = self.info  # just for compatibility with PsychoPy
        self.rp = parent.rp
        if info is not None:
            self.info.update(info)
        if rp is not None:
            self.rp.update(rp)

    def setup_task(self):
        """
        Does all the dirty setup before running the experiment.

        Steps include:
            - Logging file setup (:func:`set_logging`)
            - Creating a :class:`~psychopy.visual.Window` (:func:`create_window`)
            - Creating stimuli (:func:`create_stimuli`)
            - Creating trial structure (:func:`create_trial`)
            - Combining trials into a trial list  (:func:`create_triaList`)
            - Creating a :class:`~psychopy.data.TrialHandler` using the
              defined trialList  (:func:`create_TrialHandler`)

        :Kwargs:
            create_win (bool, default: True)
                If False, a window is not created. This is useful when you have
                an experiment consisting of a couple of separate sessions. For
                the first one you create a window and want everything to be
                presented on that window without closing and reopening it
                between the sessions.
        """
        if self.parent._initialized:
            self.win = self.parent.win
            self.logfile = self.parent.logfile
            self.info = self.parent.info
            self.rp = self.parent.rp
            self.seed = self.parent.seed
            try:
                self.valid_responses = self.computer.valid_responses
            except:
                self.valid_responses = self.parent.valid_responses

            self.create_stimuli()
            self.create_trial()
            self.create_trial_list()
            if not hasattr(self, 'trial_dur'):
                self.trial_dur = sum(ev['dur'] for ev in self.trial)
            if self.rp['autorun']:
                self.trialList = self.set_autorun(self.trialList)

            #self.set_TrialHandler()
        else:
            raise Exception('You must first call Experiment.setup()')
        #dataFileName=self.paths['data']%self.info['subjid'])

        ## guess participant ID based on the already completed sessions
        #self.info['subjid'] = self.guess_participant(
            #self.paths['data'],
            #default_subjid=self.info['subjid'])

        #self.dataFileName = self.paths['data'] + '%s.csv'

    def create_fixation(self, shape='complex', color='black', size=.2):
        """Creates a fixation spot.

        :Kwargs:
            - shape: {'dot', 'complex'} (default: 'complex')
                Choose the type of fixation:
                    - dot: a simple fixation dot (.2 deg visual angle)
                    - complex: the 'best' fixation shape by `Thaler et al., 2012
                      <http://dx.doi.org/10.1016/j.visres.2012.10.012>`_ which
                      looks like a combination of s bulls eye and cross hair
                      (outer diameter: .6 deg, inner diameter: .2 deg). Note
                      that it is constructed by superimposing two rectangles on
                      a disk, so if non-uniform background will not be visible.
            - color (str, default: 'black')
                Fixation color.

        """
        if shape == 'complex':
            r1 = size  # radius of outer circle (degrees)
            r2 = size/3.  # radius of inner circle (degrees)
            oval = visual.Circle(
                self.win,
                name   = 'oval',
                fillColor  = color,
                lineColor = None,
                radius   = r1,
            )
            center = visual.Circle(
                self.win,
                name   = 'center',
                fillColor  = color,
                lineColor = None,
                radius   = r2,
            )
            cross0 = ThickShapeStim(
                self.win,
                name='cross1',
                lineColor=self.win.color,
                lineWidth=2*r2,
                vertices=[(-r1, 0), (r1, 0)]
                )
            cross90 = ThickShapeStim(
                self.win,
                name='cross1',
                lineColor=self.win.color,
                lineWidth=2*r2,
                vertices=[(-r1, 0), (r1, 0)],
                ori=90
                )
            fixation = GroupStim(stimuli=[oval, cross0, cross90, center],
                                 name='fixation')
            # when color is set, we only want the oval and the center to change
            # so here we override :func:`GroupStim.setColor`
            def _set_complex_fix_col(newColor):
                for stim in fixation.stimuli:
                    if stim.name in ['oval', 'center']:
                        stim.setFillColor(newColor)
            fixation.color = color
            fixation.setFillColor = _set_complex_fix_col
            self.fixation = fixation

        elif shape == 'dot':
            self.fixation = GroupStim(
                stimuli=visual.PatchStim(
                    self.win,
                    name   = 'fixation',
                    color  = 'red',
                    tex    = None,
                    mask   = 'circle',
                    size   = size,
                ),
                name='fixation')

    def latin_square(self, n=6):
        """
        Generates a Latin square of size n. n must be even.

        Based on
        <http://rintintin.colorado.edu/~chathach/balancedlatinsquares.html>_

        :Kwargs:
            n (int, default: 6)
                Size of Latin square. Should be equal to the number of
                conditions you have.

        .. :note: n must be even. For an odd n, I am not aware of a
                  general method to produce a Latin square.

        :Returns:
            A `numpy.array` with each row representing one possible ordering
            of stimuli.
        """
        if n%2 != 0: sys.exit('n is not even!')

        latin = []
        col = np.arange(1,n+1)

        first_line = []
        for i in range(n):
            if i%2 == 0: first_line.append((n-i/2)%n + 1)
            else: first_line.append((i+1)/2+1)

        latin = np.array([np.roll(col,i-1) for i in first_line])

        return latin.T

    def make_para(self, n=6):
        """
        Generates a symmetric para file with fixation periods approximately 25%
        of the time.

        :Kwargs:
            n (int, default: 6)
                Size of Latin square. Should be equal to the number of
                conditions you have.
                :note: n must be even. For an odd n, I am not aware of a
                general method to produce a Latin square.

        :Returns:
            A `numpy.array` with each row representing one possible ordering
            of stimuli (fixations are coded as 0).
        """
        latin = self.latin_square(n=n).tolist()
        out = []
        for j, this_latin in enumerate(latin):
            this_latin = this_latin + this_latin[::-1]
            temp = []
            for i, item in enumerate(this_latin):
                if i%4 == 0: temp.append(0)
                temp.append(item)
            temp.append(0)
            out.append(temp)

        return np.array(out)

    def create_stimuli(self):
        """
        Define stimuli as a dictionary

        Example::

            self.create_fixation(color='white')
            line1 = visual.Line(self.win, name='line1')
            line2 = visual.Line(self.win, fillColor='DarkRed')
            self.s = {
                'fix': self.fixation,
                'stim1': [visual.ImageStim(self.win, name='stim1')],
                'stim2': GroupStim(stimuli=[line1, line2], name='lines')
                }
        """
        raise NotImplementedError

    def create_trial(self):
        """
        Create a list of events that constitute a trial.

        Example::

            self.trial = [{'dur': .100,
                           'display': self.s['fix'],
                           'func': self.idle_event},

                           {'dur': .300,
                           'display': self.s['stim1'],
                           'func': self.during_trial},
                           ]
        """
        raise NotImplementedError

    def create_trial_list(self):
        """
        Put together trials into a trialList.

        Example::

            exp_plan = OrderedDict([
                ('cond', cond),
                ('name', names[cond]),
                ('onset', ''),
                ('dur', self.trial_dur),
                ('corr_resp', corr_resp),
                ('subj_resp', ''),
                ('accuracy', ''),
                ('rt', ''),
                ])

            self.trialList = exp_plan
        """
        raise NotImplementedError

    def idle_event(self, glob_clock=None, trial_clock=None, event_clock=None,
                  this_trial=None, this_event=None, **kwargs):
        """
        Default idle function for the event.

        Sits idle catching key input of default keys (escape and trigger).

        :Kwargs:
            - glob_clock (:class:`psychopy.core.Clock`, default: None)
                A clock that started with the experiment (currently does nothing)
            - trial_clock (:class:`psychopy.core.Clock`, default: None)
                A clock that started with the trial
            - event_clock (:class:`psychopy.core.Clock`, default: None)
                A clock that started with the event within the trial
            - this_trial (dict)
                A dictionary of trial properties
            - this_event (dict)
                A dictionary with event properties
        """
        if not isinstance(this_event['display'], tuple) and \
        not isinstance(this_event['display'], list):
            display = [this_event['display']]
        else:
            display = this_event['display']

        if this_event['dur'] == 0:
            self.last_keypress()
            for stim in display: stim.draw()
            self.win.flip()

        else:
            for stim in display: stim.draw()
            self.win.flip()

            while event_clock.getTime() < this_event['dur'] and \
            trial_clock.getTime() < this_trial['dur']:# and \
            # globClock.getTime() < thisTrial['onset'] + thisTrial['dur']:
                #self.win.getMovieFrame()
                self.last_keypress()

    def feedback(self, trial_clock=None, event_clock=None,
        this_trial=None, this_event=None, all_keys=None, *args, **kwargs):
        """
        Gives feedback:
            - correct: fixation change to green
            - wrong: fixation change to red
        """
        this_resp = all_keys[-1]
        subj_resp = self.valid_responses[this_resp[0]]
        if not isinstance(this_event['display'], tuple) and \
            not isinstance(this_event['display'], list):
                display = [this_event['display']]

        for stim in display:
            if stim.name in ['fixation', 'fix']:
                orig_color = stim.color
                break
        for stim in display:
            if stim.name in ['fixation', 'fix']:
                if this_trial['corr_resp'] == subj_resp:
                    stim.setFillColor('DarkGreen')  # correct response
                else:
                    stim.setFillColor('DarkRed')  # incorrect response
            stim.draw()
        self.win.flip()

        # sit idle
        while event_clock.getTime() < this_event['dur']:
            self.last_keypress()

        for stim in display:  # reset fixation color
            if stim.name == 'fixation':
                stim.setFillColor(orig_color)

    def set_autorun(self, trial_list):
        """
        Automatically runs experiment by simulating key responses.

        This is just the absolute minimum for autorunning. Best practice would
        be extend this function to simulate responses according to your
        hypothesis.

        :Args:
            trial_list (list of dict)
                A list of trial definitions.

        :Returns:
            trial_list with ``autoResp`` and ``autoRT`` columns included.
        """
        def rt(mean):
            add = np.random.normal(mean,scale=.2)/self.rp['autorun']
            return self.trial[0]['dur'] + add

        inverse_resp = invert_dict(self.valid_responses)
        # speed up the experiment
        for ev in self.trial:
            ev['dur'] /= self.rp['autorun']
        self.trial_dur /= self.rp['autorun']

        for trial in trial_list:
            # here you could do if/else to assign different values to
            # different conditions according to your hypothesis
            trial['autoresp'] = random.choice(inverse_resp.values())
            trial['autort'] = rt(.5)
        return trial_list


    def set_TrialHandler(self, trial_list):
        """
        Converts a list of trials into a `~psychopy.data.TrialHandler`,
        finalizing the experimental setup procedure.

        Creates ``self.trialDur`` if not present yet.
        Appends information for autorun.
        """
        TrialHandler.__init__(self,
            trial_list,
            nReps=self.nReps,
            method=self.method,
            extraInfo=self.info,
            name=self.name,
            seed=self.seed)
    #def run(self):
        #"""
        #Setup and go!
        #"""
        #self.setup()
        #self.go(
            #datafile=self.paths['data'] + self.info['subjid'] + '.csv',
            #no_output=self.rp['no_output'])
        #if self.rp['register']:
            #self.register()
        #elif self.rp['push']:
            #self.commitpush()

    def get_breaks(self, breakcol=None):
        if breakcol is not None:
            breaks = []
            last = self.trialList[0][breakcol]
            for n, trial in enumerate(self.trialList):
                if trial[breakcol] != last:
                    if len(breaks) > 0:
                        breaks.append((breaks[-1][-1], n))
                    else:
                        breaks.append([0,n])
                    last = trial[breakcol]
            try:
                breaks.append((breaks[-1][-1], len(self.trialList)))
            except:
                breaks = [(0, len(self.trialList))]
        else:
            breaks = [(0, len(self.trialList))]
        if self.method == 'fullRandom':
            np.random.shuffle(breaks)
        return breaks

    def autorun(self):
        """
        Automatically runs the experiment just like it would normally work but
        responding automatically (as defined in :func:`self.set_autorun`) and
        at the speed specified by `self.rp['autorun']` parameter. If
        speed is not specified, it is set to 100.
        """
        self.rp['autorun'] = 100
        self.run()

    def run(self):
        self.setup_task()
        datafile = self.paths['data'] + self.info['subjid'] + '.csv'
        breaks = self.get_breaks(breakcol=None)
        self.show_instructions(**self.instructions)
        pause_instr = self.instructions.copy()
        pause_instr['text'] = '<pause>'
        for i, br in enumerate(breaks):
            self.set_TrialHandler(self.trialList[br[0]:br[1]])
            #for rep in range(self.nReps):
            self._loop_trials(datafile=datafile)
            if len(breaks) > 1:
                self.show_instructions(**pause_instr)

    def _loop_trials(self, datafile='data.csv'):
        """
        Iterate over the sequence of trials and events.

        .. note:: In the output file, floats are formatted to 1 ms precision so
                  that output files are nice.

        :Kwargs:
            - datafile (str, default: 'data.csv')
                Data file name to store experiment information and responses.
            - no_output (bool, default: False)
                If True, the data file will not be written. Useful for checking
                how the experiment looks like and for debugging.

        :Raises:
            :py:exc:`IOError` if `datafile` is not found.
        """
        if not self.rp['no_output']:
            try_makedirs(os.path.dirname(datafile))
            try:
                dfile = open(datafile, 'ab')
                datawriter = csv.writer(dfile, lineterminator = '\n')
            except IOError:
                raise IOError('Cannot write to the data file %s!' % datafile)
            else:
                write_head = True
        else:
            datawriter = None
            write_head = None

        # set up clocks
        glob_clock = core.Clock()
        trial_clock = core.Clock()
        event_clock = core.Clock()
        trialno = 0
        # go over the trial sequence
        for this_trial in self:
            if self.nReps > 1:
                this_trial['session'] = self.thisRepN
            trial_clock.reset()
            this_trial['onset'] = glob_clock.getTime()
            sys.stdout.write('\rtrial %s' % (trialno+1))
            sys.stdout.flush()

            # go over each event in a trial
            all_keys = []
            for j, this_event in enumerate(self.trial):
                event_clock.reset()
                event_keys = this_event['func'](glob_clock=glob_clock,
                    trial_clock=trial_clock, event_clock=event_clock,
                    this_trial=this_trial, this_event=this_event, j=j,
                    all_keys=all_keys)
                if event_keys is not None:
                    all_keys += event_keys
                # this is to get keys if we did not do that during trial
                all_keys += event.getKeys(
                    keyList=self.valid_responses.keys(),
                    timeStamped=trial_clock)

            if len(all_keys) == 0 and self.rp['autorun'] > 0:
                all_keys += [(this_trial['autoresp'], this_trial['autort'])]

            this_trial = self.post_trial(this_trial, all_keys)
            if self.rp['autorun'] > 0:  # correct the timing
                try:
                    this_trial['autort'] *= self.rp['autorun']
                    this_trial['rt'] *= self.rp['autorun']
                except:  # maybe not all keys are present
                    pass
                this_trial['onset'] *= self.rp['autorun']

            if datawriter is not None:
                header = self.info.keys() + this_trial.keys()
                if write_head:  # will write the header the first time
                    write_head = self._write_header(datafile, header, datawriter)
                out = self.info.values() + this_trial.values()
                # cut down floats to 1 ms precision
                outf = ['%.3f'%i if isinstance(i,float) else i for i in out]
                datawriter.writerow(outf)

            trialno += 1
            #if trialNo == self.nTotal/self.nReps:
                #break

        # clear trial counting in the terminal
        sys.stdout.write('\r          ')
        sys.stdout.write('\r')
        sys.stdout.flush()
        if not self.rp['no_output']: dfile.close()

    def _write_header(self, datafile, header, datawriter):
        """Determines if a header should be writen in a csv data file.

        Works by reading the first line and comparing it to the given header.
        If the header already is present, then a new one is not written.

        :Args:
            - datafile (str)
                Name of the data file
            - header (list of str)
                A list of column names
            - datawriter (:class:`csv.writer`)
                A CSV writer for writing data to the data file

        :Returns:
            False, so that it is never called again during :func:`loop_trials`
        """
        write_head = True
        # no header needed if the file already exists and has one
        try:
            dataf_r = open(datafile, 'rb')
            dataread = csv.reader(dataf_r)
        except:
            pass
        else:
            try:
                header_file = dataread.next()
            except:  # empty file
                write_head = True
            else:
                if header == header_file:
                    write_head = False
                else:
                    write_head = True
            dataf_r.close()
        if write_head:
            datawriter.writerow(header)
        return False


    def wait_for_response(self, rt_clock, this_trial):
        """
        Waits for response. Returns last key pressed, timestamped.

        :Args:
            - rt_clock (`psychopy.core.Clock`)
                A clock used as a reference for measuring response time
            - this_trial
                Current trial from `self.trialList`

        :Returns:
            A list of tuples with a key name (str) and a response time (float).
        """
        all_keys = []
        event.clearEvents() # key presses might be stored from before
        while len(all_keys) == 0: # if the participant did not respond earlier
            if 'autort' in this_trial:
                if rt_clock.getTime() > this_trial['autort']:
                    all_keys = [(this_trial['autoresp'], this_trial['autort'])]
            else:
                all_keys = event.getKeys(
                    keyList=self.valid_responses.keys(),
                    timeStamped=rt_clock)
            self.last_keypress()
        return all_keys

    def post_trial(self, this_trial, all_keys):
        """A default function what to do after a trial is over.

        It records the participant's response as the last key pressed,
        calculates accuracy based on the expected (correct) response value,
        and records the time of the last key press with respect to the onset
        of a trial. If no key was pressed, participant's response and response
        time are recorded as an empty string, while accuracy is assigned a
        'No response'.

        :Args:
            - this_trial (dict)
                A dictionary of trial properties
            - all_keys (list of tuples)
                A list of tuples with the name of the pressed key and the time
                of the key press.

        :Returns:
            this_trial with ``subj_resp``, ``accuracy``, and ``rt`` filled in.

        """
        if len(all_keys) > 0:
            this_resp = all_keys.pop()
            this_trial['subj_resp'] = self.valid_responses[this_resp[0]]
            acc = signal_det(this_trial['corr_resp'], this_trial['subj_resp'])
            this_trial['accuracy'] = acc
            this_trial['rt'] = this_resp[1]
        else:
            this_trial['subj_resp'] = ''
            acc = signal_det(this_trial['corr_resp'], this_trial['subj_resp'])
            this_trial['accuracy'] = acc
            this_trial['rt'] = ''

        return this_trial

    def _astype(self,type='pandas'):
        """
        Converts data into a requested type.

        Mostly reused :func:`psychopy.data.TrialHandler.saveAsWideText`

        :Kwargs:
            type
        """
        # collect parameter names related to the stimuli:
        header = self.trialList[0].keys()
        # and then add parameter names related to data (e.g. RT)
        header.extend(self.data.dataTypes)

        # loop through each trial, gathering the actual values:
        dataOut = []
        trialCount = 0
        # total number of trials = number of trialtypes * number of repetitions:
        repsPerType={}
        for rep in range(self.nReps):
            for trialN in range(len(self.trialList)):
                #find out what trial type was on this trial
                trialTypeIndex = self.sequenceIndices[trialN, rep]
                #determine which repeat it is for this trial
                if trialTypeIndex not in repsPerType.keys():
                    repsPerType[trialTypeIndex]=0
                else:
                    repsPerType[trialTypeIndex]+=1
                repThisType=repsPerType[trialTypeIndex]#what repeat are we on for this trial type?

                # create a dictionary representing each trial:
                # this is wide format, so we want fixed information (e.g. subject ID, date, etc) repeated every line if it exists:
                if (self.extraInfo != None):
                    nextEntry = self.extraInfo.copy()
                else:
                    nextEntry = {}

                # add a trial number so the original order of the data can always be recovered if sorted during analysis:
                trialCount += 1
                nextEntry["TrialNumber"] = trialCount

                # now collect the value from each trial of the variables named in the header:
                for parameterName in header:
                    # the header includes both trial and data variables, so need to check before accessing:
                    if self.trialList[trialTypeIndex] and self.trialList[trialTypeIndex].has_key(parameterName):
                        nextEntry[parameterName] = self.trialList[trialTypeIndex][parameterName]
                    elif self.data.has_key(parameterName):
                        nextEntry[parameterName] = self.data[parameterName][trialTypeIndex][repThisType]
                    else: # allow a null value if this parameter wasn't explicitly stored on this trial:
                        nextEntry[parameterName] = ''

                #store this trial's data
                dataOut.append(nextEntry)

        # get the extra 'wide' parameter names into the header line:
        header.insert(0,"TrialNumber")
        if (self.info != None):
            for key in self.info:
                header.insert(0, key)

        if type in [list, 'list']:
            import pdb; pdb.set_trace()
        elif type in [dict, 'dict']:
            import pdb; pdb.set_trace()
        elif type == 'pandas':
            df = pandas.DataFrame(dataOut, columns=header)

        return df

    def aspandas(self):
        """
        Convert trialList into a pandas DataFrame object
        """
        return self._astype(type='pandas')

    #def accuracy(self):
        #df = self._astype(list)
        #for line in df:
            #if line['accuracy']=='Correct':
                #accuracy += 1
        #acc = accuracy * 100 / len(df)
        #return acc
    def weighted_sample(self, probs):
        warnings.warn("weighted_sample is deprecated; "
                      "use weighted_choice instead")
        return self.weighted_choice(weights=probs)

    def weighted_choice(self, choices=None, weights=None):
        """
        Chooses an element from a list based on it's weight.

        :Kwargs:
            - choices (list, default: None)
                If None, an index between 0 and ``len(weights)`` is returned.
            - weights (list, default: None)
                If None, all choices get equal weights.

        :Returns:
            An element from ``choices``
        """
        if choices is None:
            if weights is None:
                raise Exception('Please specify either choices or weights.')
            else:
                choices = range(len(weights))
        elif weights is None:
            weights = np.ones(len(choices)) / float(len(choices))
        if not np.allclose(np.sum(weights), 1):
            raise Exception('Weights must add up to one.')
        which = np.random.random()
        ind = 0
        while which>0:
            which -= weights[ind]
            ind +=1
        ind -= 1
        return choices[ind]

    def get_behav_df(self, pattern='%s'):
        """
        Extracts data from files for data analysis.

        :Kwargs:
            pattern (str, default: '%s')
                A string with formatter information. Usually it contains a path
                to where data is and a formatter such as '%s' to indicate where
                participant ID should be incorporated.

        :Returns:
            A `pandas.DataFrame` of data for the requested participants.
        """
        return get_behav_df(self.info['subjid'], pattern=pattern)


class Experiment(ExperimentHandler, Task):
    """An extension of an TrialHandler with many useful functions.
    """
    def __init__(self,
                 name='',
                 version='0.1',
                 info=None,
                 rp=None,
                 instructions={'text': None, 'wait': 0},
                 computer=default_computer,
                 paths=set_paths,
                 **kwargs
                 ):

                 #runtimeInfo=None,
                 #originPath=None,
                 #savePickle=True,
                 #saveWideText=True,
                 #dataFileName=''):

        ExperimentHandler.__init__(self,
            name=name,
            version=version,
            extraInfo=info,
            dataFileName='.empty'
            )
        _Common.__init__(self)

        self.name = name
        self.version = version

        self.instructions = instructions
        if hasattr(self.instructions, 'text'):
            if self.instructions['text'] is None:
                self.instructions['text'] = self.__doc__
        else:
            self.instructions['text'] = self.__doc__
        if not hasattr(self.instructions, 'wait'):
            self.instructions['wait'] = 0

        #self.paths = set_paths('.')

        #self.nReps = nReps
        #self.method = method
        self.computer = computer
        self.paths = paths
        #if self.computer is None:
            #self.computer =
        ##self.dataTypes = dataTypes
        #self.originPath = originPath

        self._initialized = False

        #self.signalDet = {False: 'Incorrect', True: 'Correct'}

        # minimal parameters that Experiment expects in extraInfo and runParams
        self.info = OrderedDict([('subjid', 'subj')])
        self.rp = OrderedDict([  # these control how the experiment is run
            ('no_output', False),  # do you want output? or just playing around?
            ('debug', False),  # not fullscreen presentation etc
            ('autorun', 0),  # if >0, will autorun at the specified speed
            ('unittest', False),  # like autorun but no breaks at show_instructions
            ('register', False),  # add and commit changes, like new data files?
            ('push', False),  # add, commit and push to a hg repo?
            ])
        if info is not None:
            if isinstance(info, (list, tuple)):
                try:
                    info = OrderedDict(info)
                except:
                    info = OrderedDict([info])
            self.info.update(info)
        if rp is not None:
            if isinstance(rp, (list, tuple)):
                try:
                    rp = OrderedDict(rp)
                except:
                    rp = OrderedDict([rp])
            self.rp.update(rp)

        if self.rp['unittest']:
            self.rp['autorun'] = 100

        #sysinfo = info.RunTimeInfo(verbose=True, win=False,
                #randomSeed='set:time')
        #seed = sysinfo['experimentRandomSeed.string']

        #self.seed = 10#int(seed)
        self.tasks = []

        Task.__init__(self,
            self,
            name=name,
            version=version,
            instructions=instructions,
            **kwargs
            )


    def add_tasks(self, tasks):
        if isinstance(tasks, str):
            tasks = [tasks]

        for task in tasks:
            task = task()
            task.computer = self.computer
            task.win = self.win
            if task.info is not None:
                task.info.update(self.info)
            if task.rp is not None:
                task.rp.update(self.rp)
            self.tasks.append(task)

    def set_logging(self, logname='log.log', level=logging.WARNING):
        """Setup files for saving logging information.

        New folders might be created.

        :Kwargs:
            logname (str, default: 'log.log')
                The log file name.
        """

        if not self.rp['no_output']:
            # add .log if no extension given
            if len(logname.split('.')) < 2: logname += '.log'

            # Setup logging file
            try_makedirs(os.path.dirname(logname))
            if os.path.isfile(logname):
                writesys = False  # we already have sysinfo there
            else:
                writesys = True
            self.logfile = logging.LogFile(logname, filemode='a', level=level)

            # Write system information first
            if writesys:
                self.logfile.write('%s\n' % self.runtime_info)
                self.logfile.write('\n\n' + '#'*40 + '\n\n')
                self.logfile.write('$ python %s\n' % ' '.join(sys.argv))
        else:
            self.logfile = None

        # output to the screen
        logging.console.setLevel(level)

    def create_seed(self, seed=None):
        """
        SUPERSEDED by `psychopy.info.RunTimeInfo`
        Creates or assigns a seed for a reproducible randomization.

        When a seed is set, you can, for example, rerun the experiment with
        trials in exactly the same order as before.

        :Kwargs:
            seed (int, default: None)
                Pass a seed if you already have one.

        :Returns:
            self.seed (int)
        """
        if seed is None:
            try:
                self.seed = np.sum([ord(d) for d in self.info['date']])
            except:
                self.seed = 1
                logging.warning('No seed provided. Setting seed to 1.')
        else:
            self.seed = seed
        return self.seed

    def _guess_participant(self, data_path, default_subjid='01'):
        """Attempts to guess participant ID (it must be int).

        .. :Warning:: Not usable yet

        First lists all csv files in the data_path, then finds a maximum.
        Returns maximum+1 or an empty string if nothing is found.

        """
        datafiles = glob.glob(data_path+'*.csv')
        partids = []
        #import pdb; pdb.set_trace()
        for d in datafiles:
            filename = os.path.split(d)[1]  # remove the path
            filename = filename.split('.')[0]  # remove the extension
            partid = filename.split('_')[-1]  # take the numbers at the end
            try:
                partids.append(int(partid))
            except:
                logging.warning('Participant ID %s is invalid.' %partid)

        if len(partids) > 0: return '%02d' %(max(partids) + 1)
        else: return default_subjid

    def _guess_runno(self, data_path, default_runno = 1):
        """Attempts to guess run number.

        .. :Warning:: Not usable yet

        First lists all csv files in the data_path, then finds a maximum.
        Returns maximum+1 or an empty string if nothing is found.

        """
        if not os.path.isdir(data_path): runno = default_runno
        else:
            datafiles = glob.glob(data_path + '*.csv')
            # Splits file names into ['data', %number%, 'runType.csv']
            allnums = [int(os.path.basename(thisfile).split('_')[1]) for thisfile in datafiles]

            if allnums == []: # no data files yet
                runno = default_runno
            else:
                runno = max(allnums) + 1
                # print 'Guessing runNo: %d' %runNo

        return runno

    def get_mon_sizes(self, screen=None):
        warnings.warn('get_mon_sizes is deprecated; '
                      'use exp.get_mon_sizes instead')
        return get_mon_sizes(screen=screen)

    def create_win(self, debug=False, color='DimGray'):
        """Generates a :class:`psychopy.visual.Window` for presenting stimuli.

        :Kwargs:
            - debug (bool, default: False)
                - If True, then the window is half the screen size.
                - If False, then the windon is full screen.
            - color (str, str with a hexadecimal value, or a tuple of 3 values, default: "DimGray')
                Window background color. Default is dark gray. (`See accepted
                color names <http://www.w3schools.com/html/html_colornames.asp>`_
        """
        current_level = logging.getLevel(logging.console.level)
        logging.console.setLevel(logging.ERROR)
        monitor = monitors.Monitor(self.computer.name,
            distance=self.computer.distance,
            width=self.computer.width)
        logging.console.setLevel(current_level)
        res = get_mon_sizes(self.computer.screen)
        monitor.setSizePix(res)
        try:
            size = self.computer.win_size
        except:
            if not debug:
                size = tuple(res)
            else:
                size = (res[0]/2, res[1]/2)
        self.win = visual.Window(
            size=size,
            monitor = monitor,
            units = 'deg',
            fullscr = not debug,
            allowGUI = debug, # mouse will not be seen unless debugging
            color = color,
            winType = 'pyglet',
            screen = self.computer.screen,
            viewScale = self.computer.view_scale
        )

    def show_intro(self, text, **kwargs):
        #if text is not None:
        #self.create_win(debug=self.rp['debug'])
        self.show_instructions(text=text, **kwargs)

    def setup(self):
        try:
            self.valid_responses = self.computer.valid_responses
        except:
            self.valid_responses = {'0': 0, '1': 1}
        if not self.rp['no_output']:
            self.runtime_info = psychopy.info.RunTimeInfo(verbose=True, win=False,
                    randomSeed='set:time')
            self.seed = int(self.runtime_info['experimentRandomSeed.string'])
            np.random.seed(self.seed)
        else:
            self.runtime_info = None
            self.seed = None

        self.set_logging(self.paths['logs'] + self.info['subjid'])
        self.create_win(debug=self.rp['debug'])
        self._initialized = True
        #if len(self.tasks) == 0:
            ##self.setup = Task.setup
            #Task.setup(self)

    def run(self):
        self.setup()
        if len(self.tasks) == 0:
            Task.run(self)
        else:
            self.show_intro(**self.instructions)
            for task in self.tasks:
                task.run()
        text = ('End of Experiment. Thank you!\n\n'
                'Press space bar to exit.')
        self.show_instructions(text=text)
        if self.rp['register']:
            self.register()
        elif self.rp['push']:
            self.commitpush()
        self.quit()

    def commit(self, message=None):
        """
        Add and commit changes in a repository.

        TODO: How to set this up.
        """
        if message is None:
            message = 'data for participant %s' % self.info['subjid']
        cmd, out, err = ui._repo_action('commit', message=message)
        self.logfile.write('\n'.join([cmd, out, err]))

        return err

    def commitpush(self, message=None):
        """
        Add, commit, and push changes to a remote repository.

        Currently, only Mercurial repositories are supported.

        TODO: How to set this up.
        TODO: `git` support
        """
        err = self.commit(message=message)
        if err == '':
            out = ui._repo_action('push')
            self.logfile.write('\n'.join(out))


def get_behav_df(subjid, pattern='%s'):
    """
    Extracts data from files for data analysis.

    :Kwargs:
        pattern (str, default: '%s')
            A string with formatter information. Usually it contains a path
            to where data is and a formatter such as '%s' to indicate where
            participant ID should be incorporated.

    :Returns:
        A `pandas.DataFrame` of data for the requested participants.
    """
    if type(subjid) not in [list, tuple]:
        subjid_list = [subjid]
    else:
        subjid_list = subjid

    df_fnames = []
    for subjid in subjid_list:
        fnames = glob.glob(pattern % subjid)
        fnames.sort()
        df_fnames += fnames
    dfs = []
    for dtf in df_fnames:
        data = pandas.read_csv(dtf)
        if data is not None:
            dfs.append(data)
    if dfs == []:
        print df_fnames
        raise IOError('Behavioral data files not found.\n'
            'Tried to look for %s' % (pattern % subjid))
    df = pandas.concat(dfs, ignore_index=True)

    return df


class ThickShapeStim(visual.ShapeStim):
    """
    Draws thick shape stimuli as a collection of lines.

    PsychoPy has a bug in some configurations of not drawing lines thicker
    than 2px. This class fixes the issue. Note that it's really just a
    collection of rectanges so corners will not look nice.
    """
    def __init__(self,
                 win,
                 units  ='',
                 lineWidth=1.0,
                 lineColor=(1.0,1.0,1.0),
                 lineColorSpace='rgb',
                 fillColor=None,
                 fillColorSpace='rgb',
                 vertices=((-0.5,0),(0,+0.5),(+0.5,0)),
                 closeShape=True,
                 pos= (0,0),
                 size=1,
                 ori=0.0,
                 opacity=1.0,
                 depth  =0,
                 interpolate=True,
                 lineRGB=None,
                 fillRGB=None,
                 name='', autoLog=True):

        visual._BaseVisualStim.__init__(self, win, units=units, name=name, autoLog=autoLog)

        self.opacity = opacity
        self.pos = np.array(pos, float)
        self.closeShape=closeShape
        self.lineWidth=lineWidth
        self.interpolate=interpolate

        self._useShaders=False  #since we don't need to combine textures with colors
        self.lineColorSpace=lineColorSpace
        if lineRGB!=None:
            logging.warning("Use of rgb arguments to stimuli are deprecated. Please use color and colorSpace args instead")
            self.setLineColor(lineRGB, colorSpace='rgb')
        else:
            self.setLineColor(lineColor, colorSpace=lineColorSpace)

        self.fillColorSpace=fillColorSpace
        if fillRGB!=None:
            logging.warning("Use of rgb arguments to stimuli are deprecated. Please use color and colorSpace args instead")
            self.setFillColor(fillRGB, colorSpace='rgb')
        else:
            self.setFillColor(fillColor, colorSpace=fillColorSpace)

        self.depth=depth
        self.ori = np.array(ori,float)
        self.size = np.array([0.0,0.0])
        self.setSize(size)
        #self.size=size
        self.setVertices(vertices)
        # self._calcVerticesRendered()
        # if len(self.stimulus) == 1: self.stimulus = self.stimulus[0]

    #def __init__(self, *args, **kwargs):
        #try:
            #orig_vertices = kwargs['vertices']
            #kwargs['vertices'] = [(-0.5,0),(0,+0.5)]#,(+0.5,0)),
        #except:
            #pass
        ##import pdb; pdb.set_trace()
        #visual.ShapeStim.__init__(self, *args, **kwargs)
        #self.vertices = orig_vertices

    def draw(self):
        for stim in self.stimulus:
            stim.draw()

    def setOri(self, newOri):
        # theta = (newOri - self.ori)/180.*np.pi
        # rot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        # for stim in self.stimulus:
            # newVert = []
            # for vert in stim.vertices:
                # #import pdb; pdb.set_trace()
                # newVert.append(np.dot(rot,vert))
            # stim.setVertices(newVert)
        self.ori = newOri
        self.setVertices(self.vertices)

    def setPos(self, newPos):
        #for stim in self.stimulus:
            #stim.setPos(newPos)
        self.pos = newPos
        self.setVertices(self.vertices)

    #def setSize(self, newSize):
        ##for stim in self.stimulus:
            ##stim.setPos(newPos)
        #self.size = newSize
        #self.setVertices(self.vertices)

    def setVertices(self, value=None):
        if isinstance(value[0][0], int) or isinstance(value[0][0], float):
            self.vertices = [value]
        else:
            self.vertices = value
        self.stimulus = []

        theta = self.ori/180.*np.pi #(newOri - self.ori)/180.*np.pi
        rot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

        self._rend_vertices = []

        for vertices in self.vertices:
            rend_verts = []
            if self.closeShape:
                numPairs = len(vertices)
            else:
                numPairs = len(vertices)-1

            wh = self.lineWidth/2. - misc.pix2deg(1,self.win.monitor)
            for i in range(numPairs):
                thisPair = np.array([vertices[i],vertices[(i+1)%len(vertices)]])
                thisPair_rot = np.dot(thisPair, rot.T)
                edges = [
                    thisPair_rot[1][0]-thisPair_rot[0][0],
                    thisPair_rot[1][1]-thisPair_rot[0][1]
                    ]
                lh = np.sqrt(edges[0]**2 + edges[1]**2)/2.
                rend_vert = [[-lh,-wh],[-lh,wh], [lh,wh],[lh,-wh]]
                #import pdb; pdb.set_trace()
                line = visual.ShapeStim(
                    self.win,
                    lineWidth   = 1,
                    lineColor   = self.lineColor,#None,
                    interpolate = True,
                    fillColor   = self.lineColor,
                    ori         = -np.arctan2(edges[1],edges[0])*180/np.pi,
                    pos         = np.mean(thisPair_rot,0) + self.pos,
                    # [(thisPair_rot[0][0]+thisPair_rot[1][0])/2. + self.pos[0],
                                   # (thisPair_rot[0][1]+thisPair_rot[1][1])/2. + self.pos[1]],
                    vertices    = rend_vert
                )
                #line.setOri(self.ori-np.arctan2(edges[1],edges[0])*180/np.pi)
                self.stimulus.append(line)
                rend_verts.append(rend_vert[0])
            rend_verts.append(rend_vert[1])

            self._rend_vertices.append(rend_verts)
            #import pdb; pdb.set_trace()
            #self.setSize(self.size)


class GroupStim(object):
    """
    A convenience class to put together stimuli in a single group.

    You can then do things like `stimgroup.draw()`.
    """

    def __init__(self, stimuli=None, name=None):
        if not isinstance(stimuli, tuple) and not isinstance(stimuli, list):
            self.stimuli = [stimuli]
        else:
            self.stimuli = stimuli
        if name is None:
            self.name = self.stimuli[0].name
        else:
            self.name = name

    def __getattr__(self, name):
        """Do whatever asked but per stimulus
        """
        def method(*args, **kwargs):
            outputs =[getattr(stim, name)(*args, **kwargs) for stim in self.stimuli]
            # see if only None returned, meaning that probably the function
            # doesn't return anything
            notnone = [o for o in outputs if o is not None]
            if len(notnone) != 0:
                return outputs
        try:
            return method
        except TypeError:
            return getattr(self, name)

    def __iter__(self):
        return self.stimuli.__iter__()


class OrderedDict(dict, DictMixin):
    """
    OrderedDict code (because some are stuck with Python 2.5)

    Produces an dictionary but with (key, value) pairs in the defined order.

    Created by Raymond Hettinger on Wed, 18 Mar 2009, under the MIT License
    <http://code.activestate.com/recipes/576693/>_
    """
    def __init__(self, *args, **kwds):
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        try:
            self.__end
        except AttributeError:
            self.clear()
        self.update(*args, **kwds)

    def clear(self):
        self.__end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.__map = {}                 # key --> [key, prev, next]
        dict.clear(self)

    def __setitem__(self, key, value):
        if key not in self:
            end = self.__end
            curr = end[1]
            curr[2] = end[1] = self.__map[key] = [key, curr, end]
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        key, prev, next = self.__map.pop(key)
        prev[2] = next
        next[1] = prev

    def __iter__(self):
        end = self.__end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.__end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def popitem(self, last=True):
        if not self:
            raise KeyError('dictionary is empty')
        if last:
            key = reversed(self).next()
        else:
            key = iter(self).next()
        value = self.pop(key)
        return key, value

    def __reduce__(self):
        items = [[k, self[k]] for k in self]
        tmp = self.__map, self.__end
        del self.__map, self.__end
        inst_dict = vars(self).copy()
        self.__map, self.__end = tmp
        if inst_dict:
            return (self.__class__, (items,), inst_dict)
        return self.__class__, (items,)

    def keys(self):
        return list(self)

    setdefault = DictMixin.setdefault
    update = DictMixin.update
    pop = DictMixin.pop
    values = DictMixin.values
    items = DictMixin.items
    iterkeys = DictMixin.iterkeys
    itervalues = DictMixin.itervalues
    iteritems = DictMixin.iteritems

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, self.items())

    def copy(self):
        return self.__class__(self)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        d = cls()
        for key in iterable:
            d[key] = value
        return d

    def __eq__(self, other):
        if isinstance(other, OrderedDict):
            return len(self)==len(other) and self.items() == other.items()
        return dict.__eq__(self, other)

    def __ne__(self, other):
        return not self == other

def combinations(iterable, r):
    """
    Produces combinations of `iterable` elements of lenght `r`.

    Examples:
        - combinations('ABCD', 2) --> AB AC AD BC BD CD
        - combinations(range(4), 3) --> 012 013 023 123

    `From Python 2.6 docs <http://docs.python.org/library/itertools.html#itertools.combinations>`_
    under the Python Software Foundation License

    :Args:
        - iterable
            A list-like or a str-like object that contains some elements
        - r
            Number of elements in each ouput combination

    :Returns:
        A generator yielding combinations of lenght `r`
    """
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)

def combinations_with_replacement(iterable, r):
    """
    Produces combinations of `iterable` elements of length `r` with
    replacement: identical elements can occur in together in some combinations.

    Example: combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC

    `From Python 2.6 docs <http://docs.python.org/library/itertools.html#itertools.combinations_with_replacement>`_
    under the Python Software Foundation License

    :Args:
        - iterable
            A list-like or a str-like object that contains some elements
        - r
            Number of elements in each ouput combination

    :Returns:
        A generator yielding combinations (with replacement) of length `r`
    """
    pool = tuple(iterable)
    n = len(pool)
    if not n and r:
        return
    indices = [0] * r
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != n - 1:
                break
        else:
            return
        indices[i:] = [indices[i] + 1] * (r - i)
        yield tuple(pool[i] for i in indices)

def try_makedirs(path):
        """Attempts to create a new directory.

        This function improves :func:`os.makedirs` behavior by printing an
        error to the log file if it fails and entering the debug mode
        (:mod:`pdb`) so that data would not be lost.

        :Args:
            path (str)
                A path to create.
        """
        if not os.path.isdir(path) and path not in ['','.','./']:
            try: # if this fails (e.g. permissions) we will get an error
                os.makedirs(path)
            except:
                logging.error('ERROR: Cannot create a folder for storing data %s' %path)
                # FIX: We'll enter the debugger so that we don't lose any data
                import pdb; pdb.set_trace()

def signal_det(corr_resp, subj_resp):
    """
    Returns an accuracy label according the (modified) Signal Detection Theory.

    ================  ===================  =================
                      Response present     Response absent
    ================  ===================  =================
    Stimulus present  correct / incorrect  miss
    Stimulus absent   false alarm          (empty string)
    ================  ===================  =================

    :Args:
        corr_resp
            What one should have responded. If no response expected
            (e.g., no stimulus present), then it should be an empty string
            ('')
        subj_resp
            What the observer responsed. If no response, it should be
            an empty string ('').
    :Returns:
        A string indicating the type of response.
    """
    if corr_resp == '':  # stimulus absent
        if subj_resp == '':  # response absent
            resp = ''
        else:  # response present
            resp = 'false alarm'
    else:  # stimulus present
        if subj_resp == '':  # response absent
            resp = 'miss'
        elif corr_resp == subj_resp:  # correct response present
            resp = 'correct'
        else:  # incorrect response present
            resp = 'incorrect'
    return resp

def invert_dict(d):
    """
    Inverts a dictionary: keys become values.

    This is an instance of an OrderedDict, and so the new keys are
    sorted.

    :Args:
        d: dict
    """
    inv_dict = dict([[v,k] for k,v in d.items()])
    sortkeys = sorted(inv_dict.keys())
    inv_dict = OrderedDict([(k,inv_dict[k]) for k in sortkeys])
    return inv_dict

def get_mon_sizes(screen=None):
    """Get a list of resolutions for each monitor.

    Recipe from <http://stackoverflow.com/a/10295188>_

    :Args:
        screen (int, default: None)
            Which screen's resolution to return. If None, the a list of all
            screens resolutions is returned.

    :Returns:
        a tuple or a list of tuples of each monitor's resolutions
    """
    app = wx.App(False)  # create an app if there isn't one and don't show it
    nmons = wx.Display.GetCount()  # how many monitors we have
    mon_sizes = [wx.Display(i).GetGeometry().GetSize() for i in range(nmons)]
    if screen is None:
        return mon_sizes
    else:
        return mon_sizes[screen]

def get_para_no(file_pattern, n=6):
    """Looks up used para numbers and returns a new one for this run
    """
    all_data = glob.glob(file_pattern)
    if all_data == []: paranos = random.choice(range(n))
    else:
        paranos = []
        for this_data in all_data:
            lines = csv.reader( open(this_data) )
            try:
                header = lines.next()
                ind = header.index('paraNo')
                this_parano = lines.next()[ind]
                paranos.append(int(this_parano))
            except: pass

        if paranos != []:
            count_used = np.bincount(paranos)
            count_used = np.hstack((count_used,np.zeros(n-len(count_used))))
            poss_paranos = np.arange(n)
            paranos = random.choice(poss_paranos[count_used == np.min(count_used)].tolist())
        else: paranos = random.choice(range(n))

    return paranos

def get_unique_trials(trial_list, column='cond'):
    unique = []
    conds = []
    for trial in trial_list:
        if trial[column] not in conds:
            unique.append(OrderedDict(trial))
            conds.append(trial[column])
    # this does an argsort
    order = sorted(range(len(conds)), key=conds.__getitem__)
    # return an ordered list
    return [unique[c] for c in order]
