# Part of the psychopy_ext library
# Copyright 2010-2014 Jonas Kubilius
# The program is distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.

"""
A library of helper functions for creating and running experiments.

All experiment-related methods are kept here.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, os, csv, glob, random, warnings, copy
from UserDict import DictMixin
from collections import OrderedDict

import numpy as np
import wx

# for HTML rendering
import pyglet
import textwrap
from HTMLParser import HTMLParser

# for exporting stimuli to svg
try:
    import svgwrite
except:
    no_svg = True
else:
    no_svg = False

import psychopy.info
from psychopy import visual, core, event, logging, misc, monitors, data
from psychopy.data import TrialHandler, ExperimentHandler
from psychopy.tools.attributetools import attributeSetter

from psychopy_ext import ui
from psychopy_ext.version import __version__ as psychopy_ext_version

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


class Task(TrialHandler):

    def __init__(self,
                 parent,
                 name='',
                 version='0.1',
                 method='random',
                 data_fname=None,
                 blockcol=None
                 ):
        """
        An extension of TrialHandler with many useful functions.

        :Args:
            parent (:class:`Experiment`)
                The Experiment to which this Tast belongs.

        :Kwargs:
            - name (str, default: '')
                Name of the task. Currently not used anywhere.
            - version (str, default: '0.1')
                Version of your experiment.
            - method ({'sequential', 'random'}, default: 'random')
                Order of trials:

                    - sequential: trials and blocks presented sequentially
                    - random: trials presented randomly, blocks sequentially
                    - fullRandom: converted to 'random'

                Note that there is no explicit possibility to randomize
                the order of blocks. This is intentional because you
                in fact define block order in the `blockcol`.

            - data_fname (str, default=None)
                The name of the main data file for storing output. If None,
                reuses :class:`~psychopy_ext.exp.Datafile` instance from
                its parent; otherwise, a new one is created
                (stored in ``self.datafile``).
            - blockcol (str, default: None)
                Column name in `self.exp_plan` that defines which trial
                should be presented during which block.
        """
        self.parent = parent
        self.computer = self.parent.computer
        self.paths = self.parent.paths

        self.name = name
        self.version = version
        self.nReps = 1  # fixed

        self.method = method
        if method == 'randomFull':
            self.method = 'random'

        if data_fname is None:
            self.datafile = parent.datafile
        else:
            self.datafile = Datafile(data_fname, writeable=not self.parent.rp['no_output'])
        self.blockcol = blockcol
        self.computer.valid_responses = parent.computer.valid_responses

        self._exit_key_no = 0
        self.blocks = []

        #self.info = parent.info
        #self.extraInfo = self.info  # just for compatibility with PsychoPy
        #self.rp = parent.rp

    def __str__(self, **kwargs):
        """string representation of the object"""
        return 'psychopy_ext.exp.Task'

    def quit(self, message=''):
        """What to do when exit is requested.
        """
        print  # in case there was anything without \n
        logging.warning(message)

        self.win.flip()
        #self.rectimes.append(self.win.frameIntervals[-1])
        if not self.rp['no_output']:
            named_flips = [n for n in self.win.flipnames[:-1] if n != '']
            if len(named_flips) > 0:
                framelogname = self.paths['logs'] + self.info['subjid'] + '_timing.csv'
                self.framelog = Datafile(framelogname, writeable=not self.rp['no_output'])
                with open(framelogname, 'ab') as f:
                    writer = csv.writer(f, lineterminator = '\n')
                    writer.writerow(['event', 'time'])
                    for name, time in zip(self.win.flipnames[:-1], self.win.frameIntervals):
                        if name != '':
                            writer.writerow([name, '%.6f' % time])

        self.win.close()
        if not self.rp['no_output']:
            self.logfile.write('End time: %s\n' % data.getDateStr(format="%Y-%m-%d %H:%M"))
            self.logfile.write('end')
        core.quit()

    def flip(self, name=None, *args, **kwargs):
        self.win.flip_orig(*args, **kwargs)
        if self.win.recordFrameIntervals and not self.win.recordFrameIntervalsJustTurnedOn:
            if name is None:
                try:
                    name = self.this_event.name
                except:
                    name = ''
            self.win.flipnames.append(name)

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
        if not self.parent._initialized:
            raise Exception('You must first call Experiment.setup()')

        self.win = self.parent.win
        self.logfile = self.parent.logfile
        self.info = self.parent.info
        self.rp = self.parent.rp
        self.mouse = self.parent.mouse
        self.datafile.writeable = not self.rp['no_output']

        self._set_keys_flat()
        self.set_seed()
        self.create_stimuli()
        self.create_trial()
        if not hasattr(self, 'trial'):
            raise Exception('self.trial variable must be created '
                            'with the self.create_trial() method')
        # for backward compatibility: convert event dict into Event
        if isinstance(self.trial[0], dict):
            self.trial = [Event._fromdict(self, ev) for ev in self.trial]

        self.create_exp_plan()
        if not hasattr(self, 'exp_plan'):
            raise Exception('self.exp_plan variable must be created '
                            'with the self.create_exp_plan() method')

        ## convert Event.dur to a list of exp_plan length
        #for ev in self.trial:
            #if isinstance(ev.dur, (int, float)):
                #ev.dur = [ev.dur] * len(self.exp_plan)

        # determine if syncing to global time is necessary
        self.global_timing = True
        for ev in self.trial:
            # if event sits there waiting, global time does not apply
            if np.any(ev.dur == 0) or np.any(np.isinf(ev.dur)):
                self.global_timing = False
                break

        if self.rp['autorun'] > 0:
            # speed up the experiment
            for ev in self.trial:  # speed up each event
                #ev.dur = map(lambda x: float(x)/self.rp['autorun'], ev.dur)
                ev.dur /= self.rp['autorun']
            self.exp_plan = self.set_autorun(self.exp_plan)

        self.get_blocks()
        self.win.flip = self.flip

    def _set_keys_flat(self):
        #if keylist is None:
        keylist = self.computer.default_keys.values()
        #else:
            #keylist.extend(self.computer.default_keys.values())
        keys = []
        for key in keylist:
            if isinstance(key, (tuple, list)):
                keys.append(key)
            else:
                keys.append([key])
        # keylist might have key combinations; get rid of them for now
        self.keylist_flat = []
        for key in keys:
            self.keylist_flat.extend(key)

    def set_seed(self):
        # re-initialize seed for each block of task
        # (if there is more than one task or more than one block)
        if len(self.parent.tasks) > 1 or len(self.blocks) > 1:
            self.seed = int(core.getAbsTime())  # generate a new seed
            date = data.getDateStr(format="%Y_%m_%d %H:%M (Year_Month_Day Hour:Min)")
            random.seed(self.seed)
            np.random.seed(self.seed)

            if not self.rp['no_output']:
                try:
                    message = 'Task %s: block %d' % (self.__str__, self.this_blockn+1)
                except:
                    message = 'Task %s' % self.__str__

                self.logfile.write('\n')
                self.logfile.write('#[ PsychoPy2 RuntimeInfoAppendStart ]#\n')
                self.logfile.write('  #[[ %s ]] #---------\n' % message)
                self.logfile.write('    taskRunTime: %s\n' % date)
                self.logfile.write('    taskRunTime.epoch: %d\n' % self.seed)
                self.logfile.write('#[ PsychoPy2 RuntimeInfoappendEnd ]#\n')
                self.logfile.write('\n')

        else:
            self.seed = self.parent.seed

    def show_text(self, text='', stimuli=None, wait=0, wait_stim=None, auto=0):
        """
        Presents an instructions screen.

        :Kwargs:
            - text (str, default: None)
                Text to show.
            - stimuli (obj or list, default: None)
                Any stimuli to show along with the text?
            - wait (float, default: 0)
                How long to wait after the end of showing instructions,
                in seconds.
            - wait_stim (stimulus or a list of stimuli, default: None)
                During this waiting, which stimuli should be shown.
                Usually, it would be a fixation spot.
            - auto (float, default: 0)
                Duration of time-out of the instructions screen,
                in seconds.
        """
        def _get_rect(stim):
            rect = (stim.pos[0]-stim.size[0]/2,
                    stim.pos[1]-stim.size[1]/2,
                    stim.pos[0]+stim.size[0]/2,
                    stim.pos[1]+stim.size[1]/2)
            if stim.units == 'cm':
                func = misc.cm2pix
            elif stim.units == 'deg':
                func = misc.deg2pix
            return tuple([func(r, self.win.monitor) for r in rect])

        # for some graphics drivers (e.g., mine:)
        # draw() command needs to be invoked once
        # before it can draw properly
        visual.TextStim(self.win, text='').draw()
        self.win.flip()

        rect = []
        if stimuli is not None:
            if not isinstance(stimuli, (tuple, list)):
                stimuli = [stimuli]
            rect = [_get_rect(stim) for stim in stimuli]

        if len(rect) > 0:
            rect = np.array(rect).T
            stim_height = np.max(rect[3]) - np.min(rect[1])
            #[np.max(rect[2]) - np.min(rect[0]),
                        #]
        else:
            stim_height = 0

        if text is not None:
            instructions = self._parse_instructions(text)
            text_height = instructions._pygletTextObj.content_height
            if stimuli is not None:
                gap = misc.deg2pix(1, self.win.monitor) / 2
            else:
                gap = 0
            instructions.pos = (0, stim_height/2 + gap)
        else:
            text_height = 0

        if stimuli is not None:
            for stim in stimuli:
                if stim.units == 'deg':
                    func = misc.pix2deg
                elif stim.units == 'cm':
                    func = misc.pix2cm
                y = func(-text_height/2, self.win.monitor)
                if text is not None:
                    y -= .5
                stim.pos = (stim.pos[0], y)
                stim.draw()
        if text is not None:
            instructions.draw()

        self.win.flip()

        if self.rp['unittest']:
            print(text)

        if auto > 0:  # show text and blank out
            if self.rp['autorun']:
                auto = auto / self.rp['autorun']
            core.wait(auto)
        elif not self.rp['autorun'] or not self.rp['unittest']:
            this_key = None
            while this_key != self.computer.default_keys['trigger']:
                this_key = self.last_keypress()
                if len(this_key) > 0:
                    this_key = this_key.pop()
            if self.rp['autorun']:
                wait /= self.rp['autorun']
        self.win.flip()

        if wait_stim is not None:
            if not isinstance(wait_stim, (tuple, list)):
                wait_stim = [wait_stim]
            for stim in wait_stim:
                stim.draw()
            self.win.flip()
        core.wait(wait)  # wait a little bit before starting the experiment
        event.clearEvents()  # clear keys

    def _parse_instructions(self, text):

        #instructions = visual.TextStim(self.win, text=text,
                                    #color='white', height=20, units='pix',
                                    #pos=(0, 0),  # don't know why
                                    #wrapWidth=40*20)

        text = textwrap.dedent(text)
        if text.find('\n') < 0:  # single line, no formatting
            html = '<h2><font face="sans-serif">%s</font></h2>' % text
            instr = visual.TextStim(self.win, units='pix')
            instr._pygletTextObj = pyglet.text.HTMLLabel(html)
            width = instr._pygletTextObj.content_width
            multiline = False
        else:
            try:
                import docutils.core
            except:  # will make plain formatting
                html = '<p><font face="sans-serif">%s</font></p>' % text
                html = html.replace('\n\n', '</font></p><p><font face="sans-serif">')
                html = html.replace('\n', '</font><br /><font face="sans-serif">')
                width = 40*12
                multiline = True
            else:
                html = docutils.core.publish_parts(text,
                                        writer_name='html')['html_body']
                html = _HTMLParser().feed(html)
                width = 40*12
                multiline = True

        instructions = visual.TextStim(self.win, units='pix',
                                       wrapWidth=width)
        instructions._pygletTextObj = pyglet.text.HTMLLabel(html,
                                  width=width, multiline=multiline,
                                  x=0, anchor_x='left', anchor_y='center')

        return instructions

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
            edges = 8
            d = np.pi*2 / (4*edges)
            verts = [(r1*np.sin(e*d), r1*np.cos(e*d)) for e in xrange(edges+1)]
            verts.append([0,0])
            oval_pos = [(r2,r2), (r2,-r2), (-r2,-r2), (-r2,r2)]

            oval = []
            for i in range(4):
                oval.append(visual.ShapeStim(
                    self.win,
                    name = 'oval',
                    fillColor = color,
                    lineColor = None,
                    vertices = verts,
                    ori = 90*i,
                    pos = oval_pos[i]
                ))
            center = visual.Circle(
                self.win,
                name   = 'center',
                fillColor  = color,
                lineColor = None,
                radius   = r2,
            )
            fixation = GroupStim(stimuli=oval + [center],
                                 name='fixation')
            fixation.color = color
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
        Create a list of events that constitute a trial (``self.trial``).

        Example::

            self.trial = [exp.Event(self,
                                    dur=.100,
                                    display=self.s['fix'],
                                    func=self.idle_event),
                          exp.Event(self,
                                    dur=.300,
                                    display=self.s['stim1'],
                                    func=self.during_trial),
                           ]
        """
        raise NotImplementedError

    def create_exp_plan(self):
        """
        Put together trials into ``self.exp_plan``.

        Example::

            self.exp_plan = []
            for ...:
                exp_plan.append([
                    OrderedDict([
                        ('cond', cond),
                        ('name', names[cond]),
                        ('onset', ''),
                        ('dur', trial_dur),
                        ('corr_resp', corr_resp),
                        ('subj_resp', ''),
                        ('accuracy', ''),
                        ('rt', ''),
                        ])
                    ])
        """
        raise NotImplementedError

    def get_mouse_resp(self, keyList=None, timeStamped=False):
        """
        Returns mouse clicks.

        If ``self.respmap`` is provided, records clicks only when clicked
        inside respmap. This respmap is supposed to be a list of shape
        objects that determine boundaries of where one can click.
        Might change in the future if it gets incorporated in stimuli
        themselves.

        Note that mouse implementation is a bit shaky in PsychoPy at
        the moment. In particular, ``getPressed`` method returns
        multiple key down events per click. Thus, when calling
        ``get_mouse_resp`` from a while loo[, it is best to limit
        sampling to, for example, 150 ms (see `Jeremy's response <https://groups.google.com/d/msg/psychopy-users/HG4L-UDG93Y/FvyuB-OrsqoJ>`_).
        """
        mdict = {0: 'left-click', 1: 'middle-click', 2: 'right-click'}
        valid_mouse = [k for k,v in mdict.items() if v in self.computer.valid_responses or v in keyList]
        valid_mouse.sort()

        if timeStamped:
            mpresses, mtimes = self.mouse.getPressed(getTime=True)
        else:
            mpresses = self.mouse.getPressed(getTime=False)

        resplist = []
        if sum(mpresses) > 0:
            for but in valid_mouse:
                if mpresses[but] > 0:
                    if timeStamped:
                        resplist.append([mdict[but],mtimes[but]])
                    else:
                        resplist.append([mdict[but], None])
            if hasattr(self, 'respmap'):
                clicked = False
                for box in self.respmap:
                    if box.contains(self.mouse):
                        resplist = [tuple(r+[box]) for r in resplist]
                        clicked = True
                        break
                if not clicked:
                    resplist = []

        return resplist

    def get_resp(self, keyList=None, timeStamped=False):
        resplist = event.getKeys(keyList=keyList, timeStamped=timeStamped)
        if resplist is None:
            resplist = []

        mresp = self.get_mouse_resp(keyList=keyList, timeStamped=timeStamped)

        resplist += mresp
        return resplist

    def last_keypress(self, keyList=None, timeStamped=False):
        """
        Extract the last key pressed from the event list.

        If exit key is pressed (default: 'Left Shift + Esc'), quits.

        :Returns:
            A list of keys pressed.
        """
        if keyList is None:
            keyList = self.keylist_flat
        this_keylist = self.get_resp(keyList=keyList+self.keylist_flat,
                                     timeStamped=timeStamped)
        keys = []
        for this_key in this_keylist:
            isexit = self._check_if_exit(this_key)
            if not isexit:
                self._exit_key_no = 0
                isin_keylist = self._check_if_in_keylist(this_key, keyList)
                if isin_keylist:  # don't want to accept triggers and such
                    keys.append(this_key)
        return keys

    def _check_if_exit(self, this_key):
        """
        Checks if there one of the exit keys was pressed.

        :Args:
            this_key (str or tuple)
                Key or time-stamped key to check
        :Returns:
            True if any of the ``self.computer.default_keys['exit']``
            keys were pressed, False otherwise.
        """
        exit_keys = self.computer.default_keys['exit']

        if isinstance(this_key, tuple):
            this_key_exit = this_key[0]
        else:
            this_key_exit = this_key

        if this_key_exit in exit_keys:
            if self._exit_key_no < len(exit_keys):
                if exit_keys[self._exit_key_no] == this_key_exit:
                    if self._exit_key_no == len(exit_keys) - 1:
                        self.quit('Premature exit requested by user.')
                    else:
                        self._exit_key_no += 1
                else:
                    self._exit_key_no = 0
            else:
                self._exit_key_no = 0

        return self._exit_key_no > 0

    def _check_if_in_keylist(self, this_key, keyList):
        if isinstance(this_key, tuple):
            this_key_check = this_key[0]
        else:
            this_key_check = this_key
        return this_key_check in keyList

    def before_event(self):
        for stim in self.this_event.display:
            stim.draw()
        self.win.flip()

    def after_event(self):
        pass

    def wait_until_response(self, draw_stim=True):
        """
        Waits until a response key is pressed.

        Returns last key pressed, timestamped.

        :Kwargs:
            draw_stim (bool, default: True)
                Controls if stimuli should be drawn or have already
                been drawn (useful if you only want to redefine
                the drawing bit of this function).

        :Returns:
            A list of tuples with a key name (str) and a response time (float).
        """
        if draw_stim:
            self.before_event()

        event_keys = []
        event.clearEvents() # key presses might be stored from before
        while len(event_keys) == 0: # if the participant did not respond earlier
            if 'autort' in self.this_trial:
                if self.trial_clock.getTime() > self.this_trial['autort']:
                    event_keys = [(self.this_trial['autoresp'], self.this_trial['autort'])]
            else:
                event_keys = self.last_keypress(
                    keyList=self.computer.valid_responses.keys(),
                    timeStamped=self.trial_clock)

        return event_keys

    def idle_event(self, draw_stim=True):
        """
        Default idle function for an event.

        Sits idle catching default keys (exit and trigger).

        :Kwargs:
            draw_stim (bool, default: True)
                Controls if stimuli should be drawn or have already
                been drawn (useful if you only want to redefine
                the drawing bit of this function).

        :Returns:
            A list of tuples with a key name (str) and a response time (float).
        """
        if draw_stim:
            self.before_event()

        event_keys = None
        event.clearEvents() # key presses might be stored from before
        if self.this_event.dur == 0 or self.this_event.dur == np.inf:
            event_keys = self.last_keypress()
        else:
            event_keys = self.wait()
        return event_keys

    def feedback(self):
        """
        Gives feedback by changing fixation color.

        - Correct: fixation change to green
        - Wrong: fixation change to red
        """
        this_resp = self.all_keys[-1]
        if hasattr(self, 'respmap'):
            subj_resp = this_resp[2]
        else:
            subj_resp = self.computer.valid_responses[this_resp[0]]
        #subj_resp = this_resp[2]  #self.computer.valid_responses[this_resp[0]]

        # find which stimulus is fixation
        if isinstance(self.this_event.display, (list, tuple)):
            for stim in self.this_event.display:
                if stim.name in ['fixation', 'fix']:
                    fix = stim
                    break
        else:
            if self.this_event.display.name in ['fixation', 'fix']:
                fix = self.this_event.display

        if fix is not None:
            orig_color = fix.color  # store original color
            if self.this_trial['corr_resp'] == subj_resp:
                fix.setFillColor('DarkGreen')  # correct response
            else:
                fix.setFillColor('DarkRed')  # incorrect response

        for stim in self.this_event.display:
            stim.draw()
        self.win.flip()

        # sit idle
        self.wait()

        # reset fixation color
        fix.setFillColor(orig_color)

    def wait(self):
        """
        Wait until the event is over, register key presses.

        :Returns:
            A list of tuples with a key name (str) and a response time (float).
        """
        all_keys = []
        while self.check_continue():
            keys = self.last_keypress()
            if keys is not None:
                all_keys += keys

        return all_keys

    def check_continue(self):
        """
        Check if the event is not over yet.

        Uses ``event_clock``, ``trial_clock``, and, if
        ``self.global_timing`` is True, ``glob_clock`` to check whether
        the current event is not over yet. The event cannot last longer
        than event and trial durations and also fall out of sync from
        global clock.

        :Returns:
            A list of tuples with a key name (str) and a response time (float).
        """
        event_on = self.event_clock.getTime() < self.this_event.dur
        if self.global_timing:
            trial_on = self.trial_clock.getTime() < self.this_trial['dur']
            time_on = self.glob_clock.getTime() < self.cumtime + self.this_trial['dur']
        else:
            trial_on = True
            time_on = True
        return (event_on and trial_on and time_on)

    def set_autorun(self, exp_plan):
        """
        Automatically runs experiment by simulating key responses.

        This is just the absolute minimum for autorunning. Best practice would
        be extend this function to simulate responses according to your
        hypothesis.

        :Args:
            exp_plan (list of dict)
                A list of trial definitions.

        :Returns:
            exp_plan with ``autoresp`` and ``autort`` columns included.
        """
        def rt(mean):
            add = np.random.normal(mean,scale=.2)/self.rp['autorun']
            return self.trial[0].dur + add

        inverse_resp = invert_dict(self.computer.valid_responses)

        for trial in exp_plan:
            # here you could do if/else to assign different values to
            # different conditions according to your hypothesis
            trial['autoresp'] = random.choice(inverse_resp.values())
            trial['autort'] = rt(.5)
        return exp_plan


    def set_TrialHandler(self, trial_list, trialmap=None):
        """
        Converts a list of trials into a `~psychopy.data.TrialHandler`,
        finalizing the experimental setup procedure.
        """
        if len(self.blocks) > 1:
            self.set_seed()
        TrialHandler.__init__(self,
            trial_list,
            nReps=self.nReps,
            method=self.method,
            extraInfo=self.info,
            name=self.name,
            seed=self.seed)
        if trialmap is None:
            self.trialmap = range(len(trial_list))
        else:
            self.trialmap = trialmap

    def get_blocks(self):
        """
        Finds blocks in the given column of ``self.exp_plan``.

        The relevant column is stored in ``self.blockcol`` which is
        given by the user when initializing the experiment class.

        Produces a list of trial lists and trial mapping for each block.
        Trial mapping indicates where each trial is in the original
        `exp_plan` list.

        The output is stored in ``self.blocks``.
        """
        if self.blockcol is not None:
            blocknos = np.array([trial[self.blockcol] for trial in self.exp_plan])
            _, idx = np.unique(blocknos, return_index=True)
            blocknos = blocknos[idx].tolist()
            blocks = [None] * len(blocknos)
            for trialno, trial in enumerate(self.exp_plan):
                blockno = blocknos.index(trial[self.blockcol])
                if blocks[blockno] is None:
                    blocks[blockno] = [[trial], [trialno]]
                else:
                    blocks[blockno][0].append(trial)
                    blocks[blockno][1].append(trialno)
        else:
            blocks = [[self.exp_plan, range(len(self.exp_plan))]]
        self.blocks = blocks

    def before_task(self, text=None, wait=.5, wait_stim=None, **kwargs):
        """Shows text from docstring explaining the task.

        :Kwargs:
            - text (str, default: None)
                Text to show.
            - wait (float, default: .5)
                How long to wait after the end of showing instructions,
                in seconds.
            - wait_stim (stimulus or a list of stimuli, default: None)
                During this waiting, which stimuli should be shown.
                Usually, it would be a fixation spot.
            - \*\*kwargs
                Other parameters for :func:`~psychopy_ext.exp.Task.show_text()`
        """
        if len(self.parent.tasks) > 1:
            # if there are no blocks, try to show fixation
            if wait_stim is None:
                if len(self.blocks) <= 1:
                    try:
                        wait_stim = self.s['fix']
                    except:
                        wait = 0
                else:
                    wait = 0
            if text is None:
                self.show_text(text=self.__doc__, wait=wait,
                               wait_stim=wait_stim, **kwargs)
            else:
                self.show_text(text=text, wait=wait,
                               wait_stim=wait_stim, **kwargs)

    def run_task(self):
        """Sets up the task and runs it.

        If ``self.blockcol`` is defined, then runs block-by-block.
        """
        self.setup_task()
        self.before_task()

        self.datafile.open()
        for blockno, (block, trialmap) in enumerate(self.blocks):
            self.this_blockn = blockno
            # set TrialHandler only to the current block
            self.set_TrialHandler(block, trialmap=trialmap)
            self.run_block()
        self.datafile.close()

        self.after_task()

    def after_task(self, text=None, auto=1, **kwargs):
        """Useful for showing feedback after a task is done.

        For example, you could display accuracy.

        :Kwargs:
            - text (str, default: None)
                Text to show. If None, this is skipped.
            - auto (float, default: 1)
                Duration of time-out of the instructions screen,
                in seconds.
            - \*\*kwargs
                Other parameters for :func:`~psychopy_ext.exp.Task.show_text()`
        """
        if text is not None:
            self.show_text(text, auto=auto, **kwargs)

    def before_block(self, text=None, auto=1, wait=.5, wait_stim=None):
        """Show text before the block starts.

        Will not show anything if there's only one block.

        :Kwargs:
            - text (str, default: None)
                Text to show. If None, defaults to showing block number.
            - wait (float, default: .5)
                How long to wait after the end of showing instructions,
                in seconds.
            - wait_stim (stimulus or a list of stimuli, default: None)
                During this waiting, which stimuli should be shown.
                Usually, it would be a fixation spot. If None, this
                fixation spot will be attempted to be drawn.
            - auto (float, default: 1)
                Duration of time-out of the instructions screen,
                in seconds.
        """
        if len(self.blocks) > 1:
            if wait_stim is None:
                try:
                    wait_stim = self.s['fix']
                except:
                    pass
            if text is None:
                self.show_text(text='Block %d' % (self.this_blockn+1),
                               auto=auto, wait=wait, wait_stim=wait_stim)
            else:
                self.show_text(text=text, auto=auto, wait=wait, wait_stim=wait_stim)

    def run_block(self):
        """Run a block in a task.
        """
        self.before_block()

        # set up clocks
        self.glob_clock = core.Clock()
        self.trial_clock = core.Clock()
        self.event_clock = core.Clock()
        self.cumtime = 0

        # go over the trial sequence
        for this_trial in self:
            self.this_trial = this_trial
            self.run_trial()

        self.after_block()

    def after_block(self, text=None, **kwargs):
        """Show text at the end of a block.

        Will not show this text after the last block in the task.

        :Kwargs:
            - text (str, default: None)
                Text to show. If None, will default to
                'Pause. Hit ``trigger`` to continue.'
            - \*\*kwargs
                Other parameters for :func:`~psychopy_ext.exp.Task.show_text()`
        """
        # clear trial counting in the terminal
        sys.stdout.write('\r' + ' '*70)
        sys.stdout.write('\r')
        sys.stdout.flush()
        if text is None:
            text = ('Pause. Hit %s to continue.' %
                                self.computer.default_keys['trigger'])
        # don't show this after the last block
        if self.this_blockn+1 < len(self.blocks):
            self.show_text(text=text, **kwargs)

    def before_trial(self):
        """What to do before trial -- nothing by default.
        """
        pass

    def run_trial(self):
        """Presents a trial.
        """
        self.before_trial()

        self.trial_clock.reset()
        self.this_trial['onset'] = self.glob_clock.getTime()
        sys.stdout.write('\rtrial %s' % (self.thisTrialN+1))
        sys.stdout.flush()

        self.this_trial['dur'] = 0
        for ev in self.trial:
            if ev.durcol is not None:
                ev.dur = self.this_trial[ev.durcol]
            self.this_trial['dur'] += ev.dur

        self.all_keys = []
        self.rectimes = []
        for event_no, this_event in enumerate(self.trial):
            self.this_event = this_event
            self.event_no = event_no
            self.run_event()

        # if autorun and responses were not set yet, get them now
        if len(self.all_keys) == 0 and self.rp['autorun'] > 0:
            self.all_keys += [(self.this_trial['autoresp'], self.this_trial['autort'])]

        self.post_trial()

        # correct timing if autorun
        if self.rp['autorun'] > 0:
            try:
                self.this_trial['autort'] *= self.rp['autorun']
                self.this_trial['rt'] *= self.rp['autorun']
            except:  # maybe not all keys are present
                pass
            self.this_trial['onset'] *= self.rp['autorun']
            self.this_trial['dur'] *= self.rp['autorun']

        self.datafile.write_header(self.info.keys() + self.this_trial.keys())
        self.datafile.write(self.info.values() + self.this_trial.values())

        self.cumtime += self.this_trial['dur']
        # update exp_plan with new values
        try:
            self.exp_plan[self.trialmap[self.thisIndex]] = self.this_trial
        except:  # for staircase
            self.exp_plan.append(self.this_trial)

    def after_trial(self):
        """Alias to :func:`~psychopy_ext.exp.Task.post_trial()`
        """
        self.post_trial()

    def post_trial(self):
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
        if len(self.all_keys) > 0:
            this_resp = self.all_keys.pop()
            if hasattr(self, 'respmap'):
                subj_resp = this_resp[2]
            else:
                subj_resp = self.computer.valid_responses[this_resp[0]]
            self.this_trial['subj_resp'] = subj_resp

            try:
                acc = signal_det(self.this_trial['corr_resp'], subj_resp)
            except:
                pass
            else:
                self.this_trial['accuracy'] = acc
                self.this_trial['rt'] = this_resp[1]
        else:
            self.this_trial['subj_resp'] = ''
            try:
                acc = signal_det(self.this_trial['corr_resp'], self.this_trial['subj_resp'])
            except:
                pass
            else:
                self.this_trial['accuracy'] = acc
                self.this_trial['rt'] = ''

    def run_event(self):
        """Presents a trial and catches key presses.
        """
        # go over each event in a trial
        self.event_clock.reset()
        self.mouse.clickReset()

        # show stimuli
        event_keys = self.this_event.func()

        if isinstance(event_keys, tuple):
            event_keys = [event_keys]
        elif event_keys is None:
            event_keys = []

        if len(event_keys) > 0:
            self.all_keys += event_keys
        # this is to get keys if we did not do that during trial
        self.all_keys += self.last_keypress(
            keyList=self.computer.valid_responses.keys(),
            timeStamped=self.trial_clock)

        #if self.this_event.rectime:
            #if len(self.win.frameIntervals) > 0:
                #self.rectimes.append(self.win.frameIntervals[-1])
            #self.win.frameIntervals = []

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


class SVG(object):

    def __init__(self, win, filename='image'):
        if no_svg:
            raise ImportError("Module 'svgwrite' not found.")
        #visual.helpers.setColor(win, win.color)
        win.contrast = 1
        self.win = win
        self.aspect = self.win.size[0]/float(self.win.size[1])
        self.open(filename)

    def open(self, filename):
        filename = filename.split('.svg')[0]

        self.svgfile = svgwrite.Drawing(profile='tiny',filename='%s.svg' % filename,
                                size=('%dpx' % self.win.size[0],
                                      '%dpx' % self.win.size[1]),
                                # set default units to px; from http://stackoverflow.com/a/13008664
                                viewBox=('%d %d %d %d' %
                                         (0,0,
                                          self.win.size[0],
                                          self.win.size[1]))
                                )
        bkgr = self.svgfile.rect(insert=(0,0), size=('100%','100%'),
                            fill=self.color2rgb255(self.win))
        self.svgfile.add(bkgr)

    def save(self):
        self.svgfile.save()

    def color2attr(self, stim, attr, color='black', colorSpace=None, kwargs=None):
        if kwargs is None: kwargs = {}
        col = self.color2rgb255(stim, color=color, colorSpace=colorSpace)
        if col is None:
            kwargs[attr + '_opacity'] = 0
        else:
            kwargs[attr] = col
            kwargs[attr + '_opacity'] = 1
        return kwargs

    def write(self, stim):
        if 'Circle' in str(stim):
            color_kw = self.color2attr(stim, 'stroke', color=stim.lineColor,
                                       colorSpace=stim.lineColorSpace)
            color_kw = self.color2attr(stim, 'fill', color=stim.fillColor,
                                       colorSpace=stim.fillColorSpace,
                                       kwargs=color_kw)
            svgstim = self.svgfile.circle(
                            center=self.get_pos(stim),
                            r=self.get_size(stim, stim.radius),
                            stroke_width=stim.lineWidth,
                            opacity=stim.opacity,
                            **color_kw
                            )
        elif 'ImageStim' in str(stim):
            raise NotImplemented
        elif 'Line' in str(stim):
            color_kw = self.color2attr(stim, 'stroke', color=stim.lineColor,
                                       colorSpace=stim.lineColorSpace)
            svgstim = self.svgfile.line(
                    start=self.get_pos(stim, stim.start),
                    end=self.get_pos(stim, stim.end),
                    stroke_width=stim.lineWidth,
                    opacity=stim.opacity,
                    **color_kw
                    )
        elif 'Polygon' in str(stim):
            raise NotImplemented
            #svgstim = self.svgfile.polygon(
                    #points=...,
                    #stroke_width=stim.lineWidth,
                    #stroke=self.color2rgb255(stim, color=stim.lineColor,
                                             #colorSpace=stim.lineColorSpace),
                    #fill=self.color2rgb255(stim, color=stim.fillColor,
                                           #colorSpace=stim.fillColorSpace)
                    #)
        elif 'Rect' in str(stim):
            color_kw = self.color2attr(stim, 'stroke', color=stim.lineColor,
                                       colorSpace=stim.lineColorSpace)
            color_kw = self.color2attr(stim, 'fill', color=stim.fillColor,
                                       colorSpace=stim.fillColorSpace,
                                       kwargs=color_kw)
            svgstim = self.svgfile.rect(
                    insert=self.get_pos(stim, offset=(-stim.width/2., -stim.height/2.)),
                    size=(self.get_size(stim, stim.width), self.get_size(stim, stim.height)),
                    stroke_width=stim.lineWidth,
                    opacity=stim.opacity,
                    **color_kw
                    )
        elif 'ThickShapeStim' in str(stim):
            svgstim = stim.to_svg(self)
        elif 'ShapeStim' in str(stim):
            points = self._calc_attr(stim, np.array(stim.vertices))
            points[:, 1] *= -1
            color_kw = self.color2attr(stim, 'stroke', color=stim.lineColor,
                                       colorSpace=stim.lineColorSpace)
            color_kw = self.color2attr(stim, 'fill', color=stim.fillColor,
                                       colorSpace=stim.fillColorSpace,
                                       kwargs=color_kw)
            if stim.closeShape:
                svgstim = self.svgfile.polygon(
                        points=points,
                        stroke_width=stim.lineWidth,
                        opacity=stim.opacity,
                        **color_kw
                        )
            else:
                svgstim = self.svgfile.polyline(
                        points=points,
                        stroke_width=stim.lineWidth,
                        opacity=stim.opacity,
                        **color_kw
                        )
            tr = self.get_pos(stim)
            svgstim.translate(tr[0], tr[1])
        elif 'SimpleImageStim' in str(stim):
            raise NotImplemented
        elif 'TextStim' in str(stim):
            if stim.fontname == '':
                font = 'arial'
            else:
                font = stim.fontname
            svgstim = self.svgfile.text(text=stim.text,
                                    insert=self.get_pos(stim) + np.array([0,stim.height/2.]),
                                    fill=self.color2rgb255(stim),
                                    font_family=font,
                                    font_size=stim.heightPix,
                                    text_anchor='middle',
                                    opacity=stim.opacity
                                    )

        else:
            svgstim = stim.to_svg(self)

        if not isinstance(svgstim, list):
            svgstim = [svgstim]
        for st in svgstim:
            self.svgfile.add(st)

    def get_pos(self, stim, pos=None, offset=None):
        if pos is None:
            pos = stim.pos
        if offset is not None:
            offset = self._calc_attr(stim, np.array(offset))
        else:
            offset = np.array([0,0])
        pos = self._calc_attr(stim, pos)
        pos = self.win.size/2 + np.array([pos[0], -pos[1]]) + offset
        return pos

    def get_size(self, stim, size=None):
        if size is None:
            size = stim.size
        size = self._calc_attr(stim, size)
        return size

    def _calc_attr(self, stim, attr):
        if stim.units == 'height':
            try:
                len(attr) == 2
            except:
                out = (attr * stim.win.size[1])
            else:
                out = (attr * stim.win.size * np.array([1./self.aspect, 1]))

        elif stim.units == 'norm':
            try:
                len(attr) == 2
            except:
                out = (attr * stim.win.size[1]/2)
            else:
                out = (attr * stim.win.size/2)
        elif stim.units == 'pix':
            out = attr
        elif stim.units == 'cm':
            out = misc.cm2pix(attr, stim.win.monitor)
        elif stim.units in ['deg', 'degs']:
            out = misc.deg2pix(attr, stim.win.monitor)
        else:
            raise NotImplementedError
        return out

    def color2rgb255(self, stim, color=None, colorSpace=None):
        """
        Convert color to RGB255 while adding contrast

        #Requires self.color, self.colorSpace and self.contrast
        Modified from psychopy.visual.BaseVisualStim._getDesiredRGB
        """
        if color is None:
            color = stim.color

        if isinstance(color, str) and stim.contrast == 1:
            color = color.lower()  # keep the nice name
        else:
            # Ensure that we work on 0-centered color (to make negative contrast values work)
            if colorSpace is None:
                colorSpace = stim.colorSpace

            if colorSpace not in ['rgb', 'dkl', 'lms', 'hsv']:
                color = (color / 255.0) * 2 - 1

            # Convert to RGB in range 0:1 and scaled for contrast
            # although the shader then has to convert it back it gets clamped en route otherwise
            try:
                color = (color * stim.contrast + 1) / 2.0 * 255
                color = 'rgb(%d,%d,%d)' % (color[0],color[1],color[2])
            except:
                color = None

        return color

class Datafile(object):

    def __init__(self, filename, writeable=True, header=None):
        """
        A convenience class for managing data files.

        Output is recorded in a comma-separeated (csv) file.

        .. note:: In the output file, floats are formatted to 1 ms precision so
                  that output files are nice.

        :Args:
            filename (str)
                Path to the file name
        :Kwargs:
            - writeable (bool, defualt: True)
                Can data be written in file or not. Might seem a bit silly
                but it is actually very useful because you can create
                a file and tell it to write data without thinking
                whether `no_output` is set.
            - header (list, default: None)
                If you give a header, then it will already be written
                in the datafile. Usually it's better to wait and write
                it only when the first data line is available.
        """
        self.filename = filename
        self.writeable = writeable
        self._header_written = False
        if header is not None:
            self.write_header(header)
        else:
            self.header = header

    def open(self):
        """Opens a csv file for writing data
        """
        if self.writeable:
            try_makedirs(os.path.dirname(self.filename))
            try:
                self.dfile = open(self.filename, 'ab')
                self.datawriter = csv.writer(self.dfile, lineterminator = '\n')
            except IOError:
                raise IOError('Cannot write to the data file %s!' % self.filename)

    def close(self):
        """Closes the file
        """
        if self.writeable:
            self.dfile.close()

    def write(self, data):
        """
        Writes data list to a file.

        .. note:: In the output file, floats are formatted to 1 ms precision so
                  that output files are nice.

        :Args:
            data (list)
                A list of values to write in a datafile
        """
        if self.writeable:
            # cut down floats to 1 ms precision
            dataf = ['%.3f'%i if isinstance(i,float) else i for i in data]
            self.datawriter.writerow(dataf)

    def write_header(self, header):
        """Determines if a header should be writen in a csv data file.

        Works by reading the first line and comparing it to the given header.
        If the header already is present, then a new one is not written.

        :Args:
            header (list of str)
                A list of column names
        """
        self.header = header
        if self.writeable and not self._header_written:
            write_head = False
            # no header needed if the file already exists and has one
            try:
                dataf_r = open(self.filename, 'rb')
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
                self.datawriter.writerow(header)
            self._header_written = True


class Experiment(ExperimentHandler, Task):

    def __init__(self,
                 name='',
                 version='0.1',
                 info=None,
                 rp=None,
                 actions=None,
                 computer=default_computer,
                 paths=None,
                 data_fname=None,
                 **kwargs
                 ):
        """
        An extension of ExperimentHandler and TrialHandler with many
        useful functions.

        .. note:: When you inherit this class, you must have at least
                 ``info`` and ``rp`` (or simply ``**kwargs``) keywords
                 because :class:`~psychopy.ui.Control` expects them.

        :Kwargs:
            - name (str, default: '')
                Name of the experiment. It will be used to call the
                experiment from the command-line.
            - version (str, default: '0.1')
                Version of your experiment.
            - info (tuple, list of tuples, or dict, default: None)
                Information about the experiment that you want to see in the
                output file. This is equivalent to PsychoPy's ``extraInfo``.
                It will contain at least ``('subjid', 'subj')`` even if a
                user did not specify that.
            - rp (tuple, list of tuples, or dict, default: None)
                Run parameters that apply for this particular run but need
                not be stored in the data output. It will contain at least
                the following::

                    [('no_output', False),  # do you want output? or just playing around?
                     ('debug', False),  # not fullscreen presentation etc
                     ('autorun', 0),  # if >0, will autorun at the specified speed
                     ('unittest', False),  # like autorun but no breaks at show_instructions
                     ('repository', ('do nothing', 'commit and push', 'only commit')),  # add, commit and push to a hg repo?
                                    # add and commit changes, like new data files?
                     ]

            - actions (list of function names, default: None)
                A list of function names (as ``str``) that can be called from
                GUI.
            - computer (module, default: ``default_computer``)
                Computer parameter module.
            - paths (dict, default: None)
                A dictionary of paths where to store different outputs.
                If None, :func:`~psychopy_ext.exp.set_paths()` is called.
            - data_fname (str, default=None)
                The name of the main data file for storing output. If None,
                becomes ``self.paths['data'] + self.info['subjid'] + '.csv'``.
                Then a :class:`~psychopy_ext.exp.Datafile` instance is
                created in ``self.datafile`` for easy writing to a csv
                format.
            - \*\*kwargs

        """
        ExperimentHandler.__init__(self,
            name=name,
            version=version,
            extraInfo=info,
            dataFileName='.data'  # for now so that PsychoPy doesn't complain
            )

        self.computer = computer

        if paths is None:
            self.paths = set_paths()
        else:
            self.paths = paths

        self._initialized = False

        # minimal parameters that Experiment expects in info and rp
        self.info = OrderedDict([('subjid', 'subj')])
        if info is not None:
            if isinstance(info, (list, tuple)):
                try:
                    info = OrderedDict(info)
                except:
                    info = OrderedDict([info])

            self.info.update(info)

        self.rp = OrderedDict([  # these control how the experiment is run
            ('no_output', False),  # do you want output? or just playing around?
            ('debug', False),  # not fullscreen presentation etc
            ('autorun', 0),  # if >0, will autorun at the specified speed
            ('unittest', False),  # like autorun but no breaks when instructions shown
            ('repository', ('do nothing', 'commit & push', 'only commit')),  # add, commit and push to a hg repo?
                                                                 # add and commit changes, like new data files?
            ])
        if rp is not None:
            if isinstance(rp, (tuple, list)):
                try:
                    rp = OrderedDict(rp)
                except:
                    rp = OrderedDict([rp])
            self.rp.update(rp)

        #if not self.rp['notests']:
            #run_tests(self.computer)

        self.actions = actions

        if data_fname is None:
            filename = self.paths['data'] + self.info['subjid'] + '.csv'
            self.datafile = Datafile(filename, writeable=not self.rp['no_output'])
        else:
            self.datafile = Datafile(data_fname, writeable=not self.rp['no_output'])

        if self.rp['unittest']:
            self.rp['autorun'] = 100

        self.tasks = []  # a list to store all tasks for this exp

        Task.__init__(self,
            self,
            #name=name,
            version=version,
            **kwargs
            )

    def __str__(self, **kwargs):
        """string representation of the object"""
        return 'psychopy_ext.exp.Experiment'

    #def add_tasks(self, tasks):
        #if isinstance(tasks, str):
            #tasks = [tasks]

        #for task in tasks:
            #task = task()
            #task.computer = self.computer
            #task.win = self.win
            #if task.info is not None:
                #task.info.update(self.info)
            #if task.rp is not None:
                #task.rp.update(self.rp)
            #self.tasks.append(task)

    def set_logging(self, logname='log.log', level=logging.WARNING):
        """Setup files for saving logging information.

        New folders might be created.

        :Kwargs:
            logname (str, default: 'log.log')
                The log file name.
        """

        if not self.rp['no_output']:
            # add .log if no extension given
            if not logname.endswith('.log'): logname += '.log'

            # Setup logging file
            try_makedirs(os.path.dirname(logname))
            if os.path.isfile(logname):
                writesys = False  # we already have sysinfo there
            else:
                writesys = True
            self.logfile = logging.LogFile(logname, filemode='a', level=level)

            # Write system information first
            if writesys:
                self.logfile.write('%s' % self.runtime_info)

            self.logfile.write('\n\n\n' + '#'*40 + '\n\n')
            self.logfile.write('$ python %s\n\n' % ' '.join(sys.argv))
            self.logfile.write('Start time: %s\n\n' % data.getDateStr(format="%Y-%m-%d %H:%M"))
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

    def create_win(self, debug=False, color='DimGray', units='deg',
                   winType='pyglet', **kwargs):
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

        if 'size' not in kwargs:
            try:
                kwargs['size'] = self.computer.win_size
            except:
                if not debug:
                    kwargs['size'] = tuple(res)
                else:
                    kwargs['size'] = (res[0]/2, res[1]/2)
        for key in kwargs:
            if key in ['monitor', 'fullscr', 'allowGUI', 'screen', 'viewScale']:
                del kwargs[key]

        self.win = visual.Window(
            monitor=monitor,
            units=units,
            fullscr=not debug,
            allowGUI=debug, # mouse will not be seen unless debugging
            color=color,
            winType=winType,
            screen=self.computer.screen,
            viewScale=self.computer.view_scale,
            **kwargs
        )

        self.win.flip_orig = self.win.flip
        self.win.flipnames = []

    def setup(self):
        """
        Initializes the experiment.

        A random seed is set for `random` and `numpy.random`. The seed
        is set using the 'set:time' option.
        Also, runtime information is fully recorded, log file is set
        and a window is created.
        """
        try:
            with open(sys.argv[0], 'r') as f: lines = f.read()
        except:
            author = 'None'
            version = 'None'
        else:
            author = None
            version = None
        #if not self.rp['no_output']:
        self.runtime_info = psychopy.info.RunTimeInfo(author=author,
                version=version, verbose=True, win=False)
        key, value = get_version()
        self.runtime_info[key] = value  # updates with psychopy_ext version

        self._set_keys_flat()
        self.seed = int(self.runtime_info['experimentRunTime.epoch'])
        np.random.seed(self.seed)
        #else:
            #self.runtime_info = None
            #self.seed = None

        self.set_logging(self.paths['logs'] + self.info['subjid'])
        self.create_win(debug=self.rp['debug'])
        self.mouse = event.Mouse(win=self.win)

        self._initialized = True
        #if len(self.tasks) == 0:
            ##self.setup = Task.setup
            #Task.setup(self)

    def before_exp(self, text=None, wait=.5, wait_stim=None, **kwargs):
        """
        Instructions at the beginning of the experiment.

        :Kwargs:
            - text (str, default: None)
                Text to show.
            - wait (float, default: .5)
                How long to wait after the end of showing instructions,
                in seconds.
            - wait_stim (stimulus or a list of stimuli, default: None)
                During this waiting, which stimuli should be shown.
                Usually, it would be a fixation spot.
            - \*\*kwargs
                Other parameters for :func:`~psychopy_ext.exp.Task.show_text()`
        """
        if wait_stim is None:
            if len(self.tasks) <= 1:
                try:
                    wait_stim = self.s['fix']
                except:
                    wait = 0
            else:
                wait = 0
        if text is None:
            self.show_text(text=self.__doc__, wait=wait,
                           wait_stim=wait_stim, **kwargs)
        else:
            self.show_text(text=text, wait=wait, wait_stim=wait_stim,
                           **kwargs)
        #self.win._refreshThreshold=1/85.0+0.004
        self.win.setRecordFrameIntervals(True)

    def run(self):
        """Alias to :func:`~psychopy_ext.exp.Experiment.run_exp()`
        """
        self.run_exp()

    def run_exp(self):
        """Sets everything up and calls tasks one by one.

        At the end, committing to a repository is possible. Use
        ``register`` and ``push`` flags (see
        :class:`~psychopy_ext.exp.Experiment` for more)
        """
        self.setup()
        self.before_exp()

        if len(self.tasks) == 0:
            self.run_task()
        else:
            for task in self.tasks:
                task(self).run_task()

        self.after_exp()
        self.repo_action()
        self.quit()

    def after_exp(self, text=None, auto=1, **kwargs):
        """Text after the experiment is over.

        :Kwargs:
            - text (str, default: None)
                Text to show. If None, defaults to
                'End of Experiment. Thank you!'
            - auto (float, default: 1)
                Duration of time-out of the instructions screen,
                in seconds.
            - \*\*kwargs
                Other parameters for :func:`~psychopy_ext.exp.Task.show_text()`
        """
        if text is None:
            self.show_text(text='End of Experiment. Thank you!',
                      auto=auto, **kwargs)
        else:
            self.show_text(text=text, auto=auto, **kwargs)

    def autorun(self):
        """
        Automatically runs the experiment just like it would normally
        work but automatically (as defined in
        :func:`~psychopy_ext.exp.set_autorun()`) and
        at the speed specified by `self.rp['autorun']` parameter. If
        speed is not specified, it is set to 100.
        """
        if not hasattr(self.rp, 'autorun'):
            self.rp['autorun'] = 100
        self.run()

    def repo_action(self):
        if isinstance(self.rp['repository'], tuple):
            self.rp['repository'] = self.rp['repository'][0]

        if self.rp['repository'] == 'commit & push':
            text = 'committing data and pushing to remote server...'
        elif self.rp['repository'] == 'only commit':
            text = 'commiting data...'

        if self.rp['repository'] != 'do nothing':
            textstim = visual.TextStim(self.win, text=text, height=.3)
            textstim.draw()
            timer = core.CountdownTimer(2)
            self.win.flip()

            try:
                if self.rp['repository'] == 'commit & push':
                    self.commitpush()
                elif self.rp['repository'] == 'only commit':
                    self.commit()
            except:
                pass  # no version control found

            while timer.getTime() > 0 and len(self.last_keypress()) == 0:
                pass

    def register(self, **kwargs):
        """Alias to :func:`~psychopy_ext.exp.commit()`
        """
        return self.commit(**kwargs)

    def commit(self, message=None):
        """
        Add and commit changes in a repository.

        TODO: How to set this up.
        """
        if message is None:
            message = 'data for participant %s' % self.info['subjid']
        output = ui._repo_action('commit', message=message)
        if not self.rp['no_output']:
            self.logfile.write(output)

    def commitpush(self, message=None):
        """
        Add, commit, and push changes to a remote repository.

        Currently, only Mercurial repositories are supported.

        TODO: How to set this up.
        TODO: `git` support
        """
        self.commit(message=message)
        output = ui._repo_action('push')
        if not self.rp['no_output']:
            self.logfile.write(output)


class Event(object):

    def __init__(self, parent, name='', dur=.300, durcol=None,
                 display=None, func=None):
        """
        Defines event displays.

        :Args:
            parent (:class:`~psychopy_ext.exp.Experiment` or
            :class:`~psychopy_ext.exp.Task`)
        :Kwargs:
            - name (str, default: '')
                Event name.
            - dur (int/float or a list of int/float, default: .300)
                Event duration (in seconds). If events have different
                durations throughout experiment, you can provide a list
                of durations which must be of the same length as the
                number of trials.
            - display (stimulus or a list of stimuli, default: None)
                Stimuli that are displayed during this event. If *None*,
                displays a fixation spot (or, if not created, creates
                one first).
            - func (function, default: None)
                Function to perform . If *None*, defaults to
                :func:`~psychopy_ext.exp.Task.idle_event`.
        """
        self.parent = parent
        self.name = name
        self.dur = dur  # will be converted to a list during setup
        self.durcol = durcol
        if display is None:
            try:
                self.display = parent.fixation
            except:
                parent.create_fixation()
                self.display = parent.fixation
        else:
            self.display = display

        if isinstance(self.display, tuple):
            self.display = list(self.display)
        elif not isinstance(self.display, list):
            self.display = [self.display]

        if func is None:
            self.func = parent.idle_event
        else:
            self.func = func

    @staticmethod
    def _fromdict(parent, entries):
        """
        Create an Event instance from a dictionary.

        This is only meant for backward compatibility and should not
        be used in general.
        """
        if 'defaultFun' in entries:
            entries['func'] = entries['defaultFun']
            del entries['defaultFun']
        return Event(parent, **entries)
        #self.__dict__.update(entries)
        #for key, value in dictionary.items():
            #self.key = value


class ThickShapeStim(visual.shape.ShapeStim):
    """
    Draws thick shape stimuli as a collection of lines.

    PsychoPy has a bug in some configurations of not drawing lines thicker
    than 2px. This class fixes the issue. Note that it's really just a
    collection of rectanges so corners will not look nice.

    ..note:: `lineWidth` is specified in your units, not pixels (as is default
             in PsychoPy)

    Modified from :class:`~visual.ShapeStim`.
    """
    def __init__(self, win, lineWidth=.01, **kwargs):
        super(ThickShapeStim, self).__init__(win, lineWidth=lineWidth, **kwargs)

    @attributeSetter
    def vertices(self, value=None):
        """a list of lists or a numpy array (Nx2) specifying xy positions of
        each vertex, relative to the centre of the field.

        If you're using `Polygon`, `Circle` or `Rect`, this shouldn't be used.

        :ref:`Operations <attrib-operations>` supported.
        """
        self.__dict__['vertices'] = np.array(value, float)

        # Check shape
        # if not (self.vertices.shape==(2,) or (len(self.vertices.shape) == 2 and self.vertices.shape[1] == 2)):
        #     raise ValueError("New value for setXYs should be 2x1 or Nx2")
        # self._needVertexUpdate=True

        if isinstance(value[0][0], int) or isinstance(value[0][0], float):
            self.vertices_all = [value]
        else:
            self.vertices_all = value
        self.stimulus = []

        theta = self.ori/180.*np.pi #(newOri - self.ori)/180.*np.pi
        rot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

        self._rend_vertices = []

        if self.units == 'pix':
            w = 1
        elif self.units in ['height', 'norm']:
            w = 1./self.win.size[1]
        elif self.units == 'cm':
            w = misc.pix2cm(1, self.win.monitor)
        elif self.units in ['deg', 'degs']:
            w = misc.pix2deg(1, self.win.monitor)
        wh = self.lineWidth/2. - w

        for vertices in self.vertices_all:
            rend_verts = []
            if self.closeShape:
                numPairs = len(vertices)
            else:
                numPairs = len(vertices)-1

            for i in range(numPairs):
                thisPair = np.array([vertices[i],vertices[(i+1)%len(vertices)]])
                thisPair_rot = np.dot(thisPair, rot.T)
                edges = [
                    thisPair_rot[1][0]-thisPair_rot[0][0],
                    thisPair_rot[1][1]-thisPair_rot[0][1]
                    ]
                lh = np.sqrt(edges[0]**2 + edges[1]**2)/2.
                rend_vert = [[-lh,-wh],[-lh,wh], [lh,wh],[lh,-wh]]
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

    def draw(self, **kwargs):
        for stim in self.stimulus:
            stim.draw(**kwargs)

    def to_svg(self, svg):
        rects = []
        for stim, vertices in zip(self.stimulus,self.vertices_all):
            size = svg.get_size(stim, np.abs(stim.vertices[0])*2)
            points = svg._calc_attr(stim, np.array(vertices))
            points[:, 1] *= -1
            rect = svg.svgfile.polyline(
                    points=points,
                    stroke_width=svg._calc_attr(self,self.lineWidth),
                    stroke=svg.color2rgb255(self, color=self.lineColor,
                                colorSpace=self.lineColorSpace),
                    fill_opacity=0
                    )
            tr = svg.get_pos(self)#+size/2.
            rect.translate(tr[0], tr[1])
            rects.append(rect)
        return rects

    def setSize(self, *args, **kwargs):
        super(ThickShapeStim, self).setSize(*args, **kwargs)
        self.setVertices(self.vertices_all)

    def setOri(self, *args, **kwargs):
        super(ThickShapeStim, self).setOri(*args, **kwargs)
        self.setVertices(self.vertices_all)

    def setPos(self, *args, **kwargs):
        super(ThickShapeStim, self).setPos(*args, **kwargs)
        self.setVertices(self.vertices_all)


class GroupStim(object):
    """
    A convenience class to put together stimuli in a single group.

    You can then do things like `stimgroup.draw()`.
    """

    def __init__(self, stimuli=None, name=None):
        if not isinstance(stimuli, (tuple, list)):
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


class MouseRespGroup(object):

    def __init__(self, win, stimuli, respmap=None, multisel=False,
                 on_color='#ec6b00', off_color='white', pos=(0,0),
                 height=.2, name=''):
        #super(MouseRespGroup, self).__init__(stimuli=stimuli, name=name)
        self.win = win
        self.multisel = multisel
        self.on_color = on_color
        self.off_color = off_color
        self.pos = pos
        self.name = name

        if isinstance(stimuli, str):
            stimuli = [stimuli]
        self.stimuli = []
        for i, stim in enumerate(stimuli):
            if isinstance(stim, str):
                add = np.array([0, (len(stimuli)/2-i)*height*1.5])
                stim = visual.TextStim(self.win, text=stim, height=height,
                        pos=pos+add)
                stim.size = (5*height, height)
                stim._calcSizeRendered()
                size = (stim._sizeRendered[0]*1.2, stim._sizeRendered[1]*1.2)
            else:
                stim._calcSizeRendered()
                size = stim._sizeRendered
            stim._calcPosRendered()

            stim.respbox = visual.Rect(
                                    self.win,
                                    name=stim.name,
                                    lineColor=None,
                                    fillColor=None,
                                    pos=stim._posRendered,
                                    height=size[1],
                                    width=size[0],
                                    units='pix'
                                    )
            stim.respbox.selected = False
            self.stimuli.append(stim)
        self.selected = [False for stim in self.stimuli]
        self.clicked_on = [False for stim in self.stimuli]

    def setPos(self, newPos):
        for stim in self.stimuli:
            stim.pos += self.pos - newPos
            stim.respbox.pos += self.pos - newPos

    def draw(self):
        for stim in self.stimuli:
            stim.draw()
            stim.respbox.draw()

    def contains(self, *args, **kwargs):
        self.clicked_on = [stim.respbox.contains(*args, **kwargs) for stim in self.stimuli]
        #self.state = [(s and st) for s, st in zip(sel, self.state)]
        return any(self.clicked_on)

    def select(self, stim=None):

        if stim is None:
            try:
                idx = self.clicked_on.index(True)
            except:
                return
            else:
                stim = self.stimuli[idx]
            #if any(self.state):
                #for stim, state in zip(self.stimuli, self.state):
                    #if self.multisel:
                        #self._try_set_color(stim, state)

                    #else:
                        #self._try_set_color(stim, False)
                #if not self.multisel:

                    #self._try_set_color(stim, True)
        #else:
        if not self.multisel:
            for st in self.stimuli:
                if st == stim:
                    self._try_set_color(stim)
                else:
                    self._try_set_color(st, state=False)
        else:
            self._try_set_color(stim)

    def reset(self):
        for stim in self.stimuli:
            self._try_set_color(stim, state=False)

    def _try_set_color(self, stim, state=None):
        if state is None:
            if not stim.respbox.selected:
                color = self.on_color
                stim.respbox.selected = True
            else:
                color = self.off_color
                stim.respbox.selected = False
        else:
            if state:
                color = self.on_color
                stim.respbox.selected = True
            else:
                color = self.off_color
                stim.respbox.selected = False
        self.selected = [s.respbox.selected for s in self.stimuli]

        try:
            stim.setColor(color)
        except:
            stim.setLineColor(color)
            stim.setFillColor(color)


class _HTMLParser(HTMLParser):
    def handle_starttag(self, tag, attrs):
        if tag in self.tags:
            ft = '<font face="sans-serif">'
        else:
            ft = ''
        self.output += self.get_starttag_text() + ft
    def handle_endtag(self, tag):
        if tag in self.tags:
            ft = '</font>'
        else:
            ft = ''
        self.output += ft + '</' + tag + '>'
    def handle_data(self, data):
        self.output += data

    def feed(self, data):
        self.output = ''
        self.tags = ['h%i' %(i+1) for i in range(6)] + ['p']
        HTMLParser.feed(self, data)
        return self.output

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

def get_version():
    """Get psychopy_ext version

    If using a repository, then git head information is used.
    Else version number is used.

    :Returns:
        A key where to store version in `self.runtime_info` and
        a string value of psychopy_ext version.
    """
    d = os.path.abspath(os.path.dirname(__file__))
    githash = psychopy.info._getHashGitHead(gdir=d) # should be .../psychopy/psychopy/
    if not githash:  # a workaround when Windows cmd has no git
        git_head_file = os.path.join(d, '../.git/HEAD')
        try:
            with open(git_head_file) as f:
                pointer = f.readline()
            pointer = pointer.strip('\r\n').split('ref: ')[-1]
            git_branch = pointer.split('/')[-1]
            pointer = os.path.join(d, '../.git', pointer)
            with open(pointer) as f:
                git_hash = f.readline()
            githash = git_branch + ' ' + git_hash.strip('\r\n')
        except:
            pass

    if githash:
        key = 'pythonPsychopy_extGitHead'
        value = githash
    else:
        key = 'pythonPsychopy_extVersion'
        value = psychopy_ext_version
    return key, value

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

def weighted_sample(probs):
        warnings.warn("weighted_sample is deprecated; "
                      "use weighted_choice instead")
        return weighted_choice(weights=probs)

def weighted_choice(choices=None, weights=None):
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
    if type(subjid) not in (list, tuple):
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
        print(df_fnames)
        raise IOError('Behavioral data files not found.\n'
            'Tried to look for %s' % (pattern % subjid))
    df = pandas.concat(dfs, ignore_index=True)

    return df

def latin_square(n=6):
    """
    Generates a Latin square of size n. n must be even.

    Based on `Chris Chatham's suggestion
    <http://rintintin.colorado.edu/~chathach/balancedlatinsquares.html>`_

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
    if n%2 != 0:
        raise Exception('n must be even!')

    latin = []
    col = np.arange(1,n+1)

    first_line = []
    for i in range(n):
        if i%2 == 0:
            first_line.append((n-i/2)%n + 1)
        else:
            first_line.append((i+1)/2+1)

    latin = np.array([np.roll(col,i-1) for i in first_line])

    return latin.T

def make_para(n=6):
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
    latin = latin_square(n=n).tolist()
    out = []
    for j, this_latin in enumerate(latin):
        this_latin = this_latin + this_latin[::-1]
        temp = []
        for i, item in enumerate(this_latin):
            if i%4 == 0:
                temp.append(0)
            temp.append(item)
        temp.append(0)
        out.append(temp)

    return np.array(out)


class Rectangle(object):
    '''Draws a rectangle into a batch.'''
    def __init__(self, x1, y1, x2, y2, batch):
        self.vertex_list = batch.add(4, pyglet.gl.GL_QUADS, None,
            ('v2i', [x1, y1, x2, y1, x2, y2, x1, y2]),
            ('c4B', [200, 200, 220, 255] * 4)
        )


class TextWidget(object):
    def __init__(self, text, x, y, width, batch):
        self.document = pyglet.text.document.UnformattedDocument(text)
        self.document.set_style(0, len(self.document.text),
            dict(color=(0, 0, 0, 255))
        )
        font = self.document.get_font()
        height = font.ascent - font.descent

        self.layout = pyglet.text.layout.IncrementalTextLayout(
            self.document, width, height, multiline=False, batch=batch)
        self.caret = pyglet.text.caret.Caret(self.layout)

        self.layout.x = x
        self.layout.y = y

        # Rectangular outline
        pad = 2
        self.rectangle = Rectangle(x - pad, y - pad,
                                   x + width + pad, y + height + pad, batch)

    def hit_test(self, x, y):
        return (0 < x - self.layout.x < self.layout.width and
                0 < y - self.layout.y < self.layout.height)
