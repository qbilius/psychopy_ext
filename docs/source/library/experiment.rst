====================
Create an experiment
====================

Each experiment is a module. Usually, it contains at least the following:
- imports computer settings (``import computer``)
- sets up paths (``PATH`` variable)
- defines a stimuli presentation class (inherited from :class:`exp.Experiment`)
- defines an analysis class (inherited from :class:`object` or, for fMRI analyses, inherited from  :class:`fmri.Analysis`).

Paths should be customized: ``PATHS = exp.set_paths('.', computer)``. This sets up where all data, logs etc are stored for this experiment. For a single experiment, '.' is fine -- it means data is stored in the 'data' folder where the 'run.py' file is, for example. But if you have more than one experiment, 'confsup' would be better -- data for this experiment will be in the 'data' folder inside the 'confsup' folder.

All "callable" classes should accept at least the following variables (of type :class:`exp.OrderedDict`):

- ``extraInfo`` -- all relevant information about the participant, the session she is doing etc. This information is recorded in the output data file (together with the collected data; see :ref:`create-trialList` for more). For example::

    extraInfo = OrderedDict([
        ('subjID', 'confsup_')
        ])

- ``runParams`` -- all other run parameters that you may want to manipulate via the CLI/GUI, for example::

        runParams = OrderedDict([
            ('noOutput', False),  # do you want output? or just playing around?
            ('debug', False),  # not fullscreen presentation etc
            ('autorun', 0),  # if >0, will autorun at the specified speed
            ('push', False)  # commit and push to a hg repo?
            ])

--------------
Initialization
--------------

You should specify all "global" parameters such as stimulus size during the initialization of the Experiment (:func:`__init__`).

-----
Setup
-----

Typical setup procedure consists of:

- Setting up logging
- Creating a window for stimuli presentation
- Creating stimuli
- Creating a trial structure
- Create a list of trials with each trial's properties defined

All these steps are conveniently wrapped into a :func:`exp.Experiment.setup` and you should try to use it unless you need something special.

Usually only the 'create' steps should be defined by the user (except creating a window). The rest has some clever default presets.

--------------
Create stimuli
--------------

In :func:`create_stimuli` you should define all your stimuli and put them in a ``self.s`` dictionary. This need not be a complete specification of all stimuli properties as usually some properties are

Example::

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

----------------------
Create trial structure
----------------------

Trials are composed of events, which we define in a dictionary with three keys:
- ``dur`` -- duration of an event in seconds
- ``display`` -- which stimuli are shown during an event. This might be limiting in certain cases, but you are free to directly use ``self.s`` to manipulate stimuli directly.
- ``func`` -- function controling what to do with those stimuli. There are several predefined ones for you:
    - :func:`exp.Experiment.idle_event` which simply sits and waits until its time is up while catching key presses
    - :func:`exp.Experiment.feedback` for providing feedback after the trial with a fixation color change (correct response -- green, incorrect - red)
    
Example::

    self.trial = [{'dur': 0.300,  # in seconds
                   'display': self.s['fix'],
                   'func': self.idle_event},

                  {'dur': 0,  # this means present until response
                   'display': None,  # we'll select which condition to
                                     # present during the runtime with
                                     # :func:`set_stimuli`
                   'func': self.during_trial},

                  {'dur': .300,
                   'display': self.s['fix'],
                   'func': self.feedback}
                 ]

.. _create-trialList:
-------------------
Create a trial list
-------------------

This is the place to define properties of each trial. It is advisable to start by defining the total trial duration (though if you don't, :func:`set_TrialHandler` will do that for you).

Example::

    expPlan = []
    for rep in range(self.nreps):  # repeat the defined number of times
        for cond, (context, posname) in self.paratable.items():
            pos = (cond - 1) % 4
            expPlan.append(OrderedDict([
                ('cond', cond),
                ('context', context),
                ('posname', posname),
                ('pos', pos),
                ('onset', ''),
                ('dur', self.trialDur),
                ('corrResp', pos),
                ('subjResp', ''),
                ('accuracy', ''),
                ('rt', ''),
                ]))

----------------------                
Running the experiment
----------------------

The experiment starts with the presentation of instructions.

The experimental loop is controlled by the ``loop_trials`` function which:
- sets up data output file (in a ``cvs`` format with a header)
- goes though each trial and each event in each trial
- catches key presses and records them using the `post_trial`` function (where the mapping between key input and ``self.computer.validResponses`` is done, and accuracy and response time are computed)

----------------------------
Pushing data to a repository
----------------------------

If you provide the ``push`` flag, new data (and log files) are automatically committed and pushed to their remote Mercurial repository.


--------------
Other features
--------------

- TODO: Does basic functionality "unittests" before running anything to make sure computer is recognized and that everything works as expected
- Simulate experiments by automatically running them (can be speeded up)
- ``--noOutput`` flag to run without creating or changing any output files
- Output files created only when necessary. This is useful when debugging. For example, if the experiment fails before anything is shown, no (empty) data file is generated
- ``--debug`` flag to debug not in a fullscreen mode
