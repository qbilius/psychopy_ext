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

Each dictionary entry will be recorded in a separate column in the output file
so think about good data sharing practices (`White et al. (2013) <http://dx.doi.org/10.7287/peerj.preprints.7>`_):
- One column - one value (i.e., a number, a string, or a boolean). No lists,
dictionaries etc. You don't really need them for stimulus construction during
the runtime -- instead, implement stimulus construction in the trial's `func`
function. Trust me, you can do it!
- Dates are formatted as YYYY-MM-DD per `ISO 8601 <http://www.iso.org/iso/support/faqs/faqs_widely_used_standards/widely_used_standards_other/iso8601>`_ and XKCD's `Public service announcement <http://xkcd.com/1179/>`_
- Avoid special characters and commas (as the output is a plain comma-separated
file).
- Blank values are supposed to be ''. No None, no NA or `numpy.nan`.

*What should go into a data file?* Your data file is
supposed to describe *completely* what you did so that it was perfectly
reproducible by somebody else. That should be your guiding principle.

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

-------------------------------------------
Committing and pushing data to a repository
-------------------------------------------

Motivation
^^^^^^^^^^

In the ideal world, how should data be treated? It should be registered at the
point of its acquiry meaning that one should be able to go back and see it as
it came out. Data files are not immutable. Sometimes there is a spelling mistake that
you want to fix, or an extra column that you realized only two participants later you should have included. So you want to overwrite your data which poses a problem that
it may go wrong but you notice it too late. This is where the confidence of
having the original version help.

Of course, you could keep all versions of your data files but this is both
inefficient (a year later, will you remember whether 'data_corr' or 'data_final'
was the correct one?) and unncessary. It is much easier to track (and record)
changes. Welcome revision control systems (`learn more <http://gestaltrevision.be/wiki/python/vc>`_).

This approach provides a more stringent data and source code handling.
Bonus: simple plot sharing as can be seen in the utl repo.

What's available
^^^^^^^^^^^^^^^^

``register`` when you want to put a tag at some important point of your
experiment development, for example, when you're about to test the first
participant or when you do a pilot, so that you can always go back to that
point in time and see how your code looked exactly. 'Registration' is
inspired by the `Open Science Framework <http://openscienceframework.org/>`_
``commit`` after data collection so that data files are added to the
revision control system right away
``push`` to put your data on the remote repository immediately. This is
recommended over ``commit`` unless you run experiments without the internet connection.

You can either add these flags when you run the experiment (except ``register``),
for example, so
that data is pushed right away, or, if you forgot to do so intially, just run
``python run.py --push`` (or another flag) and the operation will be completed.
(Note: for ``register`` and ``commit``, a tag or a message has to be included.)

--------------
Other features
--------------

- TODO: Does basic functionality "unittests" before running anything to make sure computer is recognized and that everything works as expected
- Simulate experiments by automatically running them (can be speeded up)
- ``--noOutput`` flag to run without creating or changing any output files
- Output files created only when necessary. This is useful when debugging. For example, if the experiment fails before anything is shown, no (empty) data file is generated
- ``--debug`` flag to debug not in a fullscreen mode
