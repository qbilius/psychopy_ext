.. _exp:

==================================================
Create an experiment with :mod:`~psychopy_ext.exp`
==================================================

Each experiment is a module (i.e., a '.py' script). Usually, it does at least the following:

- imports computer settings (``import computer``), see :ref:`computer`
- sets up paths (``PATH`` variable)
- defines a stimuli presentation class (inherited from :class:`~psychopy_ext.exp.Experiment`)
- defines an analysis class (inherited from :class:`object` or, for fMRI analyses, inherited from  :class:`~psychopy_ext.fmri.Analysis`).

Let's see how it works out in our demo file (``demos/scripts/main.py``).


---------------------------
Importing computer settings
---------------------------

As discussed in :ref:`computer`, ``psychopy_ext`` encourages users to keep computer settings in a plain text file (``computer.py``) in the same folder as your scripts (so typically in the ``scripts/`` folder). Import this file at the top of your experiment so that all classes could use it::

    import computer
    
If you need to modify certain settings (e.g., ``valid_responses`` that defines keyboard keys that are allowed for responding), do so during the initialization of your class (see :ref:`exp-init`)


.. _set-paths:

-------------------------------------------------------
Setting up paths: :func:`~psychopy_ext.exp.set_paths()`
-------------------------------------------------------

Paths specify where various bits of project information (data, logs etc) should be stored. The default is the following (found in :func:`~psychopy_ext.exp.set_paths()`)::

    def set_paths(exp_root='', computer=default_computer, fmri_rel=''):
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

``PATHS`` variable is defined at the top of your script after all imports are done::

    import computer  # computer params defined in the previous section
    PATHS = exp.set_paths(exp_root='', computer=computer)
    
``exp_root=''`` is the default setting recommended for a single experiment. It means that data is stored where the 'run.py' file is, in the 'data' folder. But if you have more than one experiment, ``exp_root='minimal'`` would be better (for the "minimal" experiment; for the "main", you would call it ``exp_root='main'``). That means that for this experiment will be in the 'data' folder inside the 'minimal' folder.


-----------------------------------------------------
The experiment: :class:`~psychopy_ext.exp.Experiment`
-----------------------------------------------------

An *experiment* is conceptualized as a wrapper of one or more *tasks* that a participant is supposed to perform. For example, first you may want to train them on a certain task, and next you want to measure their performance. Each task is defined as a class (using :class:`~psychopy_ext.exp.Task`, see :ref:`task`), and the two tasks can neatly be combined into a single continuous experiment with the :class:`~psychopy_ext.exp.Experiment` class.

Note that if you only have a single task, you do not need to work with tasks at all. For your convenience, defining stimuli and trials directly in the Experiment class is possible (and encouraged).

So the experiment is defined in a *class*. The convenience of having classes is that we can now inherit multiple handy routines from :class:`exp.Experiment`. Some of these routines are discussed below; otherwise, check the :ref:`api`.

So let's define the ``Confsup`` class::

    class Confsup(exp.Experiment):
        """
        The configural superiority effect experiment

        Explanation of the task here.
        """
        
Your experiments should always inherit from :class:`exp.Experiment`. Also, to encourage the good practice of `docstrings <http://www.python.org/dev/peps/pep-0257/>`_, task description is supposed to be provided in this docstring. :class:`exp.Experiment` will take it and use it by default.


.. _exp-init:
        
Initialization: :func:`~psychopy_ext.exp.Experiment.__init__()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here is a quick example::

    def __init__(self, name='exp', info=('subjid', 'confsup_'), actions='run', **kwargs):
        super(Confsup, self).__init__(name=name, info=info, actions=actions,
                                      computer=computer, 
                                      paths=PATHS, **kwargs)
        
        self.computer.valid_responses = {'num_4': 0, 'num_5': 1, 'num_1': 2, 'num_2': 3}
        self.stim_size = 3.  # in deg
        ...
        self.tasks = [_Train, _Test]
        

First, look at the parameters passed to the ``__init__()`` method:

* ``name``
    This is the name of this experiment which is used to call this particular experiment from CLI and which is seen as a tab label in the GUI.
* ``info``
    A dictionary or a tuple of information that you want to record in the output file. Usually, you want to record participant id, but sometimes also session number etc. Upon calling :func:`exp.Experiment.__init__()`, info is updated to be :class:`~collections.OrderedDict` (we use *OrderedDict* instead of a regular dictionary so that the order of entired is retained in the output file.) ``info`` will be updated to contain at least the following::
        
        OrderedDict([('subjid', 'subj')])
    
    So the default value for ``subjid`` is always *subj* unless, of course, you provided thsi value yourself, as in our example, in which case it becomes ``info['subjid'] = 'confsup'``.
* ``rp``
    We omitted ``rp`` in this example because the default ``rp`` was sufficient. It stores other run parameters that you may want to manipulate via the CLI/GUI. By default::

        self.rp = OrderedDict([  # these control how the experiment is run
            ('no_output', False),  # do you want output? or just playing around?
            ('debug', False),  # not fullscreen presentation etc
            ('autorun', 0),  # if >0, will autorun at the specified speed
            ('unittest', False),  # like autorun but no breaks at show_instructions
            ('register', False),  # add and commit changes, like new data files?
            ('push', False),  # add, commit and push to a hg repo?
            ])
* ``actions``
    A list of function names in this class that are "callable" from CLI and are seen as buttons in the GUI.
* ``computer``
    The computer module (see :ref:`computer`) that holds information of your computer parameters.
* ``paths``
    A dictionary of various paths for storing data (see :ref:`set-paths` above).
* ``data_fname``
    File name where data will be stored. (Similar to PsychoPy's ``dataFilename``.) This produces a :class:`~psychopy_ext.exp.Datafile` instance (stored in ``self.datafile``) which we use later to write data.
* ``**kwargs``
    This argument allows for other keywords arguments to be passed.
    
    ..note:: When you inherit this class, you must have at least ``info`` and ``rp`` (or simply ``**kwargs``) keywords because :class:`~psychopy.ui.Control` expects them. In fact, all "callable" classes (the ones that can be accessed via GUI or CLI) must accept at least ``info`` and ``rp``. Read more about this in :ref:`ui-init`.

            
Next, the parent class (i.e., :class:`~psychopy_ext.exp.Experiment`) is initialized. Observe the parameters that it accepts.

You should specify all "global" parameters such as stimulus size during the initialization of the Experiment. This is encouraged so that all parameters are in one place and defined before any other functions are called. Also notice how we use this opportunity to redefine ``valid_responses``.

Finally, if you have more than one task, you have to provide it here by defining a ``self.tasks`` variable as list of tasks (classes) that you want to run in the oreder you want to run them. If you only have a single task, don't bother with creating a :class:`~psychopy_ext.exp.Task` and simply define your stimuli and trials in the same Experiment class.


.. _exp-setup:

Setup: :func:`~psychopy_ext.exp.Experiment.setup()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Typical setup procedure consists of:

- collecting run time information
- setting up logging and seed (:func:`~psychopy_ext.exp.Experiment.set_logging()`)
- creating a window for stimuli presentation (:func:`~psychopy_ext.exp.Experiment.create_win()`)

All these steps are conveniently wrapped into a :func:`~psychopy_ext.exp.Experiment.setup()` and you should try to use it unless you need something special. This means that usually the setup function is not redefined.


.. _exp-run:

Run: :func:`~psychopy_ext.exp.Experiment.run()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When you call :func:`~psychopy_ext.exp.Experiment.run()` (or :func:`~psychopy_ext.exp.Experiment.run_exp()`), the following steps are performed:

- Setting up of the experiment (:ref:`exp-setup`)
- Instructions are shown (:func:`~psychopy_ext.exp.Experiment.before_exp()`). By default, the docstring of the experiment class is used thus encouraging you to keep up a good practice of commenting your code. Moreover, if you have ``docutils`` installed, *reST* syntax will be parsed automatically! And if you don't have it, you can still format text with HTML tags (see more on `pyglet's website <http://www.pyglet.org/doc/api/pyglet.text.formats.html-module.html>`_). You can also redefine what's shown by default simply by redefining :func:`~psychopy_ext.exp.Experiment.before_exp()`.
- Each task is called in a row
- Final message ("Thank you") is show (:func:`~psychopy_ext.exp.Experiment.after_exp()`)

This routine is usually suffient and is not redefined.


.. _task:

-----------------------------------------------
Defining tasks: :class:`~psychopy_ext.exp.Task`
-----------------------------------------------

Stimuli and trials are defined using a :class:`~psychopy_ext.exp.Task` class.

Arguments that each Task requires:

* ``parent`` (:class:`Experiment`)
    The Experiment to which this Tast belongs.
                
Other parameters:

* ``name`` (str, default: '')
    Name of the experiment. Currently not used anywhere.
* ``version`` (str, default: '0.1')
    Version of your experiment. Also not used.
* ``method`` ({'sequential', 'random'}, default: 'random')
    Order of trials:
    
        - sequential: trials and blocks presented sequentially
        - random: trials presented randomly, blocks sequentially
        - fullRandom: converted to 'random'
        
    Note that there is no explicit possibility to randomize the order of blocks. This is intentional because you in fact define block order in the `blockcol`.
        
* ``data_fname`` (str, default=None)
    The name of the main data file for storing output. If None, reuses :class:`~psychopy_ext.exp.Datafile` instance from its parent; otherwise, a new one is created (stored in ``self.datafile``).
* ``blockcol`` (str, default: None)
    Column name in `self.exp_plan` that defines which trial should be presented during which block.


Create stimuli: :func:`~psychopy_ext.exp.Task.create_stimuli()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here you should define all your stimuli and put them in a ``self.s`` dictionary. This need not be a complete specification of all stimuli properties as usually some properties are defined during the runtime. However, at the very least you should create the "placeholder" objects.

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
        'whole': exp.GroupStim(stimuli=[corner, diag45], name='whole'),  # arrow
        'whole_odd': exp.GroupStim(stimuli=[corner, diag135], name='whole_odd')  # triangle
        }


Create trial structure: :func:`~psychopy_ext.exp.Task.create_trial()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Trials are composed of events (:class:`~psychopy_ext.exp.Event`) that can be of are defined using the following parameters:

- ``dur`` -- duration of an event in seconds; could be a number or a list of numbers for each trial separately
- ``display`` -- which stimuli are shown during an event. This might not always be possible so you can also pass *None* and access stimuli later from ``self.s``
- ``func`` -- function controling what to do with those stimuli. There are several predefined for you:

  - :func:`exp.Experiment.wait_until_response` which waits until response is produced and exits then
  - :func:`exp.Experiment.idle_event` which simply sits and waits until its time is up while catching key presses
  - :func:`exp.Experiment.feedback` for providing feedback after the trial with a fixation color change (correct response -- green, incorrect - red)

Example::

    self.trial = [exp.Event(self,  # parent of this event 
                            dur=0.300,  # in seconds
                            display=self.s['fix'],
                            func=self.idle_event),
                  exp.Event(self,
                            dur=float('inf'),  # this means present until response
                            display=None,  # we'll select which condition to
                                           # present during the runtime with
                                           # :func:`set_stimuli`
                            func=self.during_trial),
                  exp.Event(self,
                            dur=.300,
                            display=self.s['fix'],
                            func=self.feedback)
                 ]
        
                 
.. _exp-plan:
                
Create an experiment plan: :func:`~psychopy_ext.exp.Task.create_exp_plan()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

This is the place to define properties of each trial. It is advisable to start by defining the total trial duration in the ``self.trial_dur`` variable (though if you don't, :func:`~psychopy_ext.exp.Experiment.set_TrialHandler()` will do that for you later).

Each dictionary entry will be recorded in a separate column in the output file so think about good data sharing practices (`White et al. (2013) <http://dx.doi.org/10.7287/peerj.preprints.7>`_):

- One column - one value (i.e., a number, a string, or a boolean). No lists, dictionaries etc. You don't really need them for stimulus construction during the runtime -- instead, implement stimulus construction in the trial's `func` function. Trust me, you can do it!
- Dates are formatted as YYYY-MM-DD per `ISO 8601 <http://www.iso.org/iso/support/faqs/faqs_widely_used_standards/widely_used_standards_other/iso8601>`_ and XKCD's `Public service announcement <http://xkcd.com/1179/>`_
- Avoid special characters and commas (as the output is a plain comma-separated file).
- Blank values are supposed to be ''. No None, no NA or `numpy.nan`.

*What should go into a data file?* Your data file is supposed to describe *completely* what you did so that it was perfectly reproducible by somebody else. That should be your guiding principle.

Example::

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
    
Notice how we already define ``onset``, ``dur``, ``subj_resp``, ``accuracy``, and ``rt``, even though currently they are empty. These values will be modified during the run time. You don't have to list them here but it's nicer to see all values that will be recorded.

.. note:: Unlike in *PsychoPy*, here we record data (responses, reaction times etc.) in ``self.exp_plan``. Nothing is kept in ``self.data`` and it is not saved in the resulting data file. This is done in order to facilitate an easy import of data to :class:`pandas.DataFrame`. You will appreciate the power of ``pandas`` in :ref:`stats`.

.. note:: If you need to break your experiment into blocks, include a column that indicates a sequence number of that block. For example, the 'rep' column could be used to break the experiment into five intervals (of eight trials each). You need to define which column holds this information during the initialization by passing ``blockcol`` parameter to :class:`~psychopy_ext.exp.Task`.
    
    
Setup: :func:`~psychopy_ext.exp.Task.setup_task()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Setup can only bi initiated after the setup of the parent Experiment has been completed. Typical task setup procedure consists of:

- creating stimuli (:func:`~psychopy_ext.exp.Experiment.create_stimuli()`)
- creating a trial structure (:func:`~psychopy_ext.exp.Experiment.create_trial()`)
- create a list of trials with each trial's properties defined (:func:`~psychopy_ext.exp.Experiment.create_exp_plan()`)
- defining `self.trial_dur` if not defined yet
- adjusting the list of trials for auto run
- blocks during the experiment are inferred (:func:`~psychopy_ext.exp.Task.get_blocks()`) using the ``self.blockcol`` parameter (read more in :ref:`exp-plan`)

All these steps are conveniently wrapped into a :func:`~psychopy_ext.exp.Task.setup()` and you should try to use it unless you need something special. This means that usually the setup function is not redefined.


Running task: :func:`~psychopy_ext.exp.Task.run_task()`
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The experiment starts by setting up everything (:func:`~psychopy_ext.exp.Task.setup()`). Then, instructions are shown just like explained before in :ref:`exp-run` using :func:`~psychopy_ext.exp.Experiment.before_task()`. A data output file is opened for writing (it's a ``cvs`` format with a header).

Once this is ready, the experiment starts with the presentation of task instructions (``self.instructions``). Then the experimental loop is controlled by the ``run_blocks`` function which goes through each block in ``self.blocks`` (from :func:`~psychopy_ext.exp.Task.get_blocks()`). Each block runs through its own TrialHandler, and each trial goes through events. Key presses are recorded throughout using the :func:`~psychopy_ext.exp.Task.post_trial()` function (where the mapping between key input and ``self.computer.valid_responses`` is done, and accuracy and response time are computed). Data is then recorded immediatelly at the end of each trial in the datafile..


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
for example, so that data is pushed right away, or, if you forgot to do so intially, just run
``python run.py --push`` (or another flag) and the operation will be completed.
(Note: for ``register`` and ``commit``, a tag or a message has to be included.)

--------------
Other features
--------------

Autorun
^^^^^^^

You can simulate experiments by automatically running them. Just provide a value for ``autorun``. I usually run them at 100x speed (thus, I enter 100). The experiment will run and record simulated responses by itself, stopping at each instructions screen and waiting for your response. If you rather have the experiment go no stop (good for making sure evertyhing works fine), select the ``unittest`` flag instead (it runs at 100x speed).

No output
^^^^^^^^^
Use ``no_output`` flag to run without creating or changing any output files. This is useful when debugging. For example, if the experiment fails before anything is shown, no (empty) data file is generated.

Debugging
^^^^^^^^^
``debug`` flag is used to open the windon not in a fullscreen mode that comes by default. This is useful when you're doing debugging since you want to be able to access the terminal.


-------------------
Data analysis class
-------------------

There is no generalized way to run an analysis, so ``psychopy_ext`` does not provide a class for that. However, it should be constructed in a similar manner to :class:`~psychopy_ext.exp.Experiment`, for example::

    class Analysis(object):
        def __init__(self, name='analysis', info={'subjid': 'confsup_'}):            
            self.name = name
            self.info = info
            self.exp = exp.Experiment(info=self.info)
            
You can then define various analysis routines, for example::

        def run(self):
            pattern = PATHS['data'] + '%s.csv'
            df = self.exp.get_behav_df(pattern=pattern)
            agg_acc = stats.accuracy(df, cols='context', values='accuracy', yerr='subjid')
            agg_rt = stats.aggregate(df[df.accuracy=='correct'], cols='context',
                                     values='rt', yerr='subjid')

            plt = plot.Plot(ncols=2)
            plt.plot(agg_acc, kind='bar')
            plt.plot(agg_rt, kind='bar')
            plt.show()

You'll notice that now we use :mod:`~psychopy_ext.stats` and :mod:`~psychopy_ext.plot`. Learn how to work with them: :ref:`stats`.
