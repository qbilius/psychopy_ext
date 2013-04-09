==================
Starting a project
==================

Let's create a new project called ``confsup``. It is best to base it on the demos provided with ``psychopy_ext``, so copy the contents of the ``demos`` folder (in ``psychopy_ext/psychopy_ext``) to ``confsup``.

Find the ``run.py`` and run it. You'll see an app appear for our *Configural superiority experiment*:

    .. image:: ../gui_simple.png

Admittedly, it is ugly but did you know it was automatically created? Let's have a peak into the source code to see how it's done.


---------------
``run.py`` file
---------------

The ``run.py`` file is simple -- it only calls the app generator (or the command line interpreter). You need to provide details about experiments within your project::

    __author__ = "Jonas Kubilius"
    __version__ = "0.1"
    exp_choices = [
        ('Experiment1',  # experiment name
         'scripts.main',  # path to the experimental script
         'main',  # alias for calling it via the command-line inteface
         ['exp','analysis'])  # order of classes in the experimental script
        ]

``exp_choices`` can handle many formats:

- ``exp_choices = 'scripts.main'`` if you've got a single experiment
- ``exp_choices = scripts.main`` but then `scripts` must be imported beforehand
- ``exp_choices = [('Exp', 'scripts.main')]`` if you don't care about alias (inferred as 'main')
- ``exp_choices = [('Exp', 'scripts.main', 'main')]``  if you don't care about class order (default: alphabetical)
- ``exp_choices = [('Localizer', 'scripts.loc'), ('scripts.main')]`` for 2 experiments

When all is set, you call the graphic user interface (GUI) or the command-line interpreter (CLI)::

    Control(exp_choices, title='Configural superiority experiment')
