.. _quickstart:

===========
Quick start
===========

Structure
---------

In ``psychopy_ext``, the structure of a project is conceptualized in
the following way:

A Project consists of
    Experiments that consist of
        Tasks that are divided into
            Blocks
        and consist of
            Trials that consist of
                Events
        and can do
            Actions (run experiment, show stimuli, ...)
    and Analyses that can do
        Actions (analyze data in one way, in another way...)

The following figure illustrates the structure of a project and the (least of) functions that you have to modify for your task.

    .. image:: scheme.png
        :width: 400px

Of course, there are many more options that you can customize, see :ref:`architecture` for an overview or :ref:`exp` for more information on creating experiments, and :ref:`stats` and :ref:`plotting` for data analysis.


What to do
----------

Start a new project ``myproject`` by copying the contents of the ``demos`` folder (in ``psychopy_ext/``) to ``myproject``. Observe the structure of this folder. Note that ``psychopy_ext`` encourages all project-related resources (scripts, data, logs, paper) to reside within a single project folder. Your experiments will reside in the ``scripts`` folder.

The easiest way to create a new experiment is by using ``scripts/main.py`` as a template. If you need something more complex, try ``scripts/twotasks.py`` or ``scripts/staircase.py``. Refer to :class:`~psychopy_Ext.exp.Experiment` and :class:`~psychopy_Ext.exp.Task` to learn about various built-in functions.

When done with the experiment, run the project by executing ``run.py`` file. (In Windows, you can simply double-click on ``run.bat`` intead.)

If you have more than a single experiment, make another Python file for that experiment, and include the path to it in the ``run.py`` file.
