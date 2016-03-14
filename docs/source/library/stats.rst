.. _stats:

==========================================
Statistics with :mod:`~psychopy_ext.stats`
==========================================

For data analysis, ``psychopy_ext`` uses the power of `pandas <http://pandas.pydata.org/>`_ package. However, ``pandas`` is somewhat cumbersome when an average or the accuracy per condition needs to be computed. ``psychopy`` provides a couple of simplified tools to perform data aggregation.


Reading in data
---------------

First, data has to be read into a `pandas.DataFrame`. Usually, :func:`~psychopy_ext.exp.Experiment.get_behav_df()` is used to find the necessary data files and concatenate them into one. You can specify the relevant file names by providing the ``pattern`` keyword. Usually, the pattern will look like this::

    pattern = PATHS['data'] + '%s.csv'

The ``PATHS`` variable is specified at the top of your experiment and controls where various files can be found. The ``%s`` is provided so that it could be substituted with the relevant participant IDs which were provided when running the experiment and are stored in ``self.info['subjid']``. You can also directly call :func:`~psychopy_ext.exp.get_behav_df()` and provide participants IDs as the first argument.

So in the end you will have the following two lines::

    pattern = PATHS['data'] + '%s.csv'
    df = self.exp.get_behav_df(pattern=pattern)


The rest
--------

Most other computations have now been deprecated in favor os ``pandas`` and ``seaborn`` or not covered in docs (like bootstrapping).


Now you may continue to learn how to make plots: :ref:`plotting`
