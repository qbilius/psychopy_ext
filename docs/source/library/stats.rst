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


Aggregating data
----------------
    
With the data ready, you can compute averages of particular columns in this DataFrame using :func:`~psychopy_ext.stats.aggregate()` in the following way::

    agg = stats.aggregate(df, values='rt', cols='cond', yerr='subjid')
    
This means that you want to compute an average of the response time ("rt") for each condition separately. Moreover, this average is computed for each participant ("subjid") separately. The resulting ``agg`` will have each participant in its DataFrame index (rows) and condition averages in its columns.

Note that ``cols`` and ``yerr`` keywords are named so not by accident. These aggregated data can then be easily plotted using the :mod:`~psychopy_ext.plot` module. ``yerr`` will be used to draw error bars and ``cols`` will represent entries in the legend. You may also want to use the ``rows`` keyword to group certain bars and ``subplots`` to plot various conditions in separate subplots. Read more about these options in the :ref:`plotting` section.


Computing accuracy
------------------

Accuracy is very similar to aggregation but additionally you may need to specify how you coded correct and incorrect responses. By default, ``psychopy_ext`` stores "correct" and "incorrect" in the "accuracy" column, so accuracy computation with :func:`~psychopy_ext.stats.accuracy()` will look like this::

    agg = stats.aggregate(df, values='accuracy', cols='cond', yerr='subjid')
    
If you store accuracy in a different manner, then it may look like::

    agg = stats.aggregate(df, values='accuracy', cols='cond', yerr='subjid',
                          correct=1, incorrect=0)
                          

.. _reordering:

Reordering aggregates
---------------------

Annoyingly, pandas is pretty bad about retaining and updating the structure of DataFrames after aggregation. However, for plotting the order of conditions is often important. ``psychopy_ext`` is pretty smart about the order. By default, when aggregating it attempts to keep the order found in the DataFrame. For example, if participants first saw condition "one", then "two", then "three", this order will be retained in the resulting aggregation. If you want to have them sorted alphabetically, then pass the ``order='sorted'`` keyword to :func:`~psychopy_ext.stats.aggregate()`.

You may also want to define the order manually by using :func:`~psychopy_ext.stats.reorder()`.


Statistics
----------

Finally, there are several options for runnning simple statistical analyses on your data:

- :func:`~psychopy_ext.stats.oneway_anova()`
- :func:`~psychopy_ext.stats.p_corr()`
- :func:`~psychopy_ext.stats.reliability()`
- :func:`~psychopy_ext.stats.mds()`

Now you may continue to learn how to make plots: :ref:`plotting`
