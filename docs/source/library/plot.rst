.. _plotting:

===========================================
Pretty plots with :mod:`~psychopy_ext.plot`
===========================================


*Basic philosophy: Plots must come out pretty by default (no effort from the user), not after hours of tweaking.*

Usually plots are generated from aggregated data produced by :func:`~psychopy_ext.stats.aggregate` or :func:`~psychopy_ext.stats.accuracy`. These DataFrames have a very rich structure allowing for a production of nicely labelled plots.


-----------
Basic usage
-----------

1. Create a new :class:`~psychopy_ext.plot.Plot` instance::

    plt = plot.Plot()
  
   Observe that figures are objects in ``psychopy_ext``. The benefit of this approach is a seamless handling of subplots: the figure knows where to put the next subplot. To determine the shape of subplots, define ``nrows_ncols`` parameter when creating a Plot object, e.g., ``nrows_ncols = (2,1)``. Active subplot is advanced after each :func:`~psychopy_ext.plot.Plot.plot()` call.
    
2. Plot any kind of a plot by calling :func:`~psychopy_ext.plot.Plot.plot()` with a ``kind`` keyword. Currently recognized kinds of plots:
   - line
   - bar
   - bean
   - scatter
   - mds (multidimensional scaling)
   - histogram
        
   For example, if the data is stored in a variable ``agg``, then a line plot is called like this::
        
      plt.plot(agg, kind='line')
        
3. Show the plot::
    
    plt.show()
    
   Note that currently all plot objects will be shown (i.e., if you have ``plt1`` and ``plt2``, and you call ``plt1.show()``, ``plt2`` will be shown as well.

   Hint: Did you want a different order of lines or bars? Check :ref:`reordering`.


.. _gallery:

-------
Gallery
-------

*(Note that these plots were generated automatically without any tweaking.)*

.. plot:: plots.py


--------------
Tweaking plots
--------------

As powerful as the ``plot`` class is, additional tweaking will be necessary. Here are a few useful functions:

- :func:`~psychopy_ext.plot.Plot.get_ax()` - returns (an axis of) a specified subplot, so that you can manipulate it
- :func:`~psychopy_ext.plot.Plot.hide_plots()` - for hiding empty subplots in a figure (e.g. if you only need 5 subplots in a 2x3 figure, the last one could be hiden)
- :func:`~psychopy_ext.plot.Plot.set_legend_pos()` - for moving the legend around, even outside the subplot

Continue to :ref:`models`.
