Pretty plots
============

:mod:`~psychopy_ext.plot`
    - Basic philosophy: Plots must come out pretty by default (no effort from the user), not after hours of tweaking
    - Figure is a :class:`~psychopy_ext.plot.Plot` object
    - A unified plotting call via :func:`~psychopy_ext.plot.Plot.plot()` with a ``kind`` keyword. Currently recognized kinds of plots:
        - line
        - bar
        - bean
        - scatter
        - mds (multidimensional scaling)
        - imshow
    - Subplots are automatically handled by defining ``nrows_ncols`` parameter when creating a Plot object, e.g., ``nrows_ncols = (2,1)``. Active subplot is advanced after each :func:`~psychopy_ext.plot.Plot.plot()` call.

