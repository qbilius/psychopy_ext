Creating an experiment
======================

:mod:`~psychopy_ext.exp`
    - TODO: Does basic functionality "unittests" before running anything to make sure computer is recognized and that everything works as expected
    - Simulate experiments by automatically running them (can be speeded up)
    - ``--noOutput`` flag to run without creating or changing any output files
    - Output files created only when necessary. This is useful when debugging. For example, if the experiment fails before anything is shown, no (empty) data file is generated
    - ``--debug`` flag to debug not in a fullscreen mode
