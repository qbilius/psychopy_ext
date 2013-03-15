User interface
==============

:mod:`~psychopy_ext.ui`
    - Dual interface: command line interface (CLI) and graphical user interface (GUI)
    - Generated with default values automatically from Experiment and Analysis classes
    - Extra information for participant information (following PsychoPy convention)
    - Run parameters for choosing other parameters
        - No output (``--noOutput``, ``--n``)
        - Automatic running (``autorun``)
        - Debug (``debug``)
    - Basic syntax: ``python <main project file name> <experiment or analysis name> <function to call> --<parameter1> <value1> ..., e.g., python run.py exp run --subjID test --debug --n``
