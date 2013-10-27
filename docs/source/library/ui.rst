===========================================
User interface with :mod:`~psychopy_ext.ui`
===========================================

    - Dual interface: command line interface (CLI) and graphical user interface (GUI)
    - Generated with default values automatically from Experiment and Analysis classes
    - Extra information for participant information (following PsychoPy convention)
    - Run parameters for choosing other parameters
        - No output (``--noOutput``, ``--n``)
        - Automatic running (``--autorun 100`` at 100x speed)
        - Debug (``--debug``)
    - Basic syntax::
    
        python <main project file name> <experiment or analysis name> <function to call> --<parameter1> <value1> ...,
        
      e.g.::
      
        python run.py exp run --subjID test --debug --n

.. _ui-init:

========================
Initialization procedure
========================

The :class:`~psychopy_ext.ui.Control` class first initializes all classes that it can find with their default parameters. Then these default parameters are updated using the arguments that you chose in the GUI or passed in the CLI.

Consider, for example, our demo script (``demos/scripts/main.py``). Suppose in the GUI you choose "Main exp." and the 'exp' tab, where enter 'subjid' as 'test' and select 'debug' and 'no_output' options (or call ``run.py main exp run --subjid test --debug --n``). :class:`~psychopy_ext.ui.Control` will first import ``scripts.main`` and initialize the 'exp' class (i.e., ``Confsup()``). The resulting initialized class will contain the default values for the ``info`` and ``rp`` parameters (check the :class:`~psychopy_ext.exp.Experiment` to see the defaults), meaning that if you could access ``info['subjid']``, its value would be ``confsup_`` (the default for the 'exp' class).

Next, the values that you provided are inserted in these parameters. So now ``info['subjid'] = 'test'``, and ``rp['debug'] = True``.

Finally, the 'exp' class is initialized *again* by passing ``info`` and ``rp`` parameters to it. This is the reason why all classes that inherit from :class:`~psychopy_ext.exp.Experiment` should access both ``info`` and ``rp`` even if you are only going to use their default values. For other classes that do not inherit from :class:`~psychopy_ext.exp.Experiment`, this is not required.
