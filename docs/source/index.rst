.. psychopy_ext documentation master file, created by
   sphinx-quickstart on Thu Mar  7 10:38:04 2013.

Welcome to psychopy_ext documentation!
======================================

``psychopy_ext`` is a framework for a rapid reproducible design, analysis and plotting of experiments in neuroscience and psychology.

Unlike ``PsychoPy``, ``PyMVPA`` or ``matplotlib`` that are very flexible and support multiple options to suit everyone’s needs, the underlying philosophy of ``psychopy_ext`` is to act as the glue at a higher level of operation by choosing reasonable defaults for these packages and providing patterns for common tasks with a minimal user intervention. Set up your stimuli, trial structure and go! Everything else is already done for you.

Features
--------

- Easy to run and rerun everything
- Neat project organization
- Templates for building and analyzing experiments (behavioral & fMRI)
- Built-in simple models of vision (Pixel-wise difference, GaborJet, and HMAX'99)
- Custom needs? Inherit & customize: everything is a class!
- Automatic running (unit testing) of experiments
- Automatic GUI and command-line interpreter
- Simplified descriptive statistics
- Pretty plotting


Introduction
------------

.. toctree::
   :maxdepth: 3

   demo
   quickstart
   architecture
   whyuse

Documentation
-------------
.. toctree::
   :maxdepth: 3

   library/index


API reference
-------------
.. toctree::
   :maxdepth: 3

   api/index
   
   
License
-------

Copyright 2010-2013 Jonas Kubilius (http://klab.lt)

Laboratories of Biological and Experimental Psychology, KU Leuven (Belgium)

[GNU General Public License v3 or later](http://www.gnu.org/licenses/)

Included external packages and functions (covered by a compatible license): combinations, combinations_with_replacement, OrderedDict, HMAX, GaborJet


Acknowledgements
----------------

I would like to thank Jonathan Peirce, Jeremy Gray and all `PsychoPy <http://www.psychopy.org/>`_ developers for the well maintained code from which I learned a lot about development, `Scott Torborg <http://www.scotttorborg.com/python-packaging>`_ and *`The Hitchhiker's guide to packaging <http://guide.python-distribute.org/>`_* for guiding me in packaging, and `ZetCode <http://zetcode.com/wxpython/>`_ for examples on dealing with the terrible logic of wxPython.

Jonas Kubilius is a Research Assistant of the Research Foundation -- Flanders (FWO).

