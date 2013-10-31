Please refer to the `online documentation <http://qbilius.github.io/psychopy_ext/index.html>`_
for a more in depth explanation how to use the package.

What is it?
===========

**psychopy_ext** is a framework for a rapid design, analysis and plotting of experiments in neuroscience and psychology.

Unlike **PsychoPy**, **PyMVPA** or **matplotlib** that are very flexible and support multiple options to suit everyone’s needs, the underlying philosophy of **psychopy_ext** is to act as the glue at a higher level of operation by choosing reasonable defaults for these packages and providing patterns for common tasks with a minimal user intervention.


Features
--------

- Easy to run and rerun everything
- Neat project organization
- Templates for building and analyzing experiments (behavioral & fMRI)
- Simplified descriptive statistics
- Pretty plotting
- Automatic running (unit testing) of experiments
- Automatic GUI and command-line interpreter
- Custom needs? Inherit & customize: everything is a class!
- Built-in simple models of vision (Pixel-wise difference, GaborJet, and HMAX'99)

Installation
============

    pip install psychopy_ext

(`no success? <http://qbilius.github.io/psychopy_ext/faq.html#pip-failing>`_)


Quick start
===========

First, find demo files in ``site-packages`` (`where is it? <http://qbilius.github.io/psychopy_ext/faq.html#where-is-demo>`_). Now check them out:

- For people who use a keyboard:

  - In a terminal, navigate to the demos folder
  - Type ``python run.py main exp run``. Do the experiment!
  - Type ``python run.py main analysis run`` to see how well you did.

- For people who use **PsychoPy** app:

  - In coder view, open **run.py** file from the demos folder
  - Click the green running man to run it.
  - Click on the run button. Do the experiment!
  - When done, click on the green running man again, choose the analysis
    tab, and click on **run** to see how well you did.

- For people who use a mouse on Windows:

  - In a file browser, navigate to the demos folder
  - Double-click click on **run.bat**
  - Click on the run button. Do the experiment!
  - When done, click on the green running man again, choose the analysis
    tab, and click on **run** to see how well you did.

When done with the demo, inspect **main.py** file to see how it works,
and build your experiment using this template, or try more demos.


Current state of affairs
========================

**psychopy_ext** is currently stable, meaning that I use it myself daily
but there are some limitations:

- fMRI analyses (``fmri`` module) have not been thoroughly tested yet (no unit tests) but
  has been used extensively in my own research.
- ``plots`` work well but might still require fine tuning and may be
  unable to handle missing values etc.

Future roadmap (a wishlist):

- README generation with the most common commands
- Automatic summary of typical commands for CLI
- More robust command-line operation
- Browser-based project management tool
- ``info`` and ``rp`` should become classes with tips, lists etc
- Full fMRI preprocessing support (maybe)
- Generate full papers via `Open Science Paper <https://github.com/cpfaff/Open-Science-Paper>`_
  and `PythonTeX <https://github.com/gpoore/pythontex>`_
- Force metadata by turning ``exp_plan`` into a class
- Integrated Bayesian statistics


Dependencies
============

*Required*

* `Python 2.6+ <http://python.org/download/>`_ (but tested only with 2.7)
* `PsychoPy 1.7+ <http://sourceforge.net/projects/psychpy/files/>`_
* `pandas 0.12+ <http://pandas.pydata.org/getpandas.html>`_
* `docutils <https://pypi.python.org/pypi/docutils>`_ (technically it is not mandatory but it will allow rendering docstrings on screen by the ``exp`` class)

*Optional*

* `PyMVPA 2.0+ <http://www.pymvpa.org/download.html>`_ (required for the `fmri` class)
* `NiBabel <http://nipy.sourceforge.net/nibabel/installation.html#installation>`_ (required for the `fmri` class)

(Note: if there isn't a binary package for your Windows platform and your Python version, try `Christoph Gohlke's Unofficial Binaries <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_)


License
=======

Copyright 2010-2013 Jonas Kubilius (http://klab.lt)

Laboratories of Biological and Experimental Psychology, KU Leuven (Belgium)

`GNU General Public License v3 or later <http://www.gnu.org/licenses/>`_

Included external packages and functions (covered by a compatible license):
combinations, combinations_with_replacement, OrderedDict, HMAX, GaborJet
