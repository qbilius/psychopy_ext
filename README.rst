Please refer to the `online documentation <http://psychopy-ext.klab.lt>`_
for a more in depth explanation how to use the package.

What is it?
===========

**psychopy_ext** is a framework for a rapid design, analysis and plotting of experiments in neuroscience and psychology.

Unlike **PsychoPy**, **PyMVPA** or **matplotlib** that are very flexible and support multiple options to suit everyone's needs, the underlying philosophy of **psychopy_ext** is to act as the glue at a higher level of operation by choosing reasonable defaults for these packages and providing patterns for common tasks with a minimal user intervention.


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

(`no success? <http://psychopy-ext.klab.lt/intro/faq.html#pip-failing>`_)


Quick start
===========

First, find demo files in ``site-packages`` (`where is it? <http://psychopy-ext.klab.lt/intro/faq.html#where-is-demo>`_). Copy them to your home folder or another location of your choice (but where you have write permission). Now check the demos:

- For people who use a keyboard:

  - In a terminal, navigate to the demos folder
  - Type ``python run.py main exp run``. Do the experiment!
  - Type ``python run.py main analysis run`` to see how well you did.

- For people who use **PsychoPy** app:

  - In coder view, open **run.py** file from the demos folder
  - Click the green running man to run it.
  - Click on the run button. Do the experiment!
  - When done, choose the analysis tab and click on **run** to see how well you did.

- For people who use a mouse on Windows:

  - In a file browser, navigate to the demos folder
  - Double-click click on **run.bat**
  - Click on the run button. Do the experiment!
  - When done, choose the analysis tab, and click on **run** to see how well you did.

When done with the demo, inspect **main.py** file to see how it works,
and build your experiment using this template, or try more demos.


Current state of affairs
========================

**psychopy_ext** is currently stable, meaning that I use it myself daily
but there are some limitations:

- I no longer actively use ``fmri`` module so it is no longer guaranteed to work.
- ``stats`` and ``plots`` work well many functions are undocumented.

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

- general
    * `numpy`
    * `scipy`
    * `pandas 0.17+ <http://pandas.pydata.org/getpandas.html>`_
- exp
    * `PsychoPy 1.83.04+ <http://sourceforge.net/projects/psychpy/files/>`_
    * `docutils <https://pypi.python.org/pypi/docutils>`_ (will allow rendering docstrings on screen by the ``exp`` class)
    * `svgwrite <https://pypi.python.org/pypi/svgwrite>`_ (used to export stimuli to SVG format)
- plots
    * `seaborn 0.7+ <https://pypi.python.org/pypi/seaborn>`_ (pretty plots; not really mandatory but required for the benefit of your eyes)
- models
    * `sklearn`_
    * `skimage`_
    * `caffe`
    * `matlab_wrapper <https://github.com/mrkrd/matlab_wrapper>`_ (for accessing MATLAB)
- fMRI
    * `PyMVPA 2.3.1+ <http://www.pymvpa.org/download.html>`_ (`Windows version for Python 2.7 here <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_)
    * `NiBabel <http://nipy.sourceforge.net/nibabel/installation.html#installation>`_ (install with ``pip install nibabel``)
    * `h5py <https://pypi.python.org/pypi/h5py>`_ (install with ``pip install h5py``)


License
=======

Copyright 2010-2016 Jonas Kubilius (http://klab.lt)

Brain and Cognition, KU Leuven (Belgium)

McGovern Institute for Brain Research, MIT (USA)

`GNU General Public License v3 or later <http://www.gnu.org/licenses/>`_

Included external packages and functions (covered by a compatible license):
combinations, combinations_with_replacement, OrderedDict, HMAX, GaborJet
