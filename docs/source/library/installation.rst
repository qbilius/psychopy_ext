.. _installation:

====================
Get ``psychopy_ext``
====================

Quick installation::

    pip install psychopy_ext

Or clone the `github repository <https://github.com/qbilius/psychopy_ext>`_::

    git clone https://github.com/qbilius/psychopy_ext.git

Notice however that often certain dependencies will fail to install via pip and you'll have to do that manually. Read on what's required and where to get that.

------------
Dependencies
------------

This is a Python 2 (`Python 2.7 <http://python.org/download/>`_) package, but due to ``__future__`` imports, it should be compatible with Python 3 (untested).

Since this is a wrapper of multiple packages, it requires multiple packages. However, as you will probably end up using only some of ``psychopy_ext`` functionality and dependencies in Python are a mess, I decided to let you install the necessary dependencies on your own. If you need an easy way to do it, go with `conda <http://conda.pydata.org/miniconda.html>`_.

Here is a list of packages based on what needs them:

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


------------
Installation
------------

Recommended way
~~~~~~~~~~~~~~~

* Get `conda <http://conda.pydata.org/miniconda.html>`_
* Install all dependencies
* Install `psychopy_ext`: ``pip install psychopy_ext`

Note: for maximal flexibility, I recommend installing ``psychopyt_ext`` using ``-e``flag which will allow you to both have the package installed and edit it according to your needs. You may also want to choose a custom installation path then by using ``--prefix=<path>``.

Another way
~~~~~~~~~~~

Debian/Ubuntu users or those who like `virtual machines with Debian on it <http://neuro.debian.net/#virtual-machine>_` can add the NeuroDebian repository using `instructions on their website <http://neuro.debian.net/#how-to-use-this-repository>`_. Then install all dependencies via `sudo apt-get instal <package>`.


Bleeding edge
~~~~~~~~~~~~~

If you want the latest, greatest, and the least stable copy of psychopy_ext:

- If you know how to use a mouse: download and unzip ``psychopy_ext`` `source code <https://github.com/qbilius/psychopy_ext/archive/master.zip>`_

- If you know how to use a keyboard::

    git clone https://github.com/qbilius/psychopy_ext.git


You are now ready to build your first project: :ref:`start-project`.
