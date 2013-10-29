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

*Required*

* `Python 2.6+ <http://python.org/download/>`_ (but tested only with 2.7)
* `PsychoPy 1.7+ <http://sourceforge.net/projects/psychpy/files/>`_
* `pandas 0.12+ <http://pandas.pydata.org/getpandas.html>`_

*Optional*

* `PyMVPA 2.0+ <http://www.pymvpa.org/download.html>`_ (required for the `fmri` class)
* `NiBabel <http://nipy.sourceforge.net/nibabel/installation.html#installation>`_ (required for the `fmri` class)
* `docutils <https://pypi.python.org/pypi/docutils>`_ (used for rendering docstrings on screen by the ``exp`` class)

(Note: if there isn't a binary package for your Windows platform and your Python version, try `Christoph Gohlke's Unofficial Binaries <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_)


----------------------------
Preparation for installation
----------------------------

Debian/Ubuntu
~~~~~~~~~~~~~

Add the NeuroDebian repository using `instructions on their wesbite <http://neuro.debian.net/#how-to-use-this-repository>`_.

Then:

* Install *PsychoPy* for running experiments: `sudo apt-get install psychopy` (all its dependencies will be installed automatically)
* Install *pandas* for data analysis: `sudo apt-get install python-pandas`
* For fMRI analyses:

    * *PyMVPA*: `sudo apt-get install python-mvpa2`
    * *NiBabel*: `sudo apt-get install python-nibabel`

Windows
~~~~~~~

Standalone PsychoPy
^^^^^^^^^^^^^^^^^^^

Total beginners might want to merely install `Standalone PsychoPy distribution <http://sourceforge.net/projects/psychpy/files/>`_ which contains most packages required by *psychopy_ext* except for *pymvpa* an *nibabel*. These packages are used for fMRI analyses and if you are planning to conduct them, then this method will not work for you (so read on).

Others
^^^^^^

First, you'll have to choose an environment to run Python. Some good options are listed in :ref:`python-ide`.

Next, install `Python 2.7 <http://www.python.org/getit/>`_ (except if you chose *Canopy*). You have to add Python to your path: right-click on My Computer, go to Properties > Advanced system settings > Environment Variables..., look for Path variable in System Variables, click to edit it and add `C:\\Python27;` (or wherever your Python is installed). Also add `C:\\Python27\\Scripts;` to have setuptools/easy_install/pip work.

Now install `PsychoPy <http://sourceforge.net/projects/psychpy/files/>`_ (not a standalone version) and `pandas <http://pandas.pydata.org/getpandas.html>`_. If you intend to do fMRI analyses, also get `PyMVPA2 <http://www.pymvpa.org/download.html>`_ and `NiBabel <http://nipy.sourceforge.net/nibabel/installation.html#installation>`_.

Alternatively, consider `installing a virtual machine with Debian on it <http://neuro.debian.net/#virtual-machine>_` and following the instructions above for Debian/Ubuntu.

------------
Installation
------------

Standalone PsychoPy
~~~~~~~~~~~~~~~~~~~

Download the zip file from `PyPI <https://pypi.python.org/pypi/psychopy_ext>`_ and follow instructions on `PsychoPy documentation <http://www.psychopy.org/recipes/addCustomModules.html>`_ to install it.

Others
~~~~~~

Many users will be satisfied by merely installing ``psychopy_ext`` from PyPI::

    pip install psychopy_ext
    
However, to have an easy access to ``psychopy_ext`` and customize it to your own needs, I recommend downloading the source code and placing it where you keep your other projects. There are two possibilities then to use the package:

- In `run.py` add a line `sys.path.insert(0, '../psychopy_ext/')` (i.e., a relative path to the ``psychopy_ext`` folder)
- Or: simply append the location of `psychopy_ext` to PYTHONPATH

Bleeding edge
~~~~~~~~~~~~~

If you want the lastest, greatest, and the least stable copy of psychopy_ext:

- If you know how to use a mouse: download and unzip ``psychopy_ext`` `source code <https://github.com/qbilius/psychopy_ext/archive/master.zip>`_

- If you know how to use a keyboard::

    git clone https://github.com/qbilius/psychopy_ext.git


You are now ready to build your first project: :ref:`start-project`.
