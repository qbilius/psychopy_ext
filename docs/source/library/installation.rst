====================
Get ``psychopy_ext``
====================

Quick installation::

    pip install psychopy_ext
    
Or clone the `github repository <https://github.com/qbilius/psychopy_ext>`_::

    git clone https://github.com/qbilius/psychopy_ext.git


------------
Dependencies
------------
* Python 2.4+ (but tested only with 2.7)
* PsychoPy
* pandas (strongly recommended; required for all but `exp` class)
* PyMVPA (optional; required for the `fmri` class)
* NiBabel (optional; required for the `fmri` class)


----------------------------
Preparation for installation
----------------------------

Debian/Ubuntu
~~~~~~~~~~~~~

Add the NeuroDebian repository using `instructions at <http://neuro.debian.net/#how-to-use-this-repository>`_.

Then:

* Install *PsychoPy* for running experiments: `sudo apt-get install psychopy` (all its dependencies will be installed automatically)
* Install *pandas* for data analysis: `sudo apt-get install python-pandas`
* For fMRI analyses:
    * *PyMVPA*: `sudo apt-get install python-mvpa2`
    * *NiBabel*: `sudo apt-get install python-nibabel`

Windows
~~~~~~~

First, you'll have to choose an environment to run Python. Some good options include:

* `Notepad++ <http://notepad-plus-plus.org/>`_: like the default Notepad, but on steroids
* `NinjaIDE <http://ninja-ide.org/>`_: a beautiful and convenient IDE dedicated to Python
* `Geany <http://www.geany.org/>`_: a powerful lightweight cross-platform IDE
* `Spyder <https://code.google.com/p/spyderlib/>`_: looks like MatLab
* `Canopy <https://www.enthought.com/products/canopy/>`_: beginner-friendly but you may have to register for it (still free)

Next, install `Python 2.7 <http://www.python.org/getit/>`_ (except if you chose *Canopy*). You have to add Python to your path: right-click on My Computer, go to Properties > Advanced system settings > Environment Variables..., look for Path variable in System Variables, click to edit it and add `C:\\Python27;` (or wherever your Python is installed). Also add `C:\\Python27\\Scripts;` to have setuptools/easy_install work.

Now install `PsychoPy <http://sourceforge.net/projects/psychpy/files/>`_ (not a standalone version) and `pandas <http://pandas.pydata.org/getpandas.html>`_. If you intend to do fMRI analyses, also get `PyMVPA2 <http://www.pymvpa.org/download.html>`_ and `NiBabel <http://nipy.sourceforge.net/nibabel/installation.html>`_.

Alternatively, consider `installing a virtual machine with Debian on it <http://neuro.debian.net/#virtual-machine>_` and following the instructions above for Debian/Ubuntu.


------------
Installation
------------

Many users will be satisfied by merely installing ``psychopy_ext`` from PyPI::

    pip install psychopy_ext
    
However, to have an easy access to ``psychopy_ext`` and customize it to your own needs, I recommend cloning the `github repository <https://github.com/qbilius/psychopy_ext>`_:

- If you know how to use a mouse: download and unzip ``psychopy_ext`` `source code <https://github.com/qbilius/psychopy_ext/archive/master.zip>`_

- If you know how to use a keyboard::

    git clone https://github.com/qbilius/psychopy_ext.git

There are two possibilities then to use the repository:

- In `run.py` add a line `sys.path.insert(0, '../psychopy_ext/')` (i.e., a relative path to the ``psychopy_ext`` folder)
- Or: simply append the location of `psychopy_ext` to PYTHONPATH

You are now ready to build your first project: :ref:`start-project`.
