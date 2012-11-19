============
Introduction
============

To guarantee highest possible standards, all stages of scientific conduct must be made transparent. Not only the final paper, but also raw data, experimental and analysis scripts and their change history must be freely available as well. I believe that full disclosure leads to more collaborative and more reliable science.

(See also `Nick Barnes. Publish your computer code: it is good enough. <http://dx.doi.org/10.1038/467753a>`_)


=================
Recommended Usage
=================

If not running Debian already:

* `Install a virtual machine with Debian on it <http://neuro.debian.net/#virtual-machine>`_

If you already have Debian/Ubuntu:

* Add the NeuroDebian repository using `instructions at <http://neuro.debian.net/#how-to-use-this-repository>`_

Then:

* Install PsychoPy for running experiments: ``sudo apt-get install psychopy`` (all its dependencies will be installed automatically)
* Install pandas for data analysis: ``sudo apt-get install python-pandas``
* For fMRI analyses:
    * PyMVPA: ``sudo apt-get install python-mvpa2``
    * NiBabel: ``sudo apt-get install python-nibabel``

Now you can run all scripts in a convenient environment.


============
Dependencies
============

* Python 2.7 (but should work with 2.5+). In Windows, you have to add Python to your path: right-click on My Computer, go to Properties > Advanced system settings > Environment Variables..., look for Path variable in System Variables, click to edit it and add C:\Python27; (or wherever your Python is installed). Also add C:\Python27\Scripts; to have your setuptools/easy_install work
* PsychoPy, which is dependent on:
    * numpy
    * scipy
    * matplotlib
    * pyglet
    * pygame
    * pyOpenGL
    * Python Imaging Library
    * wxPython
    * setuptools
    * lxmp
    * pywin32 (Windows only)
* pandas (optional)
* PyMVPA (optional)
* NiBabel (optional)
        

===============
Getting started
===============

Double-click on twolines.py. A GUI will open with a number of choices for running experiments.


=======
Credits
=======

:Author:
    Jonas Kubilius
:Insitution:
    K.U.Leuven (Belgium)
:Website:
    http://klab.lt
:License:
    `GNU General Public License v3 or later <http://www.gnu.org/licenses/>`_
:Included packages and functions:
    argparse, combinations, combinations_with_replacement, OrderedDict, HMAX, GaborJet
:Included images:
    imageafter.com, morguefile.com