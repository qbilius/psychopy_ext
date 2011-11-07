These files accompany the following publication:

*Kubilius, J., Wagemans, J., & Op de Beeck, H. (submitted) Emergence of perceptual Gestalts in the human visual cortex: The case of the configural superiority effect.*

To guarantee highest possible standards, all stages of scientific conduct must be made transparent. Publishing a paper is not sufficient anymore. Raw data, experimental and analysis scripts must be freely available as well. (Here, raw data is not included due to a massive size of fMRI files.) I believe that full disclosure leads to more collaborative and more reliable science.

(See also `Nick Barnes. Publish your computer code: it is good enough. <http://dx.doi.org/10.1038/467753a>`_)


=================
Recommended Usage
=================

If not running Linux already:

* `Install a virtual machine with Debian on it <http://neuro.debian.net/#virtual-machine>`_
* Install PsychoPy through ``sudo apt-get install psychopy``, which will install all its dependencies
* Install PyMVPA through ``sudo apt-get install python-mvpa`` (PyNIfTI is installed automatically)
* Install setuptools through ``sudo apt-get install python-setuptools``
* Install Mercurial through ``sudo apt-get install mercurial``, then clone tabular by ``sudo hg clone https://qbilius@bitbucket.org/elaine/tabular``and install with ``sudo python setup.py install``
* Now you can run all scripts in a convenient environment


============
Dependencies
============

* Python 2.7. In Windows, you have to add Python to your path: right-click on My Computer, go to Properties > Advanced system settings > Environment Variables..., look for Path variable in System Variables, click to edit it and add C:\Python27; (or wherever your Python is installed). Also add C:\Python27\Scripts; to have your setuptools/easy_install work
* PsychoPy 1.62, which is dependent on:
    * numpy 1.5
    * scipy 0.8
    * matplotlib 1.0
    * pyglet 1.1
    * pygame 1.9
    * pyOpenGL 3.0
    * Python Imaging Library 1.1
    * wxPython 2.4
    * setuptools 0.6
    * pywin32 (Windows only)    
* Tabular 0.0.8
    * It produces a number of warnings that might be annoying. To avoid this behavior:
        * *tabular/tabular/io.py*, Line 30: change ``DEFAULT_VERBOSITY = 5`` to ``DEFAULT_VERBOSITY = 0``
        * *tabular/tabular/spreadsheet.py*:
            * Line 27: add ``import tabular.io as io; DEFAULT_VERBOSITY = io.DEFAULT_VERBOSITY``
            * Line 199: put the three ``if`` statements under condition ``if DEFAULT_VERBOSITY > 1:``
            
    * There is also a problem with saving nan's to a text file. Fix:
        * *tabular/tabular/utils.py*, Line 344: change ``nan`` to ``1.#QNAN`` (the ``DEFAULT_STRINGIFIER`` definition)
            
    * To include these updates, in the terminal navigate to where tabular is and type::
    
        $ sudo python setup.py install
        
* PyMVPA 0.6.0 and NiBabel 1.0.1 (optional; only if you want to do fMRI data analysis). PyMVPA has to be built from source as binaries do not come for Python 2.7 yet. In Unix, follow `these steps <http://www.pymvpa.org/installation.html#build-it-general-instructions>`_. In Windows, building from source `requires more effort <http://www.pymvpa.org/installation.html#build-win>`_, so consider switching to NeuroDebian now.
        

===============
Getting started
===============

In a console, navigate to the folder where this file is located, then go to scripts and type::

$ python utl.py --n exp

This will run the main experimental task without saving any data on your computer. (Press 'space' to initiate the experiment.) To record data from participant utl_01, type::

$ python utl.py --subjID utl_01 exp

If you want to run parts of experiment separately (available: 'practice','pre', 'post', 'exposure'), run::

$ python utl.py --subjID utl_01 exp --block pre

Finally, you can plot you results with the following command::

$ python utl.py --subjID utl_01 --runNo 1 --n analysis behav

To run the continous flash suppresion experiment, type::

$ python utl.py --subjID utl_01 --runType cfs exp

To see all available options, type::

$ python occlusion.py --help


=======
Credits
=======

:Author:
    Jonas Kubilius
:Insitution:
    K.U.Leuven (Belgium)
:Website:
    http://jonaskubilius.mp
:License:
    `Modified BSD License <http://www.opensource.org/licenses/bsd-license.php>`_
:Included packages and functions:
    argparse, combinations, combinations_with_replacement, OrderedDict
:Included images:
    imageafter.com, morguefile.com