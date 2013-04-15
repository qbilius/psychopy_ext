*Please refer to the [online documentation](https://psychopy_ext.readthedocs.org/en/latest/)
for a more in depth explanation how to use the package.*


What is it?
===========

`psychopy_ext` is a unified framework for designing, building, and analyzing neuroscience experiments.
Unlike *PsychoPy*, *PyMVPA* or *matplotlib* that are very flexible and support multiple options to
suit everyone’s needs, the underlying philosophy of `psychopy_ext`
is to act as the glue at a higher level of operation by choosing reasonable defaults
for these packages and providing patterns for common tasks with a minimal user intervention.
An established framework saves time and helps to avoid mistakes.
Surely "one size fits all" approach can never suit everybody's needs but this
library should at least greatly simplify creation of your custom scripts.

*(Read more [in the documentation](https://psychopy_ext.readthedocs.org/en/latest/whyuse.html))*

Features
--------

- Neat project organization
- Templates for building experiments
- Automatic modes for running experiments
- Automatic GUI and command-line interpreter
- Simplified descriptive statistics
- Pretty plotting
- Built-in simple models of vision (Pixel-wise difference, GaborJet, and
HMAX'99)


Quick start
===========

Check out a demo:

- For people who use a keyboard:

    - In a terminal, navigate to the demos folder ('psychopy_ext/demos')
    - Type `python main.py exp run`. Do the experiment!
    - Type `python main.py analysis behav --plot` to see how well you did.

- For people who use `PsychoPy` app:

    - Open `run.py` file from the demos folder ('psychopy_ext/demos')
    - Click the green running man to run it.
    - Click on the run button. Do the experiment!
    - When done, click on the green running man again, choose the analysis
    tab, select `plot` option, and click on `behav` to see how well you did.

- For people who use a mouse:

    - In a file browser, navigate to the demos folder ('psychopy_ext/demos')
    - Right click on `run.py` and run it with `pythonw.py` if available. If
    not, find it in the python folder (Windows: `C:\Python27\`)
    - Click on the run button. Do the experiment!
    - When done, click on the green running man again, choose the analysis
    tab, select `plot` option, and click on `behav` to see how well you did.

- Inspect main.py file to see how it works.


Current state of affairs
========================

*psychopy_ext* is currently in *alpha*, meaning that I use it myself daily
but there are some limitations:

- The experiments building module (`exp`) is pretty much stable.
- fMRI analyses (`fmri` module) have not been thoroughly tested yet.
- `stats` and `plots` work well but might still require fine tuning and may be
  unable to handle missing values etc.
- Unit testing is only implemented for `models` (HMAX and GaborJet)


Future roadmap (a wishlist):

- Generate full papers via '[Open Science Paper](https://github.com/cpfaff/Open-Science-Paper)'
and '[PythonTeX](https://github.com/gpoore/pythontex)'
- README generation with the most common commands
- Threaded GUI app so that it would stay open after running the experiment
- Better command-line operation
- Browser-based project management tool
- extraInfo and runParams should become classes with tips, lists etc
- Full fMRI preprocessing support (maybe)
- Force metadata by turning `trialList` into a class
- Integrated Bayesian statistics


Introduction
============

Successful accumulation of knowledge is critically dependent on the ability to verify and
replicate every part of a scientific conduct. Python and its scientific packages have greatly
fostered the ability to share and build upon experimental and analysis code. However, while
open access to publications is immediately perceived as desired, open sourcing experiment and
analysis code is often ignored or met with a grain of skepticism in the neuroscience community,
and for a good reason: many publications would be difficult to reproduce from start to end
given typically poor coding skills, lack of version control habits, and the prevalence of manual
implementation of many tasks (such as statistical analyses or plotting), leading to a reproducible
research in theory but not in practice.

I argue that the primary reason of such unreproducible research is the lack of tools that would
seamlessly enact good coding and sharing standards. Here I propose a framework tailored to
the needs of the neuroscience community that ties together project organization, creation of
experiments, behavioral and functional magnetic resonance imaging (fMRI) data analyses,
and publication quality (i.e., pretty) plotting using a unified and relatively rigid interface.
Unlike *PsychoPy*, *PyMVPA* or *matplotlib* that are very flexible and support multiple options to
suit everyone’s needs, the underlying philosophy of *psychopy_ext*
is to act as the glue at a higher level of operation by choosing reasonable defaults
for these packages and providing patterns for common tasks with a minimal user intervention.

For example, each experiment is expected to be a module with classes in it representing
different parts of scientific conduct (e.g., stimulus presentation or data analysis), and methods
representing an atomic task at hand (e.g., showing experimental instructions or running a
support vector machine analysis). Such organization is not only natural and easy to follow in
an object-oriented environment but also allows an automatic generation of a command line
and graphic user interfaces for customizing and executing these tasks conveniently. Due to a
rigid structure, *psychopy_ext* can more successfully than typical packages address realistic user
cases. For instance, running a support vector machine on fMRI data involves multiple steps of
preprocessing, aggregating over relevant axes, combining results over participants, and, ideally,
unit testing. Since it is impossible to guess the particular configuration at hand, typically the user
has to implement these steps manually. However, thanks to a common design pattern in analyses
deriving from *psychopy_ext*, these operations can be performed seamlessly out of the box.

While these choices might be limiting in certain cases, the aim of *psychopy_ext* is to provide an
intuitive basic framework for building transparent and shareable research projects.


Recommended Usage
=================

## Typical usage ##
If not running Debian already: [Install a virtual machine with Debian on it](http://neuro.debian.net/#virtual-machine).

If you already have Debian/Ubuntu: Add the NeuroDebian repository using instructions at <http://neuro.debian.net/#how-to-use-this-repository>.

Then:

* Install *PsychoPy* for running experiments: `sudo apt-get install psychopy` (all its dependencies will be installed automatically)
* Install *pandas* for data analysis: `sudo apt-get install python-pandas`
* For fMRI analyses:
    * *PyMVPA*: `sudo apt-get install python-mvpa2`
    * *NiBabel*: `sudo apt-get install python-nibabel`

Now you can run all scripts in a convenient environment.

## More advanced usage ##
If you intend to either pull updates from this repository as your working on your own scripts or you want the ability to push your changes to this repository, you may want to keep `psychopy_ext` in a separate folder from the rest of your scripts and only merge the two before the actual release of your code (e.g., when you publish or when you are done with the experiment for good).

There are two possibilities then to use the repository:

- In `run.py` add a line `sys.path.insert(0, '../psychopy_ext/')` (or where
your `psychopy_ext` is)
- Simply append the location of `psychopy_ext` to PYTHONPATH


Dependencies
============

* Python 2.4+ (but tested only with 2.7). In Windows, you have to add Python to your path: right-click on My Computer, go to Properties > Advanced system settings > Environment Variables..., look for Path variable in System Variables, click to edit it and add `C:\Python27;` (or wherever your Python is installed). Also add `C:\Python27\Scripts;` to have your setuptools/easy_install work
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


License
=======

Copyright 2010-2013 Jonas Kubilius (http://klab.lt)

Laboratories of Biological and Experimental Psychology, KU Leuven (Belgium)

[GNU General Public License v3 or later](http://www.gnu.org/licenses/)

Included external packages and functions (covered by a compatible license):
combinations, combinations_with_replacement, OrderedDict, HMAX, GaborJet


Acknowledgements
================

I would like to thank Jonathan Peirce, Jeremy Gray and all
[PsychoPy](http://www.psychopy.org/) developers for the well maintained code
from which I learned a lot about development,
[Scott Torborg](http://www.scotttorborg.com/python-packaging) and
*[The Hitchhiker's guide to packaging](http://guide.python-distribute.org/)* for
guiding me in packaging.

Jonas Kubilius is a Research Assistant of the Research Foundation -- Flanders (FWO).


[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/16e03b45ccd8094b7ce857763e2b8225 "githalytics.com")](http://githalytics.com/qbilius/psychopy_ext)

