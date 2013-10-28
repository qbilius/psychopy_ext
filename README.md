*Please refer to the [online documentation](http://qbilius.github.io/psychopy_ext/index.html)
for a more in depth explanation how to use the package.*


What is it?
===========

`psychopy_ext` is a framework for a rapid design, analysis and plotting of experiments in neuroscience and psychology.

Unlike `PsychoPy`, `PyMVPA` or `matplotlib` that are very flexible and support multiple options to suit everyone’s needs, the underlying philosophy of `psychopy_ext` is to act as the glue at a higher level of operation by choosing reasonable defaults for these packages and providing patterns for common tasks with a minimal user intervention.

*(Read more [in the documentation](http://qbilius.github.io/psychopy_ext/whyuse.html))*

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


Quick start
===========

Check out a demo:

- For people who use a keyboard:

    - In a terminal, navigate to the demos folder ('psychopy_ext/demos')
    - Type `python main.py exp run`. Do the experiment!
    - Type `python main.py analysis run --plot` to see how well you did.

- For people who use `PsychoPy` app:

    - Open `run.py` file from the demos folder ('psychopy_ext/demos')
    - Click the green running man to run it.
    - Click on the run button. Do the experiment!
    - When done, click on the green running man again, choose the analysis
    tab, select `plot` option, and click on `run` to see how well you did.

- For people who use a mouse:

    - In a file browser, navigate to the demos folder ('psychopy_ext/demos')
    - Double-click click on `run.py`
    - Click on the run button. Do the experiment!
    - When done, click on the green running man again, choose the analysis
    tab, select `plot` option, and click on `run` to see how well you did.

- Inspect main.py file to see how it works.


Current state of affairs
========================

*psychopy_ext* is currently stable, meaning that I use it myself daily
but there are some limitations:

- fMRI analyses (`fmri` module) have not been thoroughly tested yet (no unit tests) but 
  has been used extensively in my own research.
- `plots` work well but might still require fine tuning and may be
  unable to handle missing values etc.

Future roadmap (a wishlist):

- README generation with the most common commands
- Automatic summary of typical commands for CLI
- More robust command-line operation
- Browser-based project management tool
- `info` and `rp` should become classes with tips, lists etc
- Full fMRI preprocessing support (maybe)
- Generate full papers via '[Open Science Paper](https://github.com/cpfaff/Open-Science-Paper)'
and '[PythonTeX](https://github.com/gpoore/pythontex)'
- Force metadata by turning `trialList` into a class
- Integrated Bayesian statistics


Dependencies
============

* Python 2.4+ (but tested only with 2.7)
* PsychoPy
* pandas
* PyMVPA (optional; required for the `fmri` class)
* NiBabel (optional; required for the `fmri` class)


License
=======

Copyright 2010-2013 Jonas Kubilius (http://klab.lt)

Laboratories of Biological and Experimental Psychology, KU Leuven (Belgium)

[GNU General Public License v3 or later](http://www.gnu.org/licenses/)

Included external packages and functions (covered by a compatible license):
combinations, combinations_with_replacement, OrderedDict, HMAX, GaborJet


[![githalytics.com alpha](https://cruel-carlota.pagodabox.com/16e03b45ccd8094b7ce857763e2b8225 "githalytics.com")](http://githalytics.com/qbilius/psychopy_ext)

