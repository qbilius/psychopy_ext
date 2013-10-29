.. _faq:

==========================
Frequently asked questions
==========================

.. _pip-failing:

--------------------------------
Installation with pip is failing
--------------------------------

Certain packages, such as numpy, scipy, pandas, and pymvpa, need to be compiled when installing via pip, and that process commonly fails. If you find yourself in this situation, try installing the packages that pip failed to install manually by following the installation procedure carefully (:ref:`installation`).


.. _where-is-demo:

---------------------------
Where do I find demo files?
---------------------------

Demos are located in the ``psychopy_ext/demos`` folder in your Python's ``site-packages`` (unless you are using ``virtualenv`` in which case you know where you put ``psychopy_ext``). Don't know where ``site-packages`` are? Try::

    import site; print site.getsitepackages()
    
    
.. _how-run-demo:

-------------------
How do I run demos?
-------------------

**Windows:** double-click ``run.bat`` (instead of ``run.py``)

**Debian/Ubuntu:**

* find the *Terminal* (also try Ctrl+Alt+T)
* navigate to the folder containing demos
* type ``python run.py``


.. fmri-fail:

--------------------------
Why can't I run fMRI demo?
--------------------------

There are a few possible problems:

1. You are too impatient! Loading the fMRI demos take a while, don't force close the program.
2. You don't have *pymvpa2* or *nibabel* installed, so follow installation instructions carefully :ref:`installation`.
3. The ``fmri`` module is still in development so maybe you stumbled across a bug. `Let me know! <https://github.com/qbilius/psychopy_ext/issues>`_


.. _python-ide:

--------------------------------------
How / where do I write my own scripts?
--------------------------------------

If you are a Standalone PsychoPy user, an editor is built-in (go to View > Go to Coder view). However, it is very basic and thus better options are recommended below:

* `Notepad++ <http://notepad-plus-plus.org/>`_: like the default Notepad, but on steroids
* `NinjaIDE <http://ninja-ide.org/>`_: a beautiful and convenient IDE dedicated to Python
* `Geany <http://www.geany.org/>`_: a powerful lightweight cross-platform IDE
* `Spyder <https://code.google.com/p/spyderlib/>`_: looks like MatLab
* `Canopy <https://www.enthought.com/products/canopy/>`_: beginner-friendly but you may have to register for it (still free)


------------------------------------------------------------------------------------
In the source code, why are some variables in camelBack, and others just lower_case?
------------------------------------------------------------------------------------

When you see camelBack, it's probably a function or a variable from PsychoPy. However, psychopy_ext follows `PEP 8 <http://www.python.org/dev/peps/pep-0008/#naming-conventions>`_ naming conventions, and you should too.
