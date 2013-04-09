Quick start
===========

Find the ``run.py`` in the psychopy_ext demos folder, and run it. You'll see an app appear for our *Configural superiority experiment*:

    .. image:: gui_simple.png
        :width: 400px

Admittedly, it is ugly but did you know it was automatically created? But ok, try to do the experiment for now. Come back when done (it's quick!).

How well did you do? Run the ``run.py`` file again, click on the analysis tab, and hit the *behav* button.

    .. image:: barplot_single.png
        :width: 400px
    
Now that's beautiful *and* generated using only a couple of lines of code: read data, aggregate over specified axes, and plot. If we had more than a single participant in this experiment, it would even plot the error bars... but I don't suppose you want to do this experiment nine more times?

Run ``run.py`` again, but this time hit ``autorun`` button.

<a few seconds later> That was fast. What happened was that we asked the computer to do the experiment for us. If you check how well it did, you'll notice that it was not responding randomly. Rather, it was responding in line with our hypothesis (that triangles are easier to detect among arrows than lines are among lines. This autorun feature is very handy when you want to make sure the experiment and analyses run as expected, from start to finish (known as *functional testing*). You can use the default autorun or (preferebly) tune it to your experimental hypothesis.   

