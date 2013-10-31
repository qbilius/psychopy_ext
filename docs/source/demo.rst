====
Demo
====

Find the ``demo`` folder in *psychopy_ext* (:ref:`where is it? <where-is-demo>`), copy it to your home folder and run the ``run.py`` file (:ref:`how? <how-run-demo>`). You'll see an app appear for this *Demo Project*:

    .. image:: gui.png
        :width: 300px

Experiments belonging to this project are listed on the left, and options for each experiment are listed in tabs. You can choose various parameters how to run the experiment, such as entering the participant ID or determining if the experiment should run in full screen. Note that ``psychopy_ext`` generates these apps completely automatically by collecting information within your project. (For command-line ninjas, a powerful command-line interface is provided too.)

Try to run "Main" experiment by clicking the "run" button. Come back when done with the experiment (it's only 40 trials).

    .. image:: exp_win.png
        :width: 400px

(If you want to learn how the experiment was constructed, go to the :ref:`quickstart` section.)

Now how well did you do? Click on the *analysis* tab, and hit the *run* button. You should see the following plot appear:

    .. image:: barplot_single.png
        :width: 400px

Again, consistent with the underlying ``psychopy_ext's`` philosophy, this was generated using only a couple of lines of code: read data, aggregate over specified columns, and plot. (Find it ugly? Check out our :ref:`gallery`.) No need to tinker with plotting options, and if we had more than a single participant in this experiment, it would even plot the error bars with stars on top of them... but I don't suppose you want to do this experiment nine more times?

Run the experiment again, but this time select the ``autorun`` option.

<a few seconds pass>

Done! That was fast. What happened was that we asked the computer to do the experiment for us. Moreover, if you check how well it did, you'll notice that it was not responding randomly. Rather, it was responding in line with our hypothesis (that triangles are easier to detect among arrows than lines are among lines. This autorun feature is very handy when you want to make sure the experiment and analyses run as expected, from start to finish (known as *functional testing*). You can use the default autorun or (preferebly) tune it to your experimental hypothesis.

Continue on :ref:`quickstart`.
