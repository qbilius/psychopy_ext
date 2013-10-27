.. _computer:

==============
Computer setup
==============

You probably have a couple of computers that you use to create and run experiments. Their parameters are kept in PsychoPy's :class:`Monitors`. However, since they are not embedded in your code, when you share your project, nobody knows the exact parameters of your setup. Moreover, it is possible that different users have different preferences for setup on the same computer, so it would be more convenient to just keep a file a complete setup for your computers together with your code.

This information is kept in the ``computers.py`` file that should be imported at the top of your experiment script. The file might look like this (also see the complete example in the demos folder)::

    import uuid, platform

    recognized = True
    # computer defaults
    root = '.'  # means store output files here
    stereo = False  # not like in Psychopy; this merely creates two Windows
    default_keys = {'exit': ('lshift', 'escape'),  # key combination to exit
                    'trigger': 'space'}  # hit to start the experiment
    valid_responses = {'0': 0, '1': 1}  # organized as input value: output value
    # monitor defaults
    distance = 80
    width = 37.5
    # window defaults
    screen = 0  # default screen is 0
    viewScale = (1,1)

    # Get computer properties
    # Computer is recognized by its mac address
    mac = uuid.getnode()
    system = platform.uname()[0]
    name = platform.uname()[1]    
    
    if mac == 153254424809:  # Lab computer
        distance = 80
        width = 37.5
        root = '/media/qbilius/Data/data/'
    elif mac == 153254424801:  # fMRI computer
        distance = 127
        width = 60
        view_scale = [1,-1]  # top-bottom inverted
        default_keys['trigger'] = 5
        valid_responses = {'9': 0, '8': 1, '7': 2, '6': 3}

You are now ready to create your first experiment: :ref:`exp`.
