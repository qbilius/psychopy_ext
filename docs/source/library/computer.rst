==============
Computer setup
==============

You probably have a couple of computers that you use to create and run experiments. Their parameters are kept in PsychoPy's :class:`Monitors`. However, since they are not embedded in your code, when you share your project, nobody knows the exact parameters of your setup. Moreover, it is possible that different users have different preferences for setup on the same computer, so it would be more convenient to just keep a file a complete setup for your computers together with your code.

This information is kept in the ``computers.py`` file that should be imported at the top of each experimental module. The file might look like this::

    import uuid, platform

    recognized = True
    # computer defaults
    root = '.'  # means store output files here
    stereo = False  # not like in Psychopy; this merely creates two Windows
    trigger = 'space'  # hit to start the experiment
    defaultKeys = ['escape', trigger]  # "special" keys
    validResponses = {'0': 0, '1': 1}  # organized as input value: output value
    # monitor defaults
    distance = 80
    width = 37.5
    # window defaults
    screen = 0  # default screen is 0
    viewScale = [1,1]

    # Get computer properties
    # Computer is recognized by its mac address
    mac = uuid.getnode()
    system = platform.uname()[0]
    name = platform.uname()[1]
    # LEP computer
    if mac == 153254424819:
        distance = 80
        width = 37.5
        root = '/media/qbilius/Data/data/'
    elif...
