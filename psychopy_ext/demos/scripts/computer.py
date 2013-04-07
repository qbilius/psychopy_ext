#!/usr/bin/env python

"""
Computer configuration file
===========================

Specify default settings for all computers where you run your experiment such
as a monitor size or root path to storing data. This is intended as a more
portable and extended version of PsychoPy's MonitorCenter.

A computer is recognized by its mac address which is dependent on its
hardware and by its name. If anything in the hardware changes, you'll see a
warning.

# TODO: split computer configuration and defaults possibly by moving to a
config file

"""

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
viewScale = (1, 1)

# Get computer properties
# Computer is recognized by its mac address
mac = uuid.getnode()
name = platform.uname()[1]
if mac == 153254424819 and name == 'qLEPu':
    distance = 80
    width = 37.5
    root = '/media/qbilius/Data/data/'

elif mac == 153254424819 and name == 'qLEP':
    root = 'D:/data/'

elif mac == 145320949993177:  # fMRI computer
    distance = 127
    width = 60
    viewScale = [1,-1]  # top-bottom inverted
    trigger = 5
    defaultKeys = ['escape', trigger]
    validResponses = {'9': 0, '8': 1, '7': 2, '6': 3}

elif mac == 269295399767497 and name == 'qDesktop':
    distance = 57
    width = 47.7
    root = '/home/qbilius/Dropbox/data/'

elif mac == 159387622736430:  # Hendrik's desktop
    distance = 65
    width = 41

elif mac == 61959089469690:  # Hendrik's netbook
    distance = 45
    width = 22

else:
    recognized = False