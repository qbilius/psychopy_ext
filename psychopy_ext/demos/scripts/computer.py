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
system = platform.uname()[0]
name = platform.uname()[1]

if mac == 153254424819 and system == 'Linux':  # Lab computer, Ubuntu booted
    distance = 80
    width = 37.5
    root = '/media/qbilius/Data/data/'

elif mac == 153254424819 and system == 'Windows':  # Lab computer, Windows booted
    root = 'D:/data/'

elif mac == 145320949993177:  # fMRI computer
    distance = 127
    width = 60
    viewScale = [1,-1]  # top-bottom inverted
    trigger = 5
    defaultKeys = ['escape', trigger]
    validResponses = {'9': 0, '8': 1, '7': 2, '6': 3}

else:
    recognized = False