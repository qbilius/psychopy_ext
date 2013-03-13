#!/usr/bin/env python

"""
Computer configuration file
===========================

Specify default settings for all computers where you run your experiment such as a monitor size or root path to storing data. This is intended as a more portable and extended version of PsychoPy's MonitorCenter.

A computer is recognized by its mac address which is dependent on its
hardware. So if anything in the hardware changes, you'll see a warning on the
terminal.

# TODO: split computer configuration and defaults possibly by moving to a config file

"""

import uuid, os
import pygame
from psychopy import monitors, logging


# computer defaults
comprc = {
    'stereo': False,  # this is not like in Psychopy; this merely creates two screens
    'trigger': 'space',  # hit to start the experiment
    'defaultKeys': ['escape', 'space'],  # "special" keys
    'validResponses': {'0': 0, '1': 1},  # organized as input value: output value
    'path': '.'  # means store output files here
    }

# monitor defaults
monrc = {'name': 'default', 'distance': 80, 'width': 37.5}

# Window parameter setup
# get display resolution
pygame.init()
dispInfo = pygame.display.Info()
pygame.quit()
dispSize = (dispInfo.current_w,dispInfo.current_h)

winrc = {}  # Window parameters are stored here
# default window is half the screen size
winrc['size'] = (int(dispSize[0] / 2), int(dispSize[1] / 2))
# center window on the display
winrc['pos'] = ((dispSize[0] - winrc['size'][0]) / 2,
                (dispSize[1] - winrc['size'][1]) / 2 )
# for stereo displays
winrc2 = {}  # for steroe displays


# recognize computer by its mac address
mac = uuid.getnode()
if mac == 153254424819:
    if os.getenv('DESKTOP_SESSION') == 'ubuntu':
        monrc['name'] = 'My LEP Ubuntu computer'
        monrc['distance'] = 80
        monrc['width'] = 37.5*2
        winrc['size'] = (dispSize[0]/4, dispSize[1]/2)
        winrc['pos'] = ((dispSize[0] / 2- winrc['size'][0]) / 2,
                        (dispSize[1] - winrc['size'][1]) / 2)
        comprc['path'] = '/media/qbilius/Data/data/'
    else:
        monrc['name'] = 'My LEP Windows computer'
        comprc['path'] = 'D:/data/'

elif mac == 145320949993177:
    monrc['name'] = 'fMRI computer'
    monrc['distance'] = 127
    monrc['width'] = 60
    winrc['viewScale'] = [1,-1]  # top-bottom inverted
    comprc['trigger'] = '5'
    comprc['defaultKeys'] = ['escape', comprc['trigger']]
    comprc['validResponses'] = {'9': 0, '8': 1, '7': 2, '6': 3}

elif os.getenv('COMPUTERNAME') == 'P101PW007':
    monrc['name'] = 'Stereo lab'
    monrc['distance'] = 100
    winrc['size'] = ( int(dispSize[0]/2/1.2), int(dispSize[1]/1.2) )
    winrc['viewScale'] = [-1,1]
    winrc['screen'] = 1
    winrc2['size'] = winrc['size']
    winrc['pos'] = ((dispSize[0] / 2 - winrc['size'][0]) / 2,
                    (dispSize[1] - winrc['size'][1]) / 2)
    winrc2['pos'] = ((3 * dispSize[0] / 2 - winrc['size'][0]) / 2,
                    (dispSize[1] - winrc['size'][1])/2 )
    comprc['stereo'] = True
    comprc['trigger'] = 'num_4'
    comprc['defaultKeys'] = ['escape', comprc['trigger']]
    comprc['validResponses'] = {'num_1': 1, 'num_0': 0}

elif mac == 269295399767497:
    monrc['name'] = 'Home computer'
    monrc['distance'] = 120
    monrc['width'] = 47.7
    winrc['size'] = (int(dispSize[0] / 2 / 1.2), int(dispSize[1] / 1.2))
    comprc['path'] = '/home/qbilius/Dropbox/data/'

elif mac == 159387622736430:
    monrc = {'name': 'Hendrik desktop', 'distance': 65, 'width': 41}

elif mac == 61959089469690:
    monrc = {'name': 'Hendrik netbook', 'distance': 45, 'width': 22}

else:
    logging.warn('Computer not recognized, will use default monitor parameters')

# Temporarily suppres the useless warnings of a new monitor creation
current_level = logging.getLevel(logging.console.level)
logging.console.setLevel(logging.ERROR)
monitor = monitors.Monitor(monrc['name'], **monrc)
logging.console.setLevel(current_level)
monitor.setSizePix(dispSize)

if comprc['stereo']:  # create parameters for stereo displays
    winrc2.update(winrc)
    winrc2['screen'] = 1
