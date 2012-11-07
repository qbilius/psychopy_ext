import uuid, os, warnings
import pygame
from psychopy import monitors, logging

# Suppres the useless warnings of a new monitor creation
logging.console.setLevel(logging.ERROR)


class Computer:
    """Applies stored monitors settings for Psychopy Monitor Center"""

    def __init__(self, mac = None):
        mac = uuid.getnode() # get mac address

        self.stereo = False
        self.trigger = 'space'
        self.defaultKeys = ['escape', self.trigger]
        self.validResponses = {'0': 0, '1': 1}

        # get display size
        pygame.init()
        dispInfo = pygame.display.Info()
        pygame.quit()
        self.dispSize = (dispInfo.current_w,dispInfo.current_h)

        self.getMonitor(mac)
        self.monitor.setSizePix(self.dispSize)
        
        if not 'size' in self.params:
            # default window is half the screen size
            self.params['size'] = (
                int(self.dispSize[0]/1.2),
                int(self.dispSize[1]/1.2) )

        if not 'pos' in self.params:
            # center window on the display
            self.params['pos'] = (
                (self.dispSize[0]-self.params['size'][0])/2,
                (self.dispSize[1]-self.params['size'][1])/2 )

        # if not debug:
            # # when debugging
            # self.params['pos'] = None
            # self.params['size'] = self.dispSize
            
        # create parameters for stereo displays
        params_tmp = {}
        for key, value in self.params.items(): params_tmp[key] = value
        params_tmp['screen'] = 1
        # set special stereo settings for this monitor
        for key, value in self.params2.items(): params_tmp[key] = value
        self.params2 = params_tmp
        
        # initial set up
        if not hasattr(self,'path'):
            self.path = '.' # indicates that working directory is here



    def getMonitor(self, mac):
        """Get parameters of stored monitors"""
        self.params = {}
        self.params2 = {}

        # recognize computer by its mac address
        if mac == 153254424819L:
            if os.getenv('DESKTOP_SESSION') == 'ubuntu':
                self.monitor = monitors.Monitor('My LEP computer', distance = 80, width = 37.5*2)
                size = ( self.dispSize[0]/4, self.dispSize[1]/2 )
                pos = ( (self.dispSize[0]/2-size[0])/2, (self.dispSize[1]-size[1])/2 )
            else:
                self.monitor = monitors.Monitor('My LEP computer', distance = 80, width = 37.5)
                size = ( self.dispSize[0]/2, self.dispSize[1]/2 )
                pos = ( (self.dispSize[0]-size[0])/2, (self.dispSize[1]-size[1])/2 )
                self.path = 'D:/data/'
            self.params = {
                'pos': pos,
                'size': size}

        elif mac == 145320949993177L: # fMRI computer
            self.monitor = monitors.Monitor('fMRI computer', distance = 127, width = 60)
            self.params = {'viewScale': [1,-1]} # top-bottom inverted
            self.trigger = '5'
            self.defaultKeys = ['escape', self.trigger]
            self.validResponses = {'9': 0, '8': 1, '7': 2, '6': 3}

        elif os.getenv('COMPUTERNAME') == 'P101PW007': # Stereo lab
            self.monitor = monitors.Monitor('Stereo Lab', distance = 100, width = 37.5)
            size = ( int(self.dispSize[0]/2/1.2), int(self.dispSize[1]/1.2) )
            self.params = {
                'viewScale': [-1,1],
                'screen': 1,
                'size': size,
                'pos': ( (self.dispSize[0]/2-size[0])/2, (self.dispSize[1]-size[1])/2 )
                }
            self.params2['pos'] = ( (3*self.dispSize[0]/2-size[0])/2, (self.dispSize[1]-size[1])/2 )

            self.stereo = True
            self.trigger = 'num_4'
            self.defaultKeys = ['escape', self.trigger]
            self.validResponses = {'num_1': 1, 'num_0': 0}

        elif mac == 144340649432L:
            self.monitor = monitors.Monitor('Storage Room', distance = 57, width = 37.5)
            self.path = 'V:/jonas/data/'

        elif mac == 269295399767497L:  # Home computer
            self.monitor = monitors.Monitor('Home computer', distance = 120, width = 47.7)
            self.path = '/home/qbilius/Dropbox/data/'

        else:
            warnings.warn('Computer not recognized, will use default monitor parameters')
            # create a default monitor first
            self.monitor = monitors.Monitor('default', distance = 80, width = 37.5)

