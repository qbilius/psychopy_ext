import uuid, os, warnings
import pygame
from psychopy import monitors


class Computer:
    stereo = False
    trigger = 'space'
    defaultKeys = ['escape', trigger]
    validResponses = {0: '1', 1: '2', 2: '3', 3: '4'}
    
    def __init__(self, mac = None):
        mac = uuid.getnode() # get mac address
        
        # get display size
        pygame.init()
        dispInfo = pygame.display.Info()
        pygame.quit()
        self.dispSize = (dispInfo.current_w,dispInfo.current_h)
        
        self.getMonitor(mac)
        self.monitor.setSizePix(self.dispSize)
        
        
    def getMonitor(self, mac):
        """
        Get parameters of stored monitors
        """
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
            
            self.params = {
                'pos': pos,
                'size': size}
            
        elif os.getenv('COMPUTERNAME') == 'U90973': # fMRI computer
            self.monitor = monitors.Monitor('fMRI computer', distance = 127, width = 60)
            self.params = {'viewScale': [1,-1]} # top-bottom inverted

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
            self.validResponses = {1: 'num_1', 0: 'num_0'}
            
        elif mac == 123456789:
            self.monitor = monitors.Monitor('Storage Room', distance = 57, width = 37.5)
                
        else:
            warnings.warn('Computer not recognized, will use default monitor parameters')
            # create a default monitor first
            self.monitor = monitors.Monitor('default', distance = 80, width = 37.5)
    
