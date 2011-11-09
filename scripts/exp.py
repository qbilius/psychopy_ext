###############################################################################

"""
Generic functions for any kind of an experiment
"""

import sys, os, csv, time

import numpy as np
from psychopy import visual, core, event, monitors
import pygame
#import winsound

# include commonly used modules and functions
import scripts.tools.qFunctions as q
import scripts.core.init as cinit
from computer import Computer


class Experiment:

    def __init__(self):
        
        self.defaultFilter = "(plotData['cond'] != 0) & (plotData['accuracy'] != 'No response')"
        self.signalDet = {
            False: 'Incorrect',
            True: 'Correct',
            '': 'No response'}
        self.keyNames = {
            1: 'index finger',
            2: 'middle finger',
            3: 'ring finger',
            4: 'little finger'} 
            
                    
    def setExpPlan(self, expPlan):
        self.expPlan = expPlan
        
    
    def exit(self):
        """
        What to do when exit is requested
        """
        self.dispRunStats()
        core.quit()
                
            
    def lastKey(self, keyList = Computer.defaultKeys):
        """
        Extract the last key pressed from the event list    
        """
        thisKeyList = event.getKeys(keyList = keyList)
        if len(thisKeyList) > 0:        
            thisKey = thisKeyList.pop()
            if thisKey == 'escape': self.exit()
            else: return thisKey
        else:
            return None
            

    def trial2struct(self,trial):
    
        trialStruct = []
        for event in trial:
            eventStruct = {}
            eventStruct['dur'] = event[0]
            eventStruct['display'] = event[1]
            eventStruct['defaultFun'] = event[2]
            trialStruct.append(eventStruct)
        
        return trialStruct
        
        
    def TrialHandler2rec(self):
    
        import numpy.lib.recfunctions as nprc
        
        data = np.array([thisTrial.values() for thisTrial in self.expPlan.trialList])
        names = self.expPlan.trialList[0].keys()
        
        # remove trialStruct as rec does not like it
        if 'trialStruct' in names:
            data = np.delete(data,names.index('trialStruct'),1)
            del names[names.index('trialStruct')]
            
        expPlanRec = np.rec.fromrecords(data, names = names)
        return expPlanRec    
        
        
    def createWin(self, debug = False):
        """
        Generates a window from presenting stimuli.
        """
        
        self.comp = Computer()        
        
        if not 'size' in self.comp.params:
            self.comp.params['size'] = (
                self.comp.dispSize[0]/2,
                self.comp.dispSize[1]/2 )
        
        if not 'pos' in self.comp.params:
            # center window on the display
            self.comp.params['pos'] = (
                (self.comp.dispSize[0]-self.comp.params['size'][0])/2,
                (self.comp.dispSize[1]-self.comp.params['size'][1])/2 )
        if not debug:
            self.comp.params['pos'] = None
            self.comp.params['size'] = self.comp.dispSize
        
        # create parameters for stereo displays
        params_tmp = {}
        for key, value in self.comp.params.items(): params_tmp[key] = value
        params_tmp['screen'] = 1
        # set special stereo settings for this monitor
        for key, value in self.comp.params2.items(): params_tmp[key] = value 
        self.comp.params2 = params_tmp      
        
        color = 100/255.*2-1
        
        self.win = visual.Window(
            monitor  = self.comp.monitor,        
            units    = 'deg',
            fullscr  = not debug,
            allowGUI = debug, # mouse will not be seen unless debugging
            color    = (color,color,color),
            winType = 'pyglet',
            **self.comp.params
        )
        
        if self.comp.stereo:
            self.win2 = visual.Window(
                monitor  = self.comp.monitor,        
                units    = 'deg',
                fullscr  = not debug,
                allowGUI = debug, # mouse will not be seen unless debugging
                color    = (color,color,color),
                winType = 'pyglet',
                **self.comp.params2
            )
        
            
    def createBasicStim(self, debug = False):
        """
        Generates a window from presenting stimuli. Prepares a fixation and a black
        box for easy use. They all can be accessed later on.
        """
        
        # Set up fixation
        self.fixation = visual.PatchStim(
            self.win,
            name   = 'fixation',
            color  = 'red',
            tex    = None,
            mask   = 'circle',
            size   = .1,
        )
        
        # A box representing the size of a stimulus
        self.box = visual.ShapeStim(
            self.win,
            lineColor   = 'white',
            fillColor   = 'white',
            closeShape  = True
        )
        
        if self.comp.stereo:
        
            # Set up fixation
            self.fixation2 = visual.PatchStim(
                self.win2,
                name   = 'fixation',
                color  = 'red',
                tex    = None,
                mask   = 'circle',
                size   = .2,
            )
            
            # A box representing the size of a stimulus
            self.box2 = visual.ShapeStim(
                self.win2,
                lineColor   = 'white',
                fillColor   = 'white',
                closeShape  = True
            )
        
        
    def showIntro(self, text = None, args = None, wait = 0):
        """
        Show instructions until trigger is received
        """
        
        if args != None:
            (vertices, p, stimSize) = args
            # add response keys and stimuli-size boxes at the stimuli position
            box.setVertices(vertices)
            for (key, value) in keyNames.items():
                box.setPos(p[key])
                box.draw()
                number = visual.TextStim(self.win, text = value,
                                         pos = p[key], wrapWidth = stimSize - 1)
                number.draw()
                if self.comp.stereo:
                    number2 = visual.TextStim(self.win2, text = value,
                                             pos = p[key], wrapWidth = stimSize - 1)
                    number2.draw()

        if text != None:
            introText = visual.TextStim(
                self.win,
                text = text,
                color = 'black',
                height = .7,
                alignHoriz = 'center')
            introText.draw()

            if self.comp.stereo:
                introText2 = visual.TextStim(
                    self.win2,
                    text = text,
                    color = 'black',
                    height = .7,
                    alignHoriz = 'center')
                introText2.draw()
            
        self.win.flip()
        if self.comp.stereo: self.win2.flip()
        
        thisKey = None
        while thisKey != self.comp.trigger: thisKey = self.lastKey()    
        self.win.flip()
        if self.comp.stereo: self.win2.flip()
        
        core.wait(wait)
        
            
    def waitEvent(self, globClock, trialClock, eventClock,
        expPlan, thisTrial, thisEvent):
        """
        Default waiting function for the event
        Does nothing but catching key input of default keys (escape and trigger)
        """
        
        if thisEvent['dur'] == 0:
            self.lastKey()
            for stim in thisEvent['display']: stim.draw()
            self.win.flip()
            if self.comp.stereo: self.win2.flip()
            
        else:
            while eventClock.getTime() < thisEvent['dur'] and \
            trialClock.getTime() < thisTrial['dur']:# and \
            # globClock.getTime() < thisTrial['onset'] + thisTrial['dur']:
                self.lastKey()
                for stim in thisEvent['display']: stim.draw()
                self.win.flip()
                if self.comp.stereo: self.win2.flip()            
                

    def run(self, noOutput = False, audioFeedback = False, captureWin = False):
        """
        Iterate over the sequence of events
        """
         
        if not noOutput:
        
            if not os.path.isdir(cinit.paths['dataBehavDir']):
                os.makedirs(cinit.paths['dataBehavDir'])
            fileName = '_'.join(['data','%02d' %int(self.expPlan.trialList[0]['runNo']),
                self.expPlan.trialList[0]['runType']])
            filePath = cinit.paths['dataBehavDir'] + '/' + fileName + '.csv'
            fileExists = os.path.isfile(filePath)
            dataFile = open(filePath, 'a')
            dataCSV = csv.writer(dataFile, lineterminator = '\n')
            
            if not fileExists: #dataCSV.writerow(expPlan.dtype.names[:-1])
                header = self.expPlan.trialList[0].keys() #+ ['subjResp', 'accuracy','RT']
                trialStructIndex = header.index('trialStruct')
                del header[trialStructIndex]
                dataCSV.writerow(header)
                
        # set up clocks
        globClock = core.Clock()
        trialClock = core.Clock()
        eventClock = core.Clock()
        
        # go over the trial sequence
        for thisTrial in self.expPlan:
#            import pdb; pdb.set_trace()
            trialClock.reset()
            thisResp = ['',0]
            thisTrial['trialNo'] = self.expPlan.thisTrialN
            thisTrial['onset'] = self.expPlan.thisTrialN*thisTrial['dur'] # NOT QUITE CORRECT
            
            # go over each event in a trial    
            for j, thisEvent in enumerate(thisTrial['trialStruct']):
                eventClock.reset()
               
                if j == 0: thisTrial['actualOnset'] = globClock.getTime()
                
                thisEvent['defaultFun'](globClock, trialClock, eventClock,
                    self.expPlan, thisTrial, thisEvent)
                
                if thisEvent['dur'] == 0: # stop and wait for a response
                    if thisResp == ['',0]:
                        while len(allKeys) == 0: # if the participant did not respond earlier
                            allKeys = event.getKeys(
                                keyList = self.comp.validResponses.values(),
                                timeStamped = trialClock)
                            self.lastKey()
                        thisResp = allKeys.pop()
                else:
                    allKeys = event.getKeys(
                        keyList = self.comp.validResponses.values(),
                        timeStamped = trialClock)
                    
                    if len(allKeys) > 0: thisResp = allKeys.pop()
                
                self.clearDisplay(thisEvent['display'])
            
            thisTrial['subjResp'] = thisResp[0]
            thisTrial['accuracy'] = self.signalDet[q.advCmp(thisTrial['corrResp'], thisTrial['subjResp'])]
            thisTrial['RT'] = thisResp[1]*1000 #convert to msec
            
            # for practice trials, provide auditory feedback
            if audioFeedback:
                if thisTrial['accuracy']:
                    winsound.PlaySound('SystemAsterisk',winsound.SND_ALIAS)
                else:
                    winsound.PlaySound('SystemHand',winsound.SND_ALIAS)
            
            if not noOutput: #expPlan.saveAsWideText(thisTrial)#list(thisTrial)[:-1])
                out = thisTrial.values()# + [val[trialNo] for val in expPlan.data.values()]            
                trialStructIndex = self.expPlan.trialList[0].keys().index('trialStruct')
                del out[trialStructIndex]
                dataCSV.writerow(out)
            
        if not noOutput: dataFile.close()
        self.win.close()
        if self.comp.stereo: self.win2.close()

    
    def clearDisplay(self, itemList):
        """
        Clears textures of given stimuli in order to free memory
        """                      
        for item in itemList:
            try:
                item[0].clearTextures
            except:
                pass

    
    def dispRunStats(self):
        """
        Prints out RT and percent correct response
        """
        
        if self.expPlan.__class__.__name__ == 'TrialHandler':
            expPlanRec = self.TrialHandler2rec()
            
        completeTrials = expPlanRec['actualOnset'] != 0
        onsetVar = expPlanRec[completeTrials]['actualOnset'] - expPlanRec[completeTrials]['onset']
        
        print
        print '==== Run Stats ===='
        print 'Onset diff: max = %.4f and min = %.4f' %(onsetVar.max(), onsetVar.min())

        print 'No response: %d' %np.sum(expPlanRec[completeTrials]['accuracy'] == 'No response')  
         
        
         
    
#class Extras:
    def drawLine(self,shapeStim):
        """
        Draws thick shape stimuli as a collection of lines
        Necessary when PsychoPy fails to draw ShapeStim due to its thickness
        """
        vertices = shapeStim.vertices
        if shapeStim.closeShape: numPairs = len(vertices)
        else: numPairs = len(vertices)-1

        
        shape = []
        
        wh = shapeStim.lineWidth/2
        for i in range(numPairs):
            thisPair = [shapeStim.vertices[i],shapeStim.vertices[(i+1)%len(shapeStim.vertices)]]
            edges = [thisPair[1][0]-thisPair[0][0], thisPair[1][1]-thisPair[0][1]]
            #direction = np.sign(thisPair[1][0]-thisPair[0][0])
            #parts = [np.sign(thisPair[0][0])*np.sqrt(thisPair[0][0]**2 + thisPair[0][1]**2),
            #         np.sign(thisPair[1][0])*np.sqrt(thisPair[1][0]**2 + thisPair[1][1]**2)]
            lh = np.sqrt(edges[0]**2 + edges[1]**2)/2#np.abs(parts[0])+np.abs(parts[1])# 
            
            line = visual.ShapeStim(
                self.win,
                lineWidth   = 0,
                lineColor   = None,

                fillColor   = shapeStim.lineColor,
                ori         = -np.arctan2(edges[1],edges[0])*180/np.pi,
                pos         = [(thisPair[0][0]+thisPair[1][0])/2, (thisPair[0][1]+thisPair[1][1])/2],
                vertices    = [[-lh,-wh],[-lh,wh],
                               [lh,wh],[lh,-wh]]
            )
            shape.append(line)
            #import pdb; pdb.set_trace()
        
        return shape
        
        
    def makePara(expPlan):
        """
        Makes a .m file for generating the SPM para file
        """
        global numTrialsInBlock, trialDur, TR
        def sortJoin(x):
            global TR
            x.sort()
            s = [str(int(round(item * 1000. / TR))) for item in x]
            return ' '.join(s)
        def sumDur(x):
            global TR
            #x.sort()
            #import pdb; pdb.set_trace()
            s = str(int(sum(x * 1000. / TR)))
            return ' '.join(s)
        
        # block = np.ones(expPlan.shape[0], dtype = np.int)    
        # if expPlan[0]['paraType'] == 'blocked':        
            # SPMblockLen = str(int(numTrialsInBlock * trialDur * 1000. / TR))
            # # lastCond = None
            # # for i, cond in enumerate(expPlan['cond']):
                # # if cond != lastCond:
                    # # block[i] = 1
                    # # lastCond = cond
                # # else:
                    # # block[i] = 0        
        # else:        
            # SPMblockLen = '0'
            
        #allData = expPlan.addcols(block, names=['block'])
        dataNames = list(expPlan.dtype.names)
        descr = dataNames[dataNames.index('dur') + 1:dataNames.index('corrResp')]
        #descr = dataNames[dataNames.index('dur') + 1]
        allData = expPlan[['blockNo','startBlock', 'cond', 'onset', 'dur'] + descr]\
            [expPlan[descr[0]] != 'Buffer']
        
        
        aggDur = allData.aggregate(
            On = ['cond','blockNo'], AggFunc = lambda x: x[0],
            AggFuncDict = {'dur': lambda x: str(int(round(sum(x * 1000. / TR))))})
        conds = aggDur.aggregate(On = 'cond')['cond'] 
        aggExtractDur = []
        for cond in conds:
            thisCond = aggDur[aggDur['cond'] == cond]
            thisCond.sort(order = ['blockNo'])
            aggExtractDur.append(' '.join(map(str,thisCond['dur'])))
        #import pdb; pdb.set_trace()
            
        # aggDur = aggDur.aggregate(
            # On = 'cond', AggFunc = lambda x: x[0],
            # AggFuncDict = {'dur': lambda x: ' '.join(x)})
        aggOnset = allData[allData['startBlock']==1].aggregate(
            On = 'cond', AggFunc = lambda x: x[0],
            AggFuncDict = {'onset': lambda x: sortJoin(x)})
        #aggExtractDur = aggDur[['dur']].extract()
        
        aggExtractOnset = aggOnset[['onset']].extract()    
        aggExtractDescr = aggOnset[q.listify(descr)].extract()
        
        out = {'names': [], 'onsets': [], 'durations': []}    
        for dur, onset, names in zip(aggExtractDur, aggExtractOnset, aggExtractDescr):
            out['durations'].append("[" + dur + "]")
            out['onsets'].append("[" + onset + "]")
            out['names'].append(' | '.join(q.listify(names)))
        
        lenName = 0
        for name in out['names']:
            if len(name) > lenName:
                lenName = len(name)
        out['names'] = ["'" + it.ljust(lenName) + "'" for it in out['names']]
        
        if not os.path.isdir(cinit.paths['paraDir']):
            os.makedirs(cinit.paths['paraDir'])
        fileName = '_'.join(['para','%02d' %(int(expPlan[0]['runNo'])),
            expPlan[0]['runType']])
        paraFile = open(cinit.paths['paraDir'] + '/' + fileName + '.m', 'w')
        for (key, item) in out.items():
            paraFile.write(key + " = {" + ", ".join(item) + "};\n")
        paraFile.write("save('" + fileName + '.mat' +
            "', 'names', 'onsets', 'durations');")
        paraFile.close()




