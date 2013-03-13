#!/usr/bin/env python

# Part of the psychopy_ext library
# Copyright 2010-2013 Jonas Kubilius
# The program is distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.

"""A library of helper functions for creating and running experiments"""

import sys, os, csv, glob, random
#import cPickle as pickle

import numpy as np
from psychopy import visual, core, event, logging, misc
from psychopy.data import TrialHandler
#import winsound
# pandas does not come by default with PsychoPy but that should not prevent
# people from running the experiment
try:
    import pandas
except:
    pass
#import wx
#import matplotlib.pyplot as plt

from computer import Computer




class Experiment(TrialHandler):
    """An extension of an TrialHandler with many useful functions.
    """
    def __init__(
        self,
        parent = None,
        name='',
        version=0.1,
        #win = None,
        extraInfo=None,
        runParams = None,
        instructions = {'text': '', 'wait': 0},
        actions = None,
        seed=None,
        nReps=1,
        method='random',
        dataTypes=None,
        originPath=None,
        ):
        """Add a loop such as a `~psychopy.data.TrialHandler` or `~psychopy.data.StairHandler`
        Data from this loop will be included in the resulting data files.
        """

        self.parent = parent
        self.name = name
        self.version = version
        self.extraInfo = extraInfo
        self.runParams = runParams
        self.instructions = instructions
        self.actions=actions
        self.nReps = nReps
        self.method = method
        self.dataTypes = dataTypes
        self.originPath = originPath

        if seed is None:
            self.seed = np.sum([ord(d) for d in self.extraInfo['date']])
        else:
            self.seed = seed

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
        self.comp = Computer()
        # if self.parent is None:
        #     Control(exp_choices=[(name,self)], title=name)

        #if self.trialList not in [None, []]: self.create_TrialHandler()

    # def __init__(self,
                # name='',
                # version='',
                # extraInfo=None,
                # runtimeInfo=None,
                # originPath=None,
                # savePickle=True,
                # saveWideText=True,
                # dataFileName=''):
#
        # ExperimentHandler.__init__(self,
                # name=name,
                # version=version,
                # extraInfo=extraInfo,
                # runtimeInfo=runtimeInfo,
                # originPath=originPath,
                # savePickle=savePickle,
                # saveWideText=saveWideText,
                # dataFileName=dataFileName)



    def guess_participant(self, data_path, default_subjID='01'):
        """Attempts to guess participant ID (it must be int).

        First lists all csv files in the data_path, then finds a maximum.
        Returns maximum+1 or an empty string if nothing is found.

        """
        datafiles = glob.glob(data_path+'*.csv')
        partids = []
        #import pdb; pdb.set_trace()
        for d in datafiles:
            filename = os.path.split(d)[1]  # remove the path
            filename = filename.split('.')[0]  # remove the extension
            partid = filename.split('_')[-1]  # take the numbers at the end
            try:
                partids.append(int(partid))
            except:
                logging.warning('Participant ID %s is invalid.' %partid)

        if len(partids) > 0: return '%02d' %(max(partids) + 1)
        else: return default_subjID


    def guess_runNo(self, data_path, default_runNo = 1):
        """Attempts to guess run number.

        First lists all csv files in the data_path, then finds a maximum.
        Returns maximum+1 or an empty string if nothing is found.

        """
        if not os.path.isdir(data_path): runNo = default_runNo
        else:
            dataFiles = glob.glob(data_path + '*.csv')
            # Splits file names into ['data', %number%, 'runType.csv']
            allNums = [int(os.path.basename(thisFile).split('_')[1]) for thisFile in dataFiles]

            if allNums == []: # no data files yet
                runNo = default_runNo
            else:
                runNo = max(allNums) + 1
                # print 'Guessing runNo: %d' %runNo

        return runNo


    #def get_input(self, info):
        #"""Creates a dialog to get user input and loads stored values.
        #"""

        #dlg = gui.DlgFromDict(dictionary=info,title=self.name)
        #if dlg.OK == False:
            #core.quit() # user pressed cancel
        #else:
            #return info


    def setup(self):
        if 'logs' in self.paths and not self.runParams['noOutput']:
            self.set_logging(self.paths['logs'] + self.extraInfo['subjID'])
        # self.try_makedirs(self.paths['data'])
        #self.dataFileName = self.dataFileName %self.extraInfo['subjID']
        self.create_win(debug=self.runParams['debug'])
        self.create_stimuli()
        self.create_trial()
        # self.trialDur = sum(event['dur'] for event in self.trial)
        trialList = self.create_trialList()
        if self.runParams['autorun']:
            trialList = self.autorun(trialList)
        self.create_TrialHandler(trialList)
        #dataFileName=self.paths['data']%self.extraInfo['subjID'])

        ## guess participant ID based on the already completed sessions
        #self.extraInfo['subjID'] = self.guess_participant(
            #self.paths['data'],
            #default_subjID=self.extraInfo['subjID'])

        #self.dataFileName = self.paths['data'] + '%s.csv'


    def try_makedirs(self, path):
        """Attempts to create a new dir. If fails, exists gracefully.
        """
        if not os.path.isdir(path) and path not in ['','.','./']:
            try: # if this fails (e.g. permissions) we will get error
                os.makedirs(path)
            except:
                logging.error('ERROR: Cannot create a folder for storing data %s' %path)
                # We'll enter the debugger so that we don't lose any data
                import pdb; pdb.set_trace()
                core.quit()


    def set_logging(self, logname='log.log'):
        """Setup files for saving. New folders might be created.
        """
        # Setup data file
        #datadir = self.paths['data_behav']%self.extraInfo['subjID']
        #self.try_makedirs(datadir)
        #filename = '_'.join(['data','%02d' %int(self.extraInfo['runNo']), self.extraInfo['runType']])
        #self.dataFilename = datadir + filename

        # add .log if no extension given
        if len(logname.split('.')) == 0: logname += '.log'

        # Setup logging file
        self.try_makedirs(os.path.dirname(logname))
        self.logFile = logging.LogFile(
            logname,
            filemode = 'a',
            level = logging.WARNING)
        # this outputs to the screen, not a file
        logging.console.setLevel(logging.ERROR)


    #def quit(self):
        #"""What to do when exit is requested.
        #"""
        #self.win.close()
        #core.quit()


    def create_win(self, debug = False, color = (100/255.*2-1,100/255.*2-1,100/255.*2-1)):
        """Generates a window from presenting stimuli.
        """
        if not hasattr(self,'comp'):
            self.comp = Computer()
        if not debug:
            self.comp.params['pos'] = (0,0)
            #self.comp.params['size'] = self.comp.params['size']
            self.comp.params2['pos'] = (0,0)
            #self.comp.params2['size'] = self.comp.params['size']/2
        self.win = visual.Window(
            monitor = self.comp.monitor,
            units = 'deg',
            fullscr = not debug,
            allowGUI = debug, # mouse will not be seen unless debugging
            color = color,
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

    def create_fixation(self, shape='complex'):
        """
        Creates a fixation.
        """
        if shape == 'complex':
            # based on the 'best' fixation shape by Thaler et al., 2012
            # (http://dx.doi.org/10.1016/j.visres.2012.10.012)
            d1 = .6  # diameter of outer circle (degrees)
            d2 = .2  # diameter of inner circle (degrees)
            oval = visual.PatchStim(
                self.win,
                name   = 'oval',
                color  = 'black',
                tex    = None,
                mask   = 'circle',
                size   = d1,
            )
            center = visual.PatchStim(
                self.win,
                name   = 'center',
                color  = 'black',
                tex    = None,
                mask   = 'circle',
                size   = d2,
            )
            cross0 = ThickShapeStim(
                self.win,
                name='cross1',
                lineColor=self.win.color,
                lineWidth=d2,
                vertices=[(-d1/2,0),(d1/2,0)]
                )
            cross90 = ThickShapeStim(
                self.win,
                name='cross1',
                lineColor=self.win.color,
                lineWidth=d2,
                vertices=[(-d1/2,0),(d1/2,0)],
                ori=90
                )
            self.fixation = [oval, cross0, cross90, center]

        elif shape == 'dot':
            self.fixation = [visual.PatchStim(
                self.win,
                name   = 'fixation',
                color  = 'red',
                tex    = None,
                mask   = 'circle',
                size   = .2,
            )]

    def latin_square(self, n = 6):
        """
        Generates a Latin square of size n. n must be even.

        Based on
        http://rintintin.colorado.edu/~chathach/balancedlatinsquares.html
        """
        if n%2 != 0: sys.exit('n is not even!')

        latin = []
        col = np.arange(1,n+1)

        firstLine = []
        for i in range(n):
            if i%2 == 0: firstLine.append((n-i/2)%n + 1)
            else: firstLine.append((i+1)/2+1)

        latin = np.array([np.roll(col,i-1) for i in firstLine])

        return latin.T

    def make_para(self, n = 6):
        """
        Generates a symmetric para file with fixation periods in between. n must be even.
        """
        latin = self.latin_square(n = n).tolist()

        out = []
        for j, thisLatin in enumerate(latin):

            thisLatin = thisLatin + thisLatin[::-1]
            # para = open('para%02d.txt' %(j+1),'w')

            temp = []
            for i, item in enumerate(thisLatin):
                if i%4 == 0: temp.append(0) #para.write('0\n')
                temp.append(item)
                # para.write(str(item)+'\n')
            # para.write('0')
            temp.append(0)
            out.append(temp)
            # para.close()
        return np.array(out)

    def last_keypress(self, keyList = None):
        """
        Extract the last key pressed from the event list.

        If escape is pressed, quits.
        """
        if keyList is None: keyList = self.comp.defaultKeys
        thisKeyList = event.getKeys(keyList = keyList)
        if len(thisKeyList) > 0:
            thisKey = thisKeyList.pop()
            if thisKey == 'escape':
                self.quit()
            else:
                return thisKey
        else:
            return None

    def wait_for_response(self, RT_clock=False, fakeKey=None):
        """
        Waits for response. Returns last key pressed, timestamped.

        :parameters:

            RT_clock: False or psychopy.core.Clock
                A clock used as a reference for measuring response time

            fakeKey: None or (key pressed, response time)
                This is used for simulating key presses in order to test that
                the experiment is working.

        """
        allKeys = []
        event.clearEvents() # key presses might be stored from before
        while len(allKeys) == 0: # if the participant did not respond earlier
            if fakeKey is not None:
                if RT_clock.getTime() > fakeKey[1]:
                    allKeys = [fakeKey]
            else:
                allKeys = event.getKeys(
                    keyList = self.comp.validResponses.keys(),
                    timeStamped = RT_clock)
            self.last_keypress()
        return allKeys


    def waitEvent(self, globClock, trialClock, eventClock,
        thisTrial, thisEvent, j):
        """
        Default waiting function for the event
        Does nothing but catching key input of default keys (escape and trigger)
        """

        if thisEvent['dur'] == 0:
            self.last_keypress()
            for stim in thisEvent['display']: stim.draw()
            self.win.flip()
            if self.comp.stereo: self.win2.flip()

        else:
            for stim in thisEvent['display']: stim.draw()
            self.win.flip()
            if self.comp.stereo: self.win2.flip()


            while eventClock.getTime() < thisEvent['dur'] and \
            trialClock.getTime() < thisTrial['dur']:# and \
            # globClock.getTime() < thisTrial['onset'] + thisTrial['dur']:
                #self.win.getMovieFrame()
                self.last_keypress()

    def postTrial(self, thisTrial, allKeys):
        if len(allKeys) > 0:
            thisResp = allKeys.pop()
            thisTrial['subjResp'] = self.comp.validResponses[thisResp[0]]
            thisTrial['accuracy'] = self.signalDet[thisTrial['corrResp']==thisTrial['subjResp']]
            thisTrial['RT'] = thisResp[1]
        else:
            thisTrial['subjResp'] = ''
            thisTrial['accuracy'] = 'No response'
            thisTrial['RT'] = ''

        return thisTrial

    def quit(self):
        """What to do when exit is requested.
        """
        #if not self.extraInfo['noOutput']:
            #filename = '_'.join(['data','%02d' %int(self.extraInfo['runNo'])])
            #datadir = self.paths['data_behav'] %self.extraInfo['subjID']
            #self.try_makedirs(datadir)
            #import pdb; pdb.set_trace()
            #self.saveAsWideText(datadir+filename+'.csv')
        #self._currentLoop.save_data()
        # if 'noOutput' in self.extraInfo:
        #     if not self.extraInfo['noOutput']: self.save_data()
        self.win.close()
        core.quit()

    def show_instructions(self, text = '', wait = 0):
        """
        Displays instructions on the screen.

        :parameters:

            text: str
                Text to be displayed

            wait: int
                Seconds to wait before flipping

        """
        instructions = visual.TextStim(self.win, text=text,
            color='white', height=20, units='pix', pos=(0,0),
            wrapWidth=30*20)
        instructions.draw()
        self.win.flip()

        if not self.runParams['autorun']:
            thisKey = None
            while thisKey != self.comp.trigger:
                thisKey = self.last_keypress()
            if self.runParams['autorun']: wait /= self.runParams['autorun']
        core.wait(wait) # wait a little bit before starting the experiment
        self.win.flip()

    def create_TrialHandler(self, trialList):
        TrialHandler.__init__(self,
            trialList,
            nReps=self.nReps,
            method=self.method,
            dataTypes=self.dataTypes,
            extraInfo=self.extraInfo,
            seed=self.seed,
            originPath=self.originPath,
            name=self.name)
        self.trialList = trialList

    def loop_trials(self, datafile='data.csv', noOutput=False):
        """
        Iterate over the sequence of events
        """
        if not noOutput:
            self.try_makedirs(os.path.dirname(datafile))
            # no header needed if the file already exists
            if os.path.isfile(datafile):
                write_head = False
            else:
                write_head = True
            try:
                dataFile = open(datafile, 'ab')
                dataCSV = csv.writer(dataFile, lineterminator = '\n')
            except:
                raise IOError('Cannot write anything to the data file %s!' %
                              datafile)
        # set up clocks
        globClock = core.Clock()
        trialClock = core.Clock()
        eventClock = core.Clock()
        trialNo = 0
        # go over the trial sequence
        for thisTrial in self:
            trialClock.reset()
            thisTrial['onset'] = globClock.getTime()
            sys.stdout.write("\rtrial %s" % (trialNo+1))
            sys.stdout.flush()

            # go over each event in a trial
            allKeys = []
            for j, thisEvent in enumerate(self.trial):
                eventClock.reset()
                eventKeys = thisEvent['defaultFun'](globClock=globClock,
                    trialClock=trialClock, eventClock=eventClock,
                    thisTrial=thisTrial, thisEvent=thisEvent, j=j)
                if eventKeys is not None:
                    allKeys += eventKeys
                # this is to get keys if we did not do that during trial
                allKeys += event.getKeys(
                    keyList = self.comp.validResponses.keys(),
                    timeStamped = trialClock)

            thisTrial = self.postTrial(thisTrial, allKeys)
            if self.runParams['autorun'] > 0:  # correct timing
                #thisTrial['autoRT'] *= self.runParams['autorun']
                thisTrial['RT'] *= self.runParams['autorun']

            if not noOutput:
                if write_head:
                    dataCSV = csv.writer(dataFile, lineterminator = '\n')
                    header = self.extraInfo.keys() + thisTrial.keys()
                    dataCSV.writerow(header)
                    write_head = False
                out = self.extraInfo.values() + thisTrial.values()
                dataCSV.writerow(out)

            trialNo += 1
        sys.stdout.write("\r")  # clean up outputs
        if not noOutput: dataFile.close()


    def autorun(self, trialList):
        """
        Automatically runs experiment by simulating key responses.

        This is just the absolute minimum for autorunning. Best practice would
        be extend this function to simulate responses according to your
        hypothesis.
        """
        def rt(mean):
            add = np.random.normal(mean,scale=.2)/self.runParams['autorun']
            return self.trial[0]['dur'] + add

        invValidResp = dict([[v,k] for k,v in self.comp.validResponses.items()])
        sortKeys = sorted(invValidResp.keys())
        invValidResp = OrderedDict([(k,invValidResp[k]) for k in sortKeys])
        # speed up the experiment
        for event in self.trial:
            event['dur'] /= self.runParams['autorun']
        self.trialDur /= self.runParams['autorun']

        for trial in trialList:
            # here you could do if/else to assign different values to
            # different conditions according to your hypothesis
            trial['autoResp'] = random.choice(invValidResp.values())
            trial['autoRT'] = rt(.5)
        return trialList

    def _astype(self,type='pandas'):
        """
        Mostly reused psychopy.data.TrialHandler.saveAsWideText
        """
        # collect parameter names related to the stimuli:
        header = self.trialList[0].keys()
        # and then add parameter names related to data (e.g. RT)
        header.extend(self.data.dataTypes)

        # loop through each trial, gathering the actual values:
        dataOut = []
        trialCount = 0
        # total number of trials = number of trialtypes * number of repetitions:
        repsPerType={}
        for rep in range(self.nReps):
            for trialN in range(len(self.trialList)):
                #find out what trial type was on this trial
                trialTypeIndex = self.sequenceIndices[trialN, rep]
                #determine which repeat it is for this trial
                if trialTypeIndex not in repsPerType.keys():
                    repsPerType[trialTypeIndex]=0
                else:
                    repsPerType[trialTypeIndex]+=1
                repThisType=repsPerType[trialTypeIndex]#what repeat are we on for this trial type?

                # create a dictionary representing each trial:
                # this is wide format, so we want fixed information (e.g. subject ID, date, etc) repeated every line if it exists:
                if (self.extraInfo != None):
                    nextEntry = self.extraInfo.copy()
                else:
                    nextEntry = {}

                # add a trial number so the original order of the data can always be recovered if sorted during analysis:
                trialCount += 1
                nextEntry["TrialNumber"] = trialCount

                # now collect the value from each trial of the variables named in the header:
                for parameterName in header:
                    # the header includes both trial and data variables, so need to check before accessing:
                    if self.trialList[trialTypeIndex].has_key(parameterName):
                        nextEntry[parameterName] = self.trialList[trialTypeIndex][parameterName]
                    elif self.data.has_key(parameterName):
                        nextEntry[parameterName] = self.data[parameterName][trialTypeIndex][repThisType]
                    else: # allow a null value if this parameter wasn't explicitly stored on this trial:
                        nextEntry[parameterName] = ''

                #store this trial's data
                dataOut.append(nextEntry)

        # get the extra 'wide' parameter names into the header line:
        header.insert(0,"TrialNumber")
        if (self.extraInfo != None):
            for key in self.extraInfo:
                header.insert(0, key)

        if type in [list, 'list']:
            import pdb; pdb.set_trace()
        elif type in [dict, 'dict']:
            import pdb; pdb.set_trace()
        elif type == 'pandas':
            df = pandas.DataFrame(dataOut, columns=header)

        return df

    def aspandas(self):
        """
        Convert trialList into a pandas DataFrame object
        """
        return self._astype(type='pandas')

    #def accuracy(self):
        #df = self._astype(list)
        #for line in df:
            #if line['accuracy']=='Correct':
                #accuracy += 1
        #acc = accuracy * 100 / len(df)
        #return acc

    def weighted_sample(self, probs):
        which = np.random.random()
        ind = 0
        while which>0:
            which -= probs[ind]
            ind +=1
        ind -= 1
        return ind

    def get_behav_df(self, pattern='%s'):
        """
        Extracts data from files for data analysis.
        """
        if type(self.extraInfo['subjID']) not in [list, tuple]:
            subjID_list = [self.extraInfo['subjID']]
        else:
            subjID_list = self.extraInfo['subjID']

        df_fnames = []
        for subjID in subjID_list:
            df_fnames += glob.glob(pattern % subjID)
        dfs = []
        for dtf in df_fnames:
            data = pandas.read_csv(dtf)
            if data is not None:
                dfs.append(data)
        if dfs == []:
            print df_fnames
            raise IOError('Behavioral data files not found')
        df = pandas.concat(dfs, ignore_index=True)

        return df


    def genPara(expPlan):
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

    def getParaNo(self,file_pattern,n = 6):
        """Looks up used para numbers and returns a new one for this run
        """
        allData = glob.glob(file_pattern)
        if allData == []: paraNo = random.choice(range(n))
        else:
            paraNos = []
            for thisData in allData:
                lines = csv.reader( open(thisData) )
                try:
                    header = lines.next()
                    ind = header.index('paraNo')
                    thisParaNo = lines.next()[ind]
                    paraNos.append(int(thisParaNo))
                except: pass

            if paraNos != []:
                countUsed = np.bincount(paraNos)
                countUsed = np.hstack((countUsed,np.zeros(n-len(countUsed))))
                possParaNos = np.arange(n)
                paraNo = random.choice(possParaNos[countUsed == np.min(countUsed)].tolist())
            else: paraNo = random.choice(range(n))

        return paraNo


class ThickShapeStim(visual.ShapeStim):
    """
    Draws thick shape stimuli as a collection of lines.

    PsychoPy has a bug in some configurations of not drawing lines thicker
    than 2px. This class fixes the issue. Note that it's really just a
    collection of rectanges so corners will not look nice.
    """
    def __init__(self,
                 win,
                 units  ='',
                 lineWidth=1.0,
                 lineColor=(1.0,1.0,1.0),
                 lineColorSpace='rgb',
                 fillColor=None,
                 fillColorSpace='rgb',
                 vertices=((-0.5,0),(0,+0.5),(+0.5,0)),
                 closeShape=True,
                 pos= (0,0),
                 size=1,
                 ori=0.0,
                 opacity=1.0,
                 depth  =0,
                 interpolate=True,
                 lineRGB=None,
                 fillRGB=None,
                 name='', autoLog=True):

        visual._BaseVisualStim.__init__(self, win, units=units, name=name, autoLog=autoLog)

        self.opacity = opacity
        self.pos = np.array(pos, float)
        self.closeShape=closeShape
        self.lineWidth=lineWidth
        self.interpolate=interpolate

        self._useShaders=False  #since we don't need to combine textures with colors
        self.lineColorSpace=lineColorSpace
        if lineRGB!=None:
            logging.warning("Use of rgb arguments to stimuli are deprecated. Please use color and colorSpace args instead")
            self.setLineColor(lineRGB, colorSpace='rgb')
        else:
            self.setLineColor(lineColor, colorSpace=lineColorSpace)

        self.fillColorSpace=fillColorSpace
        if fillRGB!=None:
            logging.warning("Use of rgb arguments to stimuli are deprecated. Please use color and colorSpace args instead")
            self.setFillColor(fillRGB, colorSpace='rgb')
        else:
            self.setFillColor(fillColor, colorSpace=fillColorSpace)

        self.depth=depth
        self.ori = np.array(ori,float)
        self.size = np.array([0.0,0.0])
        self.setSize(size)
        # self.setVertices(vertices)
        # self._calcVerticesRendered()
        self.setVertices(vertices)
        # if len(self.stimulus) == 1: self.stimulus = self.stimulus[0]

    def draw(self):
        for stim in self.stimulus:
            stim.draw()

    def setOri(self, newOri):
        # theta = (newOri - self.ori)/180.*np.pi
        # rot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
        # for stim in self.stimulus:
            # newVert = []
            # for vert in stim.vertices:
                # #import pdb; pdb.set_trace()
                # newVert.append(np.dot(rot,vert))
            # stim.setVertices(newVert)
        self.ori = newOri
        self.setVertices(self.vertices)

    def setPos(self, newPos):
        for stim in self.stimulus:
            stim.setPos(newPos)
        self.pos = newPos

    def setVertices(self, value = None):
        if isinstance(value[0][0], int) or isinstance(value[0][0], float):
            self.vertices = [value]
        else:
            self.vertices = value
        self.stimulus = []

        theta = self.ori/180.*np.pi #(newOri - self.ori)/180.*np.pi
        rot = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])

        for vertices in self.vertices:
            if self.closeShape: numPairs = len(vertices)
            else: numPairs = len(vertices)-1

            wh = self.lineWidth/2. - misc.pix2deg(1,self.win.monitor)
            for i in range(numPairs):
                thisPair = np.array([vertices[i],vertices[(i+1)%len(vertices)]])
                thisPair_rot = np.dot(thisPair, rot.T)
                edges = [
                    thisPair_rot[1][0]-thisPair_rot[0][0],
                    thisPair_rot[1][1]-thisPair_rot[0][1]
                    ]
                lh = np.sqrt(edges[0]**2 + edges[1]**2)/2.

                line = visual.ShapeStim(
                    self.win,
                    lineWidth   = 1,
                    lineColor   = self.lineColor,#None,
                    interpolate = True,
                    fillColor   = self.lineColor,
                    ori         = -np.arctan2(edges[1],edges[0])*180/np.pi,
                    pos         = np.mean(thisPair_rot,0) + self.pos,
                    # [(thisPair_rot[0][0]+thisPair_rot[1][0])/2. + self.pos[0],
                                   # (thisPair_rot[0][1]+thisPair_rot[1][1])/2. + self.pos[1]],
                    vertices    = [[-lh,-wh],[-lh,wh],
                                   [lh,wh],[lh,-wh]]
                )
                #line.setOri(self.ori-np.arctan2(edges[1],edges[0])*180/np.pi)
                self.stimulus.append(line)


from UserDict import DictMixin

class OrderedDict(dict, DictMixin):
    """
    OrderedDict code (because some are stuck with Python 2.5)
    Created by Raymond Hettinger on Wed, 18 Mar 2009, under the MIT License
    http://code.activestate.com/recipes/576693/
    """
    def __init__(self, *args, **kwds):
        if len(args) > 1:
            raise TypeError('expected at most 1 arguments, got %d' % len(args))
        try:
            self.__end
        except AttributeError:
            self.clear()
        self.update(*args, **kwds)

    def clear(self):
        self.__end = end = []
        end += [None, end, end]         # sentinel node for doubly linked list
        self.__map = {}                 # key --> [key, prev, next]
        dict.clear(self)

    def __setitem__(self, key, value):
        if key not in self:
            end = self.__end
            curr = end[1]
            curr[2] = end[1] = self.__map[key] = [key, curr, end]
        dict.__setitem__(self, key, value)

    def __delitem__(self, key):
        dict.__delitem__(self, key)
        key, prev, next = self.__map.pop(key)
        prev[2] = next
        next[1] = prev

    def __iter__(self):
        end = self.__end
        curr = end[2]
        while curr is not end:
            yield curr[0]
            curr = curr[2]

    def __reversed__(self):
        end = self.__end
        curr = end[1]
        while curr is not end:
            yield curr[0]
            curr = curr[1]

    def popitem(self, last=True):
        if not self:
            raise KeyError('dictionary is empty')
        if last:
            key = reversed(self).next()
        else:
            key = iter(self).next()
        value = self.pop(key)
        return key, value

    def __reduce__(self):
        items = [[k, self[k]] for k in self]
        tmp = self.__map, self.__end
        del self.__map, self.__end
        inst_dict = vars(self).copy()
        self.__map, self.__end = tmp
        if inst_dict:
            return (self.__class__, (items,), inst_dict)
        return self.__class__, (items,)

    def keys(self):
        return list(self)

    setdefault = DictMixin.setdefault
    update = DictMixin.update
    pop = DictMixin.pop
    values = DictMixin.values
    items = DictMixin.items
    iterkeys = DictMixin.iterkeys
    itervalues = DictMixin.itervalues
    iteritems = DictMixin.iteritems

    def __repr__(self):
        if not self:
            return '%s()' % (self.__class__.__name__,)
        return '%s(%r)' % (self.__class__.__name__, self.items())

    def copy(self):
        return self.__class__(self)

    @classmethod
    def fromkeys(cls, iterable, value=None):
        d = cls()
        for key in iterable:
            d[key] = value
        return d

    def __eq__(self, other):
        if isinstance(other, OrderedDict):
            return len(self)==len(other) and self.items() == other.items()
        return dict.__eq__(self, other)

    def __ne__(self, other):
        return not self == other


# From Python 2.7 docs under the Python Software Foundation License
# http://docs.python.org/library/itertools.html#itertools.combinations
def combinations(iterable, r):
    # combinations('ABCD', 2) --> AB AC AD BC BD CD
    # combinations(range(4), 3) --> 012 013 023 123
    # from Python 2.6
    pool = tuple(iterable)
    n = len(pool)
    if r > n:
        return
    indices = range(r)
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != i + n - r:
                break
        else:
            return
        indices[i] += 1
        for j in range(i+1, r):
            indices[j] = indices[j-1] + 1
        yield tuple(pool[i] for i in indices)


# From Python 2.7 docs under the Python Software Foundation License
# http://docs.python.org/library/itertools.html#itertools.combinations_with_replacement
def combinations_with_replacement(iterable, r):
    # combinations_with_replacement('ABC', 2) --> AA AB AC BB BC CC
    pool = tuple(iterable)
    n = len(pool)
    if not n and r:
        return
    indices = [0] * r
    yield tuple(pool[i] for i in indices)
    while True:
        for i in reversed(range(r)):
            if indices[i] != n - 1:
                break
        else:
            return
        indices[i:] = [indices[i] + 1] * (r - i)
        yield tuple(pool[i] for i in indices)
