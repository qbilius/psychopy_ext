import scripts.tools.qFunctions as q
import os.path
import sys

# Set up paths
# Should typically be modified by the accessing module
paths = {
    'paraDirSup': 'scripts/%(runType)s/para/', # modify with the module path
    'paraDir': 'data/%s/para/', # will get modified by subjID
    'dataBehavDir': 'data/%s/data_behav/', # will get modified by subjID
    'dataFmriDir': 'data/%s/data_fmri/', # where fMRI data will be
    'recDir': 'data/%s/reconstruction/', # where ROIs will be
    #'simDir': '%(thisExpPath)s/hmax', # where simulation data will be stored
    'dataPrep': 'data/%s/data_prep_event/' # where preporcessed and masked data is stored
}

            
def getParaList(runType, pos = 5):
    """
    Returns the %pos% part of the paradigm file names in the %paraDir%.
    
    The filenames should be of the form para_%number%_runType.txt
    Note the separator '_'
    """
    global paths
    # Check first if the path exists yet
    if not os.path.isdir(paths['dataBehavDir']):
        return []
    else:   
        dataFiles = q.listDir(paths['dataBehavDir'], runType + '.csv')
        paraNames = []
        for dataFile in dataFiles:
            thisFile = open(dataFile)
            thisLine = thisFile.readline()
            thisLine = thisFile.readline().split(',')
            paraNames.append(thisLine[pos])
            thisFile.close()
    return paraNames
    

def getSubjID(
    subjID, # can be a list of int or str, or a string
    subjIDpre = '', # typically, a common subjID name part, like confSup_
    thisExpPath = '.'
    ):
    """
    Generates a list of subjIDs from given parameters.
    """
    
    try: # to find a list in the input
        subjID_list = q.listify(eval(subjID))
    except:
        subjID_list = q.listify(subjID)
    
    out = []
    if subjIDpre != '':
        if os.path.isdir(thisExpPath):
            for thisSubjID in subjID_list:
                if type(thisSubjID) != str: thisSubjID = '%02d' %thisSubjID
                out.extend(q.listDir(path = thisExpPath, \
                    pattern = subjIDpre + thisSubjID, fullPath = False))
    else: out = subjID_list           
    if out <> []: use = out
    else: sys.exit('ERROR: Specified subjects not found')
    
    if subjIDpre != '' or len(use) > 1:
        print '\nINFO: Subject IDs that will be used: ' + ', '.join(use)
    return use
    

def getRunNo(rp):
    global paths
       
    if not os.path.isdir(paths['dataBehavDir']):
        if rp.runNo == None:
            runNo = 1
            print 'Choosing runNo: %d' %runNo
        else: runNo = rp.runNo    
    else:
        dataFiles = q.listDir(paths['dataBehavDir'], '.csv', fullPath = False)        
        # Splits file names into ['data', %number%, 'runType.csv']
        allNums = [int(thisFile.split('_')[1]) for thisFile in dataFiles]
        
        if allNums == []: # no data files yet
            if rp.runNo == None:
                runNo = 1
                print 'Choosing runNo: %d' %runNo   
            else: runNo = rp.runNo
        elif rp.runNo in allNums:
            if not rp.overwrite:
                print 'ERROR: Run %d already exists.' %rp.runNo
                print 'Choose another runNo or type --overwrite'
                sys.exit()
            else: runNo = max(allNums)
        else:
            runNo = max(allNums) + 1
            print 'Guessing runNo: %d' %runNo
            
    return runNo


def makePaths(subjID):
    for (key,value) in paths.items():
        if '%s' in value: paths[key] = value %subjID

def genROIs(rp):
    # define ROIs explicitly
    if rp.rois == 'all':
        rp.rois = ['V1','V2',(['V3d','V3v'], 'V3'),'V3ab',('pFs1','V4+'),'IPS','LO',('pFs2','pFs')]
    elif rp.rois == 'all_lh':
        rp.rois = ['lh_V1','lh_V2',(['lh_V3d','lh_V3v'],'lh_V3'),'lh_V3ab',('lh_pFs1','lh_V4+'),'lh_IPS','lh_LO',('lh_pFs2','lh_pFs')]
    elif rp.rois == 'all_rh':
        rp.rois = ['rh_V1','rh_V2',(['rh_V3d','rh_V3v'],'rh_V3'),'rh_V3ab',('rh_pFs1','rh_V4+'),'rh_IPS','rh_LO',('rh_pFs2','rh_pFs')]
    elif rp.rois == 'sel':
        #rp.rois = ['V1']
        rp.rois = ['V1','V2',(['V3d','V3v'], 'V3'),'LO',('pFs','pFs')]
    elif rp.rois == 'sel_lh':
        rp.rois = ['lh_V1','lh_V2',(['lh_V3d','lh_V3v'],'lh_V3'),'lh_LO',('lh_pFs2','lh_pFs')]
    elif rp.rois == 'sel_rh':
        rp.rois = ['rh_V1','rh_V2',(['rh_V3d','rh_V3v'],'rh_V3'),'rh_LO',('rh_pFs2','rh_pFs')]    
    elif rp.rois == 'lo':
        rp.rois = ['LO',('pFs2','pFs')]
    else:
        try: rp.rois = eval(rp.rois)
        except: pass
        
    # generate a list of ROI file names and output names
    ROIs = []
    rp.rois = q.listify(rp.rois)
    def makePatt(ROI):
        ROI = q.listify(ROI)
        return ['.*' + thisROI + '.*' for thisROI in ROI]
    
    for ROI in rp.rois:        
        # renaming is provided
        if type(ROI) == tuple: ROIs.append(ROI + (makePatt(ROI[0]),))
        # a list of ROIs is provived
        elif type(ROI) == list: ROIs.append((ROI,'-'.join(ROI),makePatt(ROI)))
        # just a single ROI name provided
        else: ROIs.append((ROI,ROI,makePatt(ROI)))
        
    rp.rois = ROIs
    return rp
    
    
def run(rp):
    
    # use default parameters if the flag is given
    if rp.default:
        rp.subjIDpre = 'occlusion_'
        rp.subjID = range(1,2)
        rp.runType = 'main'
    
    
    if rp.moduleName == 'analysis':
        if not os.path.isdir(paths['dataFmriDir']) and \
        (rp.analysis == 'svm' or rp.analysis == 'corrROIs' or \
        rp.analysis == 'plotPSC'):
            print 'ERROR: These types of analysis require fMRI data but it is not ' \
            'present at the expected location ' + paths['dataFmriDir']
            sys.exit()
    
    
    paths['paraDirSup'] = paths['paraDirSup'] %{'runType': rp.runType}    
    
    # generate all subjIDs that will be used
    rp.subjID_list = getSubjID(rp.subjID, rp.subjIDpre,rp.thisExpPath)
    rp.subjID = rp.subjID_list[0]
    if rp.moduleName == 'exp':
        makePaths(rp.subjID)           
        rp.runNo = getRunNo(rp)
        rp.textOnly = True
        rp.savePlot = ''
    
    rp = genROIs(rp)
    
    
    # create relevant paths for output
    if rp.moduleName in ['analysis', 'simulation']:
        modName = rp.moduleName + '/'
        if not os.path.isdir(modName) and not rp.noOutput:
            os.makedirs(modName)
        paths[rp.moduleName] = modName
        
        # decide on input and output
        if rp.i != None: # simply plot from the existing analysis file
            rp.i = paths[rp.moduleName] + rp.i + '.hdf5'
        else:
            if not rp.noOutput:
                if rp.o == None: # form an output file name
                    thisModule = q.listify(eval('rp.'+ rp.moduleName))                    
                    outfName =  '_'.join(thisModule)
                    if rp.os != None:
                        outfName += '_' + rp.os
                    rp.o = paths[rp.moduleName] + outfName + '.hdf5'
                else: rp.o = paths[rp.moduleName] + rp.o + '.hdf5'
                
                if os.path.exists(rp.o) and not rp.overwrite:
                    oldFName = rp.o
                    # generate a new file name, which is +1 of the last one
                    existList = q.listDir(paths[rp.moduleName],
                        pattern = outfName + '_.*', cropExt = True, fullPath = False)
                    if len(existList) == 0: next = 2 # there is a single file               
                    else: # find all files that end with a number
                        splList = []
                        for el in existList:
                            spl = el.split('_')[-1]
                            try: splList.append(int(spl))
                            except: pass
                        # and choose the maximal number + 1 for the output file    
                        next = max(splList) + 1
                    
                    rp.o = rp.o.rsplit('.hdf5')[0] + '_%02d.hdf5' %next            
                    print 'WARNING: File %s exists' %oldFName
                    print '         Used %s instead' %rp.o
                
                else:
                    print 'INFO: Filename for saving is set to: ' + rp.o
    
    # make a file name for the plots
    if 'savePlot' in rp:
        if rp.savePlot != '':
            paths['plots'] = 'plots/'
            if not os.path.isdir(paths['plots']): os.makedirs(paths['plots'])
            
            if rp.savePlot == None:
                thisModule = eval('rp.'+ rp.moduleName)
                outfName =  '_'.join(thisModule)
                rp.savePlot = paths['plots'] + outfName + '.svg'
            elif len(rp.savePlot.split('.')) == 1: # no extension
                rp.savePlot = paths['plots'] + rp.savePlot + '.svg'
            else:
                rp.savePlot = paths['plots'] + rp.savePlot

            print 'INFO: Filename for saving the plot is set to: ' + rp.savePlot
        
    else: rp.savePlot = None    
    

    return rp


if __name__ == "__main__":
    run()
