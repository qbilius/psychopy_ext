################################################################################
#             The Configural Superiority Effect Experiment & Analysis          #
#                              Execution script                                #
################################################################################


################################################################################
#                              Main script flow                                #
################################################################################

# include ability to read user input easily
import scripts.tools.argparse as argparse
import sys, os
#import pdb; pdb.set_trace()
#print sys.modules
import scripts.core.init as cinit
import scripts.tools.qFunctions as q

def run():

    def runExp(rp):
        name = 'scripts.'+rp.runType
        #import pdb; pdb.set_trace()
        __import__(name)
        run = sys.modules[name].Test(rp)
        run.run()
#        sys.modules[name].run(rp)
        
        
    def runAnalysis(rp):
        name = 'scripts.'+rp.moduleName + '.' + rp.analysis[0]
        __import__(name)
        runWhat = sys.modules[name]
        
        if rp.analysis[0] == 'preproc':
            runWhat.run(rp)
        
        if rp.analysis[0] == 'svm_block':
            if rp.analysis[1] == 'main':
            # do the main analysis of parts and whole decoding
                runWhat.run(rp,
                    repeat = 10,
                    allConds = q.OrderedDict([
                        ('parts', [[1,2,3,4],[1,2,3,4]]),
                        ('whole', [[5,6,7,8],[5,6,7,8]])
                    ])
                )
            elif rp.analysis[1] == 'generalization':
                runWhat.run(rp,
                    repeat = 10,
                    allConds = q.OrderedDict([
                        ('occluded -> completed', [[1,7],range(1,13)])
                    ])
                )
            elif rp.analysis[1] == 'all':
                runWhat.run(rp,
                    repeat = 100,
                    allConds = q.OrderedDict([
                        ('parts', [[1,2,3,4],[1,2,3,4]]),
                        ('whole', [[5,6,7,8],[5,6,7,8]]),
                        ('parts -> whole', [[1,2,3,4],[5,6,7,8]]),
                        ('whole -> parts', [[5,6,7,8],[1,2,3,4]])
                     ])
                )
            elif rp.analysis[1] == 'mixed':
                runWhat.run(rp,
                    allConds = q.OrderedDict([
                        ('mixed', [range(1,9),range(1,9)])
                    ])
                )
            elif rp.analysis[1] == 'hemi':
                runWhat.run(rp,
                    repeat = 10,
                    allConds = q.OrderedDict([
                        ('parts (left)', [[1,3],[1,3]]),
                        ('whole (left)', [[5,7],[5,7]]),
                        ('parts (right)', [[2,4],[2,4]]),
                        ('whole (right)', [[6,8],[6,8]])
                    ])
                )
                
        elif rp.analysis[0] == 'behav':
            runWhat.run(rp)
            
        elif rp.analysis[0] == 'psc':
            if len(rp.analysis) == 2: plotTp = int(rp.analysis[1])
            else: plotTp = None
            runWhat.run(rp, plotTp = plotTp)
            
        elif rp.analysis[0] == 'corr':
            runWhat.run(rp, data1 = 'behav.csv', data2 = 'svm_all_all.csv')
            
        else:
            print 'ERROR: Analysis option not recognized'
        
    def runTools(rp):
        name = 'scripts.'+rp.moduleName + '.' + rp.tools[0]
        __import__(name)
        sys.modules[name].run(rp)

        
    def genMaterial(rp):
        rp.numVoxels = 'all'
        rp.group = False
        rp.textOnly = False
        rp.savePlot = ''
        
        if rp.material == 'f1':
            print 'This figure was created in MS Word.'
        elif rp.material == 'f2':
            print 'This figure was created in MS Word.'
        elif rp.material == 'f3':
            rp.moduleName = 'analysis'
            rp.analysis = ['behav', 'mean']
            runAnalysis(rp)
            rp.analysis = ['svm', 'main']
            rp.numVoxels = 'minRandom'
            runAnalysis(rp)
            print 'WARNING: Due to random time point sampling, ' +\
                'the generated figure does not match precisely the published one'
        elif rp.material == 'sf1':
            rp.moduleName = 'analysis'
            rp.analysis = ['behav', 'byRun']
            runAnalysis(rp)
        elif rp.material == 'sf2':
            rp.moduleName = 'analysis'
            rp.analysis = ['psc']
            runAnalysis(rp)    
        elif rp.material == 'sf3':            
            rp.moduleName = 'analysis'
            rp.analysis = ['svm', 'hemi']
            rp.rois = 'sel_lh'
            rp = cinit.genROIs(rp)
            runAnalysis(rp)
            rp.rois = 'sel_rh'
            rp = cinit.genROIs(rp)
            runAnalysis(rp)
        elif rp.material == 'st1':
            rp.moduleName = 'analysis'
            name = rp.moduleName + '.' + 'ROIparams'
            __import__(name)
            runWhat = sys.modules[name]
            runWhat.run(rp, subROIs = True, suppressText = False)
        
        else: print 'ERROR: Figure %s does not exist' %rp.material
        
            
            
    # Take the input
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--subjIDpre', default = '',
        help = "specify a subject ID prefix")
    parser.add_argument('--subjID', default = 'utl_TEST',
        help = "specify a subject ID")
    
    parser.add_argument('--runType', default = 'main',
        help = 'Specify the kind of run.')
    parser.add_argument('--runNo',
        type = int, default = None,
        help = "specify run number")
        
    parser.add_argument('--noOutput',
        action = "store_true", default = False,
        help = "choose this if you do not want to produce any output files; "
               "useful for showing how the program or analysis runs")
    parser.add_argument('-i',
        help = 'Specify an input file name')
    parser.add_argument('-o',
        help = 'Specify an output file name')
    parser.add_argument('--os',
        help = 'Specify a suffix to the default output file name')
    parser.add_argument("--overwrite",
        action = "store_true", default = False,
        help = "choose if you want to overwrite last session's output")
    parser.add_argument('--rois', default = 'sel',
        help = "specify ROIs for analysis; "
        "if no ROIs specified, all available ROIs will be used")
    parser.add_argument('--default', action = 'store_true', default = False,
        help = 'assigns commonly used arguments')
        
        
        
    subparsers = parser.add_subparsers(dest = 'moduleName')
    
    
    experiment = subparsers.add_parser('exp',
        help = 'Options for running the experiment',
        description = 'You can provide additional information for running the experiment')    
    experiment.add_argument("--debug", action = "store_true", default = False,
        help = "enter debug mode which is not full screen")
    experiment.add_argument('--block', default = 'all',
        help = "specify which blocks to run: pre, exposure, post, all")
    experiment.set_defaults(func = runExp)
    
                
    analysis = subparsers.add_parser('analysis',
        help = 'Analysis options',
        description = 'Specify the kind of analysis you want to perform')
    analysis.add_argument('analysis', nargs = '*',
        help = 'Specify the kind of analysis you want to perform')
    analysis.add_argument('--numVoxels',default = 'all',
        help = 'Wheather use all, min, or a specified number of voxels')
    analysis.add_argument('--group', action = 'store_true', default = False,
        help = 'Wheather to group V1-V2-V3 and LO-pFs in a plot')
    analysis.add_argument('--textOnly', action = 'store_true', default = False,
        help = 'Specify if a plot should be show in addition to the the command line output')
    analysis.add_argument('--savePlot', nargs = '?', default = '',
        help = 'Specify the file name of a resulting plot')   
    analysis.set_defaults(func = runAnalysis)
    
    tools = subparsers.add_parser('tools',
        help = 'Tools options',
        description = 'Specify the kind of analysis you want to perform')
    tools.add_argument('tools', nargs = '*',
        help = 'Specify the kind of analysis you want to perform')
    tools.set_defaults(func = runTools)
    
    material = subparsers.add_parser('material',
        help = "Specify material's number",
        description = 'Plot some material (a figure, a table) from the paper')
    material.add_argument('material',
        help = 'Specify material number from the paper')
    material.set_defaults(func = genMaterial)
    
    
    rp = parser.parse_args()
    
    rp.expName = sys.argv[0].split('.')[0]
    
    # define the path where the source is
    #thisPath = os.path.dirname(os.path.abspath(__file__))
    #root = thisPath.split("\\")
    #if len(root) == 1: # no \\, so must be Unix
    #    root = thisPath.split("/")
    rp.thisExpPath = 'data/'
        #'/'.join(root[0:-2]+ \
        #[os.path.basename(__file__).split('.')[0], 'data']) + '/'  

    if rp.moduleName == 'material': rp.default = True
    # run some predefined manipulations
    rp = cinit.run(rp)
    
    # choose what to do next
    #for subjID in rp.subjID_list:
    #    rp.subjID = subjID
    rp.func(rp)

        
if __name__ == "__main__":
    run()
