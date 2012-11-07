import mvpa2.suite
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib import mpl
import scipy.stats
# import tabular as tb
import random
import csv
import os, sys, glob, shutil
import nibabel as nb
import itertools

import pandas

# import dataAnalysis as da
# import scripts.tools.qFunctions as q
# import ROIparams
# import behav
# import psc

# import svm_plot
import exp


# import Orange
import cPickle as pickle


class MVPA(object):

    def __init__(self,
                 paths,
                 tr,
                 filename = 'svm100.pkl',
                 visualize = False,
                 ):
        self.filename = filename
        self.visualize = visualize
        self.paths = paths
        self.tr = tr

    def run(self,
            subjID,
            rois,
            ds=None,
            runType='main',
            filename='svm.pkl',
            # allConds = exp.OrderedDict([
            #                     ('parts', [[1,2,3,4],[1,2,3,4]]),
            #                     ('whole', [[5,6,7,8],[5,6,7,8]])
            #                  ]),
            clf = 'LinearNuSVMC',
            repeat = 10,
            splitRule = (5,6),
            #filter = None,
            acc = None,
            plot = True,
            numVoxels = 'all',
            condsForRef = 'sel',
            noOutput = False,
        ):

        """Performs an SVM classification.

        **Parameters**

            rp: Namespace (required)
                Run parameters that are parsed from the command line
            allConds: OrderedDict
                An ordered dictionary containing the lists of conditions that will go for
                training and testing
            clf: 'SMLR', 'LinearCSVMC', 'LinearNuSVMC', 'RbfNuSVMC', or 'RbfCSVMC', or a list of them
                A name of the classifier to be used
            repeat: int
                how many times repeat SVM training+testing cycle (with a different selection of timepoints)
            splitRule: list or 'min'
                + a list of number of chunks (~runs) and size of each chunk
                + or 'min' - take the least number across all conditions, split by sqrt
            # filter: **None** or a rule
                # Specify which data should be taken.
                # + **None** means all data will be used.
                # + Rule: e.g., "thisData['accuracy'] == 'Correct'" (only correct responses taken)
            acc: **None**, 'max', or float
                + **None** means that all data will be used
                + 'max': take max accuracy, typically limited by the whole condition
                + float: e.g., .95 (take 95% of correct responses and 5% of incorrect)
            plot: **True** or **False**
                Whether to plot the output or not
            numVoxels: 'all', 'min' or int
                Specify
                + 'all': use all voxels available
                + 'min': use the minimal number of voxels across all ROIs for all ROIs
                + int: use that many voxels for all ROIs
            condsForRef: 'all', 'sel', or 'min'
                When getting accuracy or selecting voxels for analysis, specify whether
                these analyses should depend on all conditions ('all') or only on the
                ones specified at allConds ('sel')

        """
        
        clf = eval('mvpa2.suite.' + clf + '()')
        # tabData = behav.getBehavData()

        if numVoxels != None:
            try: numVoxels = int(numVoxels) #if specified as an int
            except: numVoxels = numVoxels            
        # params = ROIparams.run(rp, space = 'native')
        # params.sort(order=['subjID','numVoxels'])

        # have more iterations within an ROI if random voxels sampling is specified
        if numVoxels == 'minRandom':
            numRoiIter = int(np.sqrt(repeat))
            repeat /= numRoiIter
            print 'INFO: Repetions over ROIs changed to %d ' %numRoiIter +\
            'and repetions within ROIs changed to %d' %repeat
        else: numRoiIter = 1

        # read in scan data and perform SVMs for each subject each ROI group
        #dataOut = []
        # numT = len(np.unique(tabData['cond']))
        #results = exp.OrderedDict([(roi[1], np.zeros((len(rp.subjID_list),numT-1,numT-1))) for roi in rp.rois])
        # results = exp.OrderedDict([(roi[1], np.zeros((len(rp.subjID_list),numT-1,numT-1))) for roi in rois])
        # import pdb; pdb.set_trace()
        
        results = []
        for r, ROI_list in enumerate(rois):
            print ROI_list[1],
            ds = self.extract_samples(subjID, runType, ROI_list)
            dsmean = np.mean(ds.samples)
            mvpa2.suite.poly_detrend(ds, polyord=2, chunks_attr='chunks')
            ds.samples += dsmean # recover the detrended mean
            evds = self.ds2evds(ds)
            # self.plotChunks(ds,evds,chunks=[0])
            #self.plot_psc(evds)
            evds = evds[evds.sa.targets != 0]
            result = self.svm(evds)
            for line in result:
                line.extend([
                    ('subjID', subjID),
                    ('ROI', ROI_list[1]),
                    ])
                results.append(line)

        if not noOutput:
            # mvpa2.suite.h5save(rp.o, results)
            try: os.makedirs(self.paths['analysis'])
            except: pass
            pickle.dump(results, open(filename,'wb'))
        return results


    def extract_samples(self,
        subjID,
        # runNo,
        runType,
        ROIs,
        # tabData = None,
        # ROIs = None,
        # # filter = None,
        # shiftTp = 2,
        # rp = None,
        # roi_path = './',
        # rec_path = './',
        ):
        """
        Produces a detrended dataset with info for classifiers.

        **Parameters**
            subjID: str
            runType: str
            tabData: tabular.tabarray
                A tabarray with all information about conditions.
                Must have 'sessNo', 'cond', and 'accuracy' columns
            ROIs: list
                A pattern of ROI file patterns to be combined into one ROI
            # filter: **None** or str (a rule)
                # Filter out particular datapoints
            shiftTp: int
                Specifies how many TRs after the onset of a stimulus to extract the signal
            roi_path: str
                A path where preprocessed ROI data is stored. This is typically used after you've extracted data once.

        **Returns**
            nim: NiftiDataset

        """

        reuse = True
        roiname = self.paths['data_rois'] %subjID + ROIs[1] + '.gz.hdf5'
        # import pdb; pdb.set_trace()
        if reuse and os.path.isfile(roiname):
            ds = mvpa2.suite.h5load(roiname)

        else:
            # make a mask by combining all ROIs
            allROIs = []
            for ROI in ROIs[2]:
                theseROIs = glob.glob((self.paths['rec']+ROI+'.nii') %subjID)
                allROIs.extend(theseROIs)
            # import pdb; pdb.set_trace()
            thisROI = sum([np.squeeze(nb.load(roi).get_data()) for roi in allROIs])
            #thisROI = sum([mvpa2.suite.fmri_dataset(roi).samples for roi in allROIs])
            #thisMask = nb.load(thisROI)
            # change 4D mask to 3D by removing the first dim (time)
            #maskData = thisMask.get_data[:,:,:,::-1] # flip ROI because functional data is flipped
            #thisMask = thisMask.setDataArray(maskData.reshape(maskData.shape[1:]))
            #print nb.load("C:/Python27/Lib/site-packages/mvpa/data/bold.nii.gz").shape

            thisMask = np.squeeze(thisROI)
            # LEFT/RIGHT flip due to SPM's flipped output
            thisMask = thisMask[::-1]

            if self.visualize: self.plot_struct(self.paths['data_fmri']
                 %subjID + 'swmeanafunc_*.nii')                

            # find all functional runs of a given runType
            allImg = glob.glob((self.paths['data_fmri']+'swa*'+runType+'.nii')
                               %subjID)
            data_path = self.paths['data_behav']+'data_%02d_%s.csv'
            labels = self.extract_labels(allImg, data_path, subjID, runType)
            ds = self.fmri_dataset(allImg, labels, thisMask)

            if True:
                try: os.makedirs(self.paths['data_roi'] %subjID)
                except: pass
                mvpa2.suite.h5save(roiname, ds, compression=9)

        return ds



    def extract_labels(self, img_fnames, data_path, subjID, runType):
        """
        Extracts data labels (targets) from behavioral data files.
        """
        labels = []
        for img_fname in img_fnames:
            runNo = int(img_fname.split('_')[-2])

            behav_data = self.get_behav_data(data_path %(subjID, runNo, runType))
            # indicate which condition was present for each acquisition
            # !!!ASSUMES!!! that each block/condition is a multiple of TR
            run_labels = []
            for lineNo, line in behav_data.iterrows():
                # how many TRs per block or condition
                repeat = int(line['dur'] / self.tr) 
                run_labels.extend( [line['stim1.cond']] * repeat )
            labels.append(run_labels)

        return labels


    def fmri_dataset(self, samples, labels, thisMask = None):
        """
        Create a dataset from an fMRI timeseries image.

        Overrides mvpa2.datasets.mri.fmri_dataset which has a buggy multiple 
        images reading.
        """
        # Load in data for all runs and all ROIs
        chunkCount = 0
        first = True
        for thisImg, thisLabel in zip(samples,labels):            
            # load the appropriate func file with a mask
            if thisMask is None:
                # import pdb; pdb.set_trace()
                tempNim = mvpa2.suite.dataset_wizard(
                    samples=thisImg,
                    targets=thisLabel,
                    chunks = chunkCount
                    )

            else:
                tempNim = mvpa2.suite.fmri_dataset(
                        samples = thisImg,
                        targets = thisLabel,
                        chunks = chunkCount,
                        mask = thisMask
                        )
            # combine all functional runs into one massive NIfTI Dataset
            if first:
                ds = tempNim
                first = False
            else:
                ds = mvpa2.suite.vstack((ds,tempNim))
            chunkCount += 1

        return ds

        
    def ds2evds(self, ds):
        
        # if self.visualize: self.plotChunks(ds, chunks=[0], shiftTp=2)

        # convert to an event-related design
        events = mvpa2.suite.find_events(targets=ds.sa.targets, chunks=ds.sa.chunks)
        # import pdb; pdb.set_trace()
        # Remove the first and the last fixation period of each block        
        # We don't want any overlap between chunks
        events_temp = []
        for evNo, ev in enumerate(events):
            if evNo != 0 and evNo != len(events)-1:
                if ev['chunks'] == events[evNo-1]['chunks'] and \
                ev['chunks'] == events[evNo+1]['chunks']:
                    events_temp.append(ev)
        events = events_temp

        for ev in events:
            ev['onset'] += 2  # offset since the peak is at 6-8 sec
            ev['duration'] = 2  # use two time points as peaks since they are both high
        evds = mvpa2.suite.eventrelated_dataset(ds, events=events)
        if self.visualize: self.plotChunks(ds, evds, chunks=[0], shiftTp=0)

        return evds


    def plotChunks(self, ds, evds, chunks = None, shiftTp = 0):
        events = mvpa2.suite.find_events(targets=ds.sa.targets, chunks=ds.sa.chunks)
        # which chunks to display
        if chunks == None: chunks = ds.UC

        # get colors and assign them to targets
        numColors = len(ds.UT)
        cmap = mpl.cm.get_cmap('Paired')
        norm = mpl.colors.Normalize(0, 1)
        z = np.linspace(0, 1, numColors + 2)
        z = z[1:-1]
        colors_tmp = cmap(norm(z))
        colors = {}
        for target, color in zip(ds.UT,colors_tmp): colors[target] = color

        chunkLen = ds.shape[0] / len(ds.UC)
        #
        eventDur = evds.a.mapper[1].boxlength

        # evdsFlat = evds.a.mapper[2].reverse(evds)
        # ds = evds.a.mapper[1].reverse(evdsFlat)

        for chunkNo, chunk in enumerate(chunks):
            plt.subplot( len(chunks), 1, chunkNo+1 )
            plt.title('Runs with conditions shifted by %d' %shiftTp)
            sel = np.array([i==chunk for i in evds.sa.chunks])
            sel_ds = np.array([i==chunk for i in ds.sa.chunks])
            # import pdb; pdb.set_trace()
            meanPerChunk = np.mean(ds[sel_ds],1) # mean across voxels
            plt.plot(meanPerChunk.T, '.')
            # import pdb; pdb.set_trace()
            for onset, target in zip(evds[sel].sa.event_onsetidx,
                                     evds[sel].sa.targets):
                # import pdb;pdb.set_trace()
                plt.axvspan(
                    xmin = onset + shiftTp - .5,
                    xmax = onset + eventDur + shiftTp - .5,
                    facecolor = colors[target],
                    alpha=0.5)

            for ev in events:
                # import pdb; pdb.set_trace()
                if ev['chunks'] == chunk:
                    plt.axvline(x=ev['onset']%chunkLen + shiftTp)
                        # xmin = ev['onset']%chunkLen + shiftTp,
                        # xmax = ev['onset']%chunkLen + ev['duration'] + shiftTp,
                        # facecolor = colors[ev['targets']],
                        # alpha=0.5)

        plt.plot(meanPerChunk.T)
        plt.show()

    def plot_psc(self,evds):
        # plt.subplot(111)
        #vx_lty = ['-', '--']
        #t_col = ['b', 'r']
        
        #evds[]
        allConds = exp.OrderedDict([
            ('metric',[1,4,7,10]),
            ('non-accidental',[2,5,8,11]),
            ('other',[3,6,9,12]),
            ])
        allConds = exp.OrderedDict([(str(i+1),i+1) for i in range(12) ])
        baseline = evds[evds.sa.targets == 0].samples
        baseline = evds.a.mapper[-1].reverse(baseline)
        # average across all voxels and all blocks
        baseline = np.mean(np.mean(baseline,2),0)
        
        # now plot the mean timeseries and standard error
        for label, conds in allConds.items():
            evdsMean = evds[np.array([t in conds for t in evds.sa.targets])].samples
            evdsMean = evds.a.mapper[-1].reverse(evdsMean)
            # average across all voxels
            evdsMean = np.mean(evdsMean,2)
            # l = mvpa2.suite.plot_err_line((evdsMean-baseline)/baseline*100)
            legend = mvpa2.suite.plot_err_line(evdsMean,linestyle='-')
                              #fmt=t_col[j], linestyle='-')
            # label this plot for automatic legend generation
            legend[0][0].set_label(label)
            
        plt.ylabel('Percent signal change')
        plt.axhline(linestyle='--', color='0.6')
        plt.xlim(( 0,evdsMean.shape[1]+1 ))
        # plt.ylim((-.5,2.))
        plt.legend()


    def correlation(self,
                    evds,
                    nIter = 10  # how many iterations for
                    ):

        # calculate the mean per target per chunk (across trials)
        run_averager = mvpa2.suite.mean_group_sample(['targets','chunks'])
        evds_avg = evds.get_mapped(run_averager)
        numT = len(evds_avg.UT)

        # calculate mean across conditions per chunk per voxel
        target_averager = mvpa2.suite.mean_group_sample(['chunks'])
        mean = evds_avg.get_mapped(target_averager)
        # subtract the mean chunk-wise
        evds_avg.samples -= np.repeat(mean, numT, 0)

        results = np.zeros((nIter,numT,numT))
        runtype = [0,1] * (len(evds_avg.UC)/2) + \
                   [-1] * (len(evds_avg.UC)%2)
                   # for odd number of chunks (will get rid of one)

        for n in range(nIter):
            print n,
            np.random.shuffle(runtype)
            evds_avg.sa['runtype'] = np.repeat(runtype,numT)

            evds_split1 = evds_avg[np.array([i==0 for i in evds_avg.sa.runtype])]
            run_averager = mvpa2.suite.mean_group_sample(['targets'])
            evds_split1 = evds_split1.get_mapped(run_averager)

            evds_split2 = evds_avg[np.array([i==1 for i in evds_avg.sa.runtype])]
            run_averager = mvpa2.suite.mean_group_sample(['targets'])
            evds_split2 = evds_split2.get_mapped(run_averager)


            results[n] = mvpa2.clfs.distance.one_minus_correlation(evds_split1.samples, evds_split2.samples)/2

        print
        # meanPerIter = np.mean(np.mean(results, 2), 1)
        # cumMean = np.cumsum(meanPerIter)/range(1, len(meanPerIter)+1)
        # plt.plot(cumMean)
    #    plt.show()

        return np.mean(results, 0)


    def svm(self,
            evds,
            nIter = 100  # how many iterations for
            ):

        # calculate the mean per target per chunk (across trials)
        run_averager = mvpa2.suite.mean_group_sample(['targets','chunks'])
        evds_avg = evds.get_mapped(run_averager)
        numT = len(evds_avg.UT)

        # subtract the mean across voxels (per target per chunk)
    #    import pdb; pdb.set_trace()
        evds_avg.samples -= np.tile(np.mean(evds_avg, 1), (evds_avg.shape[1],1) ).T
        # and divide by standard deviation across voxels
        evds_avg.samples /= np.tile(np.std(evds_avg, axis=1, ddof=1), (evds_avg.shape[1],1) ).T

        # results = np.zeros((nIter,numT*(numT-1),numT*(numT-1)))
        if len(evds_avg.UC)%2:
            runtype = [0]*(len(evds_avg.UC)-9) + [1]*8 + [-1]
            # for odd number of chunks (will get rid of one)
        else:
            runtype = [0]*(len(evds_avg.UC)-8) + [1]*8

        targets = evds_avg.UT
        results = []

        for n in range(nIter):
            print n,
            np.random.shuffle(runtype)
            evds_avg.sa['runtype'] = np.repeat(runtype,numT)

            evds_train = evds_avg[np.array([i==0 for i in evds_avg.sa.runtype])]
            evds_test = evds_avg[np.array([i==1 for i in evds_avg.sa.runtype])]
            # boost results by averaging test patterns over chunks
            run_averager = mvpa2.suite.mean_group_sample(['targets'])
            evds_test = evds_test.get_mapped(run_averager)

            for i in range(0, numT):
                for j in range(0, numT):
                    targets = (evds_train.UT[i], evds_train.UT[j])
                    if i==j:
                        pred = None
                    else:
                        ind_train = np.array([k in targets for k in evds_train.sa.targets])
                        evds_train_ij = evds_train[ind_train]

                        ind_test = np.array([k in targets for k in evds_test.sa.targets])
                        evds_test_ij = evds_test[ind_test]

                        clf = mvpa2.suite.LinearNuSVMC()
                        clf.train(evds_train_ij) 
                        #fsel = mvpa2.suite.SensitivityBasedFeatureSelection(
                            #mvpa2.suite.OneWayAnova(),
                            #mvpa2.suite.FractionTailSelector(0.05, mode='select', tail='upper'))
                        #fclf = mvpa2.suite.FeatureSelectionClassifier(clf, fsel)                
                        #fclf.train(evds_train_ij)                

                        # sensana = clf.get_sensitivity_analyzer()
                        # sens = sensana(evds_train_ij)
                        # inds = np.argsort(np.abs(sens.samples))
                        # inds = np.squeeze(inds)
                        # evds_train_ij.samples = evds_train_ij.samples[:,inds>=len(inds)-100]
                        # #import pdb; pdb.set_trace()
                        # clf.train(evds_train_ij)
                        
                        # test_samp = evds_test_ij.samples[:,inds>=len(inds)-100]
                        # predictions = clf.predict(test_samp)
                        predictions = clf.predict(evds_test_ij.samples)
                        pred = np.mean(predictions == evds_test_ij.sa.targets)
                    results.append([
                        ('iter', n),
                        ('stim1.cond', targets[0]),
                        ('stim2.cond', targets[1]),
                        ('subjResp', pred)
                    ])
                    
                    

        print
    #    meanPerIter = np.mean(np.mean(results, 2), 1)
    #    cumMean = np.cumsum(meanPerIter)/range(1, len(meanPerIter)+1)
    #    plt.plot(cumMean)
    #    plt.show()

        return results



    def dissimilarity(self,
                   evds,
                   method = 'svm',
                   nIter = 10,  # how many iterations for  # have more for SVM
                   meanFunc = 'across voxels',
                   ):        
        """
        DEPRECATED.
        Computes a dissimilarity (0 - very similar, 1 - very dissimilar) between 
        two splits of data over multiple iterations. If method is correlation, 
        dataset is split in half. If svm, leave-one-chunk.
        """

        numT = len(evds.UT)
        results = np.zeros((nIter,numT,numT))

        # prepare split of data
        # runtype is either 0 (train data or split1) or 1 (test data or split2)
        if method=='corr':  # splitHalf
            # if odd, then one of the runs will be spared (value -1)
            runtype = [0,1] * (len(evds.UC)/2) + [-1] * (len(evds.UC)%2)
        elif method=='svm':  # nFold split
            if len(evds.UC)%2:
                runtype = [0]*(len(evds.UC)-3) + [1,1] + [-1] # for odd
            else:
                runtype = [0]*(len(evds.UC)-2) + [1,1]

        # if corr: cvtype = len(evds.UC)/2
        # else: cvtype = 1
        # nfolds = mvpa2.suite.NFoldPartitioner(cvtype=cvtype,count=10,selection_strategy='equidistant')
        # import pdb; pdb.set_trace()

        for n in range(nIter):
            print n,
            # we want each iteration to have a different (random) split
            np.random.shuffle(runtype)
            # for each datapoint within a chunk, assign the same runtype
            evds.sa['runtype'] = np.repeat(runtype,len(evds.sa.chunks)/len(evds.UC))
            # create an average per target per chunk (per voxel)
            run_averager = mvpa2.suite.mean_group_sample(['targets','chunks'])
            evds_avg = evds.get_mapped(run_averager)

            # calculate mean and standard deviation across conditions per voxel
            ds_split_train = evds_avg[np.array([i==0 for i in evds_avg.sa.runtype])]
            mean_train = np.mean(ds_split_train,0)  # mean per voxel
            sd_train = np.std(ds_split_train, axis=0, ddof=1)

            ds_split_test = evds_avg[np.array([i==1 for i in evds_avg.sa.runtype])]
            mean_test = np.mean(ds_split_test,0)
            sd_test = np.std(ds_split_test, axis=0, ddof=1)

            targets = ds_split_train.UT
            if np.sum(targets != ds_split_test.UT)>0:
                sys.exit("Targets on the two splits don't match. Unbalanced design?")

            # filling in the results matrix
            for index,value in np.ndenumerate(results[n]):
                # target pair for that particular matrix cell
                indexT = (targets[index[0]],  targets[index[1]])  

                ind_train = np.array([i in indexT  for i in ds_split_train.sa.targets])
                ds_train = ds_split_train[ind_train]
                ds_train.samples -= mean_train
                ds_train.samples /= sd_train

                ind_test = np.array([i in indexT  for i in ds_split_test.sa.targets])
                ds_test = ds_split_test[ind_test]
                ds_test.samples -= mean_test
                ds_test.samples /= sd_test

    #            if index[0] == index[1]:
    #                # import pdb; pdb.set_trace()
    #                halfT1 = len(ds_train.sa.targets)/2
    #                ds_train.sa.targets = np.array([1,2]*halfT1)
    #                halfT2 = len(ds_test.sa.targets)/2
    #                ds_test.sa.targets = np.array([1,2]*halfT2)

                if method=='corr':
                    cr = mvpa2.clfs.distance.one_minus_correlation(ds_train.samples,ds_test.samples)
                    # if one target then there's one correlation only
                    if index[0] == index[1]: acc = cr  
                    else: acc = np.mean([ cr[0,1], cr[1,0] ])
                    results[n,index[0],index[1]] = acc

                elif method=='svm':                    
                    if index[0] == index[1]:  # can't do svm, so assume
                        results[n,index[0],index[1]] = 1
                    else:
                        clf = mvpa2.suite.LinearNuSVMC()
                        clf.train(ds_train)
                        predictions = clf.predict(ds_test.samples)
                        results[n,index[0],index[1]] = np.mean(predictions == ds_test.sa.targets)


                    # nfold = mvpa2.suite.NFoldPartitioner(cvtype=5,count=10,selection_strategy='equidistant')
                    # cvte = mvpa2.suite.CrossValidation(clf, HalfPartitioner(attr='runtype'),
                        # errorfx = lambda p, t: np.mean(p == t),  # this makes it report accuracy, not error
                        # enable_ca=['stats'])
                    # cvte(ds_pair)


        print
        if self.visualize:
            meanPerIter = np.mean(np.mean(results, 2), 1)
            cumMean = np.cumsum(meanPerIter)/range(1, len(meanPerIter)+1)
            plt.plot(cumMean)
            plt.show()

        return np.mean(results,0) # mean across folds


    def searchlight(self, ds):
        run_averager = mvpa2.suite.mean_group_sample(['targets', 'chunks'])
        ds = ds.get_mapped(run_averager)
        clf = mvpa2.suite.LinearNuSVMC()
        cvte = mvpa2.suite.CrossValidation(clf, mvpa2.suite.NFoldPartitioner(),
                            errorfx = lambda p, t: np.mean(p == t),enable_ca=['stats'])
        sl = mvpa2.suite.sphere_searchlight(cvte, radius=3, postproc=mvpa2.suite.mean_sample())

        pairs = [
            [(1,2),(1,3),(2,3)],
            [(4,5),(4,6),(5,6)],
            [(7,8),(7,9),(8,9)],
            [(10,11),(10,12),(11,12)]
            ]
        chance_level = .5
        for pair in pairs:
            thisDS = ds[np.array([i in pair for i in ds.sa.targets])]
            res = sl(ds)
            resOrig = res.a.mapper.reverse(res.samples)
            print res_orig.shape
            fig = plt.figure()
            fig.subplot(221)
            plt.imshow(np.mean(resOrig.samples,0), interpolation='nearest')
            fig.subplot(222)
            plt.imshow(np.mean(resOrig.samples,1), interpolation='nearest')
            fig.subplot(223)
            plt.imshow(np.mean(resOrig.samples,2), interpolation='nearest')
            plt.show()
            sphere_errors = res.samples[0]
            res_mean = np.mean(res)
            res_std = np.std(res)
            import pdb; pdb.set_trace()

            sphere_errors < chance_level - 2 * res_std
        mri_args = {
            'background' : os.path.join(datapath, 'anat.nii.gz'),
            #'background_mask' : os.path.join(datapath, 'mask_brain.nii.gz'),
            #'overlay_mask' : os.path.join(datapath, 'mask_gray.nii.gz'),
            'cmap_bg' : 'gray',
            'cmap_overlay' : 'autumn', # YlOrRd_r # pl.cm.autumn
            'interactive' : cfg.getboolean('examples', 'interactive', True),
            }
        fig = plot_lightbox(overlay=map2nifti(dataset, sens),
                  vlim=(0, None), slices=18, **mri_args)


    def plot_struct(self, nim, coords=None, roi=None):
        """
        Plots structural scan from the three sides.
        """
        for i in range(3):
            plt.subplot(2,3,i+1)
            plt.imshow( np.mean(thisMask,i).T, cmap=mpl.cm.gray,
                origin='lower', interpolation='nearest' )

        meanFuncName = glob.glob()
        meanFunc = nb.load(meanFuncName[0]).get_data()
        if coords is None:
            coords = [m/2 for m in meanFunc.shape]  # middle

        plt.subplot(234)
        plt.imshow(meanFunc[coords[0]].T, cmap=mpl.cm.gray,
            origin='lower', interpolation='nearest')
        plt.subplot(235)
        plt.imshow(meanFunc[:,coords[1]].T, cmap=mpl.cm.gray,
            origin='lower', interpolation='nearest')
        plt.subplot(236)
        plt.imshow(meanFunc[:,:,coords[2]].T, cmap=mpl.cm.gray,
            origin='lower', interpolation='nearest')
        plt.show()
    

    


    def genFakeData(self, nChunks = 4):

        def fake(nConds = 12,nVoxels = 100):
            # each voxel response per condition
            fakeCond1 = np.array([0.5,1.]*(nVoxels/2))
            # fakeCond1 = np.random.normal( loc=1,scale=1,size=(nVoxels,) )
            # ROI's response to each condition
            fakeCond1 = np.tile( fakeCond1, (nConds/2,1) )
            # add noise
            fakeDS1 = fakeCond1 + np.random.random((nConds/2,nVoxels))/10.

            fakeCond2 = np.array([1.,.5,1.,5]*(nVoxels/4))
    #        fakeCond2 = np.random.normal(loc=3,scale=1,size= (nVoxels,) )
            fakeCond2 = np.tile( fakeCond2, ( nConds/2,1 ) )
            fakeDS2 = fakeCond2 + np.random.random((nConds/2,nVoxels))/10.

            fakeChunk = np.vstack((fakeDS1,fakeDS2,fakeDS2[:,::-1],fakeDS1[:,::-1]))
            targets = range(1,nConds+1)+range(nConds,0,-1)
            fakeChunk = mvpa2.suite.dataset_wizard(samples=fakeChunk, targets=targets)
            return fakeChunk

        fakeDS = mvpa2.suite.multiple_chunks(fake,nChunks)
        return fakeDS


    def get_behav_data(self, path):
        df_fnames = glob.glob(path)
        dfs = []
        for dtf in df_fnames:
            dfs.append( pandas.io.parsers.read_csv(dtf) )
        return pandas.concat(dfs, ignore_index=True)



    def roi_params(self,
        rp,
        subROIs = False,
        suppressText = True,
        space = 'talairach',
        spm = False
        ):
        
        """
        Calculates mean coordinates and the number of voxels of each given ROI.
        
        **Parameters**
            rp: Namespace (required)
                Run parameters that are parsed from the command line
            subROIs: True or False
                If True, then subROIs are not combined together into an ROI
            suppressText: True or False
                If True, then nothing will be printed out
            space: talairach or native
                Choose the output to be either in native voxel space or in Talairach coordinates
            spm: True or False
                If True, then the coordinates in the voxel space are provided with
                indices +1 to match MatLab's convention of starting arrays from 1.
        
        """
        
        if subROIs: names = ['subjID','ROI','subROI','x','y','z','numVoxels']
        else: names = ['subjID','ROI','x','y','z','numVoxels']
        recs = []
        
        # allCoords = np.zeros((1,4))
        for subjIDno, subjID in enumerate(rp.subjID_list):
        
            for ROI_list in rp.rois:
                
                allROIs = []
                for thisROI in ROI_list[2]:
                    allROIs.extend(q.listDir(scripts.core.init.paths['recDir'] %subjID,
                        pattern = thisROI + '\.nii', fullPath = True))
                #import pdb; pdb.set_trace()
                if allROIs != []:
                    SForm = nb.load(allROIs[0]).get_header().get_sform()
                    
                    # check for overlap
                    # if subjID == 'twolines_06': import pdb; pdb.set_trace()
                    print [os.path.basename(subROI) for subROI in allROIs]
                    #
                    mask = sum([np.squeeze(nb.load(subROI).get_data()) for subROI in allROIs])            
                    if not suppressText:
                        overlap = mask > 2
                        if np.sum(overlap) > 0:
                            print 'WARNING: Overlap in %(subjID)s %(ROI)s detected.'\
                            %{'subjID': subjID, 'ROI': ROI_list[1]}
                        
                    
                    if not subROIs: allROIs = [mask]            
                    for subROI in allROIs:
                        
                        if subROIs: subROIname = os.path.basename(os.path.abspath(subROI)).split('.')[0]
                        else: subROIname = ROI_list[1]
                        #import pdb; pdb.set_trace()
                        if subROIs: thisROI = nb.load(subROI).get_data()
                        else: thisROI = subROI
                        transROI = np.transpose(thisROI.nonzero())
                        
                        meanROI = np.mean(transROI,0)[1:]
                        meanROI = meanROI[::-1] # reverse the order per convention
                    
                        # convert to the Talairach coordinates 
                        if space == 'talairach':
                            meanROI = np.dot(SForm, np.concatenate((meanROI,[1]))) # convert
                            meanROI = meanROI[:-1] # remove the last coordinate (slice number)
                        else:
                            meanROI = [m+spm for m in meanROI] # +1 to correct for SPM coords
                        
                        if subROIs:
                            recs.append((subjID,ROI_list[1],subROIname)+tuple(meanROI)+(transROI.shape[0],))
                        else:
                            recs.append((subjID,subROIname)+tuple(meanROI)+(transROI.shape[0],))
        
        
            
        ROIparams = tb.tabarray(records = recs, names = names)
        
        if not suppressText:
            if subROIs: on = ['ROI','subROI']
            else: on = ['ROI']
            ROImean = ROIparams.aggregate(On = on, AggFunc = np.mean,
                AggFuncDict = {'subjID': lambda x: None})
            
            xyz = ROIparams[['x','y','z']].extract().reshape((len(rp.subjID_list),-1,3))
            xyzErr = np.std(xyz, axis = 0, ddof = 1)
            
            # sort ROImean
            numPerSubj = xyz.shape[1]
            order = ROIparams[:numPerSubj][on]
            order = order.addcols(range(len(order)), names=['order'])
            order.sort(order=on)
            ROImean.sort(order=on)
            ROImean = ROImean.addcols(order[['order']].extract(), names = 'order')
            ROImean.sort(order = 'order')
            
            lenROI = min([len(ROI) for ROI in ROImean['ROI']])
            if subROIs: lenSubROI = min([len(ROI) for ROI in ROImean['subROI']])
            print
            print ROIparams.dtype.names[1:]
            for i, line in enumerate(ROImean):
                print line['ROI'].ljust(lenROI+2),            
                if subROIs: print line['subROI'].ljust(lenSubROI+2),            
                print '%3d' %np.round(line['x']),
                print u'\xb1 %d  ' %np.round(xyzErr[i,0]),
                print '%3d' %np.round(line['y']),
                print u'\xb1 %d  ' %np.round(xyzErr[i,1]),
                print '%3d' %np.round(line['z']),
                print u'\xb1 %d  ' %np.round(xyzErr[i,2]),
                print '%4d' %np.round(line['numVoxels'])    
        
        return ROIparams




class Preproc(object):
    def __init__(self, paths):
        self.paths = paths

    def split_reg(self, subjID):
        """
        Splits the file that has realignment information by run.
        This is used for stats as each run with its covariates has to be entered separately.
        """        
        funcImg = glob.glob(self.paths['data_fmri']%subjID + 'func_*_*.nii')
        regFiles = glob.glob(self.paths['data_fmri']%subjID + 'rp_afunc_*.txt')
        
        reg = []
        for regFile in regFiles:
            with open(regFile) as f: reg.extend( f.readlines() )
            shutil.move(regFile,
                        (self.paths['fmri_root'] %subjID)+os.path.basename(regFile))
        
        last = 0
        for func in funcImg:
            runNo = func.split('.')[0].split('_')[-2]
            
            nim = nb.load(func)
            dynScans = nim.get_shape()[3] # get number of acquisitions
            
            runType = func.split('.')[0].split('_')[-1]
            outName = self.paths['data_fmri']%subjID + 'rp_%s_%s.txt' %(runNo,runType)

            with open(outName, 'w') as f:
                f.writelines(reg[last:last+dynScans])
                last += dynScans

    
    def gen_stats_batch(self, subjID, runType=None, condcol='cond'):
        if runType is None:
            runType = ['main','loc','mer']
        elif type(runType) not in [tuple, list]:
            runType = [runType]

        self.split_reg(subjID)            
        f = open(os.path.join(self.paths['fmri_root'] %subjID,'jobs','stats.m'),'w')
        f.write("spm('defaults','fmri');\nspm_jobman('initcfg');\nclear matlabbatch\n\n")
        
        for rtNo, runType in enumerate(runType):
            analysisDir = os.path.normpath(os.path.join(os.path.abspath(self.paths['fmri_root']%subjID),'analysis',runType))
            try: os.makedirs(analysisDir)
            except:
                # sys.exit('Analysis folder already exists')
                pass # the directory already exists
            dataFiles = glob.glob(self.paths['data_behav']%subjID + 'data_*_%s.csv' %runType)
            # import pdb; pdb.set_trace()
            regressorFiles = glob.glob(self.paths['data_fmri']%subjID + 'rp_*_%s.txt' %runType)        
            f.write("matlabbatch{%d}.spm.stats.fmri_spec.dir = {'%s'};\n" %(3*rtNo+1,analysisDir))
            f.write("matlabbatch{%d}.spm.stats.fmri_spec.timing.units = 'secs';\n" %(3*rtNo+1))
            f.write("matlabbatch{%d}.spm.stats.fmri_spec.timing.RT = 2;\n" %(3*rtNo+1))
            
            for rnNo, dataFile in enumerate(dataFiles):
                runNo = int(os.path.basename(dataFile).split('_')[1])
                
                data = np.recfromcsv(dataFile, case_sensitive = True)
                f.write("matlabbatch{%d}.spm.stats.fmri_spec.sess(%d).scans = " %(3*rtNo+1,rnNo+1) +
                    "cellstr(spm_select('ExtFPList','%s','^swafunc_%02d_%s\.nii$',1:168));\n" %(os.path.abspath(self.paths['data_fmri']%subjID),runNo,runType))
                
                conds = np.unique(data[condcol])
                if runType == 'mer':
                    conds = conds[conds!=0]
    #                import pdb; pdb.set_trace()

                for cNo, cond in enumerate(conds):
                    agg = data[data[condcol] == cond]
                    f.write("matlabbatch{%d}.spm.stats.fmri_spec.sess(%d).cond(%d).name = '%s';\n" %(3*rtNo+1,rnNo+1,cNo+1,agg['name'][0]))
                    # import pdb; pdb.set_trace()
                    # onsets = ' '.join(map(str,agg['onset']))
                    if 'blockNo' in agg.dtype.names:
                        onsets = []
                        durs = []
                        for block in np.unique(agg['blockNo']): 
                            onsets.append( agg[agg['blockNo']==block]['onset'][0] )
                            durs.append( np.around(sum( agg[agg['blockNo']==block]['dur'] ), decimals = 1) )
                    else:
                        onsets = agg['onset']
                        durs = agg['dur']
                    f.write("matlabbatch{%d}.spm.stats.fmri_spec.sess(%d).cond(%d).onset = %s;\n" %(3*rtNo+1,rnNo+1,cNo+1,onsets))
                    f.write("matlabbatch{%d}.spm.stats.fmri_spec.sess(%d).cond(%d).duration = %s;\n" %(3*rtNo+1,rnNo+1,cNo+1,durs))
                        
                f.write("matlabbatch{%d}.spm.stats.fmri_spec.sess(%d).multi_reg = {'%s'};\n\n" %(3*rtNo+1,rnNo+1,os.path.abspath(regressorFiles[rnNo])))

            f.write("matlabbatch{%d}.spm.stats.fmri_est.spmmat = cellstr(fullfile('%s','SPM.mat'));\n" %(3*rtNo+2,analysisDir))
            f.write("matlabbatch{%d}.spm.stats.con.spmmat = cellstr(fullfile('%s','SPM.mat'));\n" %(3*rtNo+3,analysisDir))
            
            if runType == 'loc':
                f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.name = 'all > fix';\n" %(3*rtNo+3))
                f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.convec = [-2 1 1];\n" %(3*rtNo+3))
                f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.sessrep = 'repl';\n" %(3*rtNo+3))
                f.write("matlabbatch{%d}.spm.stats.con.consess{2}.tcon.name = 'objects > scrambled';\n" %(3*rtNo+3))
                f.write("matlabbatch{%d}.spm.stats.con.consess{2}.tcon.convec = [0 1 -1];\n" %(3*rtNo+3))
                f.write("matlabbatch{%d}.spm.stats.con.consess{2}.tcon.sessrep = 'repl';\n\n\n" %(3*rtNo+3))
            elif runType == 'mer':
                f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.name = 'hor > ver';\n" %(3*rtNo+3))
                f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.convec = [1 -1];\n" %(3*rtNo+3))
                f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.sessrep = 'repl';\n\n\n" %(3*rtNo+3))
            else:
                f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.name = 'all > fix';\n" %(3*rtNo+3))
                
                conds = np.unique(data[condcol])
                descrs = []
                for cond in conds[1:]: # skip fixation condition as it's our baseline
                    descrs.append(data[data[condcol]==cond]['descr'][0])
                # descrs = np.unique(data['descr'])
                # descrs = descrs[descrs != 'fixation']
                thisCond = ' '.join(['-1']+['1']*len(descrs))
                f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.convec = [%s];\n" %(3*rtNo+3,thisCond))
                f.write("matlabbatch{%d}.spm.stats.con.consess{1}.tcon.sessrep = 'repl';\n" %(3*rtNo+3))
                for dNo, descr in enumerate(descrs):
                    f.write("matlabbatch{%d}.spm.stats.con.consess{%d}.tcon.name = '%s';\n" %(3*rtNo+3,dNo+1,descr))
                    thisCond = [-1] + [0]*dNo + [1] + [0]*(len(descrs)-dNo-1)
                    f.write("matlabbatch{%d}.spm.stats.con.consess{%d}.tcon.convec = %s;\n" %(3*rtNo+3,dNo+1,thisCond) )
                    f.write("matlabbatch{%d}.spm.stats.con.consess{%d}.tcon.sessrep = 'both';\n" %(3*rtNo+3,dNo+1))
                f.write('\n\n')

        f.write("save('stats.mat','matlabbatch');\n")
        f.write("%%spm_jobman('interactive',matlabbatch);\n")
        f.write("spm_jobman('run',matlabbatch);")
        f.close()


class Plot(object):
    def make_symmetric(self, similarity):
        res = np.zeros(similarity.shape)
        for i in range(similarity.shape[0]):
            for j in range(similarity.shape[1]):
                res[i,j] = (similarity[i,j]+similarity[j,i])/2
        return res

    def make_full(self, distance):
        res = np.nan*np.ones(distance.shape)
        iu = np.triu_indices(len(distance),k=1)  # upper triangle less diagonal
        il = np.tril_indices(len(distance),k=-1)  # lower triangle less diagonal
        res[iu] = distance[iu]
        res = res.T
        res[iu] = distance[iu]
        return res

    def plot_similarity(self, similarity, names=None, percent=False):
        similarity = make_symmetric(similarity)
        trace = similarity.trace()/len(similarity)
        offdiag = (np.sum(similarity) - similarity.trace()) / len(similarity) / (len(similarity)-1)
        print '%.2f' %trace,
        print '%.2f' %offdiag,
        iu = np.triu_indices(len(similarity),k=1)  # upper triangle less diagonal
        rel = np.corrcoef(similarity[iu],similarity.T[iu])[0,1]
        print '%.2f' %rel
    #    import pdb; pdb.set_trace()
        if percent: plot_data = similarity*100
        else: plot_data = similarity
        im = plt.imshow(plot_data,interpolation='none',vmin=.45,vmax=.86)
        plt.colorbar(im, use_gridspec=True)
        # plt.tight_layout()
        
        if not names is None:
            names = [n[1] for n in names]
            locs, labels = plt.xticks(range(plot_data.shape[1]), names)
            plt.setp(labels, 'rotation', 'vertical')
            locs, labels = plt.yticks(range(plot_data.shape[0]), names)
        for index,value in np.ndenumerate(plot_data):
            if np.isnan(value): h = ''
            else:
                if percent: h = '%d' %(value*100)
                else: h = '.%d' %(value*100)
            plt.text(index[1]-.5,index[0]+.5,h)
        return im
        
    def plot_hcluster(self, similarity, names):
        import hcluster
        similarity = make_symmetric(similarity)
        sim2 = similarity - .5
        sim2[sim2<0] = 0
        # distance = Orange.core.SymMatrix(1-similarity)
        # root = Orange.clustering.hierarchical.HierarchicalClustering(distance)
        tree = hcluster.hcluster(sim2)
        imlist = [ str(i[0]) + '-' + i[1] for i in names]
        dendogram = hcluster.drawdendrogram(tree,imlist,jpeg='sunset.jpg')
        plt.imshow(dendogram, cmap=plt.cm.gray)

    def plot_mds(self, similarity, names):
        # 
        similarity = make_symmetric(similarity)
        sim2 = similarity - .5
        sim2[sim2<0] = 0
        #import pdb; pdb.set_trace()
        distance = Orange.core.SymMatrix(sim2)
        #import pdb; pdb.set_trace()
        mds = Orange.projection.mds.MDS(distance)
        mds.run(100)
        for (x, y), name in zip(mds.points,names):
            plt.plot((x,),(y,),'ro')
            plt.text(x,y,name[1])
            
    def mean_diag_off(self, matrix):
        trace = matrix.trace()/len(matrix)
        offdiag = (np.sum(matrix) - matrix.trace()) / len(matrix) / (len(matrix)-1)
        return [trace,offdiag]

    def avg_blocks(self, matrix, coding):
        coding = np.array(coding)
        coding_int = coding[np.not_equal(coding, None)] # remove nones
    #    try:
    #        np.bincount(coding_int)>1
    #    except:
    #        import pdb; pdb.set_trace()
        coding_int = coding_int.astype(np.int)
        if not np.all(np.bincount(coding_int)>1):
            print np.bincount(coding_int)
            sys.exit('You have a single occurence of some entry')
        else:
            uniquec = np.unique(coding_int)
            avg = np.zeros( (len(uniquec),len(uniquec)) )
            for i,ui in enumerate(uniquec):
                indi = coding == ui
                for j,uj in enumerate(uniquec):
                    indj = coding == uj
                    ind = np.outer(indi,indj)
                    np.fill_diagonal(ind,False)
                    avg[i,j] = np.mean(matrix[ind])
        return avg