#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import itertools

import numpy as np
import matplotlib.pyplot as plt
import scipy.misc
import scipy.ndimage


class Model(object):

    def get_testim(self, size=(256, 256)):
        """
        Opens Lena image and resizes it to the specified size ((256, 256) by 
        default)
        """
        lena = scipy.misc.lena()
        im = scipy.misc.imresize(lena, (256, 256))
        im = im.astype(float)

        return im

    def train(self, im):
        """
        A placeholder for a function for training a model.
        If the model is not trainable, then it will default to this function 
        here that does nothing.
        """
        pass

    def test(self, im):
        """
        A placeholder for a function for testing a model.
        """
        pass

    def dissimilarity(self, out1, out2):
        """
        Calculate a dissimilarity between two outputs of the model
        """
        if out1.ndim > 1:
            out1 = out1.ravel()
            print ('WARNING: inputs for calculating dissimilarity not '
                   'one-dimensional as expected')
        if out2.ndim > 1:
            out2 = out2.ravel()
            print ('WARNING: inputs for calculating dissimilarity not '
                   'one-dimensional as expected')
        return (1-np.corrcoef(out1,out2)[0,1])/2.

    # def compare(self, ims):
    #     output = np.array([self.run(im) for im in ims])
    #     # for out1 in output:
    #     #     for out2 in output:

    #     out1 = self.run(im1)
    #     out2 = self.run(im2)
    #     return self.dissimilarity(out1, out2)


class Pixelwise(Model):
    def run(self, im):
        if im.ndim != 2:
            sys.exit('ERROR: Input image must be two-dimensional')
        return [im.ravel()]


class Zoccolan(Model):
    """
    Based on 10.1073/pnas.0811583106
    """
    def __init__(self):
        # receptive field sizes in degrees
        #self.rfs = np.array([.6,.8,1.])
        self.rfs = np.array([.2,.35,.5])
        # window size will be fixed in pixels and we'll adjust degrees accordingly
        # self.win_size_px = 300

    def get_gabors(self, rf):

        oris = np.linspace(0,np.pi,12)
        phases = [0,np.pi]
        lams =  float(rf[0])/np.arange(1,11) # lambda = 1./sf  #1./np.array([.1,.25,.4])
        #print lams
        sigma = rf[0]/2./np.pi
        # rf = [100,100]
        gabors = np.zeros(( len(oris),len(phases),len(lams), rf[0], rf[1] ))

        i = np.arange(-rf[0]/2+1,rf[0]/2+1)
        #print i
        j = np.arange(-rf[1]/2+1,rf[1]/2+1)
        ii,jj = np.meshgrid(i,j)
        for o, theta in enumerate(oris):
            x = ii*np.cos(theta) + jj*np.sin(theta)
            y = -ii*np.sin(theta) + jj*np.cos(theta)

            for p, phase in enumerate(phases):
                for s, lam in enumerate(lams):
                    sigmaq = sigma#.56*lam
                    fxx = np.cos(2*np.pi*x/lam + phase) * np.exp(-(x**2+y**2)/(2*sigmaq**2))
                    # import pdb; pdb.set_trace()

                    fxx -= np.mean(fxx)
                    fxx /= np.linalg.norm(fxx)

                    #if p==0:
                        #plt.subplot(len(oris),len(lams),count+1)
                        #plt.imshow(fxx,cmap=mpl.cm.gray,interpolation='bicubic')
                        #count+=1

                    gabors[o,p,s,:,:] = fxx
        plt.show()
        return gabors

    def run(self, im):
        field = im.shape
        # import pdb; pdb.set_trace()
        num_tiles = (15,15)#[field[0]/10.,field[0]/10.]
        size = (field[0]/num_tiles[0], field[0]/num_tiles[0])

        V1 = []#np.zeros( gabors.shape + num_tiles )

    #    tiled_im = im.reshape((num_tiles[0],size[0],num_tiles[1],size[1]))
    #    tiled_im = np.rollaxis(tiled_im, 1, start=3)
    #    flat_im = im.reshape((num_tiles[0],num_tiles[1],-1))

        for r, rf in enumerate(self.rfs):

            def apply_filter(window, this_filter):
                this_resp = np.dot(this_filter,window)/np.linalg.norm(this_filter)
    #            import pdb; pdb.set_trace()
                return np.max((0,this_resp)) # returns at least zero

            def filter_bank(this_filter,rf):
                #print 'done0'
                resp = scipy.ndimage.filters.generic_filter(
                                im, apply_filter, size=rf,mode='nearest',
                                extra_arguments = (this_filter,))
    #            import pdb; pdb.set_trace()
                #print 'done1'
                ii,jj = np.meshgrid(np.arange(0,field[0],size[0]),
                                        np.arange(0,field[1],size[1]) )
                selresp = resp[jj,ii]

    #            maxresp = scipy.ndimage.filters.maximum_filter(
    #                resp,
    #                size = size,
    #                mode = 'nearest'
    #            )
                return np.ravel(selresp)

            gabors = self.get_gabors(rf)
            #import pdb; pdb.set_trace()
            gabors = gabors.reshape(gabors.shape[:3]+(-1,))
    #        gabors_norms = np.apply_along_axis(np.linalg.norm, -1, gabors)
    #        import pdb; pdb.set_trace()
    #        V1.append( np.apply_along_axis(filter_bank, -1, gabors,rf) )
            V1resp = np.zeros(gabors.shape[:-1]+num_tiles)
    #        import pdb; pdb.set_trace()
            for i,wi in enumerate(np.arange(0,field[0]-rf[0],size[0])):
                for j,wj in enumerate(np.arange(0,field[1]-rf[1],size[1])):
                    window = im[wi:wi+rf[0],wj:wj+rf[1]]
                    resp = np.inner(gabors,np.ravel(window))
                    resp[resp<0] = 0
                    V1resp[:,:,:,i,j] = resp #/gabors_norms
    #                print 'done'
            V1.append(V1resp)
        return [V1]


class GaborJet(Model):
    """
    This is a Python implementation of the Gabor-Jet model from Biederman lab
    (http://geon.usc.edu/GWTgrid_simple.m). A given image is transformed with a
    gabor wavelet and certain values on a grid are chosen for the output.
    Further details are in http://dx.doi.org/10.1016/j.visres.2009.08.021.

    Copyright 2011-2012 Jonas Kubilius
    Original implementation copyright 2004 Xiaomin Yue
    """
    def run(self,
            im,  # input image; can be (128,128) or (256,256) px size
            cell_type = 'complex',  # 'complex': 40 output values
                                    # 'simple': 80 values
            grid_size = 0,  # how many positions within an image to take
            sigma = 2*np.pi,  # control the size of gaussion envelope
            ):
        if im.shape[0]!=im.shape[1]:
            sys.exit('The image has to be square. Please try again')
        
        # generate the grid
        if len(im) in [128,256]:
            if grid_size == 0:
                rangeXY = np.arange(20,110+1,10)  # 10x10
            elif grid_size == 1:
                rangeXY = np.arange(10,120+1,10)  # 12x12
            else:
                rangeXY = np.arange(1,128+1)  # 128x128
            rangeXY *= len(im)/128  # if len(im)==256, scale up by two
            rangeXY -= 1  # shift from MatLab indexing to Python
        else:        
            sys.exit('The image has to be 256*256 px or 128*128 px. Please try again')

        [xx,yy] = np.meshgrid(rangeXY,rangeXY)
        
        grid = xx + 1j*yy
        grid = grid.T.ravel()  # transpose just to match MatLab's grid(:) behavior
        grid_position = np.hstack([grid.imag, grid.real]).T
        
        # FFT of the image    
        im_freq = np.fft.fft2(im)
        #

        # setup the paramers
        nScale = 5  # spatial frequency scales
        nOrientation = 8  # orientation spacing; angle = 2*np.pi/nOrientations
        xHalfResL = im.shape[0]/2
        yHalfResL = im.shape[1]/2
        kxFactor = 2*np.pi/im.shape[0]
        kyFactor = 2*np.pi/im.shape[1]

        # setup space coordinate 
        [tx,ty] = np.meshgrid(np.arange(-xHalfResL,xHalfResL),np.arange(-yHalfResL,yHalfResL))
        tx = kxFactor*tx
        ty = kyFactor*(-ty)

        # initiallize useful variables
        if cell_type == 'complex':
            JetsMagnitude  = np.zeros((len(grid),nScale*nOrientation))
            JetsPhase      = np.zeros((len(grid),nScale*nOrientation))
        else:
            JetsMagnitude  = np.zeros((len(grid),2*nScale*nOrientation))
            JetsPhase      = np.zeros((len(grid),nScale*nOrientation))

        for LevelL in range(nScale):
            k0 = np.pi/2 * (1/np.sqrt(2))**LevelL
            for DirecL in range(nOrientation):
                kA = np.pi*DirecL/nOrientation  # gabor angle; SHOULD BE *2?
                # though in http://geon.usc.edu/~biederman/publications/Fiser_Biederman_Cooper_1996.pdf it's like here
                k0x = k0*np.cos(kA)
                k0y = k0*np.sin(kA)
                # generate a kernel specified scale and orientation, which has DC on the center
                # this is a FFT of a Morlet wavelet (http://en.wikipedia.org/wiki/Morlet_wavelet)
                #import pdb; pdb.set_trace()
                freq_kernel = 2*np.pi*(
                    np.exp( -(sigma/k0)**2/2 * ((k0x-tx)**2+(k0y-ty)**2) ) -\
                    np.exp( -(sigma/k0)**2/2 * (k0**2+tx**2+ty**2) )
                    )
                #import pdb; pdb.set_trace()
                # use fftshift to change DC to the corners
                freq_kernel = np.fft.fftshift(freq_kernel)
                
                # convolve the image with a kernel of the specified scale and orientation
                TmpFilterImage = im_freq*freq_kernel
                #
                # calculate magnitude and phase
                iTmpFilterImage = np.fft.ifft2(TmpFilterImage)
                #
                #eps = np.finfo(float).eps**(3./4)
                #real = np.real(iTmpFilterImage)
                #real[real<eps] = 0
                #imag = np.imag(iTmpFilterImage)
                #imag[imag<eps] = 0
                #iTmpFilterImage = real + 1j*imag
                
                TmpGWTPhase = np.angle(iTmpFilterImage)
                tmpPhase = TmpGWTPhase[rangeXY,:][:,rangeXY] + np.pi
                JetsPhase[:,LevelL*nOrientation+DirecL] = tmpPhase.ravel()
                #import pdb; pdb.set_trace()
                    
                if cell_type == 'complex':
                    TmpGWTMag = np.abs(iTmpFilterImage)
                    # get magnitude and phase at specific positions
                    tmpMag = TmpGWTMag[rangeXY,:][:,rangeXY]
                    JetsMagnitude[:,LevelL*nOrientation+DirecL] = tmpMag.ravel()
                    import pdb; pdb.set_trace()
                else:
                    TmpGWTMag_real = np.real(iTmpFilterImage)
                    TmpGWTMag_imag = np.imag(iTmpFilterImage)                
                    # get magnitude and phase at specific positions
                    tmpMag_real = TmpGWTMag_real[rangeXY,:][:,rangeXY]
                    tmpMag_imag = TmpGWTMag_imag[rangeXY,:][:,rangeXY]
                    JetsMagnitude_real[:,LevelL*nOrientation+DirecL] = tmpMag_real.ravel()
                    JetsMagnitude_imag[:,LevelL*nOrientation+DirecL] =  tmpMag_imag.ravel()

        if cell_type == 'simple':
            JetsMagnitude = np.vstack((JetsMagnitude_real, JetsMagnitude_imag))
        # use magnitude for similarity    
        return (JetsMagnitude, JetsPhase, grid_position)    


    def dissimilarity(self, outputs):
        """
        Calculate similarity between magnitudes of gabor jet.
        """
        if outputs.ndim != 3:
            sys.exit('ERROR: 3 dimensions expected')
        
        np.sum(outputs*outputs, axis=0)
        g = g.ravel()
        f = f.ravel()
        return 1 - np.dot(outputs,outputs.T) / (np.sqrt(np.dot(g,g)) * np.sqrt(np.dot(f,f)) )


    def test(self,JetsMagnitude, JetsPhase):
        for fname,jet in zip(['jet_mag.txt','jet_phase.txt'],[JetsMagnitude, JetsPhase]):
            jet_orig = np.genfromtxt(fname, delimiter=',')
            #fid = open(fname)
            #jet_orig = np.array([float(i.strip('\n')) for i in fid.readlines()])
            #fid.close()
            #C2 = self.HMAX_resp_all(old_new = 'old', C2='old')
            plt.plot(jet.ravel(),jet_orig.ravel(),'x')
            plt.show()
            print np.corrcoef(jet.ravel(),jet_orig.ravel())[0,1]


class HMAX(Model):
    """
    HMAX for Python

    Based on the original HMAX (Riesenhuber & Poggio, 1999, doi:10.1038/14819)
    Code rewritten using a Pure MatLab implementation by Minjoon Kouh at the
    MIT Center for Biological and Computational Learning. Most of the 
    structure, variable names and some of the comments come from this 
    implementation. More comments have been added and code was optimized as 
    much as possible while trying to maintain its structure close to the 
    original. View-tuned units have been added by Hans Op de Beeck.

    Code's output is tested against the Pure MatLab output which can be tested 
    agains the Standard C/MatLab code featured at     
    http://riesenhuberlab.neuro.georgetown.edu/hmax/index.html#code
    You can compare the outputs to the standard Lena image between the present 
    and C/MatLab implementation using function test_matlab()

    Note that this implementation is not the most current HMAX 
    implementation that doesn't rely on hardcoding features anymore (e.g., 
    Serre et al., 2007).

    Copyright 2011-2012 Jonas Kubilius
    Original VTU implementation copyright 2007 Hans P. Op de Beeck
    Original MatLab implementation copyright 2004 Minjoon Kouh
    Since the original code did not specify a license type, I assume GNU GPL v3
    since it is used in Jim Mutch's latest implementation of HMAX
    http://cbcl.mit.edu/jmutch/cns/    
    """
    def __init__(self, matlab=False, filt_type='gaussian'):
        """
        Initializes key HMAX parameters

        **Parameters**

            matlab: boolean
                If *True*, Gaussian filters will be implemented using the 
                original models implementation which mimicks MatLab's behavior.
                Otherwise, a more efficient numerical method is used.
        """        
        self.n_ori = 4 # number of orientations
        # S1 filter sizes for scale band 1, 2, 3, and 4
        self.filter_sizes_all = [[7, 9], [11, 13, 15], [17, 19, 21],
                                 [23, 25, 27, 29]]
        # specify (per scale band) how many S1 units will be used to pool over
        self.C1_pooling_all = [4, 6, 9, 12]
        self.S2_config = [2,2]  # how many C1 outputs to put into one "window" in S2 in each direction

        if filt_type == 'gaussian':  # "typically" used
            if matlab:  # exact replica of the MatLab implementation
                self.filts = self.get_gaussians_matlab(self.filter_sizes_all,
                                                       self.n_ori)
            else:  # a faster and more elegant implementation
                self.filts = self.get_gaussians(self.filter_sizes_all,
                                                self.n_ori)
            self.mask_name = 'square'
        elif filt_type == 'gabor':
            self.filts = self.get_gabors(self.filter_sizes_all, self.n_ori)
            self.mask_name = 'circle'
        else:
            raise ValueError, "filter type not recognized"
        
        self.istrained = False  # initially VTUs are not set up        

    def run(self, im=None, train_im=None):
        """
        This is the main function to run the model.
        First, it trains the model, i.e., sets up prototypes for VTU.
        Next, it runs the model.
        """
        if train_im is not None:
            self.train(train_im)
        if im is None:
            im = self.get_testim()
        output = self.test(im)

        return output

    def train(self, train_im):
        """
        Train the model, i.e., supply VTUs with C2 responses to 'prototype'
        images to which these units will be maximally tuned.
        """
        prots = [self.test()[3] for im in train_im]
        self.c2RespProt = np.array(prots)
        self.istrained = True

    def test(self, im):
        """
        Test the model on the given image
        """
        if im.dtype == int: im = im.astype(float)
        # Get number of filter sizes
        size_S1 = sum([len(fs) for fs in self.filter_sizes_all])
        # outputs from each layer are stored if you want to inspect them closer
        # but note that S1 is *massive*
        output = {}
        output['S1'] = np.zeros((im.shape[0],im.shape[1], size_S1, self.n_ori))
        output['C1'] = np.zeros((im.shape[0], im.shape[1], self.n_ori,
                       len(self.filter_sizes_all)))
        output['S2'] = []
        C2_tmp = np.zeros(( (self.S2_config[0]*self.S2_config[1])**self.n_ori,
            len(self.filter_sizes_all) )) 
        
        # Go through each scale band
        S1_idx = 0
        for which_band in range(len(self.filter_sizes_all)):
        
            # calculate S1 responses            
            S1_tmp = self.get_S1(im, which_band)
            num_filter = len(self.filter_sizes_all[which_band])            
            # store S1 responses for each scale band
            output['S1'][:,:,S1_idx:S1_idx+num_filter,:] = S1_tmp
            S1_idx += num_filter
            # calculate other layers
            C1_tmp = self.get_C1(S1_tmp, which_band)
            output['C1'][:,:,:,which_band] = C1_tmp  
            S2_tmp = self.get_S2(C1_tmp, which_band)
            output['S2'].append(S2_tmp)
            C2_tmp[:,which_band] = self.get_C2(S2_tmp, which_band)

        output['C2'] = np.max(C2_tmp,1) # max over all scale bands
        # try giving a VTU output
        if self.istrained:
            output['VTU'] = self.get_VTU(C2)

        return output
        
    def get_gaussians(
        self,
        filter_sizes_all,
        n_ori = 4,
        sigDivisor = 4.
        ):        
        """
        Generates 2D difference of Gaussians (DoG) filters.
        This function is a faster, more accurate and more elegant version of 
        the original gaussian_filters_matlab but will not produce identical 
        filters as the original (but very close). For practical purposes, this 
        one is prefered. In case you want to mimic the identical behavior of 
        the original HMAX, use gaussian_filters_matlab.
        
        **Parameters**
            filter_sizes_all: list (depth 2))
                A nested list (grouped by filter bands) of integer filter sizes
            n_ori: int
                A number of filter orientations (default: 4)
                Orientations are spaced by np.pi/n_ori
            sigDivisor: float
                A parameter to adjust DoG filter frequency (default: 4.)
                
        **Returns**
            gaussians: list (depth 2)
                A nested list of filters of all orientations
        
        """        
        gaussians = []
        # loop over filter bands
        for fNo, filter_sizes in enumerate(filter_sizes_all):
            gaussians.append([])
            
            # loop over filter sizes within a filter band
            for filter_size in filter_sizes:  
                fxx = np.zeros((filter_size,filter_size,n_ori))
                sigmaq = (filter_size/sigDivisor)**2
                i = np.arange(-filter_size/2+1,filter_size/2+1)
                ii,jj = np.meshgrid(i,i)
                for t in range(n_ori):
                    theta = t*np.pi/n_ori
                    x = ii*np.cos(theta) - jj*np.sin(theta)
                    y = ii*np.sin(theta) + jj*np.cos(theta)
                    # generate a 2D DoG of a particular orientation
                    gaussian = (y**2/sigmaq - 1) / sigmaq * \
                                np.exp(-(x**2 + y**2) / (2*sigmaq))
                    # normalize the filter to zero mean and unit variance
                    gaussian -= np.mean(gaussian)
                    gaussian /= np.sqrt(np.sum(gaussian**2))
                    fxx[:,:,t] = gaussian
                    
                gaussians[fNo].append(fxx)

        return gaussians
        
                            
    def get_gaussians_matlab(
        self,
        filter_sizes_all,
        n_ori = 4,
        sigDivisor = 4.):
        """
        Generates 2D difference of Gaussians (DoG) filters.
        This is the original version of DoG filters used in HMAX. It was 
        written in a very cumbersome way and thus I replaced it by the 
        gaussian_filters function. If you want to produce identical 
        numerical values of the filters, you should use this function. 
        Otherwise, gaussian_filters does the job just as well, but much nicer.
        
        **Parameters**
            filter_sizes_all: list (depth 2))
                A nested list (grouped by filter bands) of integer filter sizes
            n_ori: int
                A number of filter orientations (default: 4)
                Orientations are spaced by np.pi/n_ori
            sigDivisor: float
                A parameter to adjust DoG filter frequency (default: 4.)
                
        **Returns**
            gaussians: list (depth 2)
                A nested list of filters of all orientations
        
        """
        
        gaussians = []
        # loop over filter bands
        for fNo, filter_sizes in enumerate(filter_sizes_all):
            gaussians.append([])
            
            # loop over filter sizes within a filter band
            for filter_size in filter_sizes:
                fx1 = np.zeros((filter_size,filter_size,n_ori))
                
                # we gonna use a trick here:
                # make filters sqrt(2) times bigger so that we can rotate them
                # without getting zeros around the edges
                fieldSize = int(np.ceil(filter_size*np.sqrt(2)))
                fieldSize = fieldSize + 1 - fieldSize%2 # make odd
                filtSizeH = fieldSize/2
                cropOff = (fieldSize-filter_size)/2
                cropRange = slice(cropOff, cropOff+filter_size)
                sigmaq = (filter_size/sigDivisor)**2
                
                i = np.arange(-fieldSize/2+1,fieldSize/2+1)            
                ii,jj = np.meshgrid(i,i)
                
                theta = 0
                x = ii*np.cos(theta) - jj*np.sin(theta)
                y = ii*np.sin(theta) + jj*np.cos(theta)
                # generate a 2D DoG of 0 deg orientation
                fxx = (y**2/sigmaq-1)/sigmaq * np.exp(-(x**2+y**2)/(2*sigmaq))
                
                # now loop over the orientations, rotate and trim the filter
                for t in range(n_ori):
                    fxx = self.addZeros(fxx,cropOff)
                    fxx = scipy.ndimage.interpolation.rotate(fxx,45,reshape=False,order=1)
                    fxx = fxx[cropOff:fieldSize+cropOff,cropOff:fieldSize+cropOff]
                    # we generate first rotated versions of a filter
                    # and end up with the one having 0 deg, but now having
                    # undergonne all interpolations and rotations
                    # to make things equall
                    count = (t+1)%n_ori
                    # crop the edges
                    # note that you should assign this cropped version to sth
                    # like fx1[:,:,count], and not a variable on its own
                    # as otherwise you only pass a reference to fxx
                    # so you'd modify fxx as well when normalizing
                    # and you really don't want that
                    fx1[:,:,count] = fxx[cropRange, cropRange]
                    # normalize the filter to zero mean and unit variance
                    fx1[:,:,count] -= np.mean(fx1[:,:,count])
                    fx1[:,:,count] /= np.sqrt(np.sum(fx1[:,:,count]**2))
                gaussians[fNo].append(fx1)
                
        return gaussians

    def get_gabors(
        self,
        filter_sizes_all,
        n_ori = 4,
        k = 2.1,
        sx = 2*np.pi * 1/3.,
        sy = 2*np.pi * 1/1.8,
        phase = 0 # S1 Gabor function phase (0 for cosine and pi/2 for sine)

        ):
        """
        Generates 2D Gabor filters.
        This is the original version of Gabor filters used in HMAX. 
        
        **Parameters**
            filter_sizes_all: list (depth 2))
                A nested list (grouped by filter bands) of integer filter sizes
            n_ori: int
                A number of filter orientations (default: 4)
                Orientations are spaced by np.pi/n_ori
            k: float
                Gabor wave number (default: 2.1)
            sx: float
                Gabor sigma in x-dir (default: 2*np.pi * 1/3.)
            sy: float
                Gabor sigma in y-dir (default: 2*np.pi * 1/1.8)
            phase: int
                Gabor function phase (0 (default) for cosine (even),
                                      np.pi/2 for sine (odd))
                
        **Returns**
            gabors: list (depth 2)
                A nested list of filters of all orientations        
        """
        
        gabors = []
        # loop over filter bands
        for fNo, filter_sizes in enumerate(filter_sizes_all):
            gabors.append([])            
            # loop over filter sizes within a filter band
            for filter_size in filter_sizes:
                fxx = np.zeros((filter_size, filter_size, n_ori))
                inc = 2. / filter_size
                i = np.pi * np.arange(-1+inc/2, 1+inc/2, inc)
                ii,jj = np.meshgrid(i,i)

                circle = self.get_circle(filter_size)
                circle_sum = np.sum(circle)

                for t in range(n_ori):
                    theta = t*np.pi/n_ori
                    x = ii*np.cos(theta) - jj*np.sin(theta)
                    y = ii*np.sin(theta) + jj*np.cos(theta)
                    # generate a 2D DoG of a particular orientation
                    gabor = np.cos(k * x - phase) * \
                            np.exp(-( (x/sx)**2 + (y/sy)**2) / 2)
                    # apply circle mask                    
                    gabor *= circle
                    # normalize the filter to zero mean and unit variance
                    gabor -= circle * np.sum(gabor) / np.sum(circle)
                    gabor /= np.sqrt(np.sum(gabor**2))
                    fxx[:,:,t] = gabor
                gabors[fNo].append(fxx)

        return gabors

    def get_circle(self, filter_size, radius=1.):
            inc = 2./filter_size
            r = np.arange(-1+inc/2, 1+inc/2, inc)
            x, y = np.meshgrid(r, r)
            return x**2 + y**2 <= radius**2

        
    def addZeros(self, matrix, numZeros):
        """
        Pads matrix with zeros
        
        **Parameters**
            matrix: numpy.array
                A 2D numpy array to be padded
            numZeros: int
                Number of rows and colums of zeros to pad
                
        **Returns**
            matrix_new: numpy.array
                A zero-padded 2D numpy array                
        """        
        matrix_new = np.zeros((matrix.shape[0]+2*numZeros,
            matrix.shape[1]+2*numZeros))
        matrix_new[numZeros:matrix.shape[0]+numZeros,
            numZeros:matrix.shape[1]+numZeros] = matrix
        
        return matrix_new
    

    def get_S1(self, im, whichBand):
        """
        This function returns S1 responses,
        using the difference of the Gaussians or Gabors as S1 filters.
        Filters are based on the original HMAX model.
        """        
        filter_sizes = self.filter_sizes_all[whichBand]
        num_filter = len(filter_sizes)
        # make S1 same size as stimulus
        S1 = np.zeros((im.shape[0], im.shape[1], num_filter, self.n_ori))

        for j in range(num_filter):
            S1_filter = self.filts[whichBand][j]
            fs = filter_sizes[j]
            if self.mask_name == 'circle':
                mask = self.get_circle(fs)
            else:
                mask = np.ones((fs,fs))
            # import pdb; pdb.set_trace()
            norm = scipy.ndimage.convolve(im**2, mask, mode='constant') + \
                                          sys.float_info.epsilon
            for i in range(self.n_ori):
                S1_buf = scipy.ndimage.convolve(im, S1_filter[:,:,i],
                                                mode='constant')
                S1[:,:,j,i] = np.abs(S1_buf) / np.sqrt(norm)

        return S1

        
    def get_C1(self, S1, which_band):
        """
        Computes C1 responses given S1 as a max over a a couple of filter
        (as defined by C1_pooling)
        """
        C1_pooling = self.C1_pooling_all[which_band]
        C1 = scipy.ndimage.filters.maximum_filter(
            S1,
            size = (C1_pooling,C1_pooling,1,1),
            mode = 'constant',
            origin = -(C1_pooling/2)
            )

        # Max over scales;
        C1 = np.squeeze(np.max(C1,2))
        return C1
        
        
    def get_S2(self, C1, which_band, target=1., sigma=1.):
        """
        Calculates S2 responses given C1.

        First it pools over C1 activities over various combinations of 4 
        filters.
        Then computes a distance to /target/ using /sigma/ as its tuning 
        sharpness.
        """        
        # half overlaped S2 sampling
        S2_shift = int(np.ceil(self.C1_pooling_all[which_band]/2.))        
        # C1 afferents are adjacent for each S2
        C1_shift = S2_shift * 2 # distance/shift between C1 afferents        
        S2_buf = [C1.shape[0] - C1_shift*(self.S2_config[0]-1),
            C1.shape[1] - C1_shift*(self.S2_config[1]-1)]
        
        # produce a sequence of all possible orientation combinations
        seq = itertools.product(range(self.n_ori),
            repeat = self.S2_config[0]*self.S2_config[1])
        # we have to keep the same order as in the original model
        seq = np.fliplr([s for s in seq]) 
        
        S2_permute = np.zeros((
            (S2_buf[0]-1)/S2_shift+1,
            (S2_buf[1]-1)/S2_shift+1,
            len(seq),
            self.S2_config[0]*self.S2_config[1]))        
        
        for c1 in range(self.S2_config[0]):
            for c2 in range(self.S2_config[1]):
                c = self.S2_config[0]*c2 + c1
                r1 = np.arange(C1_shift*c1, S2_buf[0] + C1_shift*c1, S2_shift)
                r2 = np.arange(C1_shift*c2, S2_buf[1] + C1_shift*c2, S2_shift)
                ii,jj = np.meshgrid(r1, r2)
                S2_permute[:,:,:,c] = np.take(C1[jj,ii], seq[:,c], axis=2)
                # for si, s in enumerate(seq):
                #     S2_permute[:,:,si,c] = C1[jj,ii,s[c]] # the window is 
                                                         # sliding in the x-dir
        S2 = np.sum((S2_permute-target)**2,3)
        S2 = np.exp(-S2/(2.*sigma**2))
        
        return S2

    def get_C2(self, S2, which_band):
        """C2 is a max over space per an S2 filter quadruplet"""
        return  np.max(np.max(S2,0),0)            
        
    def get_VTU(self, c2RespSpec, tuningWidth = .1):
        """
        Calculate response of view-tuned units
        
        **Parameters**
            c2RespSpec: numpy.array
                C2 responses to the stimuli
            tuningWidth: float
                How sharply VTUs should be tuned; lower values are shaper 
                tuning (default: .1)
        **Returns**
            output: np.array
                An array where each column represents view-tuned units 
                responses to a particular image (stimulus)
        
        """
        if not self.istrained:
            raise ("ERROR: You must first train VTUs by providing prototype "
            "images to them using the train() function")
        else:
            c2RespProt = self.c2RespProt

        if c2RespSpec.shape[0] != c2RespProt.shape[0]:
            raise "ERROR: The size of exemplar matrix does not match that " +\
                "of the prototype matrix"            
        
        # covariance matrix
        covMat = np.eye(c2RespSpec.shape[0]) * tuningWidth
        output = np.zeros((c2RespProt.shape[1], c2RespSpec.shape[1]))
        
        for i in range(c2RespProt.shape[1]):
            for j in range(c2RespSpec.shape[1]):
                # distance between an exemplar and a VTU
                diff = c2RespSpec[:,j] - c2RespProt[:,i]
                # where on Gaussian it is
                output[i,j] = np.exp(-.5 * np.dot(
                    np.dot(diff, np.linalg.inv(covMat)),diff)) 
        
        return output        
    
    def test_matlab(self):
        """
        Compares output from this Python implementation to the original
        C/MatLab implementation
        """
        im = self.get_testim()
        python = self.run(im)
        fid = open('c2resp_matlab.txt')
        matlab = np.array([float(i.strip('\n')) for i in fid.readlines()])
        fid.close()

        plt.plot(python,matlab,'x')
        corr = np.corrcoef(python,matlab)[0,1]
        plt.xlabel('Python implementation')
        plt.ylabel('Original C/MatLab implementation')
        plt.title('Correlation = %.4f' %corr)
        plt.show()  

        
if __name__ == '__main__':
    models = {'px': Pixelwise, 'gaborjet': GaborJet, 'hmax': HMAX}
    
    if len(sys.argv) == 1:
        m = HMAX()
    else:
        model_name = sys.argv[2]
        if model_name in models: m = models[model_name]()

    if len(sys.argv) > 2:
        try:  # easy way to give more than one image to process
            ims = eval(sys.argv[3])
        except:  # otherwise the user gave a string, hopefully
            ims = [sys.argv[3]]
    else:
        ims = [m.get_testim()]

    for im in ims:
        output = m.run(im)