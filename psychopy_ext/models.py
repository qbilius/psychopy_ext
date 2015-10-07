#!/usr/bin/env python

# Part of the psychopy_ext library
# Copyright 2010-2015 Jonas Kubilius
# The program is distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.

"""
A library of simple models of vision

Simple usage::

    import glob
    from psychopy_ext import models
    ims = glob.glob('Example_set/*.jpg')  # get all jpg images
    hmax = models.HMAX()
    # if you want to see how similar your images are to each other
    hmax.compare(ims)
    # or to simply get the output and use it further
    out = hmax.run(ims)
"""

import sys, os, glob, itertools
import cPickle as pickle
from collections import OrderedDict

import numpy as np
import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import make_axes_locatable
import scipy.misc, scipy.ndimage
import pandas
import seaborn as sns

import matlab_wrapper
import sklearn.manifold, sklearn.preprocessing, sklearn.metrics
import skimage.feature, skimage.data

import stats, plot


try:
    os.environ['CAFFE']
except:
    HAS_CAFFE = False
else:
    # Suppress GLOG output for python bindings
    GLOG_minloglevel = os.environ.pop('GLOG_minloglevel', None)
    os.environ['GLOG_minloglevel'] = '5'

    # put Python bindings in the path
    sys.path.insert(0, os.path.join(os.environ['CAFFE'], 'python'))
    import caffe
    HAS_CAFFE = True

    # Turn GLOG output back on for subprocess calls
    if GLOG_minloglevel is None:
        del os.environ['GLOG_minloglevel']
    else:
        os.environ['GLOG_minloglevel'] = GLOG_minloglevel


class Model(object):

    def __init__(self):
        self.name = 'model'
        self.nice_name = 'Model'

    def get_teststim(self, size=(256, 256)):
        """
        Opens Lena image and resizes it to the specified size ((256, 256) by
        default)
        """
        lena = scipy.misc.lena()
        im = scipy.misc.imresize(lena, (256, 256))
        im = im.astype(float)

        return im

    def report(self, test_ims):
        print 'input images:', test_ims
        print 'processing:',
        output = self.run(test_ims=test_ims, train_ims=None,
                          return_dict=True)

        ax = sns.plt.subplot(131)
        ax.set_title('Dissimilarity across stimuli\n'
                     '(blue: similar, red: dissimilar)')
        dis = self.dissimilarity(output)
        sns.heatmap(dis, ax=ax)
        plt.sns.colorbar()

        ax = plt.subplot(132)
        ax.set_title('Multidimensional scaling')
        mds = self.mds(dis)
        plot.mdsplot(mds, ax=ax, icons=test_ims)

        ax = plt.subplot(133)
        ax.set_title('Linear separability')
        lin = self.linear_clf(dis)
        self.plot_linear_clf(lin, ax=ax, chance=1./len(np.unique(y)))

        sns.plt.show()
        return dis

    def run(self, test_ims=None, train_ims=None, layers=None, return_dict=True):
        """
        This is the main function to run the model.
        """
        if train_ims is not None:
            self.train(train_ims)
        output = self.test(test_ims, layers=layers, return_dict=return_dict)
        return output

    def train(self, train_ims):
        """
        A placeholder for a function for training a model.
        If the model is not trainable, then it will default to this function
        here that does nothing.
        """
        self.train_ims = train_ims

    def test(self, test_ims, layers=None, return_dict=True):
        """
        A placeholder for a function for testing a model.
        """
        self.layers = layers
        self.test_ims = test_ims

    def predict(self, im):
        """
        A placeholder for a function for predicting a label.
        """
        pass

    def dissimilarity(self, outputs, kind='mean_euclidean', **kwargs):
        """
        Computes dissimilarity between all rows in a matrix.

        :Args:
            outputs (numpy.array)
                A NxM array of model responses. Each row contains an
                output vector of length M from a model, and distances
                are computed between each pair of rows.

        :Kwargs:
            - kind (str or callable, default: 'mean_euclidean')
                Distance metric. Accepts string values or callables recognized
                by :func:`~sklearn.metrics.pairwise.pairwise_distances`, and
                    also 'mean_euclidean' that normalizes
                    Euclidean distance by the number of features (that is,
                    divided by M), as used, e.g., by Grill-Spector et al.
                    (1999), Op de Beeck et al. (2001), Panis et al. (2011).
                .. note:: Up to version 0.6, 'mean_euclidean' was called
                    'euclidean', and 'cosine' was called 'gaborjet'. Also note
                    that 'correlation' used to be called 'corr' and is now
                    returning dissimilarities in the range [0,2] per
                    scikit-learn convention.
            - \*\*kwargs
                Keyword arguments for
                :func:`~sklearn.metric.pairwise.pairwise_distances`

        :Returns:
            A square NxN matrix, typically symmetric unless otherwise defined
            by the metric.
        """
        if kind == 'mean_euclidean':
            dis_func = lambda x: sklearn.metrics.pairwise.pairwise_distances(x, metric='euclidean', **kwargs) / np.sqrt(x.shape[1])
        else:
            dis_func = lambda x: sklearn.metrics.pairwise.pairwise_distances(x, metric=kind)

        if isinstance(outputs, (dict, OrderedDict)):
            dis = OrderedDict()
            for layer, resps in outputs.items():
                sym_dis = dis_func(resps)
                sym_dis = (sym_dis + sym_dis.T) / 2.
                dis[layer] = sym_dis
        else:
            sym_dis = dis_func(resps)
            sym_dis = (sym_dis + sym_dis.T) / 2.
            dis = sym_dis

        return dis

    def input2array(self, names):
        raise Exception('DEPRECATED. Use :func:`load_images` instead.')

    def mds(self, dis, ims=None, ax=None, seed=None, kind='metric'):
        """
        Multidimensional scaling

        :Args:
            dis
                Dissimilarity matrix
        :Kwargs:
            - ims
                Image paths
            - ax
                Plot axis
            - seed
                A seed if you need to reproduce MDS results
            - kind ({'classical', 'metric'}, default: 'metric')
                'Classical' is based on MATLAB's cmdscale, 'metric' uses
                :func:`~sklearn.manifold.MDS`.

        """
        df = []
        for layer_name, this_dis in dis.items():
            if kind == 'classical':
                vals = stats.classical_mds(this_dis)
            else:
                mds_model = sklearn.manifold.MDS(n_components=2,
                                dissimilarity='precomputed', random_state=seed)
                vals = mds_model.fit_transform(this_dis)
            for im, (x,y) in zip(ims, vals):
                imname = os.path.splitext(os.path.basename(im))[0]
                df.append([layer_name, imname, x, y])

        df = pandas.DataFrame(df, columns=['layer', 'im', 'x', 'y'])
        if self.layers != 'all':
            if not isinstance(self.layers, (tuple, list)):
                self.layers = [self.layers]
            df = df[df.layer.isin(self.layers)]

        plot.mdsplot(df, ax=ax, icons=ims)

    def linear_clf(self, resps, y, clf=sklearn.svm.LinearSVC):
        df = []
        n_folds = len(y) / len(np.unique(y))
        for layer, resp in resps.items():
            # normalize to 0 mean and variance 1 for each feature (column-wise)
            resp = sklearn.preprocessing.StandardScaler().fit_transform(resp)
            cv = sklearn.cross_validation.StratifiedKFold(y,
                    n_folds=n_folds, shuffle=True)
            # from scikit-learn docs:
            # need not match cross_val_scores precisely!!!
            preds = sklearn.cross_validation.cross_val_predict(clf(),
                 resp, y, cv=cv)

            for yi, pred in zip(y, preds):
                df.append([layer, yi, pred, yi==pred])
        df = pandas.DataFrame(df, columns=['layer', 'actual', 'predicted', 'accuracy'])
        df = stats.factorize(df)
        return df

    def linear_clf_orig(self, resps, y, clf=sklearn.svm.LinearSVC):
        print
        print '### Linear separability ###'

        df = []
        n_folds = len(y) / len(np.unique(y))
        for layer, resp in resps.items():
            # normalize to 0 mean and variance 1 for each feature
            # (column-wise)
            resp = sklearn.preprocessing.StandardScaler().fit_transform(resp)
            cv = sklearn.cross_validation.StratifiedKFold(y,
                    n_folds=n_folds)
            # from scikit-learn docs:
            # need not match cross_val_scores precisely!!!
            preds = sklearn.cross_validation.cross_val_predict(clf(),
                 resp, y, cv=cv)

            # preds = []
            # for traini, testi in cv:
            #     for nset in range(1, n_folds+1):
            #         trainj = []
            #         for j in np.unique(y):
            #             sel = np.array(traini)[y[traini]==j]
            #             sel = np.random.choice(sel, nset).tolist()
            #             trainj.extend(sel)
            #     clf.fit(resps[trainj], y[trainj])
            #     preds.append(svm.predict(resps[testi], y[testi]))
            # confusion[kind][layer] = sklearn.metrics.confusion_matrix(y, preds)
            # print confusion[kind][layer]
            # perc.fit(resps, y)
            # print layer, np.mean(scores) #perc.score(resps, y)
            # import pdb; pdb.set_trace()
            # for score in scores:
            #     df.append([kind, layer, score])
            for yi, pred in zip(y, preds):
                df.append([layer, yi, pred, yi==pred])
            # preds = perc.predict(resps)
            # for pred, cat, shape in zip(preds, cats.ravel(), shapes.ravel()):
            #     df.append([layer, cat, shape, pred==cat])

        # self.df = pandas.DataFrame(df, columns=['layer', 'category',
        #                                   'shape', 'accuracy'])
        df = pandas.DataFrame(df, columns=['layer', 'actual', 'predicted', 'accuracy'])
        df = stats.factorize(df)
        return df

    def plot_linear_clf(self, df, chance=None, ax=None):
        g = sns.factorplot('layer', 'accuracy', data=df, kind='bar',ax=ax)
        ax.axhline(chance, ls='--', c='.2')


class Pixelwise(Model):

    def __init__(self, *args, **kwargs):
        """
        Pixelwise model

        The most simple model of them all. Uses pixel values only.
        """
        super(Pixelwise, self).__init__(*args, **kwargs)
        self.name = 'px'
        self.nice_name = 'Pixelwise'

    def test(self, test_ims, return_dict=False, **kwargs):
        self.layers = [self.name]
        ims = load_images(test_ims, resize=(256,256))
        resps = ims.reshape((ims.shape[0], -1))
        if return_dict:
            resps = OrderedDict([(self.layers[-1], resps)])
        return resps


class Retinex(Model):

    def __init__(self, *args, **kwargs):
        """
        Retinex algorithm

        Based on A. Torralba's implementation presented at PAVIS 2014.

        .. warning:: Experimental
        """
        super(Retinex, self).__init__(*args, **kwargs)
        self.nice_name = 'Retinex'
        self.name = 'retinex'

    def gen(self, imname, thres=20./256, plot=True, save=False):
        im = load_images(imname, flatten=False)

        # 2D derivative
        der = np.array([[0, 0, 0], [-1, 1, 0], [0, 0, 0]])

        im_paint = np.zeros(im.shape)
        im_illum = np.zeros(im.shape)
        for chno in range(3):
            ch = im[:,:,chno]

            outv = scipy.ndimage.convolve(ch, der)
            outh = scipy.ndimage.convolve(ch, der.T)
            out = np.dstack([outv, outh])

            # threshold
            paint = np.copy(out)
            paint[np.abs(paint) < thres] = 0
            illum = np.copy(out)
            illum[np.abs(illum) >= thres] = 0
            # plt.imshow(paint[:,:,0]); plt.show()
            # plt.imshow(paint[:,:,1]); plt.show()
            # plt.imshow(illum[:,:,0]); plt.show()
            # plt.imshow(illum[:,:,1]); plt.show()

            # Pseudo-inverse (using the trick from Weiss, ICCV 2001; equations 5-7)
            im_paint[:,:,chno] = self._deconvolve(paint, der)
            im_illum[:,:,chno] = self._deconvolve(illum, der)

        im_paint = (im_paint - np.min(im_paint)) / (np.max(im_paint) - np.min(im_paint))
        im_illum = (im_illum - np.min(im_illum)) / (np.max(im_illum) - np.min(im_illum))

        # paintm = scipy.misc.imread('paint2.jpg')
        # illumm = scipy.misc.imread('illum2.jpg')
        # print np.sum((im_paint-paintm)**2)
        # print np.sum((im_illum-illumm)**2)


        if plot:
            sns.plt.subplot(131)
            sns.plt.imshow(im)

            sns.plt.subplot(132)
            sns.plt.imshow(im_paint)

            sns.plt.subplot(133)
            sns.plt.imshow(im_illum)

            sns.plt.show()

        if save:
            name, ext = imname.splitext()
            scipy.misc.imsave('%s_paint.%s' %(name, ext), im_paint)
            scipy.misc.imsave('%s_illum.%s' %(name, ext), im_illum)

    def _deconvolve(self, out, der):
        # der = np.dstack([der, der.T])
        d = []
        gi = []
        for i, deri in enumerate([der, der.T]):

            d.append(scipy.ndimage.convolve(out[...,i], np.flipud(np.fliplr(deri))))
            gi.append(scipy.ndimage.convolve(deri, np.flipud(np.fliplr(deri)), mode='constant'))

        d = np.sum(d, axis=0)
        gi = np.sum(gi, axis=0)
        gi = np.pad(gi, (der.shape[0]/2, der.shape[1]/2), mode='constant')

        gi = scipy.ndimage.convolve(gi, np.array([[1,0,0], [0,0,0], [0,0,0]]))

        mxsize = np.max(out.shape[:2])
        g = np.fft.fft2(gi, s=(mxsize*2, mxsize*2))
        g[g==0] = 1
        h = 1/g
        h[g==0] = 0

        tr = h * np.fft.fft2(d, s=(mxsize*2,mxsize*2))
        ii = np.fft.fftshift(np.real(np.fft.ifft2(tr)))

        n = (gi.shape[0] - 5) / 2
        im = ii[mxsize - n : mxsize + out.shape[0] - n,
                mxsize - n : mxsize + out.shape[1] - n]
        return im


class Zoccolan(Model):
    """
    Based on 10.1073/pnas.0811583106

    .. warning:: Not implemented fully
    """
    def __init__(self):
        super(Model, self).__init__()
        self.name = 'Zoccolan'
        # receptive field sizes in degrees
        #self.rfs = np.array([.6,.8,1.])
        #self.rfs = np.array([.2,.35,.5])
        self.rfs = [10, 20, 30]  # deg visual angle
        self.oris = np.linspace(0, np.pi, 12)
        self.phases = [0, np.pi]
        self.sfs = range(1, 11)  # cycles per RF size
        self.winsize = [5, 5]  # size of each patch on the grid
        # window size will be fixed in pixels and we'll adjust degrees accordingly
        # self.win_size_px = 300

    def get_gabors(self, rf):
        lams =  float(rf[0])/self.sfs # lambda = 1./sf  #1./np.array([.1,.25,.4])
        sigma = rf[0]/2./np.pi
        # rf = [100,100]
        gabors = np.zeros(( len(oris),len(phases),len(lams), rf[0], rf[1] ))

        i = np.arange(-rf[0]/2+1,rf[0]/2+1)
        #print i
        j = np.arange(-rf[1]/2+1,rf[1]/2+1)
        ii,jj = np.meshgrid(i,j)
        for o, theta in enumerate(self.oris):
            x = ii*np.cos(theta) + jj*np.sin(theta)
            y = -ii*np.sin(theta) + jj*np.cos(theta)

            for p, phase in enumerate(self.phases):
                for s, lam in enumerate(lams):
                    fxx = np.cos(2*np.pi*x/lam + phase) * np.exp(-(x**2+y**2)/(2*sigma**2))
                    fxx -= np.mean(fxx)
                    fxx /= np.linalg.norm(fxx)

                    #if p==0:
                        #plt.subplot(len(oris),len(lams),count+1)
                        #plt.imshow(fxx,cmap=mpl.cm.gray,interpolation='bicubic')
                        #count+=1

                    gabors[o,p,s,:,:] = fxx
        plt.show()
        return gabors

    def run(self, ims):
        ims = self.input2array(ims)
        output = [self.test(im) for im in ims]

    def test(self, im):
        field = im.shape
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

    def __init__(self, nscales=5, noris=8, imsize=256, grid_size=0):
        """
        Python implementation of the Gabor-Jet model from Biederman lab.

        A given image is transformed with a
        Gabor wavelet and certain values on a grid are chosen for the output.
        Further details are in `Xu et al., 2009
        <http://dx.doi.org/10.1016/j.visres.2009.08.021>`_.

        Original implementation copyright 2004 'Xiaomin Yue
        <http://geon.usc.edu/GWTgrid_simple.m>`_.

        :Kwargs:
            - nscales (int, default: 5)
                Spatial frequency scales
            - noris (int, default: 8)
                Orientation spacing; angle = np.pi/noris
            - imsize ({128, 256}, default: 256)
                The image can only be 128x128 px or 256x256 px size.
                If the image has a different size, it will be rescaled
                **without** maintaining the original aspect ratio.
            - grid_size (int, default: 0)
                How many positions within an image to take
        """
        super(GaborJet, self).__init__()
        self.name = 'gaborjet'
        self.nice_name = 'GaborJet'
        self.nscales = nscales
        self.noris = noris
        self.imsize = imsize

        if grid_size == 0:
            s = imsize/128.
            rangeXY = np.arange(20*s, 110*s+1, 10) - 1  # 10x10
        elif grid_size == 1:
            s = imsize/128.
            rangeXY = np.arange(10*s, 120*s+1, 10) - 1  # 12x12
        else:
            rangeXY = np.arange(imsize)  # 128x128 or 256x256

        self.rangeXY = rangeXY.astype(int)
        [xx,yy] = np.meshgrid(rangeXY,rangeXY)

        self.grid = xx + 1j*yy
        self.grid = self.grid.T.ravel()  # transpose just to match MatLab's grid(:) behavior
        self.grid_pos = np.hstack([self.grid.imag, self.grid.real]).T

    def test(self,
             test_ims,
             cell_type='complex',
             sigma=2*np.pi,
             layers='magnitudes',
             return_dict=False
             ):
        """
        Apply GaborJet to given images.

        :Args:
            ims: str or list of str
                Image(s) to process with the model.

        :Kwargs:
            - cell_type (str, default: 'complex')
                Choose between 'complex'(40 output values) and 'simple' (80
                values)
            - sigma (float, default: 2*np.pi)
                Control the size of gaussian envelope
            - layers ({'all', 'phases', 'magnitudes'}, default: 'magnitudes')
                Not truly layers, but two output possibilities: either Fourier
                magnitudes or phases.
            - return_dict (bool, default: True)
                Whether only magnitude should be returned. If True, then also
                phase and grid positions are returned in a dict.

        :Returns:
            Magnitude and, depending on 'return_dict', phase.
        """
        if layers == 'all':
            self.layers = ['phases', 'magnitudes']
        elif isinstance(layers, str):
            self.layers = [layers]
        elif layers is None:
            self.layers = ['magnitudes']
        else:
            self.layers = layers

        mags = []
        phases = []
        for imno, im in enumerate(test_ims):
            sys.stdout.write("\rRunning %s... %d%%" % (self.nice_name,
                                                        100*imno/len(test_ims)))
            sys.stdout.flush()
            im = load_images(im, resize=(self.imsize, self.imsize), flatten=True)

            mag, phase = self._test(im, cell_type=cell_type, sigma=sigma)
            mags.append(mag.ravel())
            phases.append(phase.ravel())
        sys.stdout.write("\rRunning %s... done\n" % self.nice_name)

        mags = np.array(mags)
        phases = np.array(phases)

        outputs = OrderedDict()
        for layer in self.layers:
            if layer == 'magnitudes':
                outputs['magnitudes'] = mags
            elif layer == 'phases':
                outputs['phases'] = phases
        if not return_dict:
            outputs = outputs[self.layers[-1]]
        return outputs

    def _test(self, im, cell_type='complex', sigma=2*np.pi):
        if im.shape[0] != im.shape[1]:
            raise IOError('The image has to be square. Please try again.')

        # generate the grid
        # im = scipy.misc.imresize(im, (self.imsize, self.imsize))
        #if len(im) in [128, 256]:
        rangeXY = self.rangeXY
        # FFT of the image
        im_freq = np.fft.fft2(im)

        # setup the paramers
        kx_factor = 2 * np.pi / self.imsize
        ky_factor = 2 * np.pi / self.imsize

        # setup space coordinates
        xy = np.arange(-self.imsize/2, self.imsize/2)
        [tx,ty] = np.meshgrid(xy, xy)
        tx *= kx_factor
        ty *= -ky_factor

        # initiallize useful variables
        nvars = self.nscales * self.noris
        if cell_type == 'complex':
            mag = np.zeros((len(self.grid), nvars))
            phase = np.zeros((len(self.grid), nvars))
        else:
            mag = np.zeros((len(self.grid), 2*nvars))
            phase = np.zeros((len(self.grid), nvars))

        for scale in range(self.nscales):
            k0 = np.pi/2 * (1/np.sqrt(2))**scale
            for ori in range(self.noris):
                ka = np.pi * ori / self.noris
                k0x = k0 * np.cos(ka)
                k0y = k0 * np.sin(ka)
                # generate a kernel specified scale and orientation, which has DC on the center
                # this is a FFT of a Morlet wavelet (http://en.wikipedia.org/wiki/Morlet_wavelet)
                freq_kernel = 2*np.pi * (
                    np.exp( -(sigma/k0)**2/2 * ((k0x-tx)**2 + (k0y-ty)**2) ) -\
                    np.exp( -(sigma/k0)**2/2 * (k0**2+tx**2+ty**2) )
                    )
                # use fftshift to change DC to the corners
                freq_kernel = np.fft.fftshift(freq_kernel)

                # convolve the image with a kernel of the specified scale and orientation
                conv = im_freq*freq_kernel
                #
                # calculate magnitude and phase
                iconv = np.fft.ifft2(conv)
                #
                #eps = np.finfo(float).eps**(3./4)
                #real = np.real(iTmpFilterImage)
                #real[real<eps] = 0
                #imag = np.imag(iTmpFilterImage)
                #imag[imag<eps] = 0
                #iTmpFilterImage = real + 1j*imag

                ph = np.angle(iconv)
                ph = ph[rangeXY,:][:,rangeXY] + np.pi
                ind = scale*self.noris+ori
                phase[:,ind] = ph.ravel()

                if cell_type == 'complex':
                    mg = np.abs(iconv)
                    # get magnitude and phase at specific positions
                    mg = mg[rangeXY,:][:,rangeXY]
                    mag[:,ind] = mg.ravel()
                else:
                    mg_real = np.real(iconv)
                    mg_imag = np.imag(iconv)
                    # get magnitude and phase at specific positions
                    mg_real = mg_real[rangeXY,:][:,rangeXY]
                    mg_imag = mg_imag[rangeXY,:][:,rangeXY]
                    mag[:,ind] = mg_real.ravel()
                    mag[:,nvars+ind] = mg_imag.ravel()

        # use magnitude for dissimilarity measures
        return mag, phase

    def dissimilarity(self, kind='gaborjet', *args, **kwargs):
        return super(GaborJet, self).dissimilarity(kind=kind,
                                                   *args, **kwargs)


class HMAX99(Model):
    """
    HMAX for Python

    Based on the original HMAX (`Riesenhuber & Poggio, 1999
    <http://dx.doi.org/10.1038/14819>`_)
    Code rewritten using a Pure MATLAB implementation by Minjoon Kouh at the
    MIT Center for Biological and Computational Learning. Most of the
    structure, variable names and some of the comments come from this
    implementation. More comments have been added and code was optimized as
    much as possible while trying to maintain its structure close to the
    original. View-tuned units have been added by Hans Op de Beeck.

    Code's output is tested against the Pure MatLab output which can be tested
    against the Standard C/MATLAB code featured at `Riesenhuber's lab
    <http://riesenhuberlab.neuro.georgetown.edu/hmax/index.html#code>`_.
    You can compare the outputs to the standard Lena image between the present
    and C/MatLab implementation using function :mod:test_models

    .. note:: This implementation is not the most current HMAX
    implementation that doesn't rely on hardcoding features anymore (e.g.,
    Serre et al., 2007). Use :class:`HMAX` to access MATLAB interface to a
    more current version of HMAX.

    Original VTU implementation copyright 2007 Hans P. Op de Beeck

    Original MatLab implementation copyright 2004 Minjoon Kouh

    Since the original code did not specify a license type, I assume GNU GPL v3
    since it is used in `Jim Mutch's latest implementation of HMAX
    <http://cbcl.mit.edu/jmutch/cns/>`_

    :Kwargs:
        - matlab (boolean, default: False)
            If *True*, Gaussian filters will be implemented using the
            original models implementation which mimicks MatLab's behavior.
            Otherwise, a more efficient numerical method is used.
        - filter_type ({'gaussian', 'gabor'}, default: 'gaussian')
            Type of V1 filter. We default to gaussian as it was used originally
            in HMAX'99. However, many people prefer using Gabor filters as
            they presumambly model V1 better.

    """
    def __init__(self, matlab=False, filter_type='gaussian'):
        super(Model, self).__init__()
        self.name = 'hmax99'
        self.nice_name = "HMAX'99"

        self.n_ori = 4 # number of orientations
        # S1 filter sizes for scale band 1, 2, 3, and 4
        self.filter_sizes_all = [[7, 9], [11, 13, 15], [17, 19, 21],
                                 [23, 25, 27, 29]]
        # specify (per scale band) how many S1 units will be used to pool over
        self.C1_pooling_all = [4, 6, 9, 12]
        self.S2_config = [2,2]  # how many C1 outputs to put into one "window" in S2 in each direction

        if filter_type == 'gaussian':  # "typically" used
            if matlab:  # exact replica of the MatLab implementation
                self.filts = self.get_gaussians_matlab(self.filter_sizes_all,
                                                       self.n_ori)
            else:  # a faster and more elegant implementation
                self.filts = self.get_gaussians(self.filter_sizes_all,
                                                self.n_ori)
            self.mask_name = 'square'
        elif filter_type == 'gabor':
            self.filts = self.get_gabors(self.filter_sizes_all, self.n_ori)
            self.mask_name = 'circle'
        else:
            raise ValueError, "filter type not recognized"

        self.istrained = False  # initially VTUs are not set up

    def run(self, test_ims=None, train_ims=None, layers=None, return_dict=True):
        """
        This is the main function to run the model.

        First, it trains the model, i.e., sets up prototypes for VTU.
        Next, it runs the model.
        """
        if train_ims is not None:
            self.train(train_ims)
        if test_ims is None:
            test_ims = [self.get_teststim()]
        output = self.test(test_ims, layers=layers, return_dict=return_dict)
        # if oneval:
        #     return output['C2']
        # else:
        return output

    def train(self, train_ims):
        """
        Train the model

        That is, supply VTUs with C2 responses to 'prototype'
        images to which these units will be maximally tuned.

        :Args:
            train_ims (str)
                Path to training images; anything that :func:`load_images` takes
        """
        try:
            self.tuning = pickle.load(open(train_ims,'rb'))
            print 'done'
        except:
            train_ims = self.input2array(train_ims)
            self.tuning = self.test(train_ims, op='training')['C2']
        self.istrained = True

    def test(self, ims, op='testing', layers=None, return_dict=True, **kwrags):
        """
        Test the model on the given image

        :Args:
            train_ims (str)
                Path to training images; anything that :func:`load_images` takes

        """

        ims = load_images(ims)
        # Get number of filter sizes
        size_S1 = sum([len(fs) for fs in self.filter_sizes_all])
        # outputs from each layer are stored if you want to inspect them closer
        # but note that S1 is *massive*
        output = OrderedDict()
        # S1 will not be in the output because it's too large:
        # with default parameters S1 takes 256*256*12*4*64bits = 24Mb per image
        S1 = np.zeros((ims.shape[1], ims.shape[2], size_S1, self.n_ori))
        output['C1'] = np.zeros(ims.shape + (self.n_ori,
                                len(self.filter_sizes_all)))
        # S2 has an irregular shape which depends on the spatial frequency band
        # so we'll omit it as well
        S2 = []
        C2_tmp = np.zeros(((self.S2_config[0]*self.S2_config[1])**self.n_ori,
                            len(self.filter_sizes_all)))
        output['C2'] = np.zeros((len(ims),C2_tmp.shape[0]))

        for imno, im in enumerate(ims):
            sys.stdout.write("\rRunning HMAX... %s: %d%%" %(op, 100*imno/len(ims)))
            sys.stdout.flush()
            # im = load_images(im)
            # Go through each scale band
            S1_idx = 0
            for which_band in range(len(self.filter_sizes_all)):
                # calculate S1 responses
                S1_tmp = self.get_S1(im, which_band)
                num_filter = len(self.filter_sizes_all[which_band])
                # store S1 responses for each scale band
                S1[..., S1_idx:S1_idx + num_filter, :] = S1_tmp
                S1_idx += num_filter
                # calculate other layers
                C1_tmp = self.get_C1(S1_tmp, which_band)
                output['C1'][imno, ..., which_band] = C1_tmp
                S2_tmp = self.get_S2(C1_tmp, which_band)
                S2.append(S2_tmp)
                C2_tmp[:, which_band] = self.get_C2(S2_tmp, which_band)
            output['C2'][imno] = np.max(C2_tmp, -1) # max over all scale bands
        output['S2'] = S2
        # calculate VTU if trained
        if self.istrained:
            output['VTU'] = self.get_VTU(output['C2'])
        sys.stdout.write("\rRunning HMAX... %s: done\n" %op)

        if layers is None:
            output = OrderedDict([('C2', output['C2'])])
            self.layers = ['C2']
        elif layers != 'all':
            output = OrderedDict([(layer, output[layer]) for layer in layers])
            self.layers = layers
        else:
            self.layers = ['S1', 'C1', 'S2', 'C2']

        if not return_dict:
            output = output[self.layers[-1]]

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

        :Args:
            filter_sizes_all (list of depth 2)
                A nested list (grouped by filter bands) of integer filter sizes

        :Kwargs:
            - n_ori (int, default: 4)
                A number of filter orientations. Orientations are spaced by np.pi/n_ori.
            - sigDivisor (float, default: 4.)
                A parameter to adjust DoG filter frequency.

        :Returns:
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
        Generates 2D difference of Gaussians (DoG) filters, MATLAB style.

        This is the original version of DoG filters used in HMAX. It was
        written in a very cumbersome way and thus I replaced it by the
        gaussian_filters function. If you want to produce identical
        numerical values of the filters, you should use this function.
        Otherwise, :func:`gaussian_filters` does the job just as well,
        but much nicer.

        :Args:
            filter_sizes_all (list of depth 2)
                A nested list (grouped by filter bands) of integer filter sizes
        :Kwargs:
            - n_ori (int, defualt: 4)
                A number of filter orientations. Orientations are spaced by np.pi/n_ori.
            - sigDivisor (float, default: 4.)
                A parameter to adjust DoG filter frequency.

        :Returns:
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

        :Args:
            filter_sizes_all (list of depth 2)
                A nested list (grouped by filter bands) of integer filter sizes

        :Kwargs:
            - n_ori (int, default: 4)
                A number of filter orientations. Orientations are spaced by np.pi/n_ori.
            - k (float, default: 2.1)
                Gabor wave number
            - sx (float, default: 2*np.pi * 1/3.)
                Gabor sigma in x-dir
            - sy (float, default: 2*np.pi * 1/1.8)
                Gabor sigma in y-dir
            - phase (int, default: 0)
                Gabor function phase (0 for cosine (even), np.pi/2 for sine (odd))

        :Returns:
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

        :Args:
            - matrix (numpy.ndarray)
                A 2D numpy array to be padded
            - numZeros (int)
                Number of rows and colums of zeros to pad

        :*Returns:
            matrix_new (numpy.ndarray)
                A zero-padded 2D numpy array
        """
        matrix_new = np.zeros((matrix.shape[0]+2*numZeros,
            matrix.shape[1]+2*numZeros))
        matrix_new[numZeros:matrix.shape[0]+numZeros,
            numZeros:matrix.shape[1]+numZeros] = matrix

        return matrix_new

    def get_S1(self, im, whichBand):
        """
        This function returns S1 responses.

        Using the difference of the Gaussians or Gabors as S1 filters.
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

    def get_VTU(self, C2resp, tuningWidth = .1):
        """
        Calculate response of view-tuned units

        :Args:
            c2RespSpec (numpy.ndarray)
                C2 responses to the stimuli
        :Kwargs:
            tuningWidth (float, default: .1)
                How sharply VTUs should be tuned; lower values are shaper
                tuning
        :Returns:
            An array where each column represents view-tuned units
                responses to a particular image (stimulus)
        """
        def sq(c):
            return np.dot(c,c)
        def func(row):
            # difference between tuning and each C2 response
            diff = self.tuning - \
                    np.tile(row,(self.tuning.shape[0],1))
            # this difference is then square-summed and then exponentiated :)
            return np.exp(-.5 * np.apply_along_axis(sq,1,diff) / tuningWidth)

        if not self.istrained:
            raise Exception("You must first train VTUs by providing prototype "
                            "images to them using the train() function")

        if C2resp.shape[1] != self.tuning.shape[1]:
            raise Exception("The size of exemplar matrix does not match "
                            "that of the prototype matrix")
        # apply func on each row
        return np.apply_along_axis(func, 1, C2resp)


class HOG(Model):

    def __init__(self, *args, **kwargs):
        super(HOG, self).__init__(*args, **kwargs)
        self.name = 'hog'
        self.nice_name = 'HOG'

    def test(self, test_ims, return_dict=False, layers=None, **kwargs):
        self.layers = [self.name]
        resps = []
        for imno, im in enumerate(test_ims):
            sys.stdout.write("\rRunning %s... %d%%" % (self.name, 100*imno/len(ims)))
            sys.stdout.flush()

            im = load_images(im, flatten=True)
            resps.append(skimage.feature.hog(im, **kwargs))
        resps = np.array(resps)

        sys.stdout.write("\rRunning %s... done\n" % self.name)
        sys.stdout.flush()

        if kwargs.get('visualise'):
            resps, hog_image = resps

        if return_dict:
            resps = OrderedDict([(self.layers[-1], resps)])

        return resps


class Caffe(Model):

    def __init__(self, model='CaffeNet', model_path=None, mode='cpu'):

        if model in ALIASES:
            self.name = ALIASES[model]
        else:
            self.name = name.lower()

        if self.name in CAFFE_NICE_NAMES:
            self.nice_name = CAFFE_NICE_NAMES[self.name]
        else:
            self.nice_name = model

        self.model_path = model_path  # will be updated when self.test is called

        if mode == 'cpu':
            caffe.set_mode_cpu()
        elif mode == 'gpu':
            caffe.set_mode_gpu()
        else:
            raise Exception('ERROR: mode %s not recognized' % mode)

        self.istrained = False

    def _set_paths(self):
        try:
            os.environ['CAFFE']
        except:
            raise Exception("Caffe not found in your path; it must be set in "
                            "the 'CAFFE' variable")
        else:
            self.caffe_root = os.environ['CAFFE']

        if self.model_path is not None:
            if '.caffemodel' in self.model_path:
                self.weights_path = self.model_path
                self.model_path = os.path.dirname(self.model_path)
            else:
                self.weights_path = os.path.join(self.model_path, '*.caffemodel')
        else:
            try:
                self.model_path = CAFFE_MODELS[self.name]
            except:
                raise Exception('Model %s not recognized. Please provide a '
                                'model_path when calling this model.' %
                                self.nice_name)
            self.weights_path = os.path.join(self.model_path, '*.caffemodel')

    def run(self, test_ims=None, train_ims=None, layers=None, return_dict=True):
        """
        This is the main function to run the model.
        """
        if train_ims is not None:
            self.train(train_ims)
        output = self.test(test_ims, layers=layers, return_dict=return_dict)
        return output


    def train(self, *args, **kwargs):
        raise NotImplemented

    def _setup_layers(self, layers):
        if layers is None:
            self.layers = [self.net.params.keys()[-1]]  # top layer
        elif layers == 'all':
            self.layers = self.net.params.keys()
        elif not isinstance(layers, (tuple, list)):
            self.layers = [layers]
        else:
            self.layers = layers

    def test(self, ims, layers=None, return_dict=True):

        if not hasattr(self, 'net'):
            self.net = self._classifier()
        self._setup_layers(layers)
        new_shape = (1, ) + self.net.blobs['data'].data.shape[1:]
        self.net.blobs['data'].reshape(*new_shape)
        output = OrderedDict()
        for layer in self.layers:
            sh = self.net.blobs[layer].data.shape[1:]
            if not isinstance(sh, tuple):
                sh = (sh,)
            output[layer] = np.zeros((len(ims),) + sh)

        for imno, im in enumerate(ims):
            sys.stdout.write("\rRunning %s... %d%%" % (self.name, 100*imno/len(ims)))
            sys.stdout.flush()
            # im = load_images(im)
            # im = np.dstack([im,im,im])
            im = caffe.io.load_image(im)

            out = self._test(im)
            for layer in self.layers:
                output[layer][imno] = self.net.blobs[layer].data

        sys.stdout.write("\rRunning %s... done\n" % self.name)
        sys.stdout.flush()

        if not return_dict:
            output = output[self.layers[-1]]
        return output

    def _test(self, im):
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', im)
        out = self.net.forward()
        return out

    def predict(self, ims, topn=5):
        labels = self._get_labels()
        self.layers = ['prob']
        preds = self.run(test_ims=ims, return_dict=False)
        out = []
        for pred in preds:
            classno = np.argsort(pred)[::-1][:topn]
            tmp = []
            for n in classno:
                d = {'classno': n,
                     'synset': labels[n][0],
                     'label': labels[n][1]}
                tmp.append(d)
            out.append(tmp)

        return out

    def _classifier(self):
        self._set_paths()
        mn = self.read_mean()

        model_file = os.path.join(self.model_path, '*deploy*.prototxt')

        model_file = sorted(glob.glob(model_file))[0]
        if not self.istrained:
            trained_values = sorted(glob.glob(self.weights_path))[0]
        print 'model parameters loaded from', trained_values
        net = caffe.Net(model_file, trained_values, caffe.TEST)

        self.transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_mean('data', mn.mean(1).mean(1)) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

        return net

    def read_mean(self):
        meanf = os.path.join(self.model_path, '*mean*.protobinary')
        meanf = glob.glob(meanf)
        if len(meanf) > 0:
            data = open(meanf[0], 'rb').read()
            blob = caffe.proto.caffe_pb2.BlobProto()
            blob.ParseFromString(data)
            mn = np.array(caffe.io.blobproto_to_array(blob))[0]
        else:
            meanf = os.path.join(self.caffe_root, 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
            mn = np.load(meanf)
        return mn

    def _get_labels(self):
        synset_file = os.path.join(self.caffe_root, 'data/ilsvrc12/synset_words.txt')
        try:
            with open(synset_file) as f:
                lines = f.readlines()
        except:
            raise Exception('ERROR: synset file with labels not found.\n'
                            'Tried: %s' % synset_file)
        out = []
        for line in lines:
            line = line.strip('\n\r')
            out.append((line[:9], line[10:]))
        return out

    def preds2df(self, preds):
        df = []
        for predno, pred in enumerate(preds):
            tmp = []
            for p in pred:
                p.update({'n': predno})
                tmp.append(p)
            df.extend(tmp)
        return pandas.DataFrame.from_dict(df)


class MATLABModel(Model):

    def __init__(self, model_path=None, *args, **kwargs):
        """
        A base class for making an interface to MATLAB-based models
        """
        # super(MATLABModel, self).__init__(*args, **kwargs)
        self.model_path = model_path

    def _set_model_path(self):
        if self.model_path is None:
            if self.nice_name.upper() not in os.environ:
                raise Exception('Please specify model_path to the location of '
                                '%s or add it to your path '
                                'using %s as the environment variable.' %
                                (self.nice_name, self.nice_name.upper()))
            else:
                self.model_path = os.environ[self.nice_name.upper()]

    def test(self, test_ims, return_dict=True, **kwargs):
        self._set_model_path()

        matlab = matlab_wrapper.MatlabSession(options='-nojvm -nodisplay -nosplash')
        matlab.eval('addpath %s' % self.model_path)

        resps = self._test(matlab, test_ims)

        sys.stdout.write("\rRunning %s... done\n" % self.nice_name)
        sys.stdout.flush()

        matlab.eval('rmpath %s' % self.model_path)
        del matlab

        if return_dict:
            resps = OrderedDict([(self.name, resps)])
        return resps


class HMAX(MATLABModel):

    def __init__(self, *args, **kwargs):
        """
        The minimal implementation of HMAX (aka hmin)

        The model can be downloaded from
        `here <http://cbcl.mit.edu/jmutch/hmin/>`_

        """
        self.name = 'hmax'
        self.nice_name = 'HMAX'
        super(HMAX, self).__init__(*args, **kwargs)

    def _test(self, matlab, test_ims, **kwargs):
        resps = []
        for imno, im in enumerate(test_ims):
            sys.stdout.write("\rRunning %s... %d%%" % (self.nice_name,
                                                       100*imno/len(test_ims)))
            sys.stdout.flush()
            matlab.eval("example_run('%s')" % os.path.join(os.getcwd(), im))
            resps.append(matlab.get('ans'))

        return np.array(resps)


class PHOG(MATLABModel):

    def __init__(self, *args, **kwargs):
        """
        Pyramid Histogram of Oriented Gradients

        The model can be downloaded from `here <http://www.robots.ox.ac.uk/~vgg/research/caltech/phog.html>`_

        Reference:

        `Bosch A., Zisserman A., Munoz X. Representing shape with a spatial pyramid kernel. CIVR 2007 <http://dx.doi.org/10.1145/1282280.1282340>`_
        """
        self.name = 'phog'
        self.nice_name = 'PHOG'
        super(PHOG, self).__init__(*args, **kwargs)

    def _test(self, matlab, test_ims, nbins=8, angle=360, nlayers=3, **kwargs):
        if angle not in [180, 360]:
            raise Exception('PHOG angle must be either 180 or 360')
        im = load_images(test_ims[0])
        roi = np.mat([1, im.shape[0], 1, im.shape[1]]).T
        matlab.put('roi', roi)
        args = '{}, {}, {}, {}'.format(nbins, angle, nlayers, 'roi')

        resps = []
        for imno, im in enumerate(test_ims):
            sys.stdout.write("\rRunning %s... %d%%" % (self.nice_name,
                                                       100*imno/len(test_ims)))
            sys.stdout.flush()
            matlab.eval("anna_phog('%s', %s)" % (os.path.join(os.getcwd(), im), args))
            resps.append(matlab.get('ans'))

        return np.array(resps)


class PHOW(MATLABModel):

    def __init__(self, *args, **kwargs):
        """
        Pyramid Histogram of Visual Words / Spatial Pyramid Matching

        The model can be downloaded from
        `here <http://slazebni.cs.illinois.edu/research/SpatialPyramid.zip>`_.

        Reference:

        `Lazebnik, S., Schmid, C, Ponce, J. Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories. *CVPR 2006*. <http://slazebni.cs.illinois.edu/publications/cvpr06b.pdf>`_
        """
        self.name = 'phow'
        self.nice_name = 'PHOW'
        super(PHOW, self).__init__(*args, **kwargs)

    def _test(self, matlab, test_ims,
             max_imsize=1000, grid_spacing=8, patch_size=16,
             dict_size=200, ntextons=50, npyrlevs=3,
             old_sift=False, can_skip=1, save_sift=1, **kwargs):

        # get into MATLAB struct
        params = [('maxImageSize', max_imsize),
                  ('gridSpacing', grid_spacing),
                  ('patchSize', patch_size),
                  ('dictionarySize', dict_size),
                  ('numTextonImages', ntextons),
                  ('pyramidLevels', npyrlevs),
                  ('oldSift', old_sift)
                  ]
        params = np.array([tuple(p[1] for p in params)],
                          dtype=[(p[0],'<f8') for p in params])

        matlab.put('params', params.view(np.recarray))
        ims = [os.path.basename(im) for im in test_ims]
        matlab.put('ims', ims)
        image_dir = os.path.join(os.getcwd(), os.path.dirname(test_ims[0]))
        data_dir = os.path.join(os.getcwd(), self.name + '_data')
        funcstr = "BuildPyramid(ims, '%s', '%s', params, %d, %d)" % (image_dir, data_dir, can_skip, save_sift)

        sys.stdout.write("\rRunning %s..." % self.nice_name)
        sys.stdout.flush()

        matlab.eval(funcstr)
        resps = matlab.get('ans')

        return resps


class RandomFilters(MATLABModel):

    def __init__(self, *args, **kwargs):
        """
        Random Features and Supervised Classifier

        The model can be downloaded from
        `here <http://cs.nyu.edu/~koray/publis/code/randomc101.tar.gz`_

        Reference:

        `Jarrett, K., Kavukcuoglu, K., Ranzato, M., & LeCun, Y. What is the Best Multi-Stage Architecture for Object Recognition?, *ICCV 2009 <http://cs.nyu.edu/~koray/publis/jarrett-iccv-09.pdf>`_
        """

        self.name = 'randfilt'
        self.nice_name = 'RandomFilters'
        super(RandomFilters, self).__init__(*args, **kwargs)

    # def train(self, train_ims):
    #     import matlab_wrapper
    #
    #     matlab = matlab_wrapper.MatlabSession(options='-nojvm -nodisplay -nosplash')
    #     matlab.eval('addpath %s' % self.model_path)
    #
    #     matlab.eval("[trdata,tedata] = prepareData();")
    #     matlab.eval('rmpath %s' % self.model_path)
    #     del matlab

    def _test(self, matlab, test_ims, **kwargs):
        # path = os.path.join(self.model_path, name+'.mat')
        # params = scipy.io.loadmat(open(path, 'rb'))

        param_path = os.path.join(self.model_path, '../data/params.mat')
        matlab.eval("params = load('%s');" % param_path)
        matlab.eval('params.kc.layer1 = -0.11 + 0.22 * rand(size(params.ct.layer1,1),9,9);')
        matlab.eval('params.kc.layer2 = -0.11 + 0.22 * rand(size(params.ct.layer2,1),9,9);')

        funcstr = "extractRandomFeatures(pim, params.ker, params.kc, params.ct, params.bw, params.bs);"
        resps = []
        for imno, im in enumerate(test_ims):
            sys.stdout.write("\rRunning %s... %d%%" % (self.nice_name,
                                                       100*imno/len(test_ims)))
            sys.stdout.flush()
            im = load_images(im, flatten=True)
            mx = max(im.shape[:2])
            sh = (151./mx * im.shape[0], 151./mx * im.shape[1])
            imr = np.round(resize_image(im, sh) * 255)
            matlab.put('imr', imr)
            matlab.eval('pim = imPreProcess(imr,params.ker);')
            matlab.eval(funcstr)
            resp = matlab.get('ans')
            resps.append(resp.ravel())

        return np.array(resps)

def call_matlab(script_path):
    cmd = 'matlab -nojvm -nodisplay -nosplash -r {}; exit'.format(script_path)
    subprocess.call(cmd.split())

def get_model(model_name):
    if model_name in KNOWN_MODELS:
        return KNOWN_MODELS[model_name]()
    else:
        raise Exception('Model %s not recognized' %model_name)

def load_images(names, flatten=True, resize=1.):
    try:
        names = eval(names)
    except:
        pass

    if isinstance(names, str):
        array = _load_image(names, flatten=flatten, resize=resize)
    elif isinstance(names, (list, tuple)):
        if isinstance(names[0], str):
            array = [_load_image(n, flatten=flatten, resize=resize) for n in names]
            array = np.array(array)
        else:
            array = np.array(names)
    elif isinstance(names, np.ndarray):
        array = names
    else:
        raise ValueError('input image type not recognized')

    # if array.ndim == 2:
    #     array = np.reshape(array, (1,) + array.shape)
    array = np.squeeze(array)

    if np.max(array) > 1 or array.dtype == int:
        norm = 255
    else:
        norm = 1
    array = array.astype(float) / norm

    return array

def _load_image_orig(imname, flatten=False, resize=1.):
    im = scipy.misc.imread(imname, flatten=flatten)
    if len(im.shape) == 0:  # hello, Ubuntu + conda
        im = skimage.img_as_float(skimage.io.imread(imname, as_grey=flatten)).astype(np.float32)
    im = scipy.misc.imresize(im, resize).astype(np.float32)  # WARNING: rescales to [0,255]
    im /= 255.

    return im

def _load_image(filename, flatten=True, color=False, resize=1., interp_order=1):
    """
    Load an image converting from grayscale or alpha as needed.

    Adapted from
    `caffe <https://github.com/BVLC/caffe/blob/master/python/caffe/io.py>`_.

    :Args:
        filename (str)
    :Kwargs:
        - flatten (bool)
            flag for color format. True (default) loads as RGB while False
            loads as intensity (if image is already grayscale).
        - resize ()
    :Returns:
        An image with type np.float32 in range [0, 1]
        of size (H x W x 3) in RGB or of size (H x W x 1) in grayscale.
    """

    im = skimage.img_as_float(skimage.io.imread(filename, flatten=flatten)).astype(np.float32)

    if not flatten:
        if im.ndim == 2:
            im = im[:, :, np.newaxis]
            if color:
                im = np.tile(im, (1, 1, 3))
        elif im.shape[2] == 4:
            im = im[:, :, :3]

    if not isinstance(resize, (tuple, list, np.ndarray)):
        res = [resize, resize] + (im.ndim-2) * [1]
        resize = [s*r for s,r in zip(im.shape,res)]
    im = resize_image(im, resize, interp_order=interp_order)

    return im

def resize_image(im, new_dims, interp_order=1):
    """
    Resize an image array with interpolation.

    From
    `caffe <https://github.com/BVLC/caffe/blob/master/python/caffe/io.py>`_.

    :Args:
        - im (numpy.ndarray)
            (H x W x K) ndarray
        - new_dims (tuple)
            (height, width) tuple of new dimensions.
    :Kwargs:
        interp_order (int, default: 1)
            Interpolation order, default is linear.
    :Returns:
        Resized ndarray with shape (new_dims[0], new_dims[1], K)
    """
    if im.ndim == 2 or im.shape[-1] == 1 or im.shape[-1] == 3:
        im_min, im_max = im.min(), im.max()
        if im_max > im_min:
            # skimage is fast but only understands {1,3} channel images
            # in [0, 1].
            im_std = (im - im_min) / (im_max - im_min)
            resized_std = skimage.transform.resize(im_std, new_dims, order=interp_order)
            resized_im = resized_std * (im_max - im_min) + im_min
        else:
            # the image is a constant -- avoid divide by 0
            ret = np.empty((new_dims[0], new_dims[1], im.shape[-1]),
                           dtype=np.float32)
            ret.fill(im_min)
            return ret
    else:
        # ndimage interpolates anything but more slowly.
        scale = tuple(np.array(new_dims, dtype=float) / np.array(im.shape[:2]))
        resized_im = zoom(im, scale + (1,), order=interp_order)
    return resized_im.astype(np.float32)

def run(model_name='HMAX', impaths=None):
    if model_name in KNOWN_MODELS:
        m = KNOWN_MODELS[model_name]
        if m.__name__ == 'Caffe':
            m = m(model=model_name)
        else:
            m = m()
    else:
        raise Exception('ERROR: model {0} not recognized. '
                        'Choose from:\n {1}'.format(model_name,KNOWN_MODELS.keys()))


    #ims = [m.get_teststim(), m.get_teststim().T]
    if impaths is not None:
        try:
            ims = eval(impaths)
        except:
            ims = impaths
    #print ims
    print m.predict([ims], topn=5)
    #output = m.compare(ims)
    #return output


KNOWN_MODELS = {'px': Pixelwise, 'gaborjet': GaborJet, 'hmax99': HMAX99,
                'hmax': HMAX, 'hog': HOG, 'phog': PHOG, 'phow': PHOW,
                'randfilt': RandomFilters}

aliases_inv = {'vgg-19': ['vgg-19', 'VGG_ILSVRC_19_layers', 'VGG-19'],
           'places': ['places', 'Places205-CNN', 'Places'],
           'googlenet': ['googlenet', 'GoogleNet', 'GoogLeNet']}

ALIASES = dict([(v,k) for k, vals in aliases_inv.items() for v in vals])

CAFFE_NICE_NAMES = {'vgg-19': 'VGG-19',
                    'places': 'Places',
                    'googlenet': 'GoogLeNet'}

CAFFE_MODELS = {}
if HAS_CAFFE:
    caffe_models = {}
    modelpath = os.path.join(os.environ['CAFFE'], 'models')
    for path in os.listdir(modelpath):
        protocol = os.path.join(modelpath, path, '*deploy*.prototxt')
        fname = glob.glob(protocol)
        if len(fname) > 0:
            protocol = fname[0]
            try:
                with open(protocol, 'rb') as f:
                    name = f.readline().split(': ')[1].strip('"\n\r')
            except:
                pass
            else:
                if name in ALIASES:
                    basename = ALIASES[name]
                else:
                    basename = name.lower()
                    ALIASES[name] = basename
                    ALIASES[basename] = basename
                    aliases_inv[basename] = [basename, name]
                for n in aliases_inv[basename]:
                    caffe_models[n] = Caffe
                    CAFFE_MODELS[n] = os.path.join(modelpath, path)
    KNOWN_MODELS.update(caffe_models)


if __name__ == '__main__':
    if len(sys.argv) == 2:
        output = run(model_name=sys.argv[1])
    elif len(sys.argv) > 2:
        output = run(model_name=sys.argv[1], impaths=sys.argv[2])

    # np.save('results_%s.npy' % sys.argv[1], output)
