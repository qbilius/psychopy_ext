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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
# from __future__ import unicode_literals

import sys, os, glob, itertools, warnings, inspect, argparse, imp
import tempfile, shutil
import pickle
from collections import OrderedDict

import numpy as np
import scipy.ndimage
import pandas
import seaborn as sns

import matlab_wrapper
import sklearn.manifold
import sklearn.preprocessing, sklearn.metrics, sklearn.cluster
import skimage.feature, skimage.data

from psychopy_ext import stats, plot, report, utils


try:
    imp.find_module('caffe')
    HAS_CAFFE = True
except:
    try:
        os.environ['CAFFE']
        # put Python bindings in the path
        sys.path.insert(0, os.path.join(os.environ['CAFFE'], 'python'))
        HAS_CAFFE = True
    except:
        HAS_CAFFE = False

if HAS_CAFFE:
    # Suppress GLOG output for python bindings
    GLOG_minloglevel = os.environ.pop('GLOG_minloglevel', None)
    os.environ['GLOG_minloglevel'] = '5'

    import caffe
    from caffe.proto import caffe_pb2
    from google.protobuf import text_format

    HAS_CAFFE = True

    # Turn GLOG output back on for subprocess calls
    if GLOG_minloglevel is None:
        del os.environ['GLOG_minloglevel']
    else:
        os.environ['GLOG_minloglevel'] = GLOG_minloglevel


class Model(object):

    def __init__(self, model, labels=None, verbose=True, *args, **kwargs):
        self.name = ALIASES[model]
        self.nice_name = NICE_NAMES[model]
        self.safename = self.name
        self.labels = labels
        self.args = args
        self.kwargs = kwargs
        self.verbose = verbose

    def download_model(self, path=None):
        """Downloads and extracts a model

        :Kwargs:
            path (str, default: '')
                Where model should be extracted
        """
        self._setup()
        if self.model.model_url is None:
            print('Model {} is already available'.format(self.nice_name))
        elif self.model.model_url == 'manual':
            print('WARNING: Unfortunately, you need to download {} manually. '
                  'Follow the instructions in the documentation.'.format(self.nice_name))
        else:
            print('Downloading and extracting {}...'.format(self.nice_name))
            if path is None:
                path = os.getcwd()
                text = raw_input('Where do you want the model to be extracted? '
                                 '(default: {})\n'.format(path))
                if text != '': path = text

            outpath, _ = utils.extract_archive(self.model.model_url,
                                        folder_name=self.safename, path=path)
            if self.name == 'phog':
                with open(os.path.join(outpath, 'anna_phog.m')) as f:
                    text = f.read()
                with open(os.path.join(outpath, 'anna_phog.m'), 'wb') as f:
                    s = 'dlmwrite(s,p);'
                    f.write(text.replace(s, '% ' + s, 1))

            print('Model {} is available here: {}'.format(self.nice_name, outpath))
            print('If you want to use this model, either give this path when '
                  'calling the model or add it to your path '
                  'using {} as the environment variable.'.format(self.safename.upper()))

    def _setup(self):
        if not hasattr(self, 'model'):
            if self.name in CAFFE_MODELS:
                self.model = CAFFE_MODELS[self.name](model=self.name, *self.args, **self.kwargs)
            else:
                self.model = KNOWN_MODELS[self.name](*self.args, **self.kwargs)
            self.model.labels = self.labels
            self.isflat = self.model.isflat
            self.model.verbose = self.verbose

    def run(self, *args, **kwargs):
        self._setup()
        return self.model.run(*args, **kwargs)

    def train(self, *args, **kwargs):
        self._setup()
        return self.model.train(*args, **kwargs)

    def test(self, *args, **kwargs):
        self._setup()
        return self.model.test(*args, **kwargs)

    def predict(self, *args, **kwargs):
        self._setup()
        return self.model.predict(*args, **kwargs)

    def gen_report(self, *args, **kwargs):
        self._setup()
        return self.model.gen_report(*args, **kwargs)


class _Model(object):

    def __init__(self, labels=None):
        self.name = 'Model'
        self.safename = 'model'
        self.isflat = False
        self.labels = labels
        self.model_url = None

    def gen_report(self, test_ims, train_ims=None, html=None):
        print('input images:', test_ims)
        print('processing:', end=' ')
        if html is None:
            html = report.Report(path=reppath)
            html.open()
            close_html = True
        else:
            close_html = False

        resps = self.run(test_ims=test_ims, train_ims=train_ims)

        html.writeh('Dissimilarity', h=1)
        dis = dissimilarity(resps)
        plot_data(dis, kind='dis')
        html.writeimg('dis', caption='Dissimilarity across stimuli'
                     '(blue: similar, red: dissimilar)')

        html.writeh('MDS', h=1)
        mds_res = mds(dis)
        plot_data(mds_res, kind='mds', icons=test_ims)
        html.writeimg('mds', caption='Multidimensional scaling')

        if self.labels is not None:
            html.writeh('Linear separability', h=1)
            lin = linear_clf(dis, y)
            plot_data(lin, kind='linear_clf', chance=1./len(np.unique(self.labels)))
            html.writeimg('lin', caption='Linear separability')

        if close_html:
            html.close()

    def run(self, test_ims, train_ims=None, layers='output', return_dict=True):
        """
        This is the main function to run the model.

        :Args:
            test_ims (str, list, tuple, np.ndarray)
                Test images
        :Kwargs:
            - train_ims (str, list, tuple, np.ndarray)
                Training images
            - layers ('all'; 'output', 'top', None; str, int;
                      list of str or int; default: None)
                Which layers to record and return. 'output', 'top' and None
                return the output layer.
            - return_dict (bool, default: True`)
                Whether a dictionary should be returned. If False, only the last
                layer is returned as an np.ndarray.
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
        self.train_ims = im2iter(train_ims)

    def test(self, test_ims, layers='output', return_dict=True):
        """
        A placeholder for a function for testing a model.

        :Args:
            test_ims (str, list, tuple, np.ndarray)
                Test images
        :Kwargs:
            - layers ('all'; 'output', 'top', None; str, int;
                      list of str or int; default: 'output')
                Which layers to record and return. 'output', 'top' and None
                return the output layer.
            - return_dict (bool, default: True`)
                Whether a dictionary should be returned. If False, only the last
                layer is returned as an np.ndarray.
        """
        self.layers = layers
        # self.test_ims = im2iter(test_ims)

    def predict(self, ims, topn=5):
        """
        A placeholder for a function for predicting a label.
        """
        pass

    def _setup_layers(self, layers, model_keys):
        if self.safename in CAFFE_MODELS:
            filt_layers = self._filter_layers()
        else:
            filt_layers = model_keys

        if layers in [None, 'top', 'output']:
            self.layers = [filt_layers[-1]]
        elif layers == 'all':
            self.layers = filt_layers
        elif isinstance(layers, (str, unicode)):
            self.layers = [layers]
        elif isinstance(layers, int):
            self.layers = [filt_layers[layers]]
        elif isinstance(layers, (list, tuple, np.ndarray)):
            if isinstance(layers[0], int):
                self.layers = [filt_layers[layer] for layer in layers]
            elif isinstance(layers[0], (str, unicode)):
                self.layers = layers
            else:
                raise ValueError('Layers can only be: None, "all", int or str, '
                                 'list of int or str, got', layers)
        else:
            raise ValueError('Layers can only be: None, "all", int or str, '
                             'list of int or str, got', layers)

    def _fmt_output(self, output, layers, return_dict=True):
        self._setup_layers(layers, output.keys())
        outputs = [output[layer] for layer in self.layers]

        if not return_dict:
            output = output[self.layers[-1]]
        return output

    def _im2iter(self, ims):
        """
        Converts input into in iterable.

        This is used to take arbitrary input value for images and convert them to
        an iterable. If a string is passed, a list is returned with a single string
        in it. If a list or an array of anything is passed, nothing is done.
        Otherwise, if the input object does not have `len`, an Exception is thrown.
        """
        if isinstance(ims, (str, unicode)):
            out = [ims]
        else:
            try:
                len(ims)
            except:
                raise ValueError('input image data type not recognized')
            else:
                try:
                    ndim = ims.ndim
                except:
                    out = ims
                else:
                    if ndim == 1: out = ims.tolist()
                    elif self.isflat:
                        if ndim == 2: out = [ims]
                        elif ndim == 3: out = ims
                        else:
                            raise ValueError('images must be 2D or 3D, got %d '
                                             'dimensions instead' % ndim)
                    else:
                        if ndim == 3: out = [ims]
                        elif ndim == 4: out = ims
                        else:
                            raise ValueError('images must be 3D or 4D, got %d '
                                             'dimensions instead' % ndim)
        return out

    def load_image(self, *args, **kwargs):
        return utils.load_image(*args, **kwargs)

    def dissimilarity(self, resps, kind='mean_euclidean', **kwargs):
        return dissimilarity(resps, kind=kind, **kwargs)

    def mds(self, dis, ims=None, ax=None, seed=None, kind='metric'):
        return mds(dis, ims=ims, ax=ax, seed=seed, kind=kind)

    def cluster(self, *args, **kwargs):
        return cluster(*args, **kwargs)

    def linear_clf(self, resps, y, clf=None):
        return linear_clf(resps, y, clf=clf)


def plot_data(data, kind=None, **kwargs):
    if kind in ['dis', 'dissimilarity']:
        if isinstance(data, dict): data = data.values()[0]
        g = sns.heatmap(data, **kwargs)
    elif kind == 'mds':
        g = plot.mdsplot(data, **kwargs)
    elif kind in ['clust', 'cluster']:
        g = sns.factorplot('layer', 'dissimilarity', data=df, kind='point')
    elif kind in ['lin', 'linear_clf']:
        g = sns.factorplot('layer', 'accuracy', data=df, kind='point')
        if chance in kwargs:
            ax.axhline(kwargs['chance'], ls='--', c='.2')
    else:
        try:
            sns.factorplot(x='layers', y=data.columns[-1], data=data)
        except:
            raise ValueError('Plot kind "{}" not recognized.'.format(kind))
    return g

def dissimilarity(resps, kind='mean_euclidean', **kwargs):
    """
    Computes dissimilarity between all rows in a matrix.

    :Args:
        resps (numpy.array)
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
        A square NxN matrix, typically symmetric unless otherwise
        defined by the metric, and with NaN's in the diagonal.
    """
    if kind == 'mean_euclidean':
        dis_func = lambda x: sklearn.metrics.pairwise.pairwise_distances(x, metric='euclidean', **kwargs) / np.sqrt(x.shape[1])
    else:
        dis_func = lambda x: sklearn.metrics.pairwise.pairwise_distances(x, metric=kind, **kwargs)

    if isinstance(resps, (dict, OrderedDict)):
        dis = OrderedDict()
        for layer, resp in resps.items():
            dis[layer] = dis_func(resp)
            diag = np.diag_indices(dis[layer].shape[0])
            dis[layer][diag] = np.nan
    else:
        dis = dis_func(resps)
        dis[np.diag_indices(dis.shape[0])] = np.nan

    return dis

def mds(dis, ims=None, kind='metric', seed=None):
    """
    Multidimensional scaling

    :Args:
        dis
            Dissimilarity matrix
    :Kwargs:
        - ims
            Image paths
        - seed
            A seed if you need to reproduce MDS results
        - kind ({'classical', 'metric'}, default: 'metric')
            'Classical' is based on MATLAB's cmdscale, 'metric' uses
            :func:`~sklearn.manifold.MDS`.

    """
    df = []
    if ims is None:
        if isinstance(dis, dict):
            ims = map(str, range(len(dis.values()[0])))
        else:
            ims = map(str, range(len(dis)))
    for layer_name, this_dis in dis.items():
        if kind == 'classical':
            vals = stats.classical_mds(this_dis)
        else:
            mds_model = sklearn.manifold.MDS(n_components=2,
                            dissimilarity='precomputed', random_state=seed)
            this_dis[np.isnan(this_dis)] = 0
            vals = mds_model.fit_transform(this_dis)
        for im, (x,y) in zip(ims, vals):
            imname = os.path.splitext(os.path.basename(im))[0]
            df.append([layer_name, imname, x, y])

    df = pandas.DataFrame(df, columns=['layer', 'im', 'x', 'y'])
    # df = stats.factorize(df)
    # if self.layers != 'all':
    #     if not isinstance(self.layers, (tuple, list)):
    #         self.layers = [self.layers]
    #     df = df[df.layer.isin(self.layers)]
    # plot.mdsplot(df, ax=ax, icons=icons, zoom=zoom)
    return df

def cluster(resps, labels, metric=None, clust=None,
            bootstrap=True, stratified=False, niter=1000, ci=95, *func_args, **func_kwargs):
    if metric is None:
        metric = sklearn.metrics.adjusted_rand_score
    struct = labels if stratified else None
    n_clust = len(np.unique(labels))

    if clust is None:
        clust = sklearn.cluster.AgglomerativeClustering(n_clusters=n_clust, linkage='ward')
    df = []

    def mt(data, labels):
        labels_pred = clust.fit_predict(data)
        qual = metric(labels, labels_pred)
        return qual

    print('clustering...', end=' ')
    for layer, data in resps.items():
        labels_pred = clust.fit_predict(data)
        qualo = metric(labels, labels_pred)
        if bootstrap:
            pct = stats.bootstrap_resample(data1=data, data2=labels,
                    niter=niter, func=mt, struct=struct, ci=None,
                    *func_args, **func_kwargs)
            for i, p in enumerate(pct):
                df.append([layer, qualo, i, p])
        else:
            pct = [np.nan, np.nan]
            df.append([layer, qualo, 0, np.nan])

    df = pandas.DataFrame(df, columns=['layer', 'iter', 'bootstrap',
                                       'dissimilarity'])
    # df = stats.factorize(df)
    return df

def linear_clf(resps, y, clf=None):
    if clf is None: clf = sklearn.svm.LinearSVC
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
    # df = stats.factorize(df)
    return df


class Pixelwise(_Model):

    def __init__(self):
        """
        Pixelwise model

        The most simple model of them all. Uses pixel values only.
        """
        super(Pixelwise, self).__init__()
        self.name = 'Pixelwise'
        self.safename = 'px'

    def test(self, test_ims, layers='output', return_dict=False):
        self.layers = [self.safename]
        ims = self._im2iter(test_ims)
        resps = np.vstack([self.load_image(im).ravel() for im in ims])
        resps = self._fmt_output(OrderedDict([(self.safename, resps)]), layers,
                                 return_dict=return_dict)
        return resps


class Retinex(_Model):

    def __init__(self):
        """
        Retinex algorithm

        Based on A. Torralba's implementation presented at PAVIS 2014.

        .. warning:: Experimental
        """
        super(Retinex, self).__init__()
        self.name = 'Retinex'
        self.safename = 'retinex'

    def gen(self, im, thres=20./256, plot=True, save=False):
        im = self.load_image(im)

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


class Zoccolan(_Model):
    """
    Based on 10.1073/pnas.0811583106

    .. warning:: Not implemented fully
    """
    def __init__(self):
        super(Zoccolan, self).__init__()
        self.name = 'Zoccolan'
        self.safename = 'zoccolan'
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


class GaborJet(_Model):

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
                How many positions within an image to take:
                  - 0: grid of 10x10
                  - 1: grid of 12x12
                  - else: grid of imsize x imsize
        """
        super(GaborJet, self).__init__()
        self.name = 'GaborJet'
        self.safename = 'gaborjet'
        self.isflat = True
        self.nscales = nscales
        self.noris = noris
        self.imsize = imsize

        # generate the grid
        if grid_size == 0:
            s = imsize/128.
            rangeXY = np.arange(20*s, 110*s+1, 10*s) - 1  # 10x10
        elif grid_size == 1:
            s = imsize/128.
            rangeXY = np.arange(10*s, 120*s+1, 10*s) - 1  # 12x12
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
            test_ims: str or list of str
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
        mags = []
        phases = []
        imlist = self._im2iter(test_ims)
        for imno, im in enumerate(imlist):
            sys.stdout.write("\rRunning %s... %d%%" % (self.name,
                                                        100*imno/len(imlist)))
            sys.stdout.flush()
            im = self.load_image(im, resize=(self.imsize, self.imsize), flatten=True)
            mag, phase = self._test(im, cell_type=cell_type, sigma=sigma)
            mags.append(mag.ravel())
            phases.append(phase.ravel())
        sys.stdout.write("\rRunning %s... done\n" % self.name)

        output = OrderedDict([('phases', np.array(phases)),
                              ('magnitudes', np.array(mags))])
        output = self._fmt_output(output, layers, return_dict=return_dict)
        return output

    def _test(self, im, cell_type='complex', sigma=2*np.pi):
        # FFT of the image
        im_freq = np.fft.fft2(im)

        # setup the paramers
        kx_factor = 2 * np.pi / self.imsize
        ky_factor = 2 * np.pi / self.imsize

        # setup space coordinates
        xy = np.arange(-self.imsize/2, self.imsize/2).astype(float)
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

                # calculate magnitude and phase
                iconv = np.fft.ifft2(conv)
                # import ipdb; ipdb.set_trace()

                #eps = np.finfo(float).eps**(3./4)
                #real = np.real(iTmpFilterImage)
                #real[real<eps] = 0
                #imag = np.imag(iTmpFilterImage)
                #imag[imag<eps] = 0
                #iTmpFilterImage = real + 1j*imag

                ph = np.angle(iconv)
                ph = ph[self.rangeXY,:][:,self.rangeXY] + np.pi
                ind = scale*self.noris+ori
                phase[:,ind] = ph.ravel()

                if cell_type == 'complex':
                    mg = np.abs(iconv)
                    # get magnitude and phase at specific positions
                    mg = mg[self.rangeXY,:][:,self.rangeXY]
                    mag[:,ind] = mg.ravel()
                else:
                    mg_real = np.real(iconv)
                    mg_imag = np.imag(iconv)
                    # get magnitude and phase at specific positions
                    mg_real = mg_real[self.rangeXY,:][:,self.rangeXY]
                    mg_imag = mg_imag[self.rangeXY,:][:,self.rangeXY]
                    mag[:,ind] = mg_real.ravel()
                    mag[:,nvars+ind] = mg_imag.ravel()

        # use magnitude for dissimilarity measures
        return mag, phase

    def dissimilarity(self, kind='cosine', *args, **kwargs):
        """
        Default dissimilarity for :class:`GaborJet` is `cosine`.
        """
        return super(GaborJet, self).dissimilarity(kind=kind, *args, **kwargs)


class HMAX99(_Model):
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

    The output was tested against the Pure MatLab output which can be tested
    against the Standard C/MATLAB code featured at `Riesenhuber's lab
    <http://riesenhuberlab.neuro.georgetown.edu/hmax/index.html#code>`_.

    .. note:: This implementation is not the most current HMAX
              implementation that doesn't rely on hardcoding features anymore
              (e.g., Serre et al., 2007). Use :class:`HMAX_HMIN` or :class:`HMAX_PNAS` to access MATLAB
              interface to a more current version of HMAX.

    .. note:: Images are resized to 256 x 256 as required by the original
              implementation

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
        super(HMAX99, self).__init__()
        self.name = "HMAX'99"
        self.safename = 'hmax99'
        self.isflat = True

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
            raise ValueError("filter type not recognized")

        self.istrained = False  # initially VTUs are not set up

    def train(self, train_ims):
        """
        Train the model

        That is, supply view-tuned units (VTUs) with C2 responses to
        'prototype' images, to which these VTUs will be maximally tuned.

        :Args:
            train_ims (str, list, tuple, np.ndarray)
                Training images
        """
        try:
            self.tuning = pickle.load(open(train_ims,'rb'))
            print('done')
        except:
            self.tuning = self.test(train_ims, op='training', layers='C2',
                                    return_dict=False)
        self.istrained = True

    def test(self, test_ims, op='testing', layers='output', return_dict=True):
        """
        Test the model on the given image

        :Args:
            test_ims (str, list, tuple, np.ndarray)
                Test images.

        """
        ims = self._im2iter(test_ims)
        # Get number of filter sizes
        out = OrderedDict()

        size_S1 = sum([len(fs) for fs in self.filter_sizes_all])
        S1 = np.zeros((256, 256, size_S1, self.n_ori))
        out['C1'] = np.zeros((len(ims), 256, 256, self.n_ori,
                                len(self.filter_sizes_all)))
        # S2 has an irregular shape which depends on the spatial frequency band
        S2 = []
        C2_tmp = np.zeros(((self.S2_config[0]*self.S2_config[1])**self.n_ori,
                            len(self.filter_sizes_all)))
        out['C2'] = np.zeros((len(ims), C2_tmp.shape[0]))

        for imno, im in enumerate(ims):
            # im *= 255
            sys.stdout.write("\rRunning HMAX'99... %s: %d%%" %(op, 100*imno/len(ims)))
            sys.stdout.flush()
            im = self.load_image(im, flatten=True, resize=(256,256))

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
                out['C1'][imno, ..., which_band] = C1_tmp
                S2_tmp = self.get_S2(C1_tmp, which_band)
                S2.append(S2_tmp)
                C2_tmp[:, which_band] = self.get_C2(S2_tmp, which_band)
            out['C2'][imno] = np.max(C2_tmp, -1) # max over all scale bands

        # calculate VTU if trained
        if self.istrained:
            out['VTU'] = self.get_VTU(out['C2'])

        sys.stdout.write("\rRunning HMAX'99... %s: done\n" %op)
        output = self._fmt_output(out, layers, return_dict=return_dict)
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
        .. warning:: Does not pass unittest, meaning that the outputs differ
        slightly from MATLAB implementation. I recommend not using this option.

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
            origin = -(C1_pooling // 2)
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
        S2_shift = int(np.ceil(self.C1_pooling_all[which_band] / 2.))
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
        S2 = np.sum((S2_permute-target)**2, 3)
        S2 = np.exp(-S2 / (2. * sigma**2))

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


class HOG(_Model):

    def __init__(self):
        super(HOG, self).__init__()
        self.name = 'HOG'
        self.safename = 'hog'
        self.isflat = True

    def test(self, test_ims, layers='output', return_dict=False, **kwargs):
        """
        You can also pass keyworded arguments to skimage.feature.hog.
        """
        self.layers = [self.safename]
        resps = []
        ims = self._im2iter(test_ims)
        for imno, im in enumerate(ims):
            sys.stdout.write("\rRunning %s... %d%%" % (self.name, 100*imno/len(ims)))
            sys.stdout.flush()
            im = self.load_image(im, flatten=True)
            resps.append(skimage.feature.hog(im, **kwargs))
        resps = np.array(resps)

        sys.stdout.write("\rRunning %s... done\n" % self.name)
        sys.stdout.flush()

        if kwargs.get('visualise'):
            resps, hog_image = resps

        output = self._fmt_output(OrderedDict([(self.safename, resps)]), layers,
                                  return_dict=return_dict)
        return output


class Caffe(_Model):

    def __init__(self, model='caffenet', mode='gpu', weight_file=None):
        super(Caffe, self).__init__()

        if model in ALIASES:
            self.safename = ALIASES[model]
        else:
            self.safename = model.lower()

        if self.safename in NICE_NAMES:
            self.name = NICE_NAMES[self.safename]
        else:
            self.name = model

        # self.model_path = model_path  # will be updated when self.test is called
        # self.model_file = self.model_file
        self.weight_file = weight_file

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

        if self.weight_file is not None:
            if '.caffemodel' in self.weight_file:
                model_path = os.path.dirname(self.weight_file)
            else:
                model_path = self.weight_file
        else:
            try:
                model_path = CAFFE_PATHS[self.safename]
            except:
                raise Exception('Model %s not recognized. Please provide '
                                'weight_file when calling this model.' %
                                self.name)
            path = os.path.join(model_path, '*.caffemodel')
            self.weight_file = sorted(glob.glob(path))[0]

        # self.model_path =
        path = os.path.join(model_path, '*deploy*.prototxt')
        self.model_file = sorted(glob.glob(path))[0]
        print('model parameters loaded from', self.model_file)

    def layers_from_prototxt(self, keep=['Convolution', 'InnerProduct']):
        self._set_paths()
        net = caffe_pb2.NetParameter()
        model_file = glob.glob(os.path.join(self.model_file,
                               '*deploy*.prototxt'))[0]
        text_format.Merge(open(model_file).read(), net)
        layers = []
        for layer in net.layer:
            if layer.type in keep:
                filt_layers.append(layer.name)
        return layers

    def _filter_layers(self, keep=['Convolution', 'InnerProduct']):
        layers = []
        for name, layer in zip(self.net._layer_names, self.net.layers):
            if layer.type in keep:
                if name in self.net.blobs:
                    layers.append(name)
                else:
                    raise Exception('Layer %s not accessible' % name)

        return layers

    def train(self, *args, **kwargs):
        raise NotImplemented

    def _classifier(self):
        self._set_paths()
        mn = self._read_mean()
        net = caffe.Net(self.model_file, self.weight_file, caffe.TEST)
        self.transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_mean('data', mn.mean(1).mean(1)) # mean pixel
        self.transformer.set_raw_scale('data', 255)  # the reference model operates on images in [0,255] range instead of [0,1]
        self.transformer.set_channel_swap('data', (2,1,0))  # the reference model has channels in BGR order instead of RGB

        return net

    def test(self, ims, layers='output', return_dict=True, filt_layers=True):
        if not hasattr(self, 'net'):
            self.net = self._classifier()

        ims = self._im2iter(ims)
        new_shape = (1, ) + self.net.blobs['data'].data.shape[1:]
        self.net.blobs['data'].reshape(*new_shape)
        output = OrderedDict()
        self._setup_layers(layers, self.net.blobs.keys())

        for layer in self.layers:
            sh = self.net.blobs[layer].data.shape[1:]
            if not isinstance(sh, tuple):
                sh = (sh,)
            output[layer] = np.zeros((len(ims),) + sh)

        for imno, im in enumerate(ims):
            if self.verbose:
                sys.stdout.write("\rRunning %s... %d%%" % (self.name, 100*imno/len(ims)))
                sys.stdout.flush()
            im = self.load_image(im, color=True)
            out = self._test(im)
            for layer in self.layers:
                output[layer][imno] = self.net.blobs[layer].data

        if self.verbose:
            sys.stdout.write("\rRunning %s... done\n" % self.name)
            sys.stdout.flush()

        if filt_layers:
            output = self._fmt_output(output, layers, return_dict=return_dict)
        elif not return_dict:
            output = output[self.layers[-1]]
        return output

    def _test(self, im):
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', im)
        out = self.net.forward()
        return out

    def confidence(self, ims, topn=1):
        preds = self.test(ims, layers='prob', return_dict=False, filt_layers=False)
        if topn is not None:
            preds.sort(axis=1)
            return np.squeeze(preds[:, ::-1][:topn])
        else:
            return preds

    def predict(self, ims, topn=5):
        labels = self._get_labels()
        preds = self.confidence(ims, topn=None)
        out = []

        for pred in preds:
            classno = np.argsort(pred)[::-1][:topn]
            tmp = []
            for n in classno:
                d = {'classno': n,
                     'synset': labels[n][0],
                     'label': labels[n][1],
                     'confidence': pred[n]}
                tmp.append(d)
            out.append(tmp)
        return out

    def _read_mean(self):
        model_path = os.path.dirname(self.weight_file)
        meanf = os.path.join(model_path, '*mean*.binaryproto')
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
        synset_file = os.path.join(os.environ['CAFFE'],
                                   'data/ilsvrc12/synset_words.txt')
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


class MATLABModel(_Model):

    def __init__(self, model_path=None, matlab_root=None):
        """
        A base class for making an interface to MATLAB-based models
        """
        super(MATLABModel, self).__init__()
        self.model_path = model_path
        self.matlab_root = matlab_root

    def _set_model_path(self):
        if self.model_path is None:
            try:
                safename = getattr(self, 'safename')
            except:
                safename = self.name
            finally:
                safename = safename.upper()
            if safename not in os.environ:
                raise Exception('Please specify model_path to the location of '
                                '%s or add it to your path '
                                'using %s as the environment variable.' %
                                (self.name, safename))
            else:
                self.model_path = os.environ[safename]

    def test(self, test_ims, layers='output', return_dict=True):
        self._set_model_path()

        matlab = matlab_wrapper.MatlabSession(options='-nodisplay -nosplash', matlab_root=self.matlab_root)
        matlab.eval('addpath %s' % self.model_path)

        resps = self._test(matlab, test_ims, layers=layers)

        sys.stdout.write("\rRunning %s... done\n" % self.name)
        sys.stdout.flush()

        matlab.eval('rmpath %s' % self.model_path)
        del matlab

        resps = self._fmt_output(resps, layers,
                                 return_dict=return_dict)
        return resps


class HMAX_HMIN(MATLABModel):

    def __init__(self, model_path=None, matlab_root=None):
        """
        The minimal implementation of HMAX (aka hmin).

        This is a simple reference implementation of HMAX that only provides
        output of the C2 layer.

        1. The model can be downloaded from
        `here <http://cbcl.mit.edu/jmutch/hmin/>`_.
        2. Compile it by running ``matlab -r "mex example.cpp"``.
        3. You will have to add ``HMAX_HMIN`` variable that is pointing to the directory where you extracted the model to your path so that it could be found. an easy way to do it permanently is to add the following line in your ``.bashrc`` file::

            export HMAX_HMIN=<path to the model>

        """
        super(HMAX_HMIN, self).__init__(model_path=model_path,
                                        matlab_root=matlab_root)
        self.name = 'HMAX-HMIN'
        self.safename = 'hmax_hmin'
        self.isflat = True
        self.model_url = 'manual'

    def _test(self, matlab, test_ims, **kwargs):
        resps = []
        ims = self._im2iter(test_ims)
        for imno, im in enumerate(ims):
            sys.stdout.write("\rRunning %s... %d%%" % (self.name,
                                                       100*imno/len(ims)))
            sys.stdout.flush()
            im = self.load_image(im, flatten=True)
            matlab.put('im', im)
            matlab.eval("example_run(im)")
            resps.append(matlab.get('ans'))

        return OrderedDict([('C2', np.array(resps))])


class HMAX_PNAS(MATLABModel):

    def __init__(self, model_path=None, matlab_root=None):
        """
        HMAX implementation by Serre et al. (2007)

        Installation:

        1. `Download FHLib <http://www.mit.edu/~jmutch/fhlib/>`_.
        2. Compile it by opening MATLAB, setting paths via ``run GLSetPath``, and running ``GLCompile``.
        3. `Download PNAS version of HMAX <http://cbcl.mit.edu/software-datasets/pnas07/index.html>`_, place it in FHLib such that ``pnas_package`` folder is directly inside FHLib folder.
        4. You will have to add ``HMAX_PNAS`` variable that is pointing to the directory where you extracted the model to your path so that it could be found. an easy way to do it permanently is to add the following line in your ``.bashrc`` file::

            export HMAX_PNAS=<path to fullModel directory>

        (you don't need the `SVMlight package <http://svmlight.joachims.org>`_ or `animal/non-animal dataset <http://cbcl.mit.edu/software-datasets/serre/SerreOlivaPoggioPNAS07/index.htm>`_)

        """
        super(HMAX_PNAS, self).__init__(model_path=model_path,
                                        matlab_root=matlab_root)
        self.name = 'HMAX-PNAS'
        self.safename = 'hmax_pnas'
        self.model_url = 'manual'
        # self.layer_sizes = {'C1': 78344, 'C2': 3124000, 'C2b': 2000, 'C3': 2000}
        self.layer_sizes = {'C1': 78344, 'C2': 3124000, 'C2b': 2000, 'C3': 2000}

    def train(self, **kwargs):
        """
        Train the model

        .. note:: You must modify the script 'featuresNatural_newGRBF.m' that is
                  placed in the model path. Specifically, BASE_PATH variable
                  must be defined, and IMAGE_DIR must point to your training
                  images.
        """
        self._set_model_path()

        matlab = matlab_wrapper.MatlabSession(options='-nojvm -nodisplay -nosplash')
        matlab.eval('addpath %s' % self.model_path)
        # base_path = os.path.dirname(train_ims[0])
        # matlab.put('BASE_PATH', base_path)

        sys.stdout.write("\Training %s..." % self.name)
        sys.stdout.flush()

        matlab.eval("run featuresNatural_newGRBF")

        sys.stdout.write("\Training %s... done\n" % self.name)
        sys.stdout.flush()

        matlab.eval('rmpath %s' % self.model_path)
        del matlab


    def _test(self, matlab, test_ims, layers=['C3'], nfeatures=2000, **kwargs):
        """
        Test model

        Note that S layers are inaccesible because at least S1 and S2 are
        massive.
        """
        if layers == 'all':
            self.layers = ['C1', 'C2', 'C2b', 'C3']
        elif layers in [None, 'top', 'output']:
            self.layers = ['C3']
        else:
            self.layers = layers

        gl_path = os.path.normpath(os.path.join(self.model_path, '../../../GLSetPath'))
        matlab.eval("run %s;" % gl_path)
        matlab.eval("config = FHConfig_PNAS07;")
        matlab.eval("config = FHSetupConfig(config);")
        matlab.put('featureFile', os.path.join(self.model_path,
                   'featureSets/featuresNatural_newGRBF.mat'))
        matlab.eval("load(featureFile, 'lib');")

        ims = self._im2iter(test_ims)
        resps = OrderedDict([(l, np.zeros((len(ims), nfeatures))) for l in self.layers])
        sel = OrderedDict([(l, np.random.choice(self.layer_sizes[l], size=nfeatures, replace=False)) for l in self.layers])

        for imno, im in enumerate(ims):
            sys.stdout.write("\rRunning %s... %d%%" % (self.name,
                                                       100*imno/len(ims)))
            sys.stdout.flush()
            if not isinstance(im, (str, unicode)):
                f = array2tempfile(im)
                name = f.name
            else:
                name = im
            impath = os.path.join(os.getcwd(), name)
            matlab.eval("stream = FHCreateStream(config, lib, '%s', 'all');" % impath)
            if not isinstance(im, (str, unicode)): f.close()

            for layer, resp in resps.items():
                matlab.eval("resps = FHGetResponses(config, lib, stream, '%s');" % layer.lower())
                resp[imno] = matlab.get('resps')[sel[layer]]

        return resps


class PHOG(MATLABModel):

    def __init__(self, model_path=None, matlab_root=None):
        """
        Pyramid Histogram of Oriented Gradients

        The model can be downloaded from `here <http://www.robots.ox.ac.uk/~vgg/research/caltech/phog.html>`_. You will have to add ``PHOG`` variable that is pointing to the directory where you extracted the model to your path so that it could be found. an easy way to do it permanently is to add the following line in your ``.bashrc`` file::

            export PHOG=<path to the model>

        Reference:

        `Bosch A., Zisserman A., Munoz X. Representing shape with a spatial pyramid kernel. CIVR 2007 <http://dx.doi.org/10.1145/1282280.1282340>`_
        """
        super(PHOG, self).__init__(model_path=model_path,
                                   matlab_root=matlab_root)
        self.name = 'PHOG'
        self.safename = 'phog'
        self.model_url = 'http://www.robots.ox.ac.uk/~vgg/research/caltech/phog/phog.zip'

    def _test(self, matlab, test_ims, nbins=8, angle=360, nlayers=3, **kwargs):
        if angle not in [180, 360]:
            raise Exception('PHOG angle must be either 180 or 360')
        ims = self._im2iter(test_ims)
        im = self.load_image(ims[0])
        roi = np.mat([1, im.shape[0], 1, im.shape[1]]).T
        matlab.put('roi', roi)
        args = '{}, {}, {}, {}'.format(nbins, angle, nlayers, 'roi')

        resps = []
        for imno, im in enumerate(ims):
            sys.stdout.write("\rRunning %s... %d%%" % (self.name,
                                                       100*imno/len(ims)))
            sys.stdout.flush()
            if not isinstance(im, (str, unicode)):
                f = array2tempfile(im)
                name = f.name
            else:
                name = im
            impath = os.path.join(os.getcwd(), name)
            matlab.eval("anna_phog('%s', %s)" % (impath, args))
            sys.stdout.flush()
            resps.append(matlab.get('ans'))
            if not isinstance(im, (str, unicode)): f.close()

        return OrderedDict([(self.safename, np.array(resps))])


class PHOW(MATLABModel):

    def __init__(self, model_path=None, matlab_root=None):
        """
        Pyramid Histogram of Visual Words / Spatial Pyramid Matching

        The model can be downloaded from
        `here <http://slazebni.cs.illinois.edu/research/SpatialPyramid.zip>`_. You will have to add ``PHOW`` variable that is pointing to the directory where you extracted the model to your path so that it could be found. an easy way to do it permanently is to add the following line in your ``.bashrc`` file::

            export PHOW=<path to the model>

        Note that all images must be in the same folder. If you can't do it,
        consider passing an array of loaded images rather than filenames. These
        images will then be temporarily saved in a single folder.

        Reference:

        `Lazebnik, S., Schmid, C, Ponce, J. Beyond Bags of Features: Spatial Pyramid Matching for Recognizing Natural Scene Categories. *CVPR 2006*. <http://slazebni.cs.illinois.edu/publications/cvpr06b.pdf>`_
        """
        super(PHOW, self).__init__(model_path=model_path,
                                   matlab_root=matlab_root)
        self.name = 'PHOW'
        self.safename = 'phow'
        self.model_url = 'http://slazebni.cs.illinois.edu/research/SpatialPyramid.zip'

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
        test_ims = self._im2iter(test_ims)

        ims = []
        dirs = []
        fs = []
        for im in test_ims:
            if not isinstance(im, (str, unicode)):
                f = array2tempfile(im)
                name = f.name
                fs.append(f)
            else:
                name = im
            ims.append(os.path.basename(name))
            dirs.append(os.path.dirname(name))

        if not all([d==dirs[0] for d in dirs]):
            raise Exception('All images must be in the same folder')
        matlab.put('ims', ims)
        image_dir = dirs[0]

        data_dir = tempfile.mkdtemp()
        funcstr = "BuildPyramid(ims, '%s', '%s', params, %d, %d)" % (image_dir, data_dir, can_skip, save_sift)

        sys.stdout.write("\rRunning %s..." % self.name)
        sys.stdout.flush()

        matlab.eval(funcstr)
        resps = matlab.get('ans')
        shutil.rmtree(data_dir)
        for f in fs: f.close()

        return OrderedDict([(self.safename, resps)])


class RandomFilters(MATLABModel):

    def __init__(self, model_path=None, matlab_root=None):
        """
        Random Features and Supervised Classifier

        The model can be downloaded from
        `here <http://cs.nyu.edu/~koray/publis/code/randomc101.tar.gz`_

        Reference:

        `Jarrett, K., Kavukcuoglu, K., Ranzato, M., & LeCun, Y. What is the Best Multi-Stage Architecture for Object Recognition?, *ICCV 2009 <http://cs.nyu.edu/~koray/publis/jarrett-iccv-09.pdf>`_
        """

        super(RandomFilters, self).__init__(model_path=model_path,
                                            matlab_root=matlab_root)
        self.name = 'RandomFilters'
        self.safename = 'randfilt'
        self.isflat = True
        self.model_url = 'http://cs.nyu.edu/~koray/publis/code/randomc101.tar.gz'

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

        matlab.eval('addpath %s' % os.path.join(self.model_path, 'code'))

        param_path = os.path.join(self.model_path, 'data/params.mat')
        matlab.eval("params = load('%s');" % param_path)
        matlab.eval('params.kc.layer1 = -0.11 + 0.22 * rand(size(params.ct.layer1,1),9,9);')
        matlab.eval('params.kc.layer2 = -0.11 + 0.22 * rand(size(params.ct.layer2,1),9,9);')

        funcstr = "extractRandomFeatures(pim, params.ker, params.kc, params.ct, params.bw, params.bs);"
        resps = []
        ims = self._im2iter(test_ims)
        for imno, im in enumerate(ims):
            sys.stdout.write("\rRunning %s... %d%%" % (self.name,
                                                       100*imno/len(ims)))
            sys.stdout.flush()
            im = self.load_image(im, flatten=True)
            mx = max(im.shape[:2])
            sh = (151./mx * im.shape[0], 151./mx * im.shape[1])
            imr = np.round(utils.resize_image(im, sh) * 255)
            matlab.put('imr', imr)
            matlab.eval('pim = imPreProcess(imr,params.ker);')
            matlab.eval(funcstr)
            resp = matlab.get('ans')
            resps.append(resp.ravel())

        matlab.eval('rmpath %s' % os.path.join(self.model_path, 'code'))

        return OrderedDict([(self.safename, np.array(resps))])


def get_teststim(flatten=False):
    """
    Returns a cat image. If `flatten == True`, returns a gray scale version
    of it. Note the the image is not grayscaled on the fly but rather loaded
    from the disk.

    Test image: `CC0 license - stormbringerser
    <https://pixabay.com/en/cat-animal-cute-pet-feline-kitty-618470/>`_.
    """
    path = os.path.dirname(__file__)
    if flatten:
        im = utils.load_image(os.path.join(path, 'tests/cat-gray.png'))
    else:
        im = utils.load_image(os.path.join(path, 'tests/cat.png'))
    return im

def array2tempfile(im):
    import tempfile
    f = tempfile.NamedTemporaryFile()
    scipy.misc.imsave(f, im, format='png')
    return f

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
    print(m.predict([ims], topn=5))
    #output = m.compare(ims)
    #return output

def _detect_args(mfunc, *args, **kwargs):

    var = inspect.getargspec(mfunc)
    if len(args) > 0:
        if var.varargs is None:
            args = args[:-len(var.defaults)]
    if len(kwargs) > 0:
        if var.keywords is None:
            if var.defaults is None:
                kwargs = {}
            else:
                kwargs = {k:kwargs[k] for k in var.args[-len(var.defaults):] if k in kwargs}
    return args, kwargs

def get_model(model_name, *args, **kwargs):
    if model_name in CAFFE_MODELS:
        ModelClass = Caffe
        kwargs['model'] = model_name
    elif model_name in MATLAB_MODELS:
        ModelClass = MATLAB_MODELS[model_name]
    elif model_name in KNOWN_MODELS:
        ModelClass = KNOWN_MODELS[model_name]
    else:
        raise ValueError('model {} not recognized'.format(model_name))
    margs, mkwargs = _detect_args(ModelClass.__init__, *args, **kwargs)
    m = ModelClass(*margs, **mkwargs)
    return m

def get_model_from_obj(obj, *args, **kwargs):
    if isinstance(obj, (str, unicode)):
        try:
            m = get_model(obj)
        except:
            pass
    else:
        try:
            margs, mkwargs = _detect_args(obj.__init__, *args, **kwargs)
            m = obj(*margs, **mkwargs)
        except:
            m = obj
    return m

def compare(func, test_ims, layer='output', models=None,
            plot=True, save=False, html=None,
            *func_args, **func_kwargs):

    if models is None: models = KNOWN_MODELS.keys()

    df = []
    mnames = []
    for model in models:
        m = Model(model)
        resps = m.run(test_ims=test_ims, layers=layer)
        out = func(resps, *func_args, **func_kwargs)
        df.append(pandas.DataFrame(out))
        mnames.extend([m.name]*len(out))

    df = pandas.concat(df)

    df.insert(0, 'model', mnames)
    if plot:
        sns.factorplot(x='model', y=df.columns[-1], data=df, kind='bar')
        if html is not None:
            html.writeimg(str(func))
        else:
            sns.plt.show()
    if save: df.to_csv(df)
    return df

def gen_report(models, test_ims, path='', labels=None):
    html = report.Report(path=path)
    html.open()

    if len(labels) == 0: labels = np.arange(len(test_ims))

    # for model in models:
    #     m = Model(model)
    #     m.gen_report(html=html, test_ims=impath)
    # compare(dissimilarity, html=html)
    # compare(mds, html=html)
    html.writeh('Linear classification', h=1)
    compare(linear_clf, test_ims, models=models, html=html, y=labels)

    html.writeh('Clustering', h=1)
    compare(cluster, test_ims, models=models, html=html, labels=labels)
    # compare(linear_clf, html=html)
    html.close()

class Compare(object):

    def __init__(self):
        pass

    def compare(self, func, models=None, plot=False, *func_args, **func_kwargs):
        if models is None:
            models = KNOWN_MODELS.keys() + [Caffe() for m,name in CAFFE_MODELS]
        # elif models

        df = []
        for model in models:
            m = get_model_from_obj(model)
            out = getattr(m, func)(*func_args, **func_kwargs)
            out = pandas.DataFrame(out)
            for rno, v in out.iterrows():
                df.append([m.name] + v.values.tolist())

        df = pandas.DataFrame(df, columns=['model'] + out.columns.tolist())
        if plot:
            self.plot(df)
        return df

    def get_value_from_model_name(self, name, func, *args, **kwargs):
        imodel = get_model_from_obj(name)
        f = getattr(imodel, func)
        data = f(*args, **kwargs)
        return data

    def pairwise_stats(self, models1, models2=None, func=None,
                       bootstrap=True, plot=False, niter=1000, ci=95,struct=None,
                       *func_args, **func_kwargs):
        if models2 is None:
            models2 = models1

        if func is None:
            func = stats.corr

        if bootstrap:
            print('bootstrapping...')

        df = []
        for name1, data1 in models1.items():
            for layer1, d1 in data1.items():
                for name2, data2 in models2.items():
                    for layer2, d2 in data2.items():
                        c = func(d1, d2, *func_args, **func_kwargs)
                        if bootstrap:

                            pct = stats.bootstrap_resample(d1, data2=d2,
                                func=func, niter=niter, ci=ci, struct=struct, *func_args, **func_kwargs)
                        else:
                            pct = (np.nan, np.nan)
                        df.append([name1, layer1, name2, layer2,
                                   c, pct[0], pct[1]])

        cols = ['model1', 'layer1','model2', 'layer2', 'correlation',
                'ci_low', 'ci_high']
        df = pandas.DataFrame(df, columns=cols)

        # if plot:
            # self.plot_corr(df)
        return df

    def corr(self, data1, data2=None, func=None, **kwargs):
        return self.pairwise_stats(data1, models2=data2, func=stats.corr, **kwargs)

def _get_model_from_str(name):
    return ALIASES[name.lower()]

MATLAB_MODELS = {'hmax_hmin': HMAX_HMIN, 'hmax_pnas': HMAX_PNAS,
                 'phog': PHOG, 'phow': PHOW, 'randfilt': RandomFilters}

KNOWN_MODELS = {'px': Pixelwise, 'gaborjet': GaborJet, 'hmax99': HMAX99,
                'hog': HOG}
KNOWN_MODELS.update(MATLAB_MODELS)

aliases_inv = {'px': ['px', 'pixelwise', 'pixel-wise'],
               'gaborjet': ['gaborjet', 'gj'],
               'vgg-19': ['vgg-19', 'VGG_ILSVRC_19_layers', 'VGG-19'],
               'places': ['places', 'Places205-CNN', 'Places'],
               'googlenet': ['googlenet', 'GoogleNet', 'GoogLeNet'],
               'cifar10': ['cifar10', 'CIFAR10_full_deploy']}


ALIASES = {v:k for k, vals in aliases_inv.items() for v in vals}
ALIASES.update({k:k for k in KNOWN_MODELS})

NICE_NAMES = {'px': 'Pixelwise',
              'gaborjet': 'GaborJet',
              'retinex': 'Retinex',
              'hog': 'HOG',
              'hmax99': "HMAX'99",
              'hmax_hmin': 'HMAX-HMIN',
              'hmax_pnas': 'HMAX-PNAS',
              'phog': 'PHOG',
              'phow': 'PHOW',
              'randfilt': 'Random Filters',
              'alexnet': 'AlexNet',
              'caffenet': 'CaffeNet',
              'vgg-19': 'VGG-19',
              'places': 'Places',
              'googlenet': 'GoogLeNet',
              'cifar10': 'CIFAR10',
              'resnet-152': 'ResNet-152'}

CAFFE_MODELS = {}
CAFFE_PATHS = {}
if HAS_CAFFE:
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
                CAFFE_MODELS[basename] = Caffe
                CAFFE_PATHS[basename] = os.path.dirname(protocol)
    KNOWN_MODELS.update(CAFFE_MODELS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('task', choices=['run', 'dissimilarity', 'mds', 'cluster', 'linear_clf', 'report'])
    parser.add_argument('-m', '--models', nargs='*', default=['gaborjet'])
    parser.add_argument('-i', '--impath', default=None)
    parser.add_argument('-o', '--output', default=None)
    parser.add_argument('--labels', nargs='*')
    parser.add_argument('-p', '--plot', action='store_true', default=True)
    parsed = parser.parse_args()

    if parsed.impath is None:
        parsed.impath = os.getcwd()
    if parsed.output is None:
        parsed.output = os.getcwd()

    ims = sorted(glob.glob(os.path.join(parsed.impath, '*.*')))

    if parsed.task == 'report':
        gen_report(models=parsed.models, test_ims=ims, path=parsed.output,
                   labels=parsed.labels)
    else:
        m = Model(model=parsed.models[0])
        outputs = m.run(test_ims=ims)
        if parsed.task != 'run':
            func = globals()[parsed.task]
            if parsed.task in ['cluster', 'linear_clf']:
                df = func(outputs, labels=parsed.labels)
            elif parsed.task == 'mds':
                dis = dissimilarity(outputs)
                df = mds(dis)
            else:
                df = func(outputs)

            if parsed.plot:
                if parsed.task == 'mds':
                    plot_data(df, kind=parsed.task, icons=ims)
                else:
                    plot_data(df, kind=parsed.task)
                sns.plt.show()
            if parsed.output is not None:
                fname = os.path.join(parsed.output, parsed.task + '.pkl')
                pickle.dump(df, open(fname, 'wb'))
