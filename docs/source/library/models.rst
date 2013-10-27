.. _models:

=============================================
Test stimuli with :mod:`~psychopy_ext.models`
=============================================

In many vision experiments, it is important to verify that the observed effects are not a mere outcome of some low-level image properties that are not related to the investigated effect. Several simple models have been used in the literature to rule out such alternative explanations, including computing pixel-wise differences between conditions (Op de Beeck et al., 2001), applying a simple model of V1, such as the GaborJet model (Lades et al., 1993), or applying a more complex model of the visual system, such as HMAX (Riesenhuber & Poggio, 1999). ``Psychopy_ext`` provides wrapper to these models so that they could be accessed with the same syntax, namely, by passing the filenames or numpy arrays of the images that should be compared. In case the screenshots of the exact displays (as seen by the parcipants) need to be processed, screen captures of the relevant conditions can be taken using Psychopy's :func:`~psychopy.visual.Window.getMovieFrame()` function.


Pixel-wise model: :class:`~psychopy_ext.models.Pixelwise`
---------------------------------------------------------

Pixel-wise differences model is the simplest model for estimating differences between images. Images are converted to grayscale and a Euclidean distance is then computed between all pairs of stimuli, resulting in an n-by-n dissimilarity matrix if there are n images.

GaborJet model: :class:`~psychopy_ext.models.GaborJet`
------------------------------------------------------

GaborJet model (Lades et al., 1993) belongs to the family of minimal V1-like models where image decomposition is performed by convolving an image with Gabor filters of different orientation and spatial frequency. In the GaborJet model, convolution is performed using 8 orientations (in the steps of 22.5 deg) and 5 spatial frequencies on a 10-by-10 grid in the Fourier domain. The output consists of the magnitude and phase of this convolution (arrays of 4000 elements), and the sampled grid positions. For comparing model outputs, only magnitudes are usually used (`Xu et al., 2009 <http://dx.doi.org/10.1016/j.visres.2009.08.021>`_). In psychopy_ext, the code has been implemented and verified in Python by following Xiaomin Yue's MatLab implementation available on `Irving Biederman's website <http://geon.usc.edu/GWTgrid_simple.m>`_.

HMAX model: :class:`~psychopy_ext.models.HMAX`
----------------------------------------------

HMAX model (`Riesenhuber & Poggio, 1999 <http://dx.doi.org/10.1038/14819>`_) has been proposed as a generic architecture of the visual cortex. It consists of four image processing layers and an output layer. Initially, a convolution between the image and Gabors of four orientations (in the steps of 45 deg) and 12 spatial frequencies (grouped into 4 channels) is computed. Next, a maximum of outputs of the same orientation over a local patch is taken. Outputs of this operation are pooled together in 256 distinct four-orientation configurations, and a final maximum across all these ##...## is computed. Thus, the output is an array with 256 elements. In ``psychopy_ext``, the code has been implemented and verified in Python by following the original 1999 code available on `Max Riesenhuber's website <http://riesenhuberlab.neuro.georgetown.edu/hmax/index.html#code>`_. 
