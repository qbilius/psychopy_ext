from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from setuptools import setup
from distutils.version import LooseVersion
import sys


description = ('A framework for a rapid reproducible experimental design, '
               'analysis and modeling of data in neuroscience and psychology.')
exec(open('psychopy_ext/version.py').read())

# required pip-installable packages
pip_reqs = []
# required non-pip-installable packages
reqs = []
# recommended non-pip-installable packages
recs = [('psychopy', '1.83.4', 'experiments'),
        ('pandas', '0.17', 'general purposes'),
        ('seaborn', '0.7', 'plotting'),
        ('nibabel', None, 'fMRI analyses'),
        ('h5py', None, 'fMRI analyses'),
        ('mvpa2', '2.3.1', 'fMRI analyses'),
        ('sklearn', None, 'models'),
        ('skimage', None, 'models'),
        ('caffe', None, 'models'),
        ('docutils', None, 'experiments'),
        ('svgwrite', None, 'experiments'),
        ('matlab_wrapper', None, 'models')
        ]

def try_import(package, version=None, descr=None):
    """Looks for a requested package
    """
    missing = False

    if version is None:
        version = '0'

    try:
        imported = __import__(package)
    except:
        missing = [package, descr]
    else:
        try:
            found_version = getattr(imported, '__version__')
        except:
            missing = [package, version, 'unknown version']
        else:
            if LooseVersion(found_version) < LooseVersion(version):
                missing = [package, version, found_version]

    return missing

# check for required dependencies and exit gracefully if missing
missing_reqs = [try_import(*info) for info in reqs]
if any(missing_reqs):
    print()
    print('=============')
    print('ERROR: Some REQUIRED packages are missing or not up-to-date. '
           'Please install them manually as they are often tricky to '
           'install via pip.')
    print()
    for miss in missing_reqs:
        if miss != False:
            if len(miss) < 3:
                print('%s: missing' % miss[0])
            else:
                print('%s: update to at least version %s (found %s)' % tuple(miss))

    print('=============')
    sys.exit(1)


fs = ['README.rst', 'CHANGES.txt']
long_description = [open(f).read() for f in fs]
long_description = '\n'.join(long_description)

setup(
    name='psychopy_ext',
    version=__version__,
    author='Jonas Kubilius',
    author_email='qbilius@gmail.com',
    packages=['docs', 'psychopy_ext', 'psychopy_ext.demos',
              'psychopy_ext.demos.scripts', 'psychopy_ext.tests'],
    package_data={str(''): [str('*.png'), str('*.py*'), str('*.bat'), str('*.csv'), str('*.txt')]},
    url='https://github.com/qbilius/psychopy_ext/',
    license='GNU General Public License v3 or later',
    description=description,
    long_description=long_description,
    install_requires=pip_reqs,
    # scripts=['bin/psycho-models.py'],
    #test_suite='nose.collector',
    #tests_require=['nose'],
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 2.7',
        'Topic :: Scientific/Engineering'
    ],
    keywords=('psychology experiments plotting data visualization analysis ',
              'fMRI simulations deep neural networks models hmax gaborjet hog '
              'phog reproducible research open science'),
)

# check for recommendeed dependencies and suggest them at the end if missing
missing_recs = [try_import(*info) for info in recs]
if any(missing_recs):
    print()
    print('=============')
    print("Installation complete. However, in order to actually use "
           "psychopy_ext, you need extra dependencies (listed below) that are "
           "often tricky to install. You'll have to install them manually. If "
           "you are not sure about the best way to go, I recommend using "
           "Conda.")
    print()
    for miss in missing_recs:
        if miss != False:
            if len(miss) == 1:
                print('%s: missing' % miss[0])
            elif len(miss) == 2:
                print('%s: missing; used for %s' % (miss[0], miss[1]))
            else:
                print('%s: update to at least version %s (found %s)' % tuple(miss))

    print('=============')
