from distutils.core import setup
from distutils.version import LooseVersion
import sys


description = ('A framework for a rapid reproducible design, analysis and '
               'plotting of  experiments in neuroscience and psychology.')
exec(open('psychopy_ext/version.py').read())

# check non-pip-installable dependencies and exit gracefully if missing
missing = []
reqs = [('psychopy', '1.7'), ('pandas', '0.12')]
for package, version in reqs:    
    try:
        imported = __import__(package)
    except:
        missing.append([package])
    else:
        found_version = getattr(imported, '__version__')
        if LooseVersion(found_version) < LooseVersion(version):
            missing.append([package, version, found_version])
            
if len(missing) > 0:
    print
    print '============='
    print ('ERROR: Some required packages are missing or not up-to-date. '
           'Please install them manually as they are often tricky to '
           'install via pip.')
    print
    for miss in missing:
        if len(miss) == 1:
            print '%s: missing' % miss[0]
        else:
            print '%s: update to at least version %s (found %s)' % tuple(miss)
            
    print '============='
    sys.exit(1)
    
setup(
    name='psychopy_ext',
    version=__version__,
    author='Jonas Kubilius',
    author_email='qbilius@gmail.com',
    packages=['docs', 'psychopy_ext', 'psychopy_ext.demos',
              'psychopy_ext.demos.scripts', 'psychopy_ext.tests'],
    package_data={'': ['*.png', '*.py*', '*.bat', '*.csv', '*.txt']},
    url='https://github.com/qbilius/psychopy_ext/',
    license='GNU General Public License v3 or later',
    description=description,
    long_description=open('README.rst').read(),
    install_requires=[
        "docutils",
        "seaborn>=.2",
        "svgwrite"
    ],
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
              'fMRI simulations hmax gaborjet reproducible research open science'),
)
