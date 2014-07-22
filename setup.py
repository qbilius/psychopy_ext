from distutils.core import setup
from distutils.version import LooseVersion
import sys


description = ('A framework for a rapid reproducible design, analysis and '
               'plotting of  experiments in neuroscience and psychology.')
exec(open('psychopy_ext/version.py').read())

# required pip-installable packages
pip_reqs = ['docutils', 'svgwrite', 'seaborn>=0.3']
# required non-pip-installable packages
reqs = [('psychopy', '1.80'), ('pandas', '0.12')]
# recommended non-pip-installable packages
recs = [('nibabel', None, 'fMRI analyses'),
        ('mvpa2', '2.0', 'fMRI analyses')]

def try_import(package, version, descr=None):
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
        found_version = getattr(imported, '__version__')
        if LooseVersion(found_version) < LooseVersion(version):
            missing = [package, version, found_version]

    return missing

# check for required dependencies and exit gracefully if missing
missing_reqs = [try_import(*info) for info in reqs]
if any(missing_reqs):
    print
    print '============='
    print ('ERROR: Some REQUIRED packages are missing or not up-to-date. '
           'Please install them manually as they are often tricky to '
           'install via pip.')
    print
    for miss in missing_reqs:
        if miss != False:
            if len(miss) < 3:
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
    install_requires=pip_reqs,
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

# check for recommendeed dependencies and suggest them at the end if missing
missing_recs = [try_import(*info) for info in recs]
if any(missing_recs):
    print
    print '============='
    print ('WARNING: Some RECOMMENDED packages are missing or not up-to-date. '
           'Please install them manually as they are often tricky to '
           'install via pip.')
    print
    for miss in missing_recs:
        if miss != False:
            if len(miss) == 1:
                print '%s: missing' % miss[0]
            elif len(miss) == 2:
                print '%s: missing; used for %s' % (miss[0], miss[1])
            else:
                print '%s: update to at least version %s (found %s)' % tuple(miss)

    print '============='
