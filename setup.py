from distutils.core import setup


description = ('A framework for a rapid reproducible design, analysis and '
               'plotting of  experiments in neuroscience and psychology.')
setup(
    name='psychopy_ext',
    version='0.5',
    author='Jonas Kubilius',
    author_email='qbilius@gmail.com',
    packages=['psychopy_ext', 'psychopy_ext.demos', 'psychopy_ext.demos.scripts'],
    package_data={'': ['*.png', '*.bat']},
    url='https://github.com/qbilius/psychopy_ext/',
    license='LICENSE',
    description=description,
    long_description=open('README.md').read(),
    install_requires=[
        "psychopy >= 1.7",
        "pandas >= 0.12",
        "pymvpa2 >= 2.0",
        "docutils"
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
