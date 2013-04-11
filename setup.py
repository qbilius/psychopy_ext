from distutils.core import setup


description = ('A framework for reproducible neuroscience research')
setup(
    name='psychopy_ext',
    version='0.4a2',
    author='Jonas Kubilius',
    author_email='qbilius@gmail.com',
    packages=['psychopy_ext', 'psychopy_ext.tests'],
    url='http://klab.lt/psychopy_ext/',
    license='LICENSE',
    description=description,
    long_description=open('README').read(),
    install_requires=[
        "psychopy >= 1.6",
        "pandas >= 0.10",
        "pymvpa2 >= 2.0"
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
    keywords='psychology experiments plotting data visualization analysis fMRI',
)
