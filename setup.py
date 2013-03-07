from distutils.core import setup
import os

class sdist_hg(sdist):

    user_options = sdist.user_options + [
            ('dev', None, "Add a dev marker")
            ]

    def initialize_options(self):
        sdist.initialize_options(self)
        self.dev = 0

    def run(self):
        if self.dev:
            suffix = '.dev%d' % self.get_tip_revision()
            self.distribution.metadata.version += suffix
        sdist.run(self)

    def get_tip_revision(self, path=os.getcwd()):
        from mercurial.hg import repository
        from mercurial.ui import ui
        from mercurial import node
        repo = repository(ui(), path)
        tip = repo.changelog.tip()
        return repo.changelog.rev(tip)


description = ('Extension of PsychoPy for an easier experimental setup via ' +
                'design patterns.')
setup(
    name='PsychoPy_ext',
    version='0.4a2',
    author='Jonas Kubilius',
    author_email='qbilius@gmail.com',
    packages=['psychopy_ext', 'psychopy_ext.test'],
    cmdclass={'sdist': sdist_hg},
    url='http://klab.lt/psychopy_ext/',
    license='LICENSE',
    description=description,
    long_description=open('README.md').read(),
    install_requires=[
        "psychopy >= 1.6",
        "pandas >= 0.10",
        "pymvpa2 >= 2.0"
    ],
    test_suite='nose.collector',
    tests_require=['nose'],
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
