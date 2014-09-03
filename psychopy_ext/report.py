# Part of the psychopy_ext library
# Copyright 2010-2014 Jonas Kubilius
# The program is distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.

"""Creates reports"""

import sys, glob, os
from cStringIO import StringIO


class Report(object):

    def __init__(self, info=None, rp=None, paths=None, actions='make', output='html'):
        self.info = info
        self.rp = rp
        self.paths = paths
        self.actions = actions
        self.replist = []

    def make(self, reports=None):
        if not os.path.isdir(self.paths['report']):
            os.makedirs(self.paths['report'])
        else:
            for root, dirs, files in os.walk(self.paths['report']):
                #for f in files:
                    #try:
                          #os.unlink(os.path.join(root, f))
                    #except:
                        #pass
                for d in dirs:
                    try:
                          shutil.rmtree(os.path.join(root, d))
                    except:
                        pass

        src = os.path.dirname(__file__)
        src = os.path.join(src, 'resources/')
        for f in glob.glob(src+'*'):
            if os.path.isfile(f):
                if os.path.basename(f) != 'index.html':
                    shutil.copy2(f)
            else:
                dst = os.path.join(self.paths['report'], os.path.basename(f))
                shutil.copytree(f, dst,
                    ignore=shutil.ignore_patterns('index.html'))
        with open(src + 'index.html', 'rb') as tmp:
            template = tmp.read().split('####REPLACE####')
        self.htmlfile = open(self.paths['report'] + 'index.html', 'wb')
        self.write(template[0])


        old_stdout = sys.stdout
        mystdout = StringIO()
        sys.stdout = mystdout

        if reports is None:
            self.report()
        else:
            for name, report in reports:
                self.writeh(name, h='h1')
                report.report()

        sys.stdout = old_stdout
        self.write(template[1])
        self.htmlfile.close()

    def write(self, text):
        self.htmlfile.write(text)

    def writeh(self, text, h='h1'):
        self.htmlfile.write('<%s>%s</%s>\n' % (h, text, h))

    def writeimg(self, names, caption=None, plt=None, win=None):
        if isinstance(names, str):
            names = [names]
        fname = '_'.join(names) + '.png'
        imgdir = self.paths['exp_root']
        if plt is not None or win is not None:
            if not os.path.isdir(imgdir):
                os.makedirs(imgdir)
            if plt is not None:
                plt.savefig(imgdir + fname, bbox_inches='tight')
            elif win is not None:
                win.saveMovieFrames(imgdir + fname)
        if caption is None:
            caption = ' '.join(names)
        self.htmlfile.write(
            '<figure>\n'
            '    <img src="%s" />\n'
            '    <figcaption><strong>Figure.</strong> %s</figcaption>\n'
            '</figure>\n' % (self.imgpath + fname, caption)
            )
        self.replist.append([anlname, self.info.copy(), self.rp.copy()])

    def writetable(self, agg, caption='', fmt=None):
        if fmt is None:
            fmt = '%.3f'
        fmt_lam = lambda x: fmt % x

        import pandas
        table = pandas.DataFrame(agg).to_html(float_format=fmt_lam)
        self.htmlfile.write(
            '<figure>\n'
            '    <figcaption><strong>Table.</strong> %s</figcaption>\n'
            '    <div>%s</div>\n'
            '</figure>\n' % (caption, table)
            )
