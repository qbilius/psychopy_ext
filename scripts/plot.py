import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
"""
A library of simple and beautiful plots.
"""

class Plot(object):
    def __init__(self, ax=None):
        if ax is None:
            self.ax = plt.subplot(111)
        else:
            self.ax = ax

    def subplots(self, **kwargs):
        fig, axes = plt.subplots(**kwargs)
        return (fig, axes)

    def show(self, *args, **kwargs):
        plt.show(*args, **kwargs)

    def sample_paired(self, ncolors=2):
        """
        Returns colors for matplotlib.cm.Paired.
        """
        if ncolors <= 12:
            colors_full = [mpl.cm.Paired(i * 1. / 11) for i in range(1, 12, 2)]
            colors_pale = [mpl.cm.Paired(i * 1. / 11) for i in range(12, 2)]
            colors = colors_full + colors_pale
            return colors[:ncolors]
        else:
            return [mpl.cm.Paired(c) for c in np.linspace(ncolors)]

        
    def sample_cmap(self, cmap='Paired', ncolors=2):
        cmap = mpl.cm.get_cmap(thisCmap)      
        norm = mpl.colors.Normalize(0, 1)
        z = np.linspace(0, 1, numColors + 2)
        z = z[1:-1]
        colors = cmap(norm(z))
        return colors


    def plot(self, x,y=None, ax=None, **kwargs):
        if ax is None:
            ax = plt.subplot(111)

        if y is None:
            x_values = np.arange(len(x))
            y = x
        else:
            x_values = x
        ax.errorbar(x_values,y, **kwargs)

    # def subplot():


    def pivot_plot(self,df,rows=None,cols=None,values=None,yerr=None,ax=None,
                   **kwargs):

        if ax is None: ax = plt.subplot(111)
        if type(rows) != list: rows = [rows]
        if type(cols) != list: cols = [cols]
        if type(yerr) != list: yerr = [yerr]
        if df[values].dtype == str:  # calculate accuracy
            agg = df.groupby(rows+cols+yerr, as_index=False)[values].size()
        else:
            agg = df.groupby(rows+cols+yerr,
                             as_index=False)[values].aggregate(np.mean)

        avg = agg.groupby(rows+cols)[values].mean().unstack()
        std = agg.groupby(rows+cols)[values].std(ddof=1).unstack()
        size = agg.groupby(rows+cols)[values].size().unstack()
        p_yerr = np.asarray(std/np.sqrt(size))
        p_yerr_zeros = np.zeros((p_yerr.shape[0],))
        colors = self.sample_paired(len(avg.columns))
        edgecolors = []
        for c in colors:
            edgecolors.extend([c]*avg.shape[0])
        
        avg.plot(kind='bar', ax=ax, **{
                'yerr':p_yerr_zeros,  # otherwise get_lines doesn't work
                'color': colors,
                #'edgecolor': edgecolors,  # get rid of ugly black edges
                })
        ax.set_ylabel(values)
        if 'title' in kwargs: ax.set_title(kwargs['title'])

        for i, col in enumerate(avg.columns):
            x = ax.get_lines()[i * len(avg.columns)].get_xdata()
            y = avg[col]
            ax.errorbar(x, y, yerr=p_yerr[:, i], fmt=None,
                ecolor='black')
        ax.legend(loc=8)
        plt.tight_layout()
        return ax

    def pivot_plot_old(self, pivot, persubj=None, kind='bar', ncols=2, ax=None, title='',
                xlabels=True, yerr_type='sem', legend=True, **kwargs):
        """
        Generates a bar plot from pandas pivot table.

        Not very stable due to a hack of ncols.
        """
        if ax is None:
            ax = plt.subplot(111)

        if persubj is not None:
            # calculate errorbars
            p_yerr = self.errorbars(persubj, ncols, yerr_type=yerr_type)
            # generate zero errorbars for the pandas plot
            p_yerr_zeros = np.zeros((ncols, p_yerr.shape[0]))

            # plot data with zero errorbars
            pplot = pivot.plot(kind=kind, ax=ax, legend=legend,
                            **{'yerr': p_yerr_zeros,
                            'color': self.sample_paired(ncols)})
            # put appropriate errorbars for each bar
            for i, col in enumerate(pivot.columns):
                x = pplot.get_lines()[i * ncols].get_xdata()
                y = pivot[col]
                pplot.errorbar(x, y, yerr=p_yerr[:, i], fmt=None,
                    ecolor='b')
        else:
            # plot data without errorbars
            pplot = pivot.plot(kind=kind, ax=ax, legend=legend)#,
                            #**{'color': sample_paired(ncols)})
        ax.legend(loc=8)
        ax.set_title(title)
        if not xlabels:
            ax.set_xticklabels([''] * pivot.shape[0])
        plt.tight_layout()

        return pplot


    def matrix(self, matrix, matrix_labels, ax=None, title='',
                xlabels=True, ylabels=True, **kwargs):
        """
        Plots a matrix.
        """
        if ax is None:
            ax = plt.subplot(111)
        im = ax.imshow(matrix, interpolation='none', **kwargs)
        ax.set_title(title)
        plt.colorbar(im, ax=ax, use_gridspec=True)

        xnames = ['%s|%s' %
            (str(n[0]), str(n[1])) for n in matrix_labels.columns]
        if xlabels:
            loc, labels = plt.xticks(range(matrix.shape[1]), xnames)
            plt.setp(labels, 'rotation', 'vertical')
        else:
            ax.set_xticklabels([''] * matrix.shape[1])
        if ylabels:
            ynames = ['%d|%s' % (n[0], n[1]) for n in matrix_labels.index]
            plt.yticks(range(matrix.shape[0]), ynames)
        else:
            ax.set_yticklabels([''] * matrix.shape[0])

        return ax


    def errorbars(self, persubj, ncols, yerr_type='sem'):
        # Set up error bar information
        pvalues = None
        if yerr_type == 'sem':
            p_yerr = np.std(persubj, axis=1, ddof=1) / np.sqrt(persubj.shape[1])
            p_yerr = p_yerr.reshape((-1, ncols))
        elif yerr_type == 'binomial':
            pass
            # alpha = .05
            # z = stats.norm.ppf(1-alpha/2.)
            # count = np.mean(persubj, axis=1, ddof=1)
            # p_yerr = z*np.sqrt(mean*(1-mean)/persubj.shape[1])

        return p_yerr

    def stats_test(self, agg, test='ttest'):
        d = agg.shape[0]
        
        if test == 'ttest':
            # 2-tail T-Test
            ttest = (np.zeros((agg.shape[1]*(agg.shape[1]-1)/2, agg.shape[2])),
                     np.zeros((agg.shape[1]*(agg.shape[1]-1)/2, agg.shape[2])))
            ii = 0
            for c1 in range(agg.shape[1]):
                for c2 in range(c1+1,agg.shape[1]):
                    thisTtest = stats.ttest_rel(agg[:,c1,:], agg[:,c2,:], axis = 0)
                    ttest[0][ii,:] = thisTtest[0]
                    ttest[1][ii,:] = thisTtest[1]
                    ii += 1
            ttestPrint(title = '**** 2-tail T-Test of related samples ****',
                values = ttest, plotOpt = plotOpt,
                type = 2)
        
        elif test == 'ttest_1samp':
            # One-sample t-test
            m = .5
            oneSample = stats.ttest_1samp(agg, m, axis = 0)
            ttestPrint(title = '**** One-sample t-test: difference from %.2f ****' %m,
                values = oneSample, plotOpt = plotOpt, type = 1)

        elif test == 'binomial':
            # Binomial test
            binom = np.apply_along_axis(stats.binom_test,0,agg)
            print binom
            return binom


    def ttestPrint(self, title = '****', values = None, xticklabels = None, legend = None, bon = None):
        
            d = 8
            # check if there are any negative t values (for formatting purposes)
            if np.any([np.any(val < 0) for val in values]): neg = True            
            else: neg = False
                
            print '\n' + title
            for xi, xticklabel in enumerate(xticklabels):
                print xticklabel
                
                maxleg = max([len(leg) for leg in legend])
    #            if type == 1: legendnames = ['%*s' %(maxleg,p) for p in plotOpt['subplot']['legend.names']]
    #            elif type == 2:
                pairs = q.combinations(legend,2)                
                legendnames = ['%*s' %(maxleg,p[0]) + ' vs ' + '%*s' %(maxleg,p[1]) for p in pairs]                
                #print legendnames    
                for yi, legendname in enumerate(legendnames):
                    if values[0].ndim == 1:
                        t = values[0][xi]
                        p = values[1][xi]
                    else:
                        t = values[0][yi,xi]
                        p = values[1][yi,xi]
                    if p < .001/bon: star = '***'
                    elif p < .01/bon: star = '**'
                    elif p < .05/bon: star = '*'
                    else: star = ''
                    
                    if neg and t > 0:
                        outputStr = '    %(s)s: t(%(d)d) =  %(t).3f, p = %(p).3f %(star)s'
                    else:
                        outputStr = '    %(s)s: t(%(d)d) = %(t).3f, p = %(p).3f %(star)s'
                        
                    print outputStr \
                        %{'s': legendname, 'd':(d-1), 't': t,
                        'p': p, 'star': star}


    def plotrc(self, theme = 'default'):
        """
        Reads in the rc file
        Returns: plotOpt   
        """
        
        if theme == None or theme == 'default': themeFile = 'default'
        else: themeFile = theme
        
        rcFile = open(os.path.dirname(__file__) + '/' + themeFile + '.py')
        
        plotOpt = {}
        for line in rcFile.readlines():        
            line = line.strip(' \n\t')
            line = line.split('#')[0] # getting rid of comments
            if line != '': # skipping blank and commented lines
                
                linesplit = line.split(':',1)
                (key, value) = linesplit
                key = key.strip()
                value = value.strip()
                
                # try recognizing numbers, lists etc.
                try: value = eval(value)
                except: pass
                
                linesplit = key.rsplit('.',1)
                if len(linesplit) == 2:
                    (plotOptKey, optKey) = linesplit
                    if not plotOptKey in plotOpt: plotOpt[plotOptKey] = {}
                    plotOpt[plotOptKey][optKey] = value
                elif key in ['cmap', 'simpleCmap']: plotOpt[key] = value
                else: print 'Option %s not recognized and will be ignored' %line
                            
        rcFile.close
        return plotOpt


    def prettyPlot(self,
        x,
        y = None,    
        yerr = None,
        plotType = 'errorbar',
        fig = None,
        subplotno = 111,
        max_y = None,
        theme = 'default',
        userOpt = None
        ):
        
        """
        Automatically makes beautiful plots and with error bars.
        
        **Parameters**
        
            x: numpy.ndarray
                Either x or y values
            y: **None** or numpy.ndarray
                + if **None**, then x value is assumed to be y, while x is generated automatically    
                + else it is simpy a y value
            yerr: **None** a 2D numpy.ndarray or list
                Specifies y error bars. If **None**, then no error bars are plotted
            plotType: 'errorbar' or 'bar'
                + 'errorbar' plots lines with error bars
                + 'bar' plots bars with error bars
            subplotno: int
                Use 111 for a single plot; otherwise specify subplot position
            max_y: **None** or int
                Use to indicate the maximum allowed ytick value.
                Useful when plotting accuracy since yticks should not go above 1
            theme: str
                Specifies the file name where all the custom plotting parameters are stores
            userOpt: dict
                Any plotting options that should override default options
            
        """
        
        if y == None: # meaning: x not specified
            y = x
            if y.ndim == 1:
                x = np.arange(y.shape[0])+1
                numCat = 1
            else:
                x = np.arange(y.shape[1])+1
                numCat = y.shape[0]
            # this is because if y.ndim == 1, y.shape = (numb,) instead of (1,num)
            
        numXTicks = x.shape[0]
        
        plotOpt = plotrc(theme)
        
        if not 'cmap' in plotOpt: plotOpt['cmap'] = 'Paired'    
        if not 'hatch' in plotOpt: plotOpt['hatch'] = ['']*numCat
            
        if not 'subplot' in plotOpt: plotOpt['subplot'] = {}
        if not 'xticks' in plotOpt['subplot']:
            if numXTicks > 1:
                plotOpt['subplot']['xticks'] = x
            else:
                plotOpt['subplot']['xticklabels'] = ['']
        if not 'legend' in plotOpt: plotOpt['legend'] = ['']    
        
        # x limits        
        xMin = x.min()
        xMax = x.max()
        if xMax == xMin: plotOpt['subplot']['xlim'] = (-.5,2.5)
        elif plotType == 'errorbar':
            plotOpt['subplot']['xlim'] = (xMin - (xMax - xMin)/(2.*(len(x)-1)),
                xMax + (xMax - xMin)/(2.*(len(x)-1)))
        elif plotType == 'bar':
            plotOpt['subplot']['xlim'] = (xMin - (xMax - xMin)/(1.5*(len(x)-1)),
                xMax + (xMax - xMin)/(1.5*(len(x)-1)))
        
        # y limits
        if yerr == None or not np.isfinite(yerr.max()):
            yMin = y.min()
            yMax = y.max()
        else:
            yMin = (y - yerr).min()
            yMax = (y + yerr).max()
        if yMax == yMin: plotOpt['subplot']['ylim'] = (0, yMax + yMax/4.)
        else: plotOpt['subplot']['ylim'] = (yMin - (yMax - yMin)/4., yMax + (yMax - yMin)/2.)
            
        
        # overwrite plotOpt by userOpt
        if userOpt != None:
            for (key, value) in userOpt.items():
                if type(value)==dict:
                    if not key in plotOpt: plotOpt[key] = {}
                    for (key2, value2) in value.items():
                        plotOpt[key][key2] = value2
                else: plotOpt[key] = value
        
        
        # set all values recognized by mpl
        # this is stupid but not all values are recognized by mpl
        # thus, we later set the unrecognized ones manually
        for key, value in plotOpt.items():
            if not key in ['cmap', 'colors', 'hatch', 'legend.names', 'other']:            
                for k, v in value.items():            
                    fullKey = key + '.' + k            
                    if fullKey in mpl.rcParams.keys():
                        try: mpl.rcParams[fullKey] = v
                        except: mpl.rcParams[fullKey] = str(v) # one more stupidity in mpl
                            # for ytick, a string has to be passed, even if it is a number
        plotOpt['colors'] = getColors(numCat, plotOpt, userOpt)                    
        
        
        # make a new figure
        if fig == None:
            fig = plt.figure()
            fig.canvas.set_window_title('Results')    
        ax = fig.add_subplot(subplotno)
        
        output = []    
        
        # generate plots
        if plotType == 'errorbar':
            
            if yerr != None:
                for i in range(numCat):
                    output.append(ax.errorbar(
                        x,
                        y[i,:],
                        yerr = yerr[i,:],
                        color = plotOpt['colors'][i],
                        markerfacecolor = plotOpt['colors'][i],                    
                        **plotOpt[plotType]
                        )
                    )
            else:
                if numCat == 1:
                    output.append(ax.plot(
                        x,
                        y[:],
                        color = plotOpt['colors'][0],
                        markerfacecolor = plotOpt['colors'][0])
                        )
                else:
                    for i in range(numCat):                
                        output.append(ax.plot(
                            x,
                            y[i,:],
                            color = plotOpt['colors'][i],
                            markerfacecolor = plotOpt['colors'][i]
                            )
                        )
                    
        elif plotType == 'bar':
        
            barWidth = (1-.2)/numCat       # the width of the bars
            middle = x - numCat*barWidth/2.
            if numCat == 1:
                output.append(ax.bar(
                    middle,
                    y[:],
                    barWidth,
                    facecolor = plotOpt['colors'][0],
                    yerr = None,#yerr[:],
                    hatch = plotOpt['hatch'][0],
                    **plotOpt[plotType]
                    )
                )
            else:
                for i in range(numCat):            
                    output.append(ax.bar(
                        middle+i*barWidth,
                        y[i,:],
                        barWidth,
                        facecolor = plotOpt['colors'][i],
                        yerr = yerr[i,:],
                        hatch = plotOpt['hatch'][i],
                        **plotOpt[plotType]
                        )
                    )
                
        
        # set up legend
        if plotOpt['subplot'].get('legend.names'):
            if len(plotOpt['subplot']['legend.names']) > 1:
                outLeg = [i[0] for i in output]
                leg = ax.legend(outLeg, plotOpt['subplot']['legend.names'])
                leg.draw_frame(False)
            del plotOpt['subplot']['legend.names']
            
        # draw other things
        if 'other' in plotOpt['subplot']:
            for item in q.listify(plotOpt['subplot']['other']): eval(item)
            del plotOpt['subplot']['other']
            
        # set the remaining subplot options    
        ax.set(**plotOpt['subplot'])
        
        if not 'yticks' in plotOpt['subplot'] and not 'yticklabels' in plotOpt['subplot']:
            import decimal
            yticks = ax.get_yticks()
            if max_y != None: yticks = yticks[yticks <= max_y]
            else: yticks = yticks[yticks <= yticks[-2]]
            # now some fancy way to make pretty labels
            kk = min([decimal.Decimal(str(t)).adjusted() for t in yticks])        
            lens2 = np.array([t for t in yticks if len(('%g' %(t/10**kk)).split('.')) == 1])
            spacing = len(lens2)/6 + 1
            if max_y != None: use = lens2[np.arange(len(lens2)-1,-1,-spacing)]
            else: use = lens2[np.arange(0,len(lens2),spacing)]
            
            ax.set_yticks(use)
        
        return fig



