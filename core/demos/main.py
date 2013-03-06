import sys, os, glob
import cPickle as pickle

from psychopy import core, data
import numpy as np
# pandas does not come by default with PsychoPy but that should not prevent
# people from running the experiment
try:
    import pandas
except:
    pass

from scripts.core import as exp, fmri, gui, models, plot

# some modules are only available in Python 2.7
try:
    from collections import OrderedDict
except:
    from scripts.core.exp import OrderedDict


class MyExperiment(Experiment):
    """
    This is an example class for your experiment.
    """
    def __init__(self, parent=None):
        _BaseTwolines.__init__(
            self,
            parent=parent,
            name='myexp',
            version='0.2',
            extraInfo=OrderedDict([
                ('subjID', 'myexp_'),
                ('date', data.getDateStr(format="%Y-%m-%d %H:%M"))
                ]),
            runParams=OrderedDict([
                ('noOutput', False),
                ('debug', False),
                ('autorun', 0)  # if >0, will autorun at the specified speed
                ]),
            actions=[
                ('run experiment', 'run'),
                ('display stimuli', 'displayAllStim'),
                ],
            instructions={'text':
                "Task:\n"
                "Please remember to fixate on the central dot.\n"
                "Please press spacebar to begin.\n",
                'wait': 0},
            method='random',
            )

        Experiment.__init__(self,
            parent = parent,
            name = name,
            version = version,
            extraInfo = extraInfo,
            runParams = runParams,
            actions = actions,
            instructions = instructions,
            method = method,
            )   

        # set up path for data storing
        exp_root = 'utl2/'
        self.paths = {
            'exp_root': exp_root,
            'data_behav': os.path.join(exp_root, 'data/'),
            'logs': os.path.join(exp_root, 'logs/'),
            }

        self.comp.validResponses = {
            '9': -1,  # dissimilar or very dissimilar if hit twice
            '8': 1  # similar or very similar if hit twice
            }
        self.stimSize = 3.  # in deg
        self.stimDist = 3.  # from the fixation in x or y dir

    def create_stimuli(self):
        # Define stimuli
        self.create_fixation() 
        stim = visual.SimpleImageStim(
            self.win,
            name = 'image',
            )
        self.s = {
            'fix': self.fixation,
            'stim': [stimuli]
            }

    def create_trial(self):
        self.trial = [{'dur': 8.000,
                       'display': self.s['stim'] + self.s['fix'],
                       'defaultFun': self.during_trial}]

    def create_trialList(self):
        self.trialDur = sum(event['dur'] for event in self.trial)

        n = len(self.paraTable)  # fixation is included to have an even number
                                 # of conditions so that we can generate
                                 # latin squares easily
        if paraNo is None:
            parafile = (self.paths['data_behav'] + 'data_*_%s.csv')
            paraNo = self.getParaNo(parafile % (self.extraInfo['subjID'],
                                    self.extraInfo['runType']), n=n)
        paralist = self.latin_square(n)[paraNo] - 1  # because it did not have
                                                     # fixation by default
        paralist = paralist.tolist()
        paralist = [0] + paralist + [0] + paralist[::-1] + [0]

        expPlan = []
        for para in range(len(paralist)):
            this_cond = paralist[para]
            if para == 0:
                prev_cond = -1
            else:
                prev_cond = paralist[para - 1]
            stim = self.paraTable[this_cond]
            triplet = self.assign_props(prev_cond, this_cond)
            expPlan.append(OrderedDict([
                ('paraNo', paraNo),
                ('stim1.cond', this_cond),  # current condition
                ('stim2.cond', prev_cond),  # previous condition
                ('name', stim['name']),
                ('junction', stim['junction']),
                ('degree', stim['degree']),
                ('part1', stim['part1']),
                ('part2', stim['part2']),
                ('angle', stim['angle']),
                ('triplet.name', triplet['name']),
                ('triplet.relation', triplet['relation']),
                ('triplet.distance', triplet['distance']),
                ('triplet.cond', triplet['condNo']),
                ('onset', ''),
                ('dur', self.trialDur),
                ('subjResp', ''),
                ('RT', '')
                ]))

        return expPlan

    def set_stimuli(self, thisTrial, thisEvent, *args):
        # setup only images
        for sNo, stim in enumerate(thisEvent['display']):
            if stim.name == 'image':
                newPos = np.random.randint(-self.stimDist, self.stimDist)
                stim.setPos((newPos, 0
                    ))

    def update_stimuli(self, cond, display, paths, oris, count): 
        for sNo, stim in enumerate(display):
            if stim.name != 'image':
                stim.draw()
            else:
                stim.setPos((paths[sNo][0][count], paths[sNo][1][count]))
                stim.setOri(oris[sNo][1][count])
                stim.draw()

    def during_trial(self, trialClock, thisTrial, thisEvent, *args, **kwargs):
        self.set_stimuli(thisTrial, thisEvent)
        self.update_stimuli(thisTrial['cond'], thisEvent['display'])
        self.win.flip()
        if self.runParams['autorun']:
            eventKeys = self.wait_for_response(RT_clock = trialClock,
                fakeKey=[thisTrial['autoResp'],thisTrial['autoRT']])
        else:
            eventKeys = self.wait_for_response(RT_clock = trialClock)
        return eventKeys

    def postTrial(self, thisTrial, allKeys):
        resp = []
        poss = self.comp.validResponses.values()
        kkk = self.comp.validResponses.keys()
        for thisKey in allKeys:
            resp.append(poss[kkk.index(thisKey[0])])
        total = sum(resp)
        if len(resp) > 0:
            if resp[0] == -1:
                startValue = 3
            else:
                startValue = 2
            total += startValue
            if total > 4:
                out = 4
            elif total < 1:
                out = 1
            else:
                out = total
        else:
            out = ''
        thisTrial['subjResp'] = out
        if len(allKeys) > 0:
            thisTrial['RT'] = allKeys[0][1]
        return thisTrial

    def run(self):
        self.setup()
        self.show_instructions(**self.instructions)
        self.loop_trials(
            datafile=self.paths['data'] + self.extraInfo['subjID'] + '.csv',
            noOutput=self.runParams['noOutput']
            )
        self.win.close()

    def autorun(self, trialList):
        def sample(probs):
            which = np.random.random()
            ind = 0
            while which>0:
                which -= probs[ind]
                ind +=1
            ind -= 1
            return ind

        def rt(mean):
            add = np.random.normal(mean,scale=.2)/self.runParams['autorun']
            return self.trial[0]['dur'] + add            

        invValidResp = dict([[v,k] for k,v in self.comp.validResponses.items()])
        sortKeys = sorted(invValidResp.keys())
        invValidResp = OrderedDict([(k,invValidResp[k]) for k in sortKeys])

        # speed up the experiment
        for event in self.trial:
            event['dur'] /= self.runParams['autorun']
        self.trialDur /= self.runParams['autorun']

        for trial in trialList:
            pos = trial['pos']
            if trial['relation'] == 'angle':                
                if trial['distance'] == 'metric':     
                    meanrt = .8
                    probs = [1/10.,1/10.,1/10.,1/10.]
                    probs[pos] = 7/10.
                elif trial['distance'] == 'non-accidental':
                    meanrt = .5
                    probs = [1/12.,1/12.,1/12.,1/12.]
                    probs[pos] = 9/12.

            elif trial['relation'] == 'position':                
                if trial['distance'] == 'metric':     
                    meanrt = .5
                    probs = [1/12.,1/12.,1/12.,1/12.]
                    probs[pos] = 9/12.                    
                elif trial['distance'] == 'non-accidental':
                    meanrt = .7
                    probs = [1/10.,1/10.,1/10.,1/10.]
                    probs[pos] = 7/10.    

            elif trial['relation'] == 'right angle position':                
                meanrt = .5
                probs = [1/12.,1/12.,1/12.,1/12.]
                probs[pos] = 9/12.

            ind = sample(probs)
            trial['autoResp'] = invValidResp.values()[ind]
            trial['autoRT'] = rt(meanrt)
        return trialList
    

class Simulation(MyExperiemnt):
    """
    This class a convenience class where we redefine a few methods from
    MyExperiment to make it suitable for simulations.
    """
    def __init__(self,
                 name='simulation',
                 actions = [('run simulation', 'plot_sim')]
                 ):        
        MyExperiemnt.__init__(self)
        self.name = name
        self.actions = actions

    def run(self, model_name='GaborJet', num_it=1):
        dataFileName_df = self.paths['sim'] + model_name + '_df.pkl'
        try:
            
            df = pickle.load(open(dataFileName_df, 'rb'))
            print 'loaded stored dataset of model responses'
        except:
            dataFileName = self.paths['sim'] + model_name + '.pkl'

            try:
                model = eval('models.' + model_name + '()')
            except:
                sys.exit('Model %s not recognized' % model_name)


            try:
                model_resp = pickle.load(open(dataFileName, 'rb'))
                print 'loaded stored model responses'
            except:
                print 'Running ' + model_name
                model_resp = []
                for it in range(num_it):  # many iterations so that stim pos and ori would vary
                    responses = []
                    trialList = self.gen_stimuli()
                    for trial in trialList:
                        resp = model.run(trial['stim'])
                        trial[model_name] = resp[0].ravel()  # use magnitude only
                    model_resp.append(trialList)
                pickle.dump(model_resp, open(dataFileName, 'wb'))

            results = []
            header = ['iter', 'subjID', 'stim1.cond', 'stim2.cond', 'subjResp']
            import matplotlib.pyplot as plt

            for it, trialList in enumerate(model_resp):
                for cond1 in trialList:
                    for cond2 in trialList:
                        # import pdb; pdb.set_trace()
                        c1 = cond1[model_name].reshape((5,8,100)).reshape((5,2,4,100))
                        c1 = c1[:,0,:,:]+c1[:,1,:,:]
                        c1 = c1**3
                        c2 = cond2[model_name].reshape((5,8,100)).reshape((5,2,4,100))
                        c2 = c2[:,0,:,:]+c2[:,1,:,:]
                        c2 = c2**3
                        # import pdb; pdb.set_trace()
                        # plt.imshow(c1[0,0].reshape((10,10)),interpolation='none');plt.show()
                        # import pdb; pdb.set_trace()
                        # corr = model.dissimilarity(cond1[model_name],
                        #                            cond2[model_name])
                        corr = model.dissimilarity(c1, c2)
                        results.append([it, 'sim%02d' % it, cond1['stim1.cond'],
                            cond2['stim1.cond'],  # no typo here!
                            corr])

            df = pandas.DataFrame(results, columns=header)
            # df = df.groupby(['stim1.cond', 'stim2.cond'],
            #         as_index=False)['subjResp'].mean()
            # df['stim1.cond'] = df['stim1.cond'].astype(int)
            # df['stim2.cond'] = df['stim2.cond'].astype(int)
            analysis = MVPA()
            # import pdb; pdb.set_trace()
            df = analysis.add_info(df)
            # pickle.dump(df, open(dataFileName_df, 'wb'))

        return df

    def gen_stimuli(self,
                    win_size_px=256,  # fixed size, rescale everything to it
                    seeStim=False,  # show the window for a while or not
                    ):

        hs = self.stimSize * np.sqrt(2)  # max stimulus box size
        self.stimPos = self.get_stimPos(self.stimDist)
        stim_box = self.stimPos + np.array([(-hs, hs), (hs, hs),
                                           (-hs, -hs), (hs, -hs)]) + self.w
        # now find the largest distance betwen the edges of the box
        win_box_deg = np.array([abs(i - j) for i, j in itertools.combinations(stim_box, 2)])
        win_size_deg = np.max(win_box_deg)
        # now we take these values and calculate what distance from
        # the current screen would yield these values
        #self.comp.monitor.getWidth()
        scr_width_cm = self.comp.monitor.getWidth()
        scr_width_pix = self.comp.monitor.getSizePix()
        win_width_cm = win_size_px * scr_width_cm / scr_width_pix[0]
        # we use approximation arctan(x)=x, but x has to be converted to radians
        distance = 180 / np.pi * win_width_cm / win_size_deg
        self.comp.monitor = monitors.Monitor('simulation', distance=distance,
                                             width=scr_width_cm)
        self.comp.params['size'] = (win_size_px, win_size_px)
        self.comp.monitor.setSizePix(scr_width_pix)

        self.create_win(debug=True)
        self.create_stimuli()
        self.create_trial()
        trialList = self.create_trialList()

        # select each condition only once
        unique_tr = []
        conds = []
        for trial in trialList:
            if trial['stim1.cond'] not in conds and trial['stim1.cond'] != 0:
                unique_tr.append(trial)
                conds.append(trial['stim1.cond'])

        self.create_TrialHandler(unique_tr)

        for trialNo, thisTrial in enumerate(self):
            thisEvent = self.trial[0]
            self.set_stimuli(thisTrial, thisEvent)
            paths, oris = self.gen_motion_params(thisEvent['display'])
            count = 0
            self.update_stimuli(thisTrial['stim1.cond'], thisEvent['display'], paths, oris, count)
            self.win.getMovieFrame(buffer='back')
            if seeStim:  # show the window for a while
                self.win.flip()
                core.wait(1 / 60. * 5)
            self.win.clearBuffer()

            stim = self.win.movieFrames[0]
            unique_tr[trialNo]['stim'] = np.asarray(stim.convert('L')) * 1.
            self.win.movieFrames = []

        self.win.close()

        return unique_tr

    def set_stimuli(self, thisTrial, thisEvent):
        if thisTrial['stim1.cond'] != 0:  # no fixation for simulation
            for sNo, stim in enumerate(thisEvent['display']):
                if stim.name == 'twolines':
                    vertices = self.xjunction(self.stimSize,
                        part1=thisTrial['part1'],
                        part2=thisTrial['part2'],
                        angle=thisTrial['angle'])
                    stim.setVertices(vertices)

    def plot_sim(self):
        df = self.run()
        analysis = Analysis()
        subset = analysis.get_subset(df)
        analysis._plot(subset)
        # plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.3)


class Analysis(MyExperiment, plot.Plot):
    def __init__(self,
                 name='analysis',
                 actions = [('behavioral analysis', 'behav')],
                 subj = 'one'
                 ):        
        MyExperiment.__init__(self)
        self.name = name
        self.actions = actions
        if subj == 'all':
            self.set_all_subj()

    def set_all_subj(self):
        self.extraInfo['subjID'] = ['myexp_%02d' % i for i in range(1,9)]

    def plot_method(self, df, method='subset', **kwargs):
        if method == 'subset':
            fig, axes = self.subplots(nrows=3, ncols=2)#, sharey=True)
            for r, ROI_list in enumerate(self.rois):
                self.subset(df[df['ROI']==ROI_list[1]],
                    ax=axes[r / 2][r % 2], title=ROI_list[1],
                    xtickson=(r==3),
                    **kwargs)
            axes[2][1].axis('off')
        # elif method = 'avg':
        #     self.avg_svm(df)
        else:
            fig = self.figure()
            axes = self.ImageGrid(fig, 111,
                                  nrows_ncols = (3, 2),
                                  direction="row",
                                  axes_pad = 0.05,
                                  add_all=True,
                                  label_mode = "L",
                                  share_all = True,
                                  cbar_location="right",
                                  cbar_mode="single",
                                  cbar_size="10%",
                                  cbar_pad=0.05,
                                  )

            for r, ROI_list in enumerate(self.rois):
                self.matrix(df[df['ROI']==ROI_list[1]], method=method,
                    ax=axes[r], title=ROI_list[1], **kwargs)
            axes[5].axis('off')
        self.show()

    def get_subset(self, df, bysubj=False, grouping='relation'):
        """
        Selects and plots metric and non-accidental stimuli only
        """
        subset = df[np.logical_or(df['triplet.distance'] == 'metric',
                                  df['triplet.distance'] == 'non-accidental')]
        if grouping == 'all':
            agg = self.aggregate(subset, values='subjResp',
                    rows=['triplet.relation','triplet.name'],
                    cols='triplet.distance', yerr='subjID')
        else:
            agg = self.aggregate(subset, values='subjResp',
                    rows='triplet.relation',
                    cols='triplet.distance', yerr='subjID')
        # if not bysubj:
        #     agg.mean(1)
        return agg


modules = [MyExperiemnt, Simulation, Analysis]

if __name__ == "__main__":
    ctrl = gui.Control()
    if len(sys.argv) > 1:        
        ctrl.cmd(modules)
    else:
        ctrl.app([('Experiment 1', modules)])