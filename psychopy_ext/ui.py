#!/usr/bin/env python

# Part of the psychopy_ext library
# Copyright 2010-2013 Jonas Kubilius
# The program is distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.

"""Basic command-line and graphic user interface"""

import wx, sys, inspect
import argparse
# some modules are only available in Python 2.6
try:
    from collections import OrderedDict
except:
    from exp import OrderedDict


class StaticBox(wx.StaticBox):
    def __init__(self, parent, label='', content=None):
        '''
        Partially taken from psychopy.gui.Dlg
        '''
        wx.StaticBox.__init__(self, parent, label=label)
        self.sizer = wx.StaticBoxSizer(self)
        grid = wx.GridSizer(rows=len(content), cols=2)
        choices=False
        self.inputFields = []
        for label, initial in content.items():
            #import pdb; pdb.set_trace()
            #create label
            labelLength = wx.Size(9*len(label)+16,25)#was 8*until v0.91.4
            inputLabel = wx.StaticText(parent,-1,label,
                                            size=labelLength,
                                           )
            #if len(color): inputLabel.SetForegroundColour(color)
            grid.Add(inputLabel, 1, wx.ALIGN_LEFT)
            #create input control
            if type(initial)==bool:
                inputBox = wx.CheckBox(parent, -1)
                inputBox.SetValue(initial)
            elif type(initial)==int:
                inputBox = wx.SpinCtrl(parent, size=(60, -1), initial=initial)
            elif not choices:
                inputLength = wx.Size(max(50, 9*len(unicode(initial))+16), 25)
                inputBox = wx.TextCtrl(parent,-1,unicode(initial),size=inputLength)
            else:
                inputBox = wx.Choice(parent, -1, choices=[str(option) for option in list(choices)])
                # Somewhat dirty hack that allows us to treat the choice just like
                # an input box when retrieving the data
                inputBox.GetValue = inputBox.GetStringSelection
                initial = choices.index(initial) if initial in choices else 0
                inputBox.SetSelection(initial)
            #if len(color): inputBox.SetForegroundColour(color)
            #if len(tip): inputBox.SetToolTip(wx.ToolTip(tip))
            self.inputFields.append(inputBox)#store this to get data back on button click
            grid.Add(inputBox, 1, wx.ALIGN_LEFT)

        self.sizer.Add(grid)
        #self.SetSizer(self.sizer)

class Page(wx.Panel):
    def __init__(self, parent, module):
        wx.Panel.__init__(self, parent,-1)
        self.sb1 = StaticBox(self, label="Information",
            content=module.extraInfo)
        self.sb2 = StaticBox(self, label="Parameters",
            content=module.runParams)

        buttons = wx.GridSizer(rows=1,cols=len(module.actions))
        for i, (label,action) in enumerate(module.actions):
            run = wx.Button(self, label=label, size=(150, 30))
            buttons.Add(run, 1)
            run.module = module  # when clicked, what to do
            run.action = action
            run.Bind(wx.EVT_BUTTON,self.OnButtonClick)
            if i==0: run.SetFocus()

        pagesizer = wx.BoxSizer(wx.VERTICAL)
        pagesizer.Add(self.sb1.sizer)
        pagesizer.Add(self.sb2.sizer)
        pagesizer.Add(buttons, 1, wx.EXPAND|wx.ALL)

        self.SetSizer(pagesizer)

    def OnButtonClick(self, event):
        module = event.GetEventObject().module
        # first update extraInfo and runParams
        for key, field in zip(module.extraInfo.keys(),self.sb1.inputFields):
            module.extraInfo[key] = field.GetValue()
        for key, field in zip(module.runParams.keys(),self.sb2.inputFields):
            module.runParams[key] = field.GetValue()

        rp = event.GetEventObject()
        getattr(module, rp.action)()


class Notebook(wx.Notebook):
    """
    Notebook class
    """
    def __init__(self, parent):
        wx.Notebook.__init__(self, parent, id=wx.ID_ANY, style=
                             wx.BK_DEFAULT)
        self.Bind(wx.EVT_NOTEBOOK_PAGE_CHANGING, self.OnPageChanging)

    def OnPageChanging(self, event):
        old = event.GetOldSelection()
        new = event.GetSelection()
        self.GetPage(new).module

        if old != -1:
            setup_page(self.exp_choices[new], self.GetPage(new))


class Listbook(wx.Listbook):
    """
    Listbook class
    """
    def __init__(self, parent, exp_choices):
        wx.Listbook.__init__(self, parent, id=wx.ID_ANY, style=
                             wx.BK_DEFAULT)
        self.exp_choices = exp_choices
        self.Bind(wx.EVT_LISTBOOK_PAGE_CHANGING, self.OnPageChanging)

    def OnPageChanging(self, event):
        old = event.GetOldSelection()
        new = event.GetSelection()
        # sel = self.GetSelection()
        # print 'OnPageChanging, old:%d, new:%d, sel:%d\n' % (old, new, sel)
        if old != -1:
            setup_page(self.exp_choices[new], self.GetPage(new))


class Control(object):
    def __init__(self, exp_choices,
                 title='Experiment',
                 #modcall=False  # Set true when calling a module directly like
                                # python -m path.to.module
                 ):
        #if modcall:
            #if len(sys.argv) > 1:
                #self.cmd(exp_choices[1])
            #else:
                #self.app(exp_choices)
        #else:
        if len(exp_choices) == 0:
            raise SyntaxError('exp_choices is not supposed to be empty')
        else:
            if len(sys.argv) > 1:  # command line interface desired
                self.cmd(exp_choices)
            else:
                self.app(exp_choices, title=title)

    def app(self, exp_choices=[], title='Experiment'):
        app = wx.App()
        frame = wx.Frame(None, title=title, size = (650,300))
        # Here we create a panel and a listbook on the panel
        panel = wx.Panel(frame)
        lb = Listbook(panel, exp_choices)
        # add pages to the listbook
        for num, choice in enumerate(exp_choices):
            pagepanel = wx.Panel(lb)
            lb.AddPage(pagepanel, choice[0], select=num==0)
        # lb.ChangeSelection(0)

        # nicely size the entire window
        panelsizer = wx.BoxSizer()
        panelsizer.Add(lb, 1,  wx.EXPAND|wx.ALL)
        panel.SetSizer(panelsizer)

        frame.Layout()
        frame.Centre()
        frame.Show()
        app.MainLoop()

    def cmd(self, exp_choices):
        """
        Heavily stripped-down version of argparse.
        """
        if len(exp_choices) == 1:  # if just a single choice then just take it
            input_mod = None
            input_class = sys.argv[1]
            input_func = sys.argv[2]
            mod_str = exp_choices[0][1]
            arg_start = 3
        else:
            input_mod = sys.argv[1]
            input_class = sys.argv[2]
            input_func = sys.argv[3]
            arg_start = 4
            avail_mods = [e[2] for e in exp_choices]
            try:
                idx = avail_mods.index(input_mod)
            except:
                raise Exception("module '%s' not recognized" % input_mod)
            mod_str = exp_choices[idx][1]

        try:
            __import__(mod_str)
        #except ImportError:
            #raise Exception('Module %s not found or has errors' % mod_str)
        except:
            raise

        module = sys.modules[mod_str]
        class_names = []
        runclass = None
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj):
                # make sure this obj is defined in module
                # and not imported
                if obj.__module__ == module.__name__:
                    try:
                        ar = inspect.getargspec(obj.__init__)
                    except:
                        pass
                    else:
                        try:
                            nameidx = ar.args.index('name')
                        except:
                            pass
                        else:
                            objname = ar.defaults[nameidx - len(ar.args)]
                            class_names.append(objname)
                            if objname == input_class:
                                runclass = obj

        if runclass is None:
            raise Exception('Class %s not found. Choose from:\n%s' %
                (input_class, ', '.join(class_names)))

        try:
            init_mod = runclass()
        except:
            #import pdb; pdb.set_trace()
            raise #SyntaxError('This module appears to require some arguments but that'
                #'should not be the case.' )

        extraInfo = {}
        runParams = {}
        i = arg_start
        if len(sys.argv) > i:
            if sys.argv[i][0] != '-':
                raise IOError('%s should be followed by function arguments '
                'that start with a - or --' % ' '.join(sys.argv[:i]))
            while i < len(sys.argv):
                input_key = sys.argv[i].lstrip('-')
                if input_key == '':
                    raise IOError('There cannot be any - just by themselves in '
                                  'the input')
                item = None
                for key, value in init_mod.extraInfo.items():
                    if key.startswith(input_key):
                        item = (key, value)
                        params = extraInfo
                        break
                if item is None:
                    for key, value in init_mod.runParams.items():
                        if key.startswith(input_key):
                            item = (key, value)
                            params = runParams
                            break
                if item is None:
                    raise IOError('Argument %s is not recognized' % input_key)

                key = item[0]
                if isinstance(value, bool):
                    try:
                        if sys.argv[i+1][0] != '-':
                            raise IOError('Expected no value after %s because it '
                                            'is bool' % input_key)
                        else:
                            params[key] = True
                    except IndexError:  # this was the last argument
                        params[key] = True

                else:
                    try:
                        input_value = sys.argv[i+1]
                    except IndexError:
                        raise Exception('Expected a value after %s but got nothing'
                             % input_key)

                    try:
                        # not safe but fine in this context
                        params[key] = eval(input_value)
                    except:
                        if input_value[0] == '-':
                            raise IOError('Expected a value after %s but got '
                                        'another argument' % input_key)
                        else:
                            params[key] = input_value
                    i += 1
                i += 1

        init_mod.extraInfo.update(extraInfo)
        init_mod.runParams.update(runParams)
        init_mod = runclass(extraInfo=init_mod.extraInfo,
                            runParams=init_mod.runParams)
        try:
            func = getattr(init_mod, input_func)
        except AttributeError:
            print 'Function %s not defined in class %s' % (input_func, input_class)
        else:
            if hasattr(func, '__call__'):
                func()
            else:
                raise Exception('Object %s not callable; is it really a function?' %
                                input_func)

    def _type(self, input_key, input_value, value, exp_type):
        if isinstance(value, exp_type):
            try:
                input_value = int(input_value)
            except:
                Exception('Expected %s for %s'
                           % (exp_type, input_key))
            return input_value


    def cmd_old(self, modules, raw_args=None):
        def add_arg(mod_parse, arg, default):
            if type(default) == bool:
                if default:
                    action = 'store_false'
                else:
                    action = 'store_true'
                mod_parse.add_argument('--' + arg, default=default,
                                       action=action)
            else:
                mod_parse.add_argument('--' + arg, default=default,
                                       type=type(default))

        parser = argparse.ArgumentParser()

        subparsers = parser.add_subparsers(dest='moduleName')
        mods = {}
        for module_uninit in modules:
            mod = module_uninit()
            mods[mod.name] = mod
            mod_parse = subparsers.add_parser(mod.name)
            action_choices = [a[1] for a in mod.actions]
            mod_parse.add_argument('action', choices=action_choices)
            for arg, default in mod.extraInfo.items():
                add_arg(mod_parse, arg, default)
            for arg, default in mod.runParams.items():
                add_arg(mod_parse, arg, default)

        if raw_args is not None:
            raw_args = raw_args.split(' ')
        rp = parser.parse_args(args=raw_args)
        module = mods[rp.moduleName]


        for key in module.extraInfo.keys():
            try:
                value = eval(rp.__dict__[key])
            except:
                value = rp.__dict__[key]
            module.extraInfo[key] = value

        for key in module.runParams.keys():
            if key in rp.__dict__:
                module.runParams[key] = rp.__dict__[key]

            #else:
                #module.runParams[key] =
        # mod, module = mods[rp.moduleName]
        # extraInfo = []
        # runParams = []
        # for key in mod.extraInfo.keys():
        #     try:
        #         value = eval(rp.__dict__[key])
        #     except:
        #         value = rp.__dict__[key]
        #     extraInfo.append((key,value))
        # for key in mod.runParams.keys():
        #     runParams.append((key,rp.__dict__[key]))
        # module = module(extraInfo=OrderedDict(extraInfo),
        #                 runParams=OrderedDict(runParams))
        getattr(module, rp.action)()
        # args.func(args)

def setup_page(choice, pagepanel):
    label, exp_path = choice
    if type(exp_path) == str:
        __import__(exp_path)
        mods = sys.modules[exp_path].MODULES
    else:
        mods = exp_path
    #if type(exp) in [tuple,list]:
        #pagepanel = wx.Panel(lb)
        #lb.AddPage(pagepanel, label)
    nb = wx.Notebook(pagepanel)
    for mod in mods:
        #m = mod(parent=exp)
        m = mod()
        nb.AddPage(Page(nb,m), m.name)
    panelsizer = wx.BoxSizer()
    panelsizer.Add(nb, 1,  wx.EXPAND|wx.ALL)
    pagepanel.SetSizer(panelsizer)
    # else:
        #m = exp(parent=exp_parent)
        # lb.AddPage(Page(lb, exp), label)
