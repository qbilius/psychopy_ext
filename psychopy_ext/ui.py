#!/usr/bin/env python

# Part of the psychopy_ext library
# Copyright 2010-2013 Jonas Kubilius
# The program is distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.

"""Basic command-line and graphic user interface"""

import wx, sys, inspect
from types import ModuleType
# some modules are only available in Python 2.6
try:
    from collections import OrderedDict
except:
    from exp import OrderedDict

class Control(object):
    def __init__(self, exp_choices,
                 title='Experiment',
                 size=(500,300),
                 #modcall=False  # Set true when calling a module directly like
                                # python -m path.to.module
                 ):
        #if modcall:
            #if len(sys.argv) > 1:
                #self.cmd(exp_choices[1])
            #else:
                #self.app(exp_choices)
        #else:
        """
        :Args:
            exp_choices: could be one of the following
                - `module`
                - a path to a module (str)
                - a list of tuples (name, path, alias for cmd)
        :Kwargs:
            title
            size
        """
        # direct path to the experiment
        if isinstance(exp_choices, str) or isinstance(exp_choices, ModuleType):
            exp_choices = [('Experiment', exp_choices, 'main')]
        elif len(exp_choices) == 0:
            sys.exit('exp_choices is not supposed to be empty')

        choices = []
        for choice in exp_choices:
            if len(choice) == 0:
                sys.exit('Please give at least a path to the experiment')
            elif len(choice) == 1:  # path to experiment is given
                choices.append(('Experiment', choice, choice[1], None))
            elif len(choice) == 2:  # if 'scripts.main', then cli call alias is 'main'
                choices.append(choice + (choice[1].split('.',1)[1], None))
            elif len(choice) == 3:
                choices.append(choice + (None,))
            else:
                choices.append(choice)
        exp_choices = choices

        if len(sys.argv) > 1:  # command line interface desired
            self.cmd(exp_choices)
        else:
            self.app(exp_choices, title=title, size=size)

    def cmd(self, exp_choices):
        """
        Heavily stripped-down version of argparse.
        """
        if len(exp_choices) == 1:  # if just a single choice then just take it
            input_mod_alias = None
            input_class_alias = sys.argv[1]
            input_func = sys.argv[2]
            module = exp_choices[0][1]
            class_order = exp_choices[0][3]
            arg_start = 3
        else:
            input_mod_alias = sys.argv[1]
            input_class_alias = sys.argv[2]
            input_func = sys.argv[3]
            class_order = exp_choices[0][3]
            arg_start = 4
            avail_mods = [e[2] for e in exp_choices]
            try:
                idx = avail_mods.index(input_mod_alias)
            except:
                sys.exit("module '%s' not recognized" % input_mod_alias)
            module = exp_choices[idx][1]

        if input_mod_alias is not None:
            if input_mod_alias.startswith('-'):
                sys.exit('You have to specify the name of the experiment after %s'
                         % sys.argv[0])
        if input_class_alias.startswith('-') or input_func.startswith('-'):
            sys.exit('You have to specify properly what you want to run. '
                     "Got '%s %s' instead." % (input_class_alias, input_func))

        if class_order is not None:
            if input_class_alias not in class_order:
                sys.exit('Class %s not available. Choose from:\n%s' %
                    (input_class_alias, ', '.join(class_order)))

        if isinstance(module, str):
            print 'initializing...'
            try:
                __import__(module)
            except:
                raise
            module = sys.modules[module]

        class_aliases, class_obj = _get_classes(module,
            input_class_alias=input_class_alias, class_order=class_order)
        if class_obj is None:
            sys.exit('Class %s not found. Choose from: %s' %
                (input_class_alias, ', '.join([c[0] for c in class_aliases])))

        try:
            class_init = class_obj()
        except:
            #import pdb; pdb.set_trace()
            raise #SyntaxError('This module appears to require some arguments but that'
                #'should not be the case.' )

        extraInfo = {}
        runParams = {}
        i = arg_start
        if len(sys.argv) > i:
            if sys.argv[i][0] != '-':
                sys.exit('%s should be followed by function arguments '
                'that start with a - or --' % ' '.join(sys.argv[:i]))

            while i < len(sys.argv):
                input_key = sys.argv[i].lstrip('-')
                if input_key == '':
                    sys.exit("There cannot be any '-' just by themselves "
                                  "in the input")
                item = None
                for key, value in class_init.extraInfo.items():
                    if key == input_key or key[0] == input_key:
                        item = (key, value)
                        params = extraInfo
                        break
                if item is None:
                    for key, value in class_init.runParams.items():
                        if key == input_key or key[0] == input_key:
                            item = (key, value)
                            params = runParams
                            break
                if item is None:
                    sys.exit('Argument %s is not recognized' % input_key)

                key = item[0]
                if isinstance(value, bool):
                    try:
                        if sys.argv[i+1][0] != '-':
                            sys.exit('Expected no value after %s because it '
                                            'is bool' % input_key)
                        else:
                            params[key] = True
                    except IndexError:  # this was the last argument
                        params[key] = True

                else:
                    try:
                        input_value = sys.argv[i+1]
                    except IndexError:
                        sys.exit('Expected a value after %s but got nothing'
                             % input_key)

                    try:
                        ## not safe but fine in this context
                        params[key] = eval(input_value)
                    except:
                        if input_value[0] == '-':
                            sys.exit('Expected a value after %s but got '
                                        'another argument' % input_key)
                        else:
                            params[key] = input_value
                    i += 1
                i += 1
        class_init.extraInfo.update(extraInfo)
        class_init.runParams.update(runParams)
        class_init = class_obj(extraInfo=class_init.extraInfo,
            runParams=class_init.runParams)
        try:
            func = getattr(class_init, input_func)
        except AttributeError:
            sys.exit('Function %s not recognized in class %s. Check spelling?' %
                     (input_func, class_obj.__name__))
        else:
            if hasattr(func, '__call__'):
                func()
            else:
                sys.exit('Object %s not callable; is it really a function?' %
                                input_func)

    def app(self, exp_choices=[], title='Experiment', size=(400,300)):
        app = wx.App()
        frame = wx.Frame(None, title=title, size=size)
        # Here we create a panel and a listbook on the panel
        panel = wx.Panel(frame)
        if len(exp_choices) > 1:
            lb = Listbook(panel, exp_choices)
            # add pages to the listbook
            for num, choice in enumerate(exp_choices):
                pagepanel = wx.Panel(lb)
                lb.AddPage(pagepanel, choice[0], select=num==0)
            lb.ChangeSelection(0)
            booktype = lb
            panelsizer = wx.BoxSizer()
            panelsizer.Add(booktype, 1,  wx.EXPAND|wx.ALL)
            panel.SetSizer(panelsizer)
        else:  # if there's only one Notebook, don't create a listbook
            setup_page(exp_choices[0], panel)
        # nicely size the entire window
        frame.Layout()
        frame.Centre()
        frame.Show()
        app.MainLoop()

    def _type(self, input_key, input_value, value, exp_type):
        if isinstance(value, exp_type):
            try:
                input_value = int(input_value)
            except:
                Exception('Expected %s for %s'
                           % (exp_type, input_key))
            return input_value


    def _cmd_old(self, modules, raw_args=None):
        import argparse
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

def _get_classes(module, input_class_alias=None, class_order=None):
    """
    Finds all useable classes in a given module.
    """
    if class_order is None:
        class_aliases = []
    else:
        class_aliases = [None] * len(class_order)
    class_obj = None
    found_classes = inspect.getmembers(module, inspect.isclass)
    for name, obj in found_classes:
        init_vars = inspect.getargspec(obj.__init__)
        try:
            init_vars.args.index('extraInfo')
            init_vars.args.index('runParams')
        except:
            pass
        else:
            if name[0] != '_':  # avoid private classes
                class_alias = _get_class_alias(module, obj)
                if class_alias == input_class_alias:
                    class_obj = obj
                if class_order is not None:
                    try:
                        idx = class_order.index(class_alias)
                        class_aliases[idx] = (class_alias, obj)
                    except:
                        pass
                else:
                    class_aliases.append((class_alias, obj))
    # if some class not found; get rid of it
    class_aliases = [c for c in class_aliases if c is not None]
    return class_aliases, class_obj

def _get_class_alias(module, obj):
    # make sure this obj is defined in module rather than imported
    if obj.__module__ == module.__name__:
        try:
            init_vars = inspect.getargspec(obj.__init__)
        except:
            pass
        else:
            try:  # must have a name, extraInfo, and runParams
                nameidx = init_vars.args.index('name')
            except:
                pass
            else:
                class_alias = init_vars.defaults[nameidx - len(init_vars.args)]
                return class_alias

def _get_methods(myclass):
    """
    Finds all functions inside a class that are callable without any parameters.
    """
    methods = []
    for name, method in inspect.getmembers(myclass, inspect.ismethod):
        if name[0] != '_':  # avoid private methods
            mvars = inspect.getargspec(method)
            if len(mvars.args) == 1:  # avoid methods with input variables
                if mvars.args[0] == 'self':
                    methods.append((name, method))
    return methods

class StaticBox(wx.StaticBox):
    def __init__(self, parent, label='', content=None):
        """
        Partially taken from :class:`psychopy.gui.Dlg`
        """
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
                inputBox = wx.Choice(parent, -1,
                            choices=[str(option) for option in list(choices)])
                # Somewhat dirty hack that allows us to treat the choice just like
                # an input box when retrieving the data
                inputBox.GetValue = inputBox.GetStringSelection
                if initial in choices:
                    initial = choices.index(initial)
                else:
                    initial = 0
                inputBox.SetSelection(initial)
            #if len(color): inputBox.SetForegroundColour(color)
            #if len(tip): inputBox.SetToolTip(wx.ToolTip(tip))
            self.inputFields.append(inputBox)#store this to get data back on button click
            grid.Add(inputBox, 1, wx.ALIGN_LEFT)

        self.sizer.Add(grid)
        #self.SetSizer(self.sizer)

class Page(wx.Panel):
    """
    Creates a page inside a Notebook with two boxes, Information and Parameters,
    corresponding to extraInfo and runParams in :class:`exp.Experiment`, and
    buttons which, when clicked, runs a corresponding method.
    """
    def __init__(self, parent, class_obj):
        wx.Panel.__init__(self, parent, -1)
        self.class_obj = class_obj
        class_init = class_obj()
        if class_init.extraInfo is not None:
            self.sb1 = StaticBox(self, label="Information",
                content=class_init.extraInfo)
        if class_init.runParams is not None:
            self.sb2 = StaticBox(self, label="Parameters",
                content=class_init.runParams)

        # generate buttons
        # each button launches a function in a given class
        actions = _get_methods(class_init)
        # buttons will sit on a grid of 3 columns and as many rows as necessary
        buttons = wx.GridSizer(rows=0, cols=3)
        add = False
        for i, (label, action) in enumerate(actions):
            if hasattr(class_init, 'actions'):
                if class_init.actions is not None:
                    if isinstance(class_init.actions, str):
                        class_init.actions = [class_init.actions]
                    if label in class_init.actions:
                        add = True
            else:
                add = True
            if add:
                run = wx.Button(self, label=label, size=(150, 30))
                buttons.Add(run, 1)
                run.extraInfo = class_init.extraInfo  # when clicked, what to do
                run.runParams = class_init.runParams
                run.action = label
                run.Bind(wx.EVT_BUTTON, self.OnButtonClick)
                if i==0: run.SetFocus()

        pagesizer = wx.BoxSizer(wx.VERTICAL)
        # place the two boxes for entering information
        if class_init.extraInfo is not None:
            pagesizer.Add(self.sb1.sizer)
        if class_init.runParams is not None:
            pagesizer.Add(self.sb2.sizer)
        # put the buttons in the bottom
        pagesizer.Add(buttons, 1, wx.ALL|wx.ALIGN_LEFT)

        self.SetSizer(pagesizer)

    def OnButtonClick(self, event):
        #module = event.GetEventObject().module
        ## first update extraInfo and runParams
        button = event.GetEventObject()
        for key, field in zip(button.extraInfo.keys(), self.sb1.inputFields):
            button.extraInfo[key] = field.GetValue()
        for key, field in zip(button.runParams.keys(), self.sb2.inputFields):
            button.runParams[key] = field.GetValue()

        class_init = self.class_obj(extraInfo=button.extraInfo,
            runParams=button.runParams)
        func = getattr(class_init, button.action)
        func()  # FIX: would be nice to keep it open at the end of the exp

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
        if old != -1:
            setup_page(self.exp_choices[new], self.GetPage(new))

class Listbook(wx.Listbook):
    """
    Listbook class
    """
    def __init__(self, parent, exp_choices):
        wx.Listbook.__init__(self, parent, id=wx.ID_ANY, style=wx.BK_DEFAULT)
        self.exp_choices = exp_choices
        self.Bind(wx.EVT_LISTBOOK_PAGE_CHANGING, self.OnPageChanging)

    def OnPageChanging(self, event):
        old = event.GetOldSelection()
        new = event.GetSelection()
        if old != -1:
            setup_page(self.exp_choices[new], self.GetPage(new))

def setup_page(choice, pagepanel):
    """
    Creates a :class:`Page` inside a :class:`Notebook`.

    :Args:
        - choice (tuple)
            A tuple of (name, module path, module alias)
        - pagepanel
    """
    label, mod_path, alias, order = choice
    if isinstance(mod_path, str):
        __import__(mod_path)
        class_aliases, class_obj = _get_classes(sys.modules[mod_path],
            class_order=order)
    else:
        class_aliases, class_obj = _get_classes(mod_path, class_order=order)
    nb = wx.Notebook(pagepanel)
    for class_alias, class_obj in class_aliases:
        nb.AddPage(Page(nb, class_obj), class_alias)
    panelsizer = wx.BoxSizer()
    panelsizer.Add(nb, 1,  wx.EXPAND|wx.ALL)
    pagepanel.SetSizer(panelsizer)