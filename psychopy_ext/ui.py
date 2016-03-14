# Part of the psychopy_ext library
# Copyright 2010-2014 Jonas Kubilius
# The program is distributed under the terms of the GNU General Public License,
# either version 3 of the License, or (at your option) any later version.

"""Basic command-line and graphic user interface"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import sys, os, inspect, shutil, subprocess
from types import ModuleType

try:
    import wx
    try:
        from agw import advancedsplash as AS
    except ImportError: # if it's not there locally, try the wxPython lib.
        import wx.lib.agw.advancedsplash as AS
    has_wx = True
except:
    has_wx = False

from psychopy import core

# some modules are only available in Python 2.6
try:
    from collections import OrderedDict
except:
    from exp import OrderedDict

from psychopy_ext import report


class Control(object):

    def __init__(self, exp_choices,
                 title='Project',
                 size=None
                 ):
        """
        Initializes user control interface.

        Determines automatically whether to open a Graphic User Interface (GUI)
        or operate in a Command Line Interface (CLI) based on the number of
        arguments in ``sys.argv``.

        :Args:
            exp_choices
                :class:`~psychopy_ext.ui.Choices`
        :Kwargs:
            title (str, default: 'Project')
                Title of the GUI app window.
            size (tuple of two int, default: None)
                Size of a GUI app. If None, tries to fit the contents.
                However, if you have multiple pages in the Listbook,
                it will probably do a poor job.
        """
        # Some basic built-in functions
        try:
            action = sys.argv[1]
        except: # otherwise do standard stuff
            pass
        else:
            recognized = ['--commit','--register','--push']
            if action in recognized:
                self.run_builtin()

        if not isinstance(exp_choices, (list, tuple)):
            exp_choices = [exp_choices]

        if len(sys.argv) > 1:  # command line interface desired
            if sys.argv[1] == 'report':
                self.report(exp_choices, sys.argv)
            else:
                self.cmd(exp_choices)
        else:
            self.app(exp_choices, title=title, size=size)

    def run_builtin(self, action=None):
        if action is None:
            action = sys.argv[1]
        if action == '--commit':
            try:
                message = sys.argv[2]
            except:
                sys.exit('Please provide a message for committing changes')
            else:
                _repo_action('commit', message=message)
        elif action == '--register':
            try:
                tag = sys.argv[2]
            except:
                sys.exit('Please provide a tag to register')
            else:
                _repo_action('register', tag=tag)
        elif action == '--push':
            _repo_action('push')

        sys.exit()

    def cmd(self, exp_choices):
        """
        Heavily stripped-down version of argparse.
        """
        # if just a single choice then just take it
        try:
            third_is_arg = sys.argv[3].startswith('-')
        except:
            third_is_arg = True
        if third_is_arg and len(exp_choices) == 1:
            input_mod_alias = None
            input_class_alias = sys.argv[1]
            input_func = sys.argv[2]
            module = exp_choices[0].module
            class_order = exp_choices[0].order
            arg_start = 3
        else:
            input_mod_alias = sys.argv[1]
            input_class_alias = sys.argv[2]
            input_func = sys.argv[3]
            arg_start = 4
            avail_mods = [e.alias for e in exp_choices]
            try:
                idx = avail_mods.index(input_mod_alias)
            except:
                sys.exit("module '%s' not recognized" % input_mod_alias)
            module = exp_choices[idx].module
            class_order = exp_choices[idx].order

        if input_mod_alias is not None:
            if input_mod_alias.startswith('-'):
                sys.exit('You have to specify the name of the experiment after %s'
                         % sys.argv[0])
        if input_class_alias.startswith('-') or input_func.startswith('-'):
            sys.exit('You have to specify properly the task you want to run. '
                     "Got '%s %s' instead." % (input_class_alias, input_func))

        if class_order is not None:
            if input_class_alias not in class_order:
                sys.exit('Class %s not available. Choose from:\n%s' %
                    (input_class_alias, ', '.join(class_order)))

        if isinstance(module, (str, unicode)):
            sys.stdout.write('initializing...')
            sys.stdout.flush()
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

        info = {}
        rp = {}
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
                # is input_key among info?
                if hasattr(class_init, 'info'):
                    for key, value in class_init.info.items():
                        if key == input_key or key[0] == input_key:
                            item = (key, value)
                            params = info
                            break
                # is input_key among rp then?
                if item is None and hasattr(class_init, 'rp'):
                    for key, value in class_init.rp.items():
                        if key == input_key or key[0] == input_key:
                            item = (key, value)
                            params = rp
                            break
                # not found?
                if item is None and (hasattr(class_init, 'info') or
                    hasattr(class_init, 'rp')):
                        sys.exit('Argument %s is not recognized' % input_key)
                else: # found!
                    key, value = item
                    if isinstance(value, bool):
                        try:
                            if sys.argv[i+1][0] != '-':
                                input_value = eval(sys.argv[i+1])
                                if not isinstance(input_value, bool):
                                    sys.exit('Expected True/False after %s' %
                                             input_key)
                                else:
                                    params[key] = input_value
                                i += 1
                            else:
                                params[key] = True
                        except IndexError:  # this was the last argument
                            params[key] = True

                    else:
                        try:
                            input_value = sys.argv[i+1].lstrip('"').rstrip('"')
                        except IndexError:
                            sys.exit('Expected a value after %s but got nothing'
                                 % input_key)

                        if isinstance(value, tuple):
                            try:
                                input_value = eval(input_value)
                            except:
                                pass
                            if input_value in value:
                                params[key] = input_value
                            else:
                                sys.exit('Value %s is not possible for %s.\n'
                                         'Choose from: %s'
                                         % (input_value, key, value))
                        else:
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

        if hasattr(class_init, 'info'):
            class_init.info.update(info)
            for key, value in class_init.info.items():
                if isinstance(value, tuple):
                    class_init.info[key] = value[0]
        if hasattr(class_init, 'rp'):
            class_init.rp.update(rp)
            for key, value in class_init.rp.items():
                if isinstance(value, tuple):
                    class_init.rp[key] = value[0]
        if hasattr(class_init, 'info') and hasattr(class_init, 'rp'):
            class_init = class_obj(info=class_init.info, rp=class_init.rp)
        elif hasattr(class_init, 'info'):
            class_init = class_obj(info=class_init.info)
            class_init.rp = None
        elif hasattr(class_init, 'rp'):
            class_init = class_obj(rp=class_init.rp)
            class_init.info = None
        else:
            class_init = class_obj()
            class_init.info = None
            class_init.rp = None
        sys.stdout.write('\r               ')
        sys.stdout.write('\r')
        sys.stdout.flush()
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

    def app(self, exp_choices=[], title='Experiment', size=None):
        if not has_wx:
            raise Exception('You must have wx to open a psychopy_ext app.')
        app = MyApp()

        # initial frame with a gauge on it
        frame = wx.Frame(None, title=title, size=size)
        ## Here we create a panel and a listbook on the panel
        panel = wx.Panel(frame)

        if len(exp_choices) > 1:
            lb = Listbook(panel, exp_choices, frame)
            # add pages to the listbook
            for num, choice in enumerate(exp_choices):
                pagepanel = wx.Panel(lb)
                lb.AddPage(pagepanel, choice.name, select=num==0)
            lb.ChangeSelection(0)
            booktype = lb
            panelsizer = wx.BoxSizer()
            panelsizer.Add(booktype, 1,  wx.EXPAND|wx.ALL)
            panel.SetSizer(panelsizer)
        else:  # if there's only one Notebook, don't create a listbook
            setup_page(exp_choices[0], panel, frame)
        # nicely size the entire window
        app.splash.Close()
        panel.Fit()
        if size is None:
            frame.Fit()
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


def report(exp_choices, args):
    reports = []
    if len(args) == 2:
        argnames = [ch.alias for ch in exp_choices]
    else:
        argnames = args[2:]

    for ch in exp_choices:
        if ch.alias in argnames:
            choice = ch.module
            if isinstance(choice, (str, unicode)):
                try:
                    __import__(choice)
                except:
                    raise #module = None
                else:
                    module = sys.modules[choice]
            else:
                module = choice
            if module is not None:
                classes = inspect.getmembers(module, inspect.isclass)
                for name, cls in classes:
                    if name == 'report':
                        reports.append((ch.name, cls))
                        break
    rep = report.Report()
    rep.make(reports=reports)

def _get_classes(module, input_class_alias=None, class_order=None):
    """
    Finds all useable classes in a given module.

    'Usable' means the ones that are not private
    (class name does not start with '_').

    TODO: maybe alse check if upon initialization has info and rp
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
            #init_vars.args.index('info')
            #init_vars.args.index('rp')
            init_vars.args.index('name')
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
    class_aliases = [c for c in class_aliases if c is not None and c[0] is not None]
    return class_aliases, class_obj

def _get_class_alias(module, obj):
    # make sure this obj is defined in module rather than imported
    if obj.__module__ == module.__name__:
        try:
            init_vars = inspect.getargspec(obj.__init__)
        except:
            pass
        else:
            try:  # must have a name, info, and rp
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

def _get_methods_byname(myclass):
    if hasattr(myclass, 'actions'):
        if myclass.actions is not None:
            if isinstance(myclass.actions, (str, unicode)):
                actions = [myclass.actions]
            else:
                actions = myclass.actions
            methods = []
            for action in actions:
                try:
                    func = getattr(myclass, action)
                except AttributeError:
                    pass
                else:
                    methods.append([action, func])
            if len(methods) == 0:
                return _get_methods(myclass)
            else:
                return methods
        else:
            return _get_methods(myclass)
    else:
        return _get_methods(myclass)


class MyApp(wx.App):

    def __init__(self):
        super(MyApp, self).__init__(redirect=False)
        path = os.path.join(os.path.dirname(__file__), 'importing.png')
        image = wx.Bitmap(path, wx.BITMAP_TYPE_PNG)
        self.splash = AS.AdvancedSplash(None, bitmap=image,
                                        style=AS.AS_NOTIMEOUT|wx.FRAME_SHAPED)
        self.splash.SetText(' ')  # bitmap doesn't show up without this
        wx.Yield()  # linux wants this line


class StaticBox(wx.StaticBox):
    def __init__(self, parent, label='', content=None):
        """
        Partially taken from :class:`psychopy.gui.Dlg`
        """
        wx.StaticBox.__init__(self, parent, label=label)
        self.sizer = wx.StaticBoxSizer(self)
        grid = wx.FlexGridSizer(rows=len(content), cols=2)
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
            if isinstance(initial, bool):  # check box for true/false
                inputBox = wx.CheckBox(parent, -1)
                inputBox.SetValue(initial)
            elif isinstance(initial, int): # spin field for ints
                imin = 0  # wx default
                imax = 100  # wx default
                if initial > imax:
                    imax = initial
                elif initial < imin:
                    imin = initial

                inputBox = wx.SpinCtrl(parent, size=(60, -1), initial=initial,
                                       min=imin, max=imax)
            elif isinstance(initial, tuple):  # choice for tuples
                inputBox = wx.Choice(parent, -1,
                            choices=[str(option) for option in initial])
                ## Somewhat dirty hack that allows us to treat the choice just like
                ## an input box when retrieving the data
                inputBox.GetValue = inputBox.GetStringSelection
                #if initial in choices:
                    #initial = choices.index(initial)
                #else:
                    #initial = 0
                inputBox.SetSelection(0)
            else:  # plain text label for eveything else
                inputLength = wx.Size(max(50, 9*len(unicode(initial))+16), 25)
                inputBox = wx.TextCtrl(parent,-1,unicode(initial),size=inputLength)

            #if len(color): inputBox.SetForegroundColour(color)
            #if len(tip): inputBox.SetToolTip(wx.ToolTip(tip))
            self.inputFields.append(inputBox)#store this to get data back on button click
            grid.Add(inputBox, 1, wx.ALIGN_LEFT)

        self.sizer.Add(grid)
        #self.SetSizer(self.sizer)

class Page(wx.Panel):
    """
    Creates a page inside a Notebook with two boxes, Information and Parameters,
    corresponding to info and rp in :class:`exp.Experiment`, and
    buttons which, when clicked, runs a corresponding method.
    """
    def __init__(self, parent, class_obj, alias, class_alias, frame, pagepanel):
        wx.Panel.__init__(self, parent, -1)
        self.class_obj = class_obj
        self.alias = alias
        self.class_alias = class_alias
        self.frame = frame
        self.pagepanel = pagepanel

        class_init = class_obj()
        if not hasattr(class_init, 'info'):
            class_init.info = None
        if not hasattr(class_init, 'rp'):
            class_init.rp = None
        if class_init.info is not None:
            self.sb1 = StaticBox(self, label="Information",
                content=class_init.info)
        if class_init.rp is not None:
            self.sb2 = StaticBox(self, label="Parameters",
                content=class_init.rp)

        #collpane = wx.CollapsiblePane(self, wx.ID_ANY, "Details:")#, style=wx.CP_DEFAULT_STYLE|wx.CP_NO_TLW_RESIZE)
        #collpane.Bind(wx.EVT_COLLAPSIBLEPANE_CHANGED, self.OnPaneChanged)
        ## now add a test label in the collapsible pane using a sizer to layout it:
        #win = collpane.GetPane()
        #paneSz = wx.BoxSizer(wx.VERTICAL)
        #paneSz.Add(wx.StaticText(win, wx.ID_ANY, "test!"), 1, wx.GROW | wx.ALL, 2)
        #win.SetSizer(paneSz)
        #paneSz.SetSizeHints(win)
        ##collpane.Collapse()

        # generate buttons
        # each button launches a function in a given class
        #actions = _get_methods(class_init)
        actions = _get_methods_byname(class_init)
        # buttons will sit on a grid of 2 columns and as many rows as necessary
        buttons_sizer = wx.FlexGridSizer(rows=0, cols=2)
        add = False
        self.buttons = []
        for i, (label, action) in enumerate(actions):
            if hasattr(class_init, 'actions'):
                if class_init.actions is not None:
                    if isinstance(class_init.actions, (str, unicode)):
                        class_init.actions = [class_init.actions]
                    if label in class_init.actions:
                        add = True
            else:
                add = True
            if add:
                run = wx.Button(self, label=label, size=(150, 30))
                run._proc_running = False
                buttons_sizer.Add(run, 1)
                run.info = class_init.info  # when clicked, what to do
                run.rp = class_init.rp
                run.action = label
                run.Bind(wx.EVT_BUTTON, self.OnButtonClick)
                self.buttons.append(run)
                if i==0: run.SetFocus()

        pagesizer = wx.BoxSizer(wx.VERTICAL)
        # place the two boxes for entering information
        if class_init.info is not None:
            pagesizer.Add(self.sb1.sizer)
        if class_init.rp is not None:
            pagesizer.Add(self.sb2.sizer)
        # put the buttons in the bottom
        pagesizer.Add(buttons_sizer, 1, wx.ALL|wx.ALIGN_LEFT)
        self.SetSizer(pagesizer)

        # add the pane with a zero proportion value to the 'sz' sizer which contains it
        #pagesizer.Add(collpane, 1, wx.GROW | wx.ALL, 5)
        #pagesizer.Add( collpane, 2, wx.RIGHT|wx.LEFT|wx.EXPAND, 5 )


    def OnButtonClick(self, event):
            button = event.GetEventObject()
        #if button._proc_running:
            #self.enable(button)
            #self.proc.kill()
        #else:
            # if info has tuples, select only one option
            if button.info is not None and button.info != {}:
                for key, field in zip(button.info.keys(), self.sb1.inputFields):
                    button.info[key] = field.GetValue()
            else:
                button.info = {}
            # if info has tuples, select only one option
            if button.rp is not None:
                for key, field in zip(button.rp.keys(), self.sb2.inputFields):
                    button.rp[key] = field.GetValue()
            else:
                button.rp = {}

            # call the relevant script
            opts = [self.alias, self.class_alias, button.GetLabelText()]
            params = []

            for k,v in button.info.items() + button.rp.items():
                params.append('--%s' % k)
                vstr = '%s' % v
                if len(vstr.split(' ')) > 1:
                    vstr = '"%s"' % vstr
                params.append(vstr)
            command = [sys.executable, sys.argv[0]] + opts + params

            #button._origlabel = opts[2]
            #button.SetLabel('kill')
            #button._proc_running = True
            button.proc = subprocess.Popen(command, shell=False)  # no shell is safer

            #if button.proc.poll() is not None:  # done yet?
                #self.enable(button)

    def enable(self, button):
        button.SetLabel(button._origlabel)
        button._proc_running = False

    #def OnPaneChanged(self, evt):
        ## redo the layout
        #self.GetSizer().Layout()
        #self.Fit()
        #self.pagepanel.GetSizer().Layout()
        #self.pagepanel.Layout()
        #self.pagepanel.Fit()
        #self.frame.Layout()
        #self.frame.Fit()



class Listbook(wx.Listbook):
    """
    Listbook class
    """
    def __init__(self, parent, exp_choices, frame):
        wx.Listbook.__init__(self, parent, id=wx.ID_ANY)
        self.exp_choices = exp_choices
        self.ready = []
        self.Bind(wx.EVT_LISTBOOK_PAGE_CHANGING, self.OnPageChanging)
        self.frame = frame

    def OnPageChanging(self, event):
        new = event.GetSelection()
        if new not in self.ready:
            success = setup_page(self.exp_choices[new], self.GetPage(new), self.frame)
            if success:
                self.ready.append(new)

def setup_page(choice, pagepanel, frame):
    """
    Creates a :class:`Page` inside a :class:`Notebook`.

    :Args:
        - choice (tuple)
            A tuple of (name, module path, module alias)
        - pagepanel
    """
    if isinstance(choice.module, str):
        try:
            __import__(choice.module)
        except ImportError as e:
            wx.MessageBox('%s' % e, 'Info', wx.OK | wx.ICON_ERROR)
            return False
        else:
            class_aliases, class_obj = _get_classes(sys.modules[choice.module],
                                                    class_order=choice.order)
    else:
        class_aliases, class_obj = _get_classes(choice.module, class_order=choice.order)

    nb = wx.Notebook(pagepanel)


    for class_alias, class_obj in class_aliases:
        nb.AddPage(Page(nb, class_obj, choice.alias, class_alias, frame, pagepanel), class_alias)
    panelsizer = wx.BoxSizer()
    panelsizer.Add(nb, 1,  wx.EXPAND|wx.ALL)
    pagepanel.SetSizer(panelsizer)
    pagepanel.Layout()
    pagepanel.Fit()
    return True

def _detect_rev():
    """
    Detects revision control system.

    Recognizes: git, hg
    """
    revs = ['git', 'hg']
    for rev in revs:
        try:
            out, err = core.shellCall(rev + ' status', stderr=True)
        except:  # revision control is not installed
            pass
        else:
            if err[:5] not in ['abort', 'fatal']:
                return rev

def _repo_action(cmd, **kwargs):
    """
    Detects revision control system and performs a specified action.

    Currently supported: committing changes, tagging the current version of the
    repository (registration), and pushing.

    'Registration' is inspired by the `Open Science Framework
    <http://openscienceframework.org/>`_. Useful when you start running
    participants so that you can always go back to that version.

    To set up ``git`` on Windows, you should download it from the
    ``git-scm.com`` website, install it choosing to add git to the registry
    such that it could be used from cmd too (Option 2),
    use their Git Bash to generate an SSH key
    (`follow these instructions <https://help.github.com/articles/generating-ssh-keys>`_),
    and create a ``HOME`` variable in your environment that hold a path
    to your user on Windows (because in it there is .ssh folder where you
    saved your SSH key). This ``HOME`` variable is used by git to know
    where to look for keys.
    """
    rev = _detect_rev()
    if rev == 'hg':
        if cmd == 'push':
            call = 'hg push'
        elif cmd == 'commit':
            if 'message' in kwargs:
                call = 'hg commit -A -m "%s"' % kwargs['message']
            else:
                raise Exception('Please provide a message for committing changes')
        elif cmd == 'register':
            if 'tag' in kwargs:
                call = 'hg tag %s' % kwargs['tag']
            else:
                raise Exception('Please provide a tag to register')
        else:
            raise Exception("%s is not supported for %s yet" % (rev, cmd))

    elif rev == 'git':
        if cmd == 'push':
            call = 'git push'
        elif cmd == 'commit':
            if 'message' in kwargs:
                call = ['git add -A', 'git commit -m "%s"' % kwargs['message']]
            else:
                raise Exception('Please provide a message for committing changes')
        else:
            raise Exception("%s is not supported for %s yet" % (rev, cmd))

    elif rev is None:
        raise Exception("no revision control detected")
    else:
        raise Exception("%s is not supported for %s yet" % (rev, cmd))

    if not isinstance(call, (list, tuple)):
        call = [call]
    tmp_out = ''
    for c in call:
        out, err = core.shellCall(c, stderr=True)
        tmp_out += '\n'.join([out, err])
    output = ['$ ' + c for c in call]
    if tmp_out != '':
        output.append(tmp_out)
    output = '\n'.join(output) + '\n'
    sys.stdout.write(output)
    return output


class Choices(object):

    def __init__(self, module, name='', alias=None, order=None):
        """
        Holds choices for calling experiments.

        :Args:
            module (str or module)
                The module you want to call. If your script is in
                'scripts/main.py', then module should be 'scripts.main'.
                You can also give the module itself if you already have
                it imported::

                    import scripts.main
                    exp_choices = Choices(scripts.main)

        :Kwargs:
            - name (str, default: '')
                Name of the experiment.
            - alias (str, default: None)
                For CLI: alias for calling this experiment. If *None*,
                will be inferred from ``module``.
            - order (list, default: None)
                For GUI: Order of tabs (classes).
        """
        ## if direct path to the experiment
        #if isinstance(exp_choices, str) or isinstance(exp_choices, ModuleType):
            #exp_choices = [('Experiment', exp_choices, 'main')]
        #elif len(exp_choices) == 0:
            #sys.exit('exp_choices is not supposed to be empty')

        #choices = []
        #for choice in exp_choices:
            #if len(choice) == 0:
                #sys.exit('Please give at least a path to the experiment')
            #elif len(choice) == 1:  # path to experiment is given
                #choices.append(('Experiment', choice, choice[1], None))
            #elif len(choice) == 2:  # if 'scripts.main', then cli call alias is 'main'
                #choices.append(choice + (choice[1].split('.',1)[1], None))
            #elif len(choice) == 3:
                #choices.append(choice + (None,))
            #else:
                #choices.append(choice)
        #exp_choices = choices

        self.module = module
        self.name = name

        if alias is None:
            try:
                self.alias = module.split('.',1)[1]
            except:
                import pdb; pdb.set_trace()
                self.alias = module.__name__
        else:
            self.alias = alias

        self.order = order


class Arg(dict):
    def __init__(self, key, value, advanced=True, label=None, guess=False):
        self.dict = {#key: key,
                     #value: value,
                     'advanced': advanced,
                     'label': label,
                     'guess': guess}
        super(Arg, self).__init__(self.dict)
        self.__dict__ = self
        self.key = key
        self.value = value

class Params(OrderedDict):

    def __init__(self, params):

        #for p in params:
            #self.update({p.key: p})
            #self[p.key] = p

        if isinstance(params, (dict, OrderedDict)):
            params = [Arg(k,v) for k,v in params.items()]
            #if isinstance(p, (dict, OrderedDict)):
                #p = Arg(p)

        dict_vals = [(p.key, p) for p in params]
        super(Params, self).__init__(dict_vals)
        #import pdb; pdb.set_trace()
        self.params = params

        #self.update(dict_vals)

        #self.__dict__ = self

    def __getitem__(self, key):
        return super(Params, self).__getitem__(key).value

    def __getattr__(self, key):
        if key in self:
            return super(Params, self).__getitem__(key)
        else:
            return super(Params, self).__getattr__(key)

    #def __setitem__(self, key, value):
        #super(Params,self).__setitem__(key, value)

    def update(self, new_dict):
        #import pdb; pdb.set_trace()
        #if isinstance(new_dict, (OrderedDict, dict)):
            ##import pdb; pdb.set_trace()
            #new_dict = Params(new_dict)
        for key in new_dict:
            try:
                new_item = getattr(new_dict, key)
            except:
                new_item = Arg(key, new_dict[key])
                #import pdb; pdb.set_trace()

            if key in self:
                #
                #super(Params, self).update(
                item = getattr(self, key)
                #import pdb; pdb.set_trace()
                item.update(new_item)
            else:
                #import pdb; pdb.set_trace()
                self[key] = new_item
