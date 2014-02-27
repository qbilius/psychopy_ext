import os, cPickle
import wx, numpy, pandas

import matplotlib
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wxagg import NavigationToolbar2Wx as NavigationToolbar

import stats, plot


class DragList(wx.ListCtrl):
    """
    From http://wiki.wxpython.org/ListControls#Drag_and_Drop_with_lists
    """
    def __init__(self, *arg, **kw):
        wx.ListCtrl.__init__(self, *arg, **kw)

        self.Bind(wx.EVT_LIST_BEGIN_DRAG, self._startDrag)

        dt = ListDrop(self)
        self.SetDropTarget(dt)

    def getItemInfo(self, idx):
        """Collect all relevant data of a listitem, and put it in a list"""
        l = []
        l.append(idx) # We need the original index, so it is easier to eventualy delete it
        l.append(self.GetItemData(idx)) # Itemdata
        l.append(self.GetItemText(idx)) # Text first column
        for i in range(1, self.GetColumnCount()): # Possible extra columns
            l.append(self.GetItem(idx, i).GetText())
        return l

    def _startDrag(self, e):
        """ Put together a data object for drag-and-drop _from_ this list. """
        l = []
        idx = -1
        while True: # find all the selected items and put them in a list
            idx = self.GetNextItem(idx, wx.LIST_NEXT_ALL, wx.LIST_STATE_SELECTED)
            if idx == -1:
                break
            l.append(self.getItemInfo(idx))

        # Pickle the items list.
        itemdata = cPickle.dumps(l, 1)
        # create our own data format and use it in a
        # custom data object
        ldata = wx.CustomDataObject("ListCtrlItems")
        ldata.SetData(itemdata)
        # Now make a data object for the  item list.
        data = wx.DataObjectComposite()
        data.Add(ldata)

        # Create drop source and begin drag-and-drop.
        dropSource = wx.DropSource(self)
        dropSource.SetData(data)
        res = dropSource.DoDragDrop(flags=wx.Drag_DefaultMove)

        # If move, we want to remove the item from this list.
        if res == wx.DragMove:
            # It's possible we are dragging/dropping from this list to this list.  In which case, the
            # index we are removing may have changed...

            # Find correct position.
            l.reverse() # Delete all the items, starting with the last item
            for i in l:
                pos = self.FindItem(i[0], i[2])
                self.DeleteItem(pos)

    def _insert(self, x, y, seq):
        """ Insert text at given x, y coordinates --- used with drag-and-drop. """

        # Find insertion point.
        index, flags = self.HitTest((x, y))

        if index == wx.NOT_FOUND: # not clicked on an item
            if flags & (wx.LIST_HITTEST_NOWHERE|wx.LIST_HITTEST_ABOVE|wx.LIST_HITTEST_BELOW): # empty list or below last item
                index = self.GetItemCount() # append to end of list
            elif self.GetItemCount() > 0:
                if y <= self.GetItemRect(0).y: # clicked just above first item
                    index = 0 # append to top of list
                else:
                    index = self.GetItemCount() + 1 # append to end of list
        else: # clicked on an item
            # Get bounding rectangle for the item the user is dropping over.
            rect = self.GetItemRect(index)

            # If the user is dropping into the lower half of the rect, we want to insert _after_ this item.
            # Correct for the fact that there may be a heading involved
            if y > rect.y - self.GetItemRect(0).y + rect.height/2:
                index += 1

        for i in seq: # insert the item data
            idx = self.InsertStringItem(index, i[2])
            self.SetItemData(idx, i[1])
            for j in range(1, self.GetColumnCount()):
                try: # Target list can have more columns than source
                    self.SetStringItem(idx, j, i[2+j])
                except:
                    pass # ignore the extra columns
            index += 1

    def set_data(self, df):
        self.DeleteAllItems()
        self.DeleteColumn(0)
        self.InsertColumn(0, 'data', width=-1)
        if df is not None:
            for itemcount, item in enumerate(df.columns):
                self.InsertStringItem(itemcount, str(item))

    def delete_all_items(self):
        self.DeleteAllItems()
        self.DeleteColumn(0)


class ListDrop(wx.PyDropTarget):
    """ Drop target for simple lists.
    From http://wiki.wxpython.org/ListControls#Drag_and_Drop_with_lists
    """
    def __init__(self, source):
        """ Arguments:
         - source: source listctrl.
        """
        wx.PyDropTarget.__init__(self)

        self.dv = source

        # specify the type of data we will accept
        self.data = wx.CustomDataObject("ListCtrlItems")
        self.SetDataObject(self.data)

    # Called when OnDrop returns True.  We need to get the data and
    # do something with it.
    def OnData(self, x, y, d):
        # copy the data from the drag source to our data object
        if self.GetData():
            # convert it back to a list and give it to the viewer
            ldata = self.data.GetData()
            l = cPickle.loads(ldata)
            self.dv._insert(x, y, l)

        # what is returned signals the source what to do
        # with the original data (move, copy, etc.)  In this
        # case we just return the suggested value given to us.
        return d


class MatplotPanel(wx.Panel):
    """
    From http://stackoverflow.com/a/19898295/1109980
    """

    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)

        self.subplots = None
        self.rows = None
        self.cols = None
        self.yerr = None
        self.values = None

        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)

        plt = plot.Plot()
        self.canvas = FigureCanvas(self, -1, plt.fig)
        self.toolbar = NavigationToolbar(self.canvas)

        self.sizer.Add(self.toolbar, 0, wx.EXPAND)
        self.sizer.Add(self.canvas, 1, wx.GROW)
        self.Fit()

    def redraw(self, agg):
        self.canvas.Destroy()
        self.toolbar.Destroy()
        plt = plot.Plot()
        kind = self.frame.plot_type.GetItemLabel(self.frame.plot_type.GetSelection())
        errkind = self.frame.err_type.GetItemLabel(self.frame.err_type.GetSelection())
        plt.plot(agg, kind=kind, errkind=errkind.lower())

        self.canvas = FigureCanvas(self, -1, plt.fig)
        self.toolbar = NavigationToolbar(self.canvas)

        self.sizer.Add(self.toolbar, 0, wx.EXPAND)
        self.sizer.Add(self.canvas, 1, wx.LEFT | wx.TOP | wx.GROW)
        self.Layout()

    def draw_empty(self):
        self.canvas.Destroy()
        self.toolbar.Destroy()
        plt = plot.Plot()

        self.canvas = FigureCanvas(self, -1, plt.fig)
        self.toolbar = NavigationToolbar(self.canvas)

        self.sizer.Add(self.toolbar, 0, wx.EXPAND)
        self.sizer.Add(self.canvas, 1, wx.GROW)
        self.Layout()

    def _get_items(self, parent, event):
        """
        #Parent is the list where the item is being inserted
        """
        items = []
        if parent.GetItemCount() > 0:
            items += [parent.GetItemText(i) for i in range(parent.GetItemCount())]
        if event.GetEventType() == wx.EVT_LIST_DELETE_ITEM.evtType[0]:
            if parent == event.GetEventObject():
                idx = items.index(event.GetText())
                del items[idx]

        if len(items) == 0:
            items = None
        return items

    def changePlot(self, event):
        self.subplots = self._get_items(self.frame.list_subplots, event)
        self.rows = self._get_items(self.frame.list_rows, event)
        self.cols = self._get_items(self.frame.list_cols, event)
        self.values = self._get_items(self.frame.list_values, event)
        self.yerr = self._get_items(self.frame.list_yerr, event)
        self.plot()

    def plot(self):
        if self.values is None or (self.cols is None and self.rows is None):
            self.draw_empty()
        else:
            agg = stats.aggregate(self.df, subplots=self.subplots, rows=self.rows,
                                  cols=self.cols, yerr=self.yerr, values=self.values)
            self.redraw(agg)
            self.frame.list_agg.DeleteAllItems()
            for i in range(self.frame.list_agg.GetColumnCount()):
                self.frame.list_agg.DeleteColumn(0)

            aggr = self.frame.list_agg.stack(agg)
            self.frame.list_agg.set_data(aggr)
            self.frame.aggr = aggr


class Shell(wx.Panel):

    def __init__(self, *args, **kwargs):
        wx.Panel.__init__(self, *args, **kwargs)
        from wx import py
        self.shell = py.shell.Shell(self, -1)
        #self.shell = py.crust.Crust(self, -1)
        self.sizer = wx.BoxSizer(wx.VERTICAL)
        self.SetSizer(self.sizer)

        self.sizer.Add(self.shell, 1, wx.EXPAND)
        self.Fit()


class DataList(wx.ListCtrl):

    def __init__(self, *args, **kwargs):
        super(DataList, self).__init__(*args, **kwargs)

    def append_data(self, event):
        list_df = event.GetEventObject()
        idx = event.GetIndex()
        path = list_df.items[idx]
        thisdf = pandas.read_csv(path)
        try:
            self.df = self.df.append(thisdf, ignore_index=True)
            nodf = False
        except:
            self.df = thisdf
            nodf = True

        self.set_data(thisdf)

        self.list_data_headers.set_data(self.df)
        self.window_canvas.df = self.df
        if not nodf:
            self.window_canvas.plot()

    def load_data(self, event):
        self.DeleteAllItems()
        for i in range(self.GetColumnCount()):
            self.DeleteColumn(0)

        list_df = event.GetEventObject()
        dfs = []
        for path in list_df.items:
            data = pandas.read_csv(path)
            if data is not None:
                dfs.append(data)
        if len(dfs) > 0:
            self.df = pandas.concat(dfs, ignore_index=True)
            self.set_data(self.df)
        else:
            self.df = None
            self.frame.list_subplots.delete_all_items()
            self.frame.list_rows.delete_all_items()
            self.frame.list_cols.delete_all_items()
            self.frame.list_values.delete_all_items()
            self.frame.list_yerr.delete_all_items()

        self.list_data_headers.set_data(self.df)
        self.window_canvas.df = self.df
        self.window_canvas.plot()

    def set_data(self, df):
        cols = [self.GetColumn(i).GetText() for i in range(self.GetColumnCount())]
        nrows = self.GetItemCount()

        for colcount, col in enumerate(df):
            if col not in cols:
                self.InsertColumn(colcount, col, width=-1)
            if colcount == 0:
                for itemcount, item in df[col].iteritems():
                    self.InsertStringItem(nrows + itemcount, str(item))
            else:
                for itemcount, item in df[col].iteritems():
                    self.SetStringItem(nrows + itemcount, colcount, str(item))

    def stack(self, agg):
        agg_df = plot._stack_levels(agg, 'subplots')
        agg_df = plot._stack_levels(agg_df, 'rows')
        agg_df = plot._stack_levels(agg_df, 'cols')
        agg_df = agg_df.reset_index()
        names = []
        for a in agg_df.columns:
            try:
                short_name = a.split('.')[-1]
            except:
                pass
            else:
                names.append((a, short_name))
        #print names
        #import pdb; pdb.set_trace()
        agg_df = agg_df.rename(columns=dict(names))
        agg_df = agg_df.rename(columns={0: agg.names})
        return agg_df


class DataFileList(wx.ListCtrl):

    def __init__(self, *args, **kwargs):
        super(DataFileList, self).__init__(*args, **kwargs)
        self.InsertColumn(0, 'data files', width=-1)
        self.Bind(wx.EVT_KEY_DOWN, self.delete_items)
        self.items = []

    def add_items(self, paths):
        for item in paths:
            self.items.append(item)
            self.InsertStringItem(self.GetItemCount(), os.path.basename(item))

    def delete_items(self, event):
        if event.KeyCode == 127:  # 'del'
            if self.GetSelectedItemCount() == self.GetItemCount():
                self.items = []
                self.DeleteAllItems()
            else:
                sel = []
                for i in range(self.GetSelectedItemCount()):
                    idx = self.GetFirstSelected(0)
                    del self.items[idx]
                    self.DeleteItem(idx)



