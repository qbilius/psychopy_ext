# -*- coding: utf-8 -*- 

###########################################################################
## Python code generated with wxFormBuilder (version Nov  6 2013)
## http://www.wxformbuilder.org/
##
## PLEASE DO "NOT" EDIT THIS FILE!
###########################################################################

from analyzer_comps import MatplotPanel
from analyzer_comps import DragList
from analyzer_comps import Shell
from analyzer_comps import DataList
from analyzer_comps import DataFileList
import wx
import wx.xrc

###########################################################################
## Class MyFrame
###########################################################################

class MyFrame ( wx.Frame ):
	
	def __init__( self, parent ):
		wx.Frame.__init__ ( self, parent, id = wx.ID_ANY, title = u"Analyzer", pos = wx.DefaultPosition, size = wx.Size( 1148,815 ), style = wx.DEFAULT_FRAME_STYLE )
		
		self.SetSizeHintsSz( wx.DefaultSize, wx.DefaultSize )
		
		sizer_frame = wx.BoxSizer( wx.HORIZONTAL )
		
		self.notebook = wx.Notebook( self, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, 0 )
		self.window_page = wx.Panel( self.notebook, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer12 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.window_page_splitter = wx.SplitterWindow( self.window_page, wx.ID_ANY, wx.DefaultPosition, wx.Size( 100,-1 ), wx.SP_3D|wx.SP_BORDER )
		self.window_page_splitter.SetSashGravity( 1 )
		self.window_page_splitter.Bind( wx.EVT_IDLE, self.window_page_splitterOnIdle )
		
		self.window_plotter = wx.Panel( self.window_page_splitter, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer13 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.window_3 = wx.SplitterWindow( self.window_plotter, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SP_3D|wx.SP_BORDER )
		self.window_3.SetSashGravity( 1 )
		self.window_3.Bind( wx.EVT_IDLE, self.window_3OnIdle )
		
		self.window_notrow = wx.Panel( self.window_3, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer14 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.window_row_plot_splitter = wx.SplitterWindow( self.window_notrow, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SP_3D|wx.SP_BORDER )
		self.window_row_plot_splitter.Bind( wx.EVT_IDLE, self.window_row_plot_splitterOnIdle )
		
		self.window_plot = wx.Panel( self.window_row_plot_splitter, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer131 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.window_canvas = MatplotPanel( self.window_plot, wx.ID_ANY, wx.DefaultPosition, wx.Size( 500,-1 ), wx.TAB_TRAVERSAL )
		bSizer131.Add( self.window_canvas, 1, wx.EXPAND |wx.ALL, 5 )
		
		self.m_panel12 = wx.Panel( self.window_plot, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		self.m_panel12.SetMaxSize( wx.Size( 150,-1 ) )
		
		bSizer121 = wx.BoxSizer( wx.VERTICAL )
		
		plot_typeChoices = [ u"bar", u"line", u"bean" ]
		self.plot_type = wx.RadioBox( self.m_panel12, wx.ID_ANY, u"plot type", wx.DefaultPosition, wx.DefaultSize, plot_typeChoices, 1, wx.RA_SPECIFY_COLS )
		self.plot_type.SetSelection( 0 )
		bSizer121.Add( self.plot_type, 0, wx.EXPAND|wx.ALL, 5 )
		
		self.m_panel151 = wx.Panel( self.m_panel12, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer19 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText1 = wx.StaticText( self.m_panel151, wx.ID_ANY, u"values", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText1.Wrap( -1 )
		bSizer19.Add( self.m_staticText1, 0, wx.ALL, 5 )
		
		self.list_values = DragList( self.m_panel151, wx.ID_ANY, wx.DefaultPosition, wx.Size( -1,20 ), wx.LC_LIST )
		bSizer19.Add( self.list_values, 0, wx.ALIGN_RIGHT|wx.ALL, 5 )
		
		
		self.m_panel151.SetSizer( bSizer19 )
		self.m_panel151.Layout()
		bSizer19.Fit( self.m_panel151 )
		bSizer121.Add( self.m_panel151, 0, wx.ALL|wx.ALIGN_RIGHT, 5 )
		
		self.m_panel14 = wx.Panel( self.m_panel12, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer26 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText11 = wx.StaticText( self.m_panel14, wx.ID_ANY, u"subplots", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText11.Wrap( -1 )
		bSizer26.Add( self.m_staticText11, 0, wx.ALL, 5 )
		
		self.list_subplots = DragList( self.m_panel14, wx.ID_ANY, wx.DefaultPosition, wx.Size( -1,20 ), wx.LC_LIST )
		bSizer26.Add( self.list_subplots, 0, wx.ALIGN_RIGHT|wx.ALL, 5 )
		
		
		self.m_panel14.SetSizer( bSizer26 )
		self.m_panel14.Layout()
		bSizer26.Fit( self.m_panel14 )
		bSizer121.Add( self.m_panel14, 0, wx.ALL|wx.ALIGN_RIGHT, 5 )
		
		self.m_panel152 = wx.Panel( self.m_panel12, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer24 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText12 = wx.StaticText( self.m_panel152, wx.ID_ANY, u"x axis", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText12.Wrap( -1 )
		bSizer24.Add( self.m_staticText12, 0, wx.ALL, 5 )
		
		self.list_rows = DragList( self.m_panel152, wx.ID_ANY, wx.DefaultPosition, wx.Size( -1,-1 ), wx.LC_LIST )
		bSizer24.Add( self.list_rows, 0, wx.ALIGN_RIGHT|wx.ALL, 5 )
		
		
		self.m_panel152.SetSizer( bSizer24 )
		self.m_panel152.Layout()
		bSizer24.Fit( self.m_panel152 )
		bSizer121.Add( self.m_panel152, 1, wx.EXPAND |wx.ALL, 5 )
		
		self.m_panel15 = wx.Panel( self.m_panel12, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer22 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText13 = wx.StaticText( self.m_panel15, wx.ID_ANY, u"legend", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText13.Wrap( -1 )
		bSizer22.Add( self.m_staticText13, 0, wx.ALL, 5 )
		
		self.list_cols = DragList( self.m_panel15, wx.ID_ANY, wx.DefaultPosition, wx.Size( -1,-1 ), wx.LC_LIST )
		bSizer22.Add( self.list_cols, 0, wx.ALIGN_RIGHT|wx.ALL, 5 )
		
		
		self.m_panel15.SetSizer( bSizer22 )
		self.m_panel15.Layout()
		bSizer22.Fit( self.m_panel15 )
		bSizer121.Add( self.m_panel15, 0, wx.ALL|wx.ALIGN_RIGHT, 5 )
		
		self.m_panel13 = wx.Panel( self.m_panel12, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer201 = wx.BoxSizer( wx.VERTICAL )
		
		self.m_staticText14 = wx.StaticText( self.m_panel13, wx.ID_ANY, u"error bars", wx.DefaultPosition, wx.DefaultSize, 0 )
		self.m_staticText14.Wrap( -1 )
		bSizer201.Add( self.m_staticText14, 0, wx.ALL, 5 )
		
		self.list_yerr = DragList( self.m_panel13, wx.ID_ANY, wx.DefaultPosition, wx.Size( -1,20 ), wx.LC_LIST )
		bSizer201.Add( self.list_yerr, 0, wx.ALIGN_RIGHT|wx.ALL, 5 )
		
		err_typeChoices = [ u"SEM", u"CI" ]
		self.err_type = wx.RadioBox( self.m_panel13, wx.ID_ANY, u"type", wx.DefaultPosition, wx.DefaultSize, err_typeChoices, 1, wx.RA_SPECIFY_COLS )
		self.err_type.SetSelection( 0 )
		bSizer201.Add( self.err_type, 1, wx.ALL|wx.EXPAND, 5 )
		
		
		self.m_panel13.SetSizer( bSizer201 )
		self.m_panel13.Layout()
		bSizer201.Fit( self.m_panel13 )
		bSizer121.Add( self.m_panel13, 0, wx.ALL|wx.ALIGN_RIGHT, 5 )
		
		
		self.m_panel12.SetSizer( bSizer121 )
		self.m_panel12.Layout()
		bSizer121.Fit( self.m_panel12 )
		bSizer131.Add( self.m_panel12, 0, wx.ALIGN_RIGHT|wx.ALL, 5 )
		
		
		self.window_plot.SetSizer( bSizer131 )
		self.window_plot.Layout()
		bSizer131.Fit( self.window_plot )
		self.window_row_plot_splitter.Initialize( self.window_plot )
		bSizer14.Add( self.window_row_plot_splitter, 1, wx.EXPAND, 0 )
		
		
		self.window_notrow.SetSizer( bSizer14 )
		self.window_notrow.Layout()
		bSizer14.Fit( self.window_notrow )
		self.window_shell = Shell( self.window_3, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		self.window_3.SplitHorizontally( self.window_notrow, self.window_shell, 624 )
		bSizer13.Add( self.window_3, 1, wx.EXPAND, 0 )
		
		
		self.window_plotter.SetSizer( bSizer13 )
		self.window_plotter.Layout()
		bSizer13.Fit( self.window_plotter )
		self.window_list = wx.Panel( self.window_page_splitter, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer20 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.list_data_headers = DragList( self.window_list, wx.ID_ANY, wx.DefaultPosition, wx.Size( 100,-1 ), wx.LC_REPORT )
		bSizer20.Add( self.list_data_headers, 1, wx.ALL|wx.EXPAND, 5 )
		
		
		self.window_list.SetSizer( bSizer20 )
		self.window_list.Layout()
		bSizer20.Fit( self.window_list )
		self.window_page_splitter.SplitVertically( self.window_plotter, self.window_list, 1017 )
		bSizer12.Add( self.window_page_splitter, 1, wx.EXPAND, 0 )
		
		
		self.window_page.SetSizer( bSizer12 )
		self.window_page.Layout()
		bSizer12.Fit( self.window_page )
		self.notebook.AddPage( self.window_page, u"plots", True )
		self.window_data = wx.Panel( self.notebook, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer21 = wx.BoxSizer( wx.HORIZONTAL )
		
		self.m_splitter5 = wx.SplitterWindow( self.window_data, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.SP_3D )
		self.m_splitter5.SetSashGravity( 1 )
		self.m_splitter5.Bind( wx.EVT_IDLE, self.m_splitter5OnIdle )
		
		self.m_panel34 = wx.Panel( self.m_splitter5, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer18 = wx.BoxSizer( wx.VERTICAL )
		
		self.list_data = DataList( self.m_panel34, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LC_REPORT )
		bSizer18.Add( self.list_data, 1, wx.ALL|wx.EXPAND, 5 )
		
		
		self.m_panel34.SetSizer( bSizer18 )
		self.m_panel34.Layout()
		bSizer18.Fit( self.m_panel34 )
		self.m_panel32 = wx.Panel( self.m_splitter5, wx.ID_ANY, wx.DefaultPosition, wx.Size( -1,-1 ), wx.TAB_TRAVERSAL )
		bSizer17 = wx.BoxSizer( wx.VERTICAL )
		
		self.list_datafiles = DataFileList( self.m_panel32, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LC_LIST|wx.LC_NO_HEADER )
		bSizer17.Add( self.list_datafiles, 1, wx.ALL|wx.EXPAND, 5 )
		
		self.button_addfiles = wx.Button( self.m_panel32, wx.ID_ANY, u"Choose data files", wx.DefaultPosition, wx.DefaultSize, 0 )
		bSizer17.Add( self.button_addfiles, 0, wx.ALL, 5 )
		
		
		self.m_panel32.SetSizer( bSizer17 )
		self.m_panel32.Layout()
		bSizer17.Fit( self.m_panel32 )
		self.m_splitter5.SplitVertically( self.m_panel34, self.m_panel32, 937 )
		bSizer21.Add( self.m_splitter5, 1, wx.EXPAND, 5 )
		
		
		self.window_data.SetSizer( bSizer21 )
		self.window_data.Layout()
		bSizer21.Fit( self.window_data )
		self.notebook.AddPage( self.window_data, u"data", False )
		self.window_agg = wx.Panel( self.notebook, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.TAB_TRAVERSAL )
		bSizer211 = wx.BoxSizer( wx.VERTICAL )
		
		self.list_agg = DataList( self.window_agg, wx.ID_ANY, wx.DefaultPosition, wx.DefaultSize, wx.LC_REPORT )
		bSizer211.Add( self.list_agg, 1, wx.ALL|wx.EXPAND, 5 )
		
		
		self.window_agg.SetSizer( bSizer211 )
		self.window_agg.Layout()
		bSizer211.Fit( self.window_agg )
		self.notebook.AddPage( self.window_agg, u"aggr", False )
		
		sizer_frame.Add( self.notebook, 1, wx.EXPAND, 0 )
		
		
		self.SetSizer( sizer_frame )
		self.Layout()
	
	def __del__( self ):
		pass
	
	def window_page_splitterOnIdle( self, event ):
		self.window_page_splitter.SetSashPosition( 1017 )
		self.window_page_splitter.Unbind( wx.EVT_IDLE )
	
	def window_3OnIdle( self, event ):
		self.window_3.SetSashPosition( 624 )
		self.window_3.Unbind( wx.EVT_IDLE )
	
	def window_row_plot_splitterOnIdle( self, event ):
		self.window_row_plot_splitter.SetSashPosition( 536 )
		self.window_row_plot_splitter.Unbind( wx.EVT_IDLE )
	
	def m_splitter5OnIdle( self, event ):
		self.m_splitter5.SetSashPosition( 937 )
		self.m_splitter5.Unbind( wx.EVT_IDLE )
	

