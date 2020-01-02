try:
    from backtrader_plotting.bokeh.bokeh import Bokeh
except ImportError as e:
    pass

from backtrader_plotting.bokeh.optbrowser import OptBrowser

# initialize analyzer tables
from backtrader_plotting.analyzer_tables import inject_datatables
inject_datatables()
