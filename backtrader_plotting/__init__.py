
from .bokeh.bokeh import Bokeh
from .plotly import Plotly

# initialize analyzer tables
from backtrader_plotting.analyzer_tables import inject_datatables
inject_datatables()
