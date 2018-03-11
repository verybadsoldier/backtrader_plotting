try:
    from .bokeh.bokeh import Bokeh
except ImportError as e:
    print(f"Bokeh support disabled! Reason: " + str(e))

try:
    from .plotly import Plotly
except ImportError as e:
    print(f"Plotly support disabled! Reason: " + str(e))

# initialize analyzer tables
from backtrader_plotting.analyzer_tables import inject_datatables
inject_datatables()
