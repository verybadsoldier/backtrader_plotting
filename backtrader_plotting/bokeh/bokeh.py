from ._utils import generate_stylesheet
import bisect
import os
import sys
import tempfile
from jinja2 import Environment, PackageLoader
import datetime
from typing import List, Dict, Callable
import backtrader as bt
from bokeh.models import ColumnDataSource, Model
from bokeh.models.widgets import Panel, Tabs, Select, DataTable, TableColumn
from bokeh.layouts import column, gridplot, row
from bokeh.server.server import Server
from bokeh.document import Document
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from backtrader_plotting.utils import get_data_obj

from .figure import Figure, HoverContainer
from .datatable import TableGenerator
from ..schemes import Blackly
from ..schemes.scheme import Scheme
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.util.browser import view
from typing import Optional, Union, Tuple
import logging
from array import array

_logger = logging.getLogger(__name__)


if 'ipykernel' in sys.modules:
    from IPython.core.display import display, HTML
    from bokeh.io import output_notebook, show
    output_notebook()


class FigurePage(object):
    def __init__(self):
        self.figures: List[Figure] = []
        self.cds: ColumnDataSource = None
        self.analyzers: List[bt.Analyzer, bt.MetaStrategy, Optional[bt.AutoInfoClass]] = []
        self.strategies: List[bt.Strategy] = None


class Bokeh(metaclass=bt.MetaParams):
    params = (('scheme', Blackly()),
              ('filename', None))

    def __init__(self, **kwargs):
        for pname, pvalue in kwargs.items():
            setattr(self.p.scheme, pname, pvalue)

        self._iplot: bool = None
        self._data_graph = None
        self._volume_graphs = None
        self._num_plots = 0
        self._tablegen = TableGenerator(self.p.scheme)
        if not isinstance(self.p.scheme, Scheme):
            raise Exception("Provided scheme has to be a subclass of backtrader_plotting.schemes.scheme.Scheme")

        self._fp = FigurePage()

    def _build_graph(self, datas, inds, obs):
        self._data_graph = {}
        self._volume_graphs = []
        for d in datas:
            if not d.plotinfo.plot:
                continue

            pmaster = Bokeh._resolve_plotmaster(d.plotinfo.plotmaster)
            if pmaster is None:
                self._data_graph[d] = []
            else:
                if pmaster not in self._data_graph:
                    self._data_graph[pmaster] = []
                self._data_graph[pmaster].append(d)

            if self.p.scheme.volume and self.p.scheme.voloverlay is False:
                self._volume_graphs.append(d)

        # Sort observers in the different lists/dictionaries
        for o in obs:
            if not o.plotinfo.plot or o.plotinfo.plotskip:
                continue

            if o.plotinfo.subplot:
                self._data_graph[o] = []
            else:
                pmaster = Bokeh._resolve_plotmaster(o.plotinfo.plotmaster or o.data)
                if pmaster not in self._data_graph:
                    self._data_graph[pmaster] = []
                self._data_graph[pmaster].append(o)

        for ind in inds:
            if not hasattr(ind, 'plotinfo'):
                # no plotting support - so far LineSingle derived classes
                continue

            # should this indicator be plotted?
            if not ind.plotinfo.plot or ind.plotinfo.plotskip:
                continue

            # subplot = create a new figure for this indicator
            subplot = ind.plotinfo.subplot
            if subplot:
                self._data_graph[ind] = []
            else:
                pm = ind.plotinfo.plotmaster if ind.plotinfo.plotmaster is not None else ind.data
                pm = get_data_obj(pm)
                pmaster = Bokeh._resolve_plotmaster(pm)
                if pmaster not in self._data_graph:
                    self._data_graph[pmaster] = []
                self._data_graph[pmaster].append(ind)

    @property
    def figures(self):
        return self._fp.figures

    @staticmethod
    def _resolve_plotmaster(obj):
        if obj is None:
            return None

        while True:
            pm = obj.plotinfo.plotmaster
            if pm is None:
                break
            else:
                obj = pm
        return obj

    @staticmethod
    def _get_start_end(strategy, start, end):
        st_dtime = strategy.lines.datetime.array
        if start is None:
            start = 0
        if end is None:
            end = len(st_dtime)

        if isinstance(start, datetime.date):
            start = bisect.bisect_left(st_dtime, bt.date2num(start))

        if isinstance(end, datetime.date):
            end = bisect.bisect_right(st_dtime, bt.date2num(end))

        if end < 0:
            end = len(st_dtime) + 1 + end  # -1 =  len() -2 = len() - 1

        return start, end

    @staticmethod
    def _is_valid_result(result: List):
        """Raises if result is not a list. Return False if result is empty."""
        if not isinstance(result, List):
            raise Exception("'result' has to be a List")
        return len(result) != 0

    def generate_result_model(self, result: Union[List[bt.Strategy], List[List[bt.OptReturn]]], columns=None) -> Model:
        """Generates a model from a result object"""
        if not Bokeh._is_valid_result(result):
            return

        if Bokeh._is_optresult(result):
            return self.generate_optresult_model(result, columns)
        elif Bokeh._is_btresult(result):
            for s in result:
                self.plot(s)
            return self.generate_model()
        else:
            raise Exception(f"Unsupported result type: {str(result)}")

    @staticmethod
    def _is_optresult(result: Union[List[bt.Strategy], List[List[bt.OptReturn]]]):
        return isinstance(result[0], List) and len(result[0]) > 0 and isinstance(result[0][0], (bt.OptReturn, bt.Strategy))

    @staticmethod
    def _is_btresult(result: Union[List[bt.Strategy], List[List[bt.OptReturn]]]):
        return isinstance(result[0], bt.Strategy)

    def plot_result(self, result: Union[List[bt.Strategy], List[List[bt.OptReturn]]], columns=None):
        """Plots a cerebro result. Pass either a list of strategies or a list of list of optreturns"""
        if not Bokeh._is_valid_result(result):
            return

        if Bokeh._is_optresult(result):
            self.run_optresult_server(result, columns)
        elif Bokeh._is_btresult(result):
            for s in result:
                self.plot(s)
            self.show()
        else:
            raise Exception(f"Unsupported result type: {str(result)}")

    def _blueprint_strategy(self, strategy: bt.Strategy, start=None, end=None, **kwargs):
        if not strategy.datas:
            return

        if not len(strategy):
            return

        strat_figures = []
        self._fp.analyzers = [a for _, a in strategy.analyzers.getitems()]

        st_dtime = strategy.lines.datetime.plot()
        if start is None:
            start = 0
        if end is None:
            end = len(st_dtime)

        if isinstance(start, datetime.date):
            start = bisect.bisect_left(st_dtime, bt.date2num(start))

        if isinstance(end, datetime.date):
            end = bisect.bisect_right(st_dtime, bt.date2num(end))

        if end < 0:
            end = len(st_dtime) + 1 + end  # -1 =  len() -2 = len() - 1

        # TODO: using a pandas.DataFrame is desired. On bokeh 0.12.13 this failed cause of this issue:
        # https://github.com/bokeh/bokeh/issues/7400
        strat_clk: array[float] = strategy.lines.datetime.plotrange(start, end)

        if self._fp.cds is None:
            # we use timezone of first data
            dtline = [bt.num2date(x, strategy.datas[0]._tz) for x in strat_clk]

            # add an index line to use as x-axis (instead of datetime axis) to avoid datetime gaps (e.g. weekends)
            indices = list(range(0, len(dtline)))
            self._fp.cds = ColumnDataSource(data=dict(datetime=dtline, index=indices))

        self._build_graph(strategy.datas, strategy.getindicators(), strategy.getobservers())

        start, end = Bokeh._get_start_end(strategy, start, end)

        # reset hover container to not mix hovers with other strategies
        hoverc = HoverContainer()

        for master, slaves in self._data_graph.items():
            plotabove = getattr(master.plotinfo, 'plotabove', False)
            bf = Figure(strategy, self._fp.cds, hoverc, start, end, self.p.scheme, type(master), plotabove)
            strat_figures.append(bf)

            bf.plot(master, strat_clk, None)

            for s in slaves:
                bf.plot(s, strat_clk, master)

        for v in self._volume_graphs:
            bf = Figure(strategy, self._fp.cds, hoverc, start, end, self.p.scheme)
            bf.plot_volume(v, strat_clk, 1.0, start, end)

        # apply legend click policy
        for f in strat_figures:
            f.figure.legend.click_policy = self.p.scheme.legend_click

        for f in strat_figures:
            f.figure.legend.background_fill_color = self.p.scheme.legend_background_color
            f.figure.legend.label_text_color = self.p.scheme.legend_text_color

        # link axis
        for i in range(1, len(strat_figures)):
            strat_figures[i].figure.x_range = strat_figures[0].figure.x_range

        # configure xaxis visibility
        if self.p.scheme.xaxis_pos == "bottom":
            for i, f in enumerate(strat_figures):
                f.figure.xaxis.visible = False if i <= len(strat_figures) else True

        hoverc.apply_hovertips(strat_figures)

        self._fp.figures += strat_figures

    def plot_and_generate_model(self, strats):
        self._reset()
        for strat in strats:
            self.plot(strat)
        return self.generate_model()

    # region Generator Methods
    def generate_model(self) -> Model:
        """Returns a model generated from internal blueprints"""
        if self.p.scheme.plot_mode == 'single':
            return self._generate_model_single(self._fp)
        elif self.p.scheme.plot_mode == 'tabs':
            return self._generate_model_tabs(self._fp)
        else:
            raise Exception(f"Unsupported plot mode: {self.p.scheme.plot_mode}")

    def _generate_model_single(self, fp: FigurePage):
        """Print all figures in one column. Plot observers first, then all plotabove then rest"""
        figs = list(fp.figures)
        observers = [x for x in figs if issubclass(x.master_type, bt.Observer)]
        figs = [x for x in figs if x not in observers]
        aboves = [x for x in figs if x.plotabove]
        figs = [x for x in figs if x not in aboves]
        figs = [x.figure for x in observers + aboves + figs]

        panels = []
        if len(figs) > 0:
            chart_grid = gridplot([[x] for x in figs], sizing_mode='fixed', toolbar_location='right', toolbar_options={'logo': None})
            panels.append(Panel(child=chart_grid, title="Charts"))

        panel_analyzers = self._get_analyzer_tab(fp)
        if panel_analyzers is not None:
            panels.append(panel_analyzers)

        return Tabs(tabs=panels)

    def _generate_model_tabs(self, fp: FigurePage):
        figs = list(fp.figures)
        observers = [x for x in figs if issubclass(x.master_type, bt.Observer)]
        datas = [x for x in figs if issubclass(x.master_type, bt.DataBase)]
        inds = [x for x in figs if issubclass(x.master_type, bt.Indicator)]

        panels = []

        def add_panel(obj, title):
            if len(obj) == 0:
                return
            g = gridplot([[x.figure] for x in obj], sizing_mode='fixed', toolbar_location='left', toolbar_options={'logo': None})
            panels.append(Panel(title=title, child=g))

        add_panel(datas, "Datas")
        add_panel(inds, "Indicators")
        add_panel(observers, "Observers")

        p_analyzers = self._get_analyzer_tab(fp)
        if p_analyzers is not None:
            panels.append(p_analyzers)

        return Tabs(tabs=panels)
    # endregion

    def _get_analyzer_tab(self, fp: FigurePage) -> Optional[Panel]:
        def _get_column_row_count(col) -> int:
            return sum([x.height for x in col if x.height is not None])

        if len(fp.analyzers) == 0:
            return None

        table_width = int(self.p.scheme.analyzer_tab_width / self.p.scheme.analyzer_tab_num_cols)
        col_childs = []
        for _ in range(0, self.p.scheme.analyzer_tab_num_cols):
            col_childs.append([])

        for a in fp.analyzers:
            table_header, elements = self._tablegen.get_analyzers_tables(a, table_width)

            col_childs = sorted(col_childs, key=lambda x: _get_column_row_count(x))
            col_childs[0] += [table_header] + elements

        childs = []
        for c in col_childs:
            if len(c) == 0:
                break
            childs.append(column(children=c, sizing_mode='fixed'))

        childs = row(children=childs, sizing_mode='fixed')
        return Panel(child=childs, title="Analyzers")

    def _output_stylesheet(self, template="basic.css.j2"):
        return generate_stylesheet(self.p.scheme, template)

    def _output_plot_file(self, model, filename=None, template="basic.html.j2"):
        if filename is None:
            tmpdir = tempfile.gettempdir()
            filename = os.path.join(tmpdir, f"bt_bokeh_plot_{self._num_plots}.html")

        env = Environment(loader=PackageLoader('backtrader_plotting.bokeh', 'templates'))
        templ = env.get_template(template)
        templ.globals['now'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        html = file_html(model,
                         template=templ,
                         resources=CDN,
                         template_variables=dict(
                             stylesheet=self._output_stylesheet(),
                             show_headline=self.p.scheme.show_headline,
                             )
                         )

        with open(filename, 'w') as f:
            f.write(html)

        return filename

    def savefig(self, fig, filename, width, height, dpi, tight):
        self._generate_output(fig, filename)

    #  region interface for backtrader
    def plot(self, obj: Union[bt.Strategy, bt.OptReturn], figid=0, numfigs=1, iplot=True, start=None, end=None, use=None, **kwargs):
        """Called by backtrader to plot either a strategy or an optimization results"""

        if numfigs > 1:
            raise Exception("numfigs must be 1")
        if use is not None:
            raise Exception("Different backends by 'use' not supported")

        self._iplot = iplot and 'ipykernel' in sys.modules

        if isinstance(obj, bt.Strategy):
            self._blueprint_strategy(obj, start, end, **kwargs)
        elif isinstance(obj, bt.OptReturn):
            if not hasattr(obj, 'strategycls'):
                raise Exception("Missing field 'strategycls' in OptReturn. Include this commit in your backtrader package to fix it: 'https://github.com/verybadsoldier/backtrader/commit/f03a0ed115338ed8f074a942f6520b31c630bcfb'")
            self._fp.analyzers = [a for _, a in obj.analyzers.getitems()]
        else:
            raise Exception(f'Unsupported plot source object: {type(obj)}')
        return [self._fp]

    def show(self):
        """Called by backtrader to display a figure"""
        model = self.generate_model()
        if self._iplot:
            css = self._output_stylesheet()
            display(HTML(css))
            show(model)
        else:
            filename = self._output_plot_file(model, self.p.filename)
            view(filename)

        self._reset()
        self._num_plots += 1
    #  endregion

    def _reset(self):
        self._fp = FigurePage()

    def generate_optresult_model(self, optresult: List[List[bt.OptReturn]], columns=None) -> Model:
        """Generates and returns an interactive model"""
        #o = list(self._options.keys())
        #selector = Select(title="Result:", value="result", options=o)

        cds = ColumnDataSource()
        tab_columns = []

        for idx, strat in enumerate(optresult[0]):
            # add suffix when dealing with more than 1 strategy
            strat_suffix = ''
            if len(optresult[0]):
                strat_suffix = f' [{idx}]'

            for name, val in strat.params._getitems():
                tab_columns.append(TableColumn(field=f"{idx}_{name}", title=f'{name}{strat_suffix}'))

                # get value for the current param for all results
                pvals = []
                for res in optresult:
                    pvals.append(res[idx].params._get(name))
                cds.add(pvals, f"{idx}_{name}")

        # add user columns specified by parameter 'columns'
        if columns is not None:
            for k, v in columns.items():
                ll = [str(v(x)) for x in optresult]
                cds.add(ll, k)
                tab_columns.append(TableColumn(field=k, title=k))

        selector = DataTable(source=cds, columns=tab_columns, width=1600, height=160)

        model = column([selector, self.plot_and_generate_model(optresult[idx])])

        def update(name, old, new):
            if len(new['1d']['indices']) == 0:
                return

            stratidx = new['1d']['indices'][0]
            model.children[-1] = self.plot_and_generate_model(optresult[stratidx])

        cds.on_change('selected', update)
        return model

    def run_optresult_server(self, result: List[List[bt.OptReturn]], columns: Dict[str, Callable]=None):
        """Serves an optimization resulst as a Bokeh application running on a web server"""
        if len(result) == 0:
            return

        if not isinstance(result[0], List):
            raise Exception("Passes 'result' object is no optimization result!")

        def make_document(doc: Document):
            doc.title = "Backtrader Optimization Result"

            env = Environment(loader=PackageLoader('backtrader_plotting.bokeh', 'templates'))
            doc.template = env.get_template("basic.html.j2")

            model = self.generate_optresult_model(result, columns)
            doc.add_root(model)

        Bokeh._run_server(make_document)

    @staticmethod
    def _run_server(fnc_make_document, iplot=True, notebook_url="localhost:8889", port=80):
        """Runs a Bokeh webserver application. Documents will be created using fnc_make_document"""
        handler = FunctionHandler(fnc_make_document)
        app = Application(handler)
        if iplot and 'ipykernel' in sys.modules:
            show(app, notebook_url=notebook_url)
        else:
            apps = {'/': app}

            print("Open your browser here: http://localhost")
            server = Server(apps, port=port)
            server.run_until_shutdown()
    #  endregion
