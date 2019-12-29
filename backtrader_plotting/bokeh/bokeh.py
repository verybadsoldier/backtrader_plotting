from array import array
import bisect
import datetime
import inspect
import itertools
import logging
import re
import os
import sys
import tempfile
from typing import List, Dict, Callable, Optional, Union

import backtrader as bt
from backtrader_plotting.utils import get_data_obj

from bokeh.models import ColumnDataSource, Model
from bokeh.models.widgets import Panel, Tabs, DataTable, TableColumn
from bokeh.layouts import column, gridplot, row
from bokeh.server.server import Server
from bokeh.document import Document
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.embed import file_html
from bokeh.models.widgets import NumberFormatter, StringFormatter, Div
from bokeh.resources import CDN
from bokeh.util.browser import view

from jinja2 import Environment, PackageLoader

from backtrader_plotting.bokeh.utils import generate_stylesheet
from backtrader_plotting.bokeh import label_resolver
from backtrader_plotting.utils import find_by_plotid
from backtrader_plotting import bttypes
from backtrader_plotting.bokeh.figure import Figure, HoverContainer
from backtrader_plotting.bokeh.datatable import TableGenerator
from backtrader_plotting.schemes import Blackly
from backtrader_plotting.schemes.scheme import Scheme
from backtrader_plotting.html import metadata

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


class Bokeh(metaclass=bt.MetaParams):
    params = (('scheme', Blackly()),
              ('filename', None),
              ('plotconfig', None),
              )

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

        self._initialized: bool = False
        self._figurepage: Optional[FigurePage] = None
        self._cerebro: Optional[bt.Cerebro] = None

    def _configure_plotting(self):
        for strategy in self._cerebro.runningstrats:
            datas, inds, obs = strategy.datas, strategy.getindicators(), strategy.getobservers()

            for objs in [datas, inds, obs]:
                for idx, obj in enumerate(objs):
                    self._configure_plotobject(obj, idx, strategy)

    def _configure_plotobject(self, obj, idx, strategy):
        if self.p.plotconfig is None:
            return

        def apply_config(obj, config):
            for k, v in config.items():
                if k == 'plotmaster':
                    # this needs special treatment since a string is passed but we need to set the actual obj
                    v = find_by_plotid(strategy, v)

                setattr(obj.plotinfo, k, v)

        for k, config in self.p.plotconfig.items():
            ctype, target = k.split(':', 2)

            if ctype == 'r':  # regex
                label = Figure.plotobj2label(obj)

                m = re.match(target, label)
                if m:
                    apply_config(obj, config)
            elif ctype[0] == '#':  # index
                target_type, target_idx = target.split('-')

                target_types = {
                    'i': bt.Indicator,
                    'o': bt.Observer,
                    'd': bt.AbstractDataBase,
                }

                # check if instance type matches
                if not isinstance(obj, target_types[target_type]):
                    continue

                if int(target_idx) != idx:
                    continue
                apply_config(obj, config)
            elif ctype == 'id':  # plotid
                plotid = getattr(obj.plotinfo, 'plotid', None)
                if plotid is None or plotid != target:
                    continue
                apply_config(obj, config)
            else:
                raise RuntimeError(f'Unknown config type in plotting config: {k}')

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

        for obj in itertools.chain(inds, obs):
            if not hasattr(obj, 'plotinfo'):
                # no plotting support - so far LineSingle derived classes
                continue

            # should this indicator be plotted?
            if not obj.plotinfo.plot or obj.plotinfo.plotskip:
                continue

            # subplot = create a new figure for this indicator
            subplot: bool = obj.plotinfo.subplot
            plotmaster: str = obj.plotinfo.plotmaster
            if subplot and plotmaster is None:
                self._data_graph[obj] = []
            else:
                plotmaster = plotmaster if plotmaster is not None else obj.data
                if plotmaster not in self._data_graph:
                    self._data_graph[plotmaster] = []
                self._data_graph[plotmaster].append(obj)
    @property
    def figures(self):
        return self._figurepage.figures

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

    def generate_result_model(self, result: Union[List[bt.Strategy], List[List[bt.OptReturn]]], columns=None, num_item_limit=None) -> Model:
        """Generates a model from a result object"""
        if bttypes.is_optresult(result) or bttypes.is_ordered_optresult(result):
            return self.generate_optresult_model(result, columns, num_item_limit)
        elif bttypes.is_btresult(result):
            for s in result:
                self.plot(s)
            return self.generate_model()
        else:
            raise Exception(f"Unsupported result type: {str(result)}")

    def plot_result(self, result: Union[List[bt.Strategy], List[List[bt.OptReturn]]], columns=None, ioloop=None):
        """Plot a cerebro result. Pass either a list of strategies or a list of list of optreturns."""
        if bttypes.is_optresult(result) or bttypes.is_ordered_optresult(result):
            self.run_optresult_server(result, columns, ioloop=ioloop)
        elif bttypes.is_btresult(result):
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
        self._figurepage.analyzers += [a for _, a in strategy.analyzers.getitems()]

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

        if self._figurepage.cds is None:
            # we use timezone of first data
            dtline = [bt.num2date(x, strategy.datas[0]._tz) for x in strat_clk]

            # add an index line to use as x-axis (instead of datetime axis) to avoid datetime gaps (e.g. weekends)
            indices = list(range(0, len(dtline)))
            self._figurepage.cds = ColumnDataSource(data=dict(datetime=dtline, index=indices))

        self._build_graph(strategy.datas, strategy.getindicators(), strategy.getobservers())

        start, end = Bokeh._get_start_end(strategy, start, end)

        # reset hover container to not mix hovers with other strategies
        hoverc = HoverContainer()

        for master, slaves in self._data_graph.items():
            plotabove = getattr(master.plotinfo, 'plotabove', False)
            figure = Figure(strategy, self._figurepage.cds, hoverc, start, end, self.p.scheme, master, plotabove)
            strat_figures.append(figure)

            figure.plot(master, strat_clk, None)

            for s in slaves:
                figure.plot(s, strat_clk, master)

        for v in self._volume_graphs:
            figure = Figure(strategy, self._figurepage.cds, hoverc, start, end, self.p.scheme)
            figure.plot_volume(v, strat_clk, 1.0, start, end)

        for f in strat_figures:
            f.figure.legend.click_policy = self.p.scheme.legend_click
            f.figure.legend.location = self.p.scheme.legend_location
            f.figure.legend.background_fill_color = self.p.scheme.legend_background_color
            f.figure.legend.label_text_color = self.p.scheme.legend_text_color
            f.figure.legend.orientation = self.p.scheme.legend_orientation

        # link axis
        for i in range(1, len(strat_figures)):
            strat_figures[i].figure.x_range = strat_figures[0].figure.x_range

        # configure xaxis visibility
        if self.p.scheme.xaxis_pos == "bottom":
            for i, f in enumerate(strat_figures):
                f.figure.xaxis.visible = False if i <= len(strat_figures) else True

        hoverc.apply_hovertips(strat_figures)

        self._figurepage.figures += strat_figures

    def plot_and_generate_model(self, strats):
        self._reset()
        for strat in strats:
            self.plot(strat)
        return self.generate_model()

    # region Generator Methods
    def generate_model(self) -> Model:
        """Returns a model generated from internal blueprints"""
        if self.p.scheme.plot_mode == 'single':
            panels = self._generate_model_single(self._figurepage)
        elif self.p.scheme.plot_mode == 'tabs':
            panels = self._generate_model_panels(self._figurepage)
        else:
            raise Exception(f"Unsupported plot mode: {self.p.scheme.plot_mode}")

        # now append analyzer tab(s)
        for strategy in self._cerebro.runningstrats:
            strat_analyzers = [x for x in self._figurepage.analyzers if x.strategy is strategy]

            analyzer_name_suffix = None
            if len(self._cerebro.runningstrats) > 1:
                analyzer_name_suffix = label_resolver.strategy2shortname(strategy)
            panel_analyzer = self._get_analyzer_panel(strat_analyzers, analyzer_name_suffix)
            if panel_analyzer is not None:
                panels.append(panel_analyzer)

        meta = Div(text=metadata.get_metadata_div(self._cerebro))
        metapanel = Panel(child=meta, title="Meta")
        panels.append(metapanel)

        return Tabs(tabs=panels)

    def _generate_model_single(self, fp: FigurePage) -> List[Panel]:
        """Print all figures in one column. Plot observers first, then all plotabove then rest"""
        figs = list(fp.figures)
        observers = [x for x in figs if isinstance(x.master, bt.Observer)]
        figs = [x for x in figs if x not in observers]
        aboves = [x for x in figs if x.plotabove]
        figs = [x for x in figs if x not in aboves]
        figs = [x.figure for x in observers + aboves + figs]

        panels = []
        if len(figs) > 0:
            chart_grid = gridplot([[x] for x in figs],
                                  toolbar_options={'logo': None},
                                  toolbar_location=self.p.scheme.toolbar_location,
                                  sizing_mode=self.p.scheme.plot_sizing_mode,
                                  )
            panels.append(Panel(child=chart_grid, title="Charts"))

        return panels

    def _get_nodata_panel(self):
        chart_grid = gridplot([], toolbar_location=self.p.scheme.toolbar_location, toolbar_options={'logo': None})
        return Panel(child=chart_grid, title="No Data")

    def _generate_model_panels(self, fp: FigurePage) -> List[Panel]:
        figs = list(fp.figures)
        observers = [x for x in figs if isinstance(x.master, bt.Observer)]
        datas = [x for x in figs if isinstance(x.master, bt.DataBase)]
        inds = [x for x in figs if isinstance(x.master, bt.Indicator)]

        panels = []

        def add_panel(obj, title):
            if len(obj) == 0:
                return
            g = gridplot([[x.figure] for x in obj],
                         toolbar_options={'logo': None},
                         toolbar_location=self.p.scheme.toolbar_location,
                         sizing_mode=self.p.scheme.plot_sizing_mode,
                         )
            panels.append(Panel(title=title, child=g))

        add_panel(datas, "Datas")
        add_panel(inds, "Indicators")

        # group observers by associated strategy
        for strategy in self._cerebro.runningstrats:
            strat_observers = [x for x in observers if x.master._owner is strategy]

            if len(self._cerebro.runningstrats) > 1:
                # add a a strategy suffix if we show multiple tabs
                title_suffix = f' - {label_resolver.strategy2shortname(strategy)}'
            else:
                title_suffix = ''

            add_panel(strat_observers, "Observers" + title_suffix)

        return panels
    # endregion

    def _get_analyzer_panel(self, analyzers: List[bt.Analyzer], name_suffix=None) -> Optional[Panel]:
        def _get_column_row_count(col) -> int:
            return sum([x.height for x in col if x.height is not None])

        if len(analyzers) == 0:
            return None

        table_width = int(self.p.scheme.analyzer_tab_width / self.p.scheme.analyzer_tab_num_cols)

        acolumns = []
        for analyzer in analyzers:
            table_header, elements = self._tablegen.get_analyzers_tables(analyzer, table_width)

            acolumns.append(column([table_header] + elements))

        childs = gridplot(acolumns, ncols=self.p.scheme.analyzer_tab_num_cols, toolbar_options={'logo': None})
        name = "Analyzers"
        if name_suffix is not None:
            name += f" - {name_suffix}"
        return Panel(child=childs, title=name)

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
        """Called by backtrader to plot either a strategy or an optimization result."""
        if numfigs > 1:
            raise Exception("numfigs must be 1")
        if use is not None:
            raise Exception("Different backends by 'use' not supported")

        self._iplot = iplot and 'ipykernel' in sys.modules

        # do initialization
        if not self._initialized:
            self._cerebro = obj.cerebro
            self._figurepage = FigurePage()
            self._configure_plotting()
            self._initialized = True

        if isinstance(obj, bt.Strategy):
            self._blueprint_strategy(obj, start, end, **kwargs)
        elif isinstance(obj, bt.OptReturn):
            # for optresults we only plot analyzers!
            self._figurepage.analyzers += [a for _, a in obj.analyzers.getitems()]
        else:
            raise Exception(f'Unsupported plot source object: {str(type(obj))}')
        return [self._figurepage]

    def show(self):
        """Display a figure (called by backtrader)."""
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
        self._figurepage = FigurePage()
        self._plots_configured = False

    @staticmethod
    def _get_limited_optresult(optresult: Union[bttypes.OptResult, bttypes.OrderedOptResult], num_item_limit=None):
        if num_item_limit is None:
            return optresult
        return optresult[0:num_item_limit]

    @staticmethod
    def _get_opt_count(optresult: Union[bttypes.OptResult, bttypes.OrderedOptResult]):
        if isinstance(optresult[0], dict):
            # OrderedOptResult
            return len(optresult['optresult'][0]['result'])
        else:
            # OptResult
            return len(optresult[0])

    def generate_optresult_model(self, optresult: Union[bttypes.OptResult, bttypes.OrderedOptResult], columns=None, num_item_limit=None) -> Model:
        """Generates and returns an interactive model for an OptResult or an OrderedOptResult"""
        cds = ColumnDataSource()
        tab_columns = []

        col_formatter_num = NumberFormatter(format='0.000')
        col_formatter_str = StringFormatter()
        opts = optresult if bttypes.is_optresult(optresult) else [x.result for x in optresult.optresult]
        if bttypes.is_ordered_optresult(optresult):
            benchmarks = [x.benchmark for x in Bokeh._get_limited_optresult(optresult.optresult, num_item_limit)]
            cds.add(benchmarks, "benchmark")
            tab_columns.append(TableColumn(field='benchmark', title=optresult.benchmark_label, sortable=False, formatter=col_formatter_num))

        for idx, strat in enumerate(opts[0]):
            # add suffix when dealing with more than 1 strategy
            strat_suffix = ''
            if len(opts[0]) > 1:
                strat_suffix = f' [{idx}]'

            for name, val in strat.params._getitems():
                # get value for the current param for all results
                pvals = []
                formatter = col_formatter_num
                for opt in Bokeh._get_limited_optresult(opts, num_item_limit):
                    param = opt[idx].params._get(name)
                    if inspect.isclass(param):
                        paramstr = param.__name__
                    else:
                        paramstr = param
                    pvals.append(paramstr)

                if len(pvals) > 0 and isinstance(pvals[0], str):
                    formatter = col_formatter_str
                tab_columns.append(TableColumn(field=f"{idx}_{name}", title=f'{name}{strat_suffix}', sortable=False, formatter=formatter))

                cds.add(pvals, f"{idx}_{name}")

        # add user columns specified by parameter 'columns'
        if columns is not None:
            for k, v in columns.items():
                ll = [str(v(x)) for x in Bokeh._get_limited_optresult(optresult, num_item_limit)]
                cds.add(ll, k)
                tab_columns.append(TableColumn(field=k, title=k, sortable=False, formatter=col_formatter_str))

        selector = DataTable(source=cds, columns=tab_columns, width=1600, height=150)

        model = column([selector, self.plot_and_generate_model(opts[0])])

        def update(_name, _old, new):
            if len(new) == 0:
                return

            stratidx = new[0]
            model.children[-1] = self.plot_and_generate_model(opts[stratidx])

        cds.selected.on_change('indices', update)
        return model

    def run_optresult_server(self, result: bttypes.OptResult, columns: Dict[str, Callable]=None, ioloop=None):
        """Serves an optimization resulst as a Bokeh application running on a web server"""
        def make_document(doc: Document):
            doc.title = "Backtrader Optimization Result"

            env = Environment(loader=PackageLoader('backtrader_plotting.bokeh', 'templates'))
            doc.template = env.get_template("basic.html.j2")

            model = self.generate_optresult_model(result, columns)
            doc.add_root(model)

        Bokeh._run_server(make_document, ioloop=ioloop)

    @staticmethod
    def _run_server(fnc_make_document, iplot=True, notebook_url="localhost:8889", port=80, ioloop=None):
        """Runs a Bokeh webserver application. Documents will be created using fnc_make_document"""
        handler = FunctionHandler(fnc_make_document)
        app = Application(handler)
        if iplot and 'ipykernel' in sys.modules:
            show(app, notebook_url=notebook_url)
        else:
            apps = {'/': app}

            print("Open your browser here: http://localhost")
            server = Server(apps, port=port, io_loop=ioloop)
            server.run_until_shutdown()
    #  endregion
