import bisect
import os
import tempfile
from collections import OrderedDict
from jinja2 import Environment, PackageLoader
import datetime
from typing import List
import pandas
import backtrader as bt
from ..schemes import PlotMode
from bokeh.models import ColumnDataSource, ToolbarBox, NumberFormatter
from bokeh.models.widgets import Panel, Tabs, DataTable, TableColumn, DateFormatter, Paragraph
from bokeh.layouts import column, gridplot, layout, row
from bokeh.plotting import output_file, show
from bokeh.layouts import Column
from .figure import Figure, HoverContainer
from .datatable import TableGenerator
from ..schemes import Blackly
from ..schemes.scheme import Scheme
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.util.browser import view
from typing import Optional, Union, Tuple
from backtrader_archive import Strategy as btaStrategy
from backtrader_archive.data.archive import DataBase, Indicator, Observer, Analyzer


class FigurePage(object):
    def __init__(self):
        self.figures: List[Figure] = []
        self.cds: ColumnDataSource = None
        self.analyzers: List[Tuple[str, bt.Analyzer, bt.Strategy, Optional[bt.AutoInfoClass]]] = []
        self.strategies: List[bt.Strategy] = None


class Bokeh(metaclass=bt.MetaParams):
    params = (('scheme', Blackly()),)

    def __init__(self, cerebro: bt.Cerebro=None, **kwargs):
        for pname, pvalue in kwargs.items():
            setattr(self.p.scheme, pname, pvalue)

        self._data_graph = None
        self._volume_graphs = None
        self._num_plots = 0
        self._tablegen = TableGenerator(self.p.scheme)
        if not isinstance(self.p.scheme, Scheme):
            raise Exception("Provided scheme has to be a subclass of backtrader_plogging.schemes.scheme.Scheme")

        self._cerebro: bt.Cerebro = cerebro
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

        for i in inds:
            if not hasattr(i, 'plotinfo'):
                # no plotting support - so far LineSingle derived classes
                continue

            if not i.plotinfo.plot or i.plotinfo.plotskip:
                continue

            subplot = i.plotinfo.subplot
            if subplot:
                self._data_graph[i] = []
            else:
                pmaster = Bokeh._resolve_plotmaster(i.plotinfo.plotmaster if i.plotinfo.plotmaster is not None else i.data)
                if pmaster not in self._data_graph:
                    self._data_graph[pmaster] = []
                self._data_graph[pmaster].append(i)

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

    def _find_optreturn_idx(self, optreturn: bt.OptReturn) -> Optional[int]:
        for lst in self._cerebro.runstrats:
            for idx, o in enumerate(lst):
                if o == optreturn:
                    return idx
        raise Exception("OptReturn object not found in cerebro opt list!")

    def plot(self, obj: Union[bt.Strategy, bt.OptReturn], figid=0, numfigs=1, iplot=True, start=None, end=None, use=None, **kwargs):
        if numfigs > 1:
            raise Exception("numfigs must be 1")
        if use is not None:
            raise Exception("Different backends by 'use' not supported")

        # TODO: fix classes
        if isinstance(obj, (bt.Strategy, btaStrategy)):
            self._plot_strategy(obj, start, end, **kwargs)
        elif isinstance(obj, bt.OptReturn):
            if self._cerebro is None:
                raise Exception('No cerebro object provide! Please provide one when plotting OptReturn objects!')
            # find corresponding strategy in cerebro
            optidx = self._find_optreturn_idx(obj)
            strategy = self._cerebro.runningstrats[optidx]
            for name, a in obj.analyzers.getitems():
                self._fp.analyzers.append((name, a, strategy, obj.params))
        else:
            raise Exception(f'Unsupported plot source object: {type(strategy)}')
        return [self._fp]

    def _plot_strategy(self, strategy: bt.Strategy, start=None, end=None, **kwargs):
        if not strategy.datas:
            return

        if not len(strategy):
            return

        strat_figures = []
        # reset hover container to not mix hovers with other strategies
        hoverc = HoverContainer()
        for name, a in strategy.analyzers.getitems():
            self._fp.analyzers.append((name, a, strategy, None))

        # TODO: using a pandas.DataFrame is desired. On bokeh 0.12.13 this failed cause of this issue:
        # https://github.com/bokeh/bokeh/issues/7400

        if self._fp.cds is None:
            # use datetime line of first data as master datetime. also convert it according to user provided tz
            dtline = pandas.Series([bt.num2date(x, strategy.datas[0]._tz) for x in strategy.datas[0].lines.datetime.plotrange(start, end)], name="DateTime")
            self._fp.cds = ColumnDataSource(data={'datetime': dtline})

        self._build_graph(strategy.datas, strategy.getindicators(), strategy.getobservers())

        start, end = Bokeh._get_start_end(strategy, start, end)

        for master, slaves in self._data_graph.items():
            plotabove = getattr(master.plotinfo, 'plotabove', False)
            bf = Figure(strategy, self._fp.cds, hoverc, start, end, self.p.scheme, type(master), plotabove)
            strat_figures.append(bf)

            bf.plot(master, None)

            for s in slaves:
                bf.plot(s, master)

        for v in self._volume_graphs:
            bf = Figure(strategy, self._fp.cds, hoverc, start, end, self.p.scheme)
            bf.plot_volume(v, 1.0, start, end)

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

    def show(self):
        filename = self._generate_output(self._fp)
        view(filename)

        self._reset()
        self._num_plots += 1

    def _generate_output(self, fp: FigurePage, filename:str = None) -> str:
        if self.p.scheme.plot_mode == PlotMode.Single:
            return self._show_single(fp, filename)
        elif self.p.scheme.plot_mode == PlotMode.Tabs:
            return self._show_tabs(fp, filename)
        else:
            raise Exception("Unsupported plot mode")

    def _show_single(self, fp: FigurePage, filename: str=None):
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

        tabs = Tabs(tabs=panels)
        return self._output_plot_file(tabs, filename)

    def _show_tabs(self, fp: FigurePage, filename: str=None):
        figs = list(fp.figures)
        observers = [x for x in figs if issubclass(x.master_type, (bt.Observer, Observer))]
        datas = [x for x in figs if issubclass(x.master_type, (bt.DataBase, DataBase))]
        inds = [x for x in figs if issubclass(x.master_type, (bt.Indicator, Indicator))]

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

        tabs = Tabs(tabs=panels)
        return self._output_plot_file(tabs, filename)

    def _get_analyzer_tab(self, fp: FigurePage) -> Optional[Panel]:
        def _get_column_row_count(col) -> int:
            return sum([x.height for x in col if x.height is not None])

        if len(fp.analyzers) == 0:
            return None

        col_childs = [[], []]
        for name, analyzer, strategy, params in fp.analyzers:
            table_header, elements = self._tablegen.get_analyzers_tables(analyzer, strategy, params)

            col0cnt = _get_column_row_count(col_childs[0])
            col1cnt = _get_column_row_count(col_childs[1])
            col_idx = 0 if col0cnt <= col1cnt else 1
            col_childs[col_idx] += [table_header] + elements

        column1 = column(children=col_childs[0], sizing_mode='fixed')
        childs = [column1]
        if len(col_childs[1]) > 0:
            column2 = column(children=col_childs[1], sizing_mode='fixed')
            childs.append(column2)
        childs = row(children=childs, sizing_mode='fixed')

        return Panel(child=childs, title="Analyzers")

    def _reset(self):
        self._fp = FigurePage()

    def _output_plot(self, obj):
        show(obj)

    def _output_plot_file(self, obj, filename=None, template="basic.html.j2"):
        if filename is None:
            tmpdir = tempfile.gettempdir()
            filename = os.path.join(tmpdir, f"bt_bokeh_plot_{self._num_plots}.html")

        env = Environment(loader=PackageLoader('backtrader_plotting.bokeh', 'templates'))
        templ = env.get_template(template)
        templ.globals['now'] = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        html = file_html(obj,
                         template=templ,
                         resources=CDN,
                         template_variables=dict(
                             datatable_row_color_even=self.p.scheme.table_color_even,
                             datatable_row_color_odd=self.p.scheme.table_color_odd,
                             datatable_header_color=self.p.scheme.table_header_color,
                             tab_active_background_color=self.p.scheme.tab_active_background_color,
                             tab_active_color=self.p.scheme.tab_active_color,

                             tooltip_background_color=self.p.scheme.tooltip_background_color,
                             tooltip_text_color_label=self.p.scheme.tooltip_text_label_color,
                             tooltip_text_color_value=self.p.scheme.tooltip_text_value_color,
                             body_background_color=self.p.scheme.body_fill,
                             headline_color=self.p.scheme.plot_title_text_color,
                             text_color=self.p.scheme.text_color,
                             show_headline=self.p.scheme.show_headline,
                         )
                         )

        with open(filename, 'w') as f:
            f.write(html)

        return filename

    def savefig(self, fig, filename, width, height, dpi, tight):
        self._generate_output(fig, filename)
