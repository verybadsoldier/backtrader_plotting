import bisect
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
from ..schemes import Blackly
from ..schemes.scheme import Scheme
from bokeh.embed import file_html
from bokeh.resources import CDN
from bokeh.util.browser import view
from backtrader_plotting.utils import inject_humanreadable

from bokeh.layouts import widgetbox
from bokeh.models.widgets import CheckboxButtonGroup


class Bokeh(metaclass=bt.MetaParams):
    params = (('scheme', Blackly()),)

    def __init__(self, **kwargs):
        for pname, pvalue in kwargs.items():
            setattr(self.p.scheme, pname, pvalue)

        self._figures: List = []
        self._data_graph = None
        self._volume_graphs = None
        self._cds: ColumnDataSource = None
        self._analyzers = None
        self._hoverc = HoverContainer()

        if not isinstance(self.p.scheme, Scheme):
            raise Exception("Provided scheme has to be a subclass of backtrader_plogging.schemes.scheme.Scheme")

        inject_humanreadable()

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

        return start, end

    def plot(self, strategy, figid=0, numfigs=1, iplot=True, start=None, end=None, use=None, **kwargs):
        if numfigs > 1:
            raise Exception("numfigs must be 1")
        if use is not None:
            raise Exception("Different backends by 'use' not supported")

        if not strategy.datas:
            return

        if not len(strategy):
            return

        self._analyzers = strategy.analyzers
        # use datetime line of first data as master datetime. also convert it according to user provided tz
        self._dtline = pandas.Series([bt.num2date(x, strategy.datas[0]._tz) for x in strategy.datas[0].lines.datetime.plotrange(start, end)], name="DateTime")

        # TODO: using a pandas.DataFrame is desired. On bokeh 0.12.13 this failed cause of this issue:
        # https://github.com/bokeh/bokeh/issues/7400
        self._cds = ColumnDataSource(data={'datetime': self._dtline})

        self._build_graph(strategy.datas, strategy.getindicators(), strategy.getobservers())

        start, end = Bokeh._get_start_end(strategy, start, end)

        for master, slaves in self._data_graph.items():
            plotabove = getattr(master.plotinfo, 'plotabove', False)
            bf = Figure(self._cds, self._hoverc, start, end, self.p.scheme, type(master), plotabove)
            self._figures.append(bf)

            bf.plot(master, None)

            for s in slaves:
                bf.plot(s, master)

        for v in self._volume_graphs:
            bf = Figure(self._cds, self._hoverc, start, end, self.p.scheme)
            bf.plot_volume(v, 1.0, start, end)

        # apply legend click policy
        for f in self._figures:
            f.figure.legend.click_policy = self.p.scheme.legend_click

        for f in self._figures:
            f.figure.legend.background_fill_color = self.p.scheme.legend_background_color
            f.figure.legend.label_text_color = self.p.scheme.legend_text_color

        # link axis
        for i in range(1, len(self._figures)):
            self._figures[i].figure.x_range = self._figures[0].figure.x_range

        # configure xaxis visibility
        if self.p.scheme.xaxis_pos == "bottom":
            for i, f in enumerate(self._figures):
                f.figure.xaxis.visible = False if i <= len(self._figures) else True

        self._hoverc.apply_hovertips(self._figures)

    def show(self):
        if self.p.scheme.plot_mode == PlotMode.Single:
            self._show_single()
        elif self.p.scheme.plot_mode == PlotMode.Tabs:
            self._show_tabs()
        else:
            raise Exception("Unsupported plot mode")

    def _show_single(self):
        """Print all figures in one column. Plot observers first, then all plotabove then rest"""
        figs = list(self._figures)
        observers = [x for x in figs if issubclass(x.master_type, bt.Observer)]
        figs = [x for x in figs if x not in observers]
        aboves = [x for x in figs if x.plotabove]
        figs = [x for x in figs if x not in aboves]

        figs = [x.figure for x in observers + aboves + figs]

        chart_grid = gridplot([[x] for x in figs], sizing_mode='fixed', toolbar_location='right', toolbar_options={'logo': None})
        panel_charts = Panel(child=chart_grid, title="Charts")
        panel_analyzers = self._get_analyzer_tab()

        tabs = Tabs(tabs=[panel_charts, panel_analyzers])
        self._output_plot_file(tabs)

    def _show_tabs(self):
        figs = list(self._figures)
        observers = [x for x in figs if issubclass(x.master_type, bt.Observer)]
        datas = [x for x in figs if issubclass(x.master_type, bt.DataBase)]
        inds = [x for x in figs if issubclass(x.master_type, bt.Indicator)]

        g = gridplot([[x.figure] for x in observers], sizing_mode='fixed', toolbar_location='left', toolbar_options={'logo': None})
        g2 = gridplot([[x.figure] for x in datas], sizing_mode='fixed', toolbar_location='left', toolbar_options={'logo': None})
        g3 = gridplot([[x.figure] for x in inds], sizing_mode='fixed', toolbar_location='left', toolbar_options={'logo': None})

        tab_observers = Panel(child=g, title="Observers")
        tab_datas = Panel(child=g2, title="Datas")
        tab_indicators = Panel(child=g3, title="Indicators")
        tab_analyzers = self._get_analyzer_tab()

        tabs = Tabs(tabs=[tab_observers, tab_datas, tab_indicators, tab_analyzers])
        self._output_plot_file(tabs)

    def _get_analyzer_tab(self):
        rows = []
        cur_row = []
        for name, analyzer in self._analyzers.getitems():
            acls = type(analyzer)
            hrlables = getattr(acls, 'human_labels', {})
            cds = ColumnDataSource()
            labels = []
            values = []

            def add_to_table(item, baselabel=""):
                for ak, av in item.items():
                    if ak in hrlables:
                        ak = hrlables[ak]
                    label = f"{baselabel} - {ak}" if len(baselabel) > 0 else ak
                    if isinstance(av, (bt.AutoOrderedDict, OrderedDict)):
                        add_to_table(av, label)
                    else:
                        labels.append(label)
                        values.append(av)

            title = Paragraph(text=name, width=200, style={'font-size': 'large'})
            add_to_table(analyzer.rets)

            cds.add(labels, 'labels')
            cds.add(values, 'features')

            columns = [
                TableColumn(field="labels", title="Performance"),  # 0.000% for percentage
                TableColumn(field="features", title="Value", formatter=NumberFormatter(format="0.00")),
            ]
            cur_row.append([title, DataTable(source=cds, columns=columns, width=self.p.scheme.table_width, height=self.p.scheme.table_height, row_headers=False)])
            if len(cur_row) == 2:
                rows.append(cur_row)
                cur_row = []
        return Panel(child=layout(rows), title="Analyzers")

    def _output_plot(self, obj):
        show(obj)

    def _output_plot_file(self, obj, filename="lines.html"):
        env = Environment(loader=PackageLoader('backtrader_plotting.bokeh', 'templates'))
        templ = env.get_template("basic.html.j2")
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

        view(filename)

    def savefig(self, fig, filename, width, height, dpi, tight):
        pass
