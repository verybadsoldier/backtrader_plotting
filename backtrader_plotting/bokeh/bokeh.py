from array import array
from collections import defaultdict
import bisect
import datetime
import itertools
import logging
import re
import os
import sys
import tempfile
from typing import List, Dict, Optional, Union, Tuple

import backtrader as bt

import numpy as np

import pandas as pd

from bokeh.models import ColumnDataSource, Model
from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import column, gridplot

from bokeh.embed import file_html
from bokeh.models.widgets import Div
from bokeh.resources import CDN
from bokeh.util.browser import view

from jinja2 import Environment, PackageLoader

from backtrader_plotting.bokeh.utils import generate_stylesheet, append_cds, get_indicator_data
from backtrader_plotting.utils import convert_by_line_clock, get_clock_line
from backtrader_plotting.bokeh import label_resolver
from backtrader_plotting.utils import find_by_plotid, convert_to_pandas
from backtrader_plotting.bokeh.figureenvelope import FigureEnvelope, HoverContainer
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
    def __init__(self, obj: Union[bt.Strategy, bt.OptReturn]):
        self.figure_envs: List[FigureEnvelope] = []
        self.strategy: Optional[bt.Strategy] = obj if isinstance(obj, bt.Strategy) else None
        self.cds: Optional[ColumnDataSource] = ColumnDataSource(data=dict(datetime=np.array([], dtype=np.datetime64), index=np.array([], np.float64)))
        self.analyzers: List[bt.Analyzer, bt.MetaStrategy, Optional[bt.AutoInfoClass]] = []
        self.model: Optional[Model] = None  # the whole generated model will we attached here after plotting

    def get_tradingdomains(self) -> List[str]:
        tradingdomain = set()
        for fe in self.figure_envs:
            tradingdomain = tradingdomain.union(fe.get_tradingdomains())
        return list(tradingdomain)


class Bokeh(metaclass=bt.MetaParams):
    params = (('scheme', Blackly()),
              ('filename', None),
              ('plotconfig', None),
              ('output_mode', 'show'),
              ('show', True)
              )

    def __init__(self, **kwargs):
        for pname, pvalue in kwargs.items():
            setattr(self.p.scheme, pname, pvalue)

        self._iplot: Optional[bool] = None
        self._tablegen = TableGenerator(self.p.scheme)
        if not isinstance(self.p.scheme, Scheme):
            raise Exception("Provided scheme has to be a subclass of backtrader_plotting.schemes.scheme.Scheme")

        self._initialized: bool = False
        self._is_optreturn: bool = False  # when optreturn is active during optimization then we get a thinned out result only
        self._current_fig_idx: Optional[int] = None
        self.figurepages: List[FigurePage] = []

    def _configure_plotting(self, strategy: bt.Strategy):
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
            ctype, target = k.split(':')

            if ctype == 'r':  # regex
                label = label_resolver.plotobj2label(obj)

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

    def list_tradingdomains(self, strategy: bt.Strategy):
        data_graph, volume_graph = self._build_graph(strategy.datas, strategy.getindicators(), strategy.getobservers())

        lgs = list()
        for master in itertools.chain(data_graph.keys(), volume_graph):
            lg = FigureEnvelope._resolve_tradingdomain(master)
            if isinstance(lg, str) and lg not in lgs:
                lgs.append(lg)

        return lgs

    def _build_graph(self, datas, inds, obs, tradingdomain=None) -> Tuple[Dict, List]:
        data_graph = {}
        volume_graph = []
        for d in datas:
            if not d.plotinfo.plot or not FigureEnvelope.should_filter_by_tradingdomain(d, tradingdomain):
                continue

            pmaster = Bokeh._resolve_plotmaster(d.plotinfo.plotmaster)
            if pmaster is None:
                data_graph[d] = []
            else:
                if pmaster not in data_graph:
                    data_graph[pmaster] = []
                data_graph[pmaster].append(d)

            if self.p.scheme.volume and self.p.scheme.voloverlay is False:
                volume_graph.append(d)

        for obj in itertools.chain(inds, obs):
            if not hasattr(obj, 'plotinfo'):
                # no plotting support - so far LineSingle derived classes
                continue

            # should this indicator be plotted?
            if not obj.plotinfo.plot or obj.plotinfo.plotskip or not FigureEnvelope.should_filter_by_tradingdomain(obj, tradingdomain):
                continue

            # subplot = create a new figure for this indicator
            subplot: bool = obj.plotinfo.subplot
            plotmaster: str = obj.plotinfo.plotmaster
            if subplot and plotmaster is None:
                data_graph[obj] = []
            else:
                plotmaster = plotmaster if plotmaster is not None else get_indicator_data(obj)

                if plotmaster not in data_graph:
                    data_graph[plotmaster] = []
                data_graph[plotmaster].append(obj)

        return data_graph, volume_graph

    @property
    def _cur_figurepage(self) -> FigurePage:
        return self.figurepages[self._current_fig_idx]

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
            end = len(st_dtime) + 1 + end

        return start, end

    def _blueprint_strategy(self, strategy: bt.Strategy, start=None, end=None, tradingdomain=None, **kwargs) -> None:
        if not strategy.datas:
            return

        self._cur_figurepage.analyzers += [a for _, a in strategy.analyzers.getitems()]

        data_graph, volume_graph = self._build_graph(strategy.datas, strategy.getindicators(), strategy.getobservers(), tradingdomain)

        start, end = Bokeh._get_start_end(strategy, start, end)

        # reset hover container to not mix hovers with other strategies
        hoverc = HoverContainer(hover_tooltip_config=self.p.scheme.hover_tooltip_config, is_multidata=len(strategy.datas) > 1)

        strat_figures = []
        for master, slaves in data_graph.items():
            plotorder = getattr(master.plotinfo, 'plotorder', 0)
            figure = FigureEnvelope(strategy, self._cur_figurepage.cds, hoverc, start, end, self.p.scheme, master, plotorder, len(strategy.datas) > 1)

            figure.plot(master, None)

            for s in slaves:
                figure.plot(s, master)
            strat_figures.append(figure)

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

        self._cur_figurepage.figure_envs += strat_figures

        # volume graphs
        for v in volume_graph:
            plotorder = getattr(v.plotinfo, 'plotorder', 0)
            figure = FigureEnvelope(strategy, self._cur_figurepage.cds, hoverc, start, end, self.p.scheme, v, plotorder, is_multidata=len(strategy.datas) > 1)
            figure.plot_volume(v)
            self._cur_figurepage.figure_envs.append(figure)

    def plot_and_generate_optmodel(self, obj: Union[bt.Strategy, bt.OptReturn]):
        self._reset()
        self.plot(obj)

        # we support only one strategy at a time so pass fixed zero index
        # if we ran optresults=False then we have a full strategy object -> pass it to get full plot
        return self.generate_model(0)

    @staticmethod
    def _sort_plotobjects(objs: List[FigureEnvelope]) -> None:
        objs.sort(key=lambda x: x.plotorder)

    def get_figurepage(self, idx: int = 0):
        return self.figurepages[idx]

    # region Generator Methods
    def generate_model(self, figurepage_idx: int = 0) -> Model:
        """Returns a model generated from internal blueprints"""
        if figurepage_idx >= len(self.figurepages):
            raise RuntimeError(f'Cannot generate model for FigurePage with index {figurepage_idx} as there are only {len(self.figurepages)}.')

        figurepage = self.figurepages[figurepage_idx]
        if not self._is_optreturn:
            tabs = self.generate_model_tabs(figurepage)
        else:
            tabs = []

        # now append analyzer tab(s)
        analyzers = figurepage.analyzers
        panel_analyzer = self.get_analyzer_panel(analyzers)
        if panel_analyzer is not None:
            tabs.append(panel_analyzer)

        # append meta tab
        if not self._is_optreturn:
            assert figurepage.strategy is not None
            meta = Div(text=metadata.get_metadata_div(figurepage.strategy))
            metapanel = Panel(child=meta, title="Meta")
            tabs.append(metapanel)

        model = Tabs(tabs=tabs)

        # attach the model to the underlying figure for later reference (e.g. unit test)
        figurepage.model = model

        return model

    def _get_nodata_panel(self):
        chart_grid = gridplot([], toolbar_location=self.p.scheme.toolbar_location, toolbar_options={'logo': None})
        return Panel(child=chart_grid, title="No Data")

    @property
    def is_tabs_single(self) -> bool:
        if self.p.scheme.tabs == 'single':
            return True
        elif self.p.scheme.tabs == 'multi':
            return False
        else:
            raise RuntimeError(f'Invalid tabs parameter "{self.p.scheme.tabs}"')

    def generate_model_tabs(self, fp: FigurePage, tradingdomain=None) -> List[Panel]:
        observers = [x for x in fp.figure_envs if isinstance(x.master, bt.Observer)]
        datas = [x for x in fp.figure_envs if isinstance(x.master, bt.DataBase)]
        inds = [x for x in fp.figure_envs if isinstance(x.master, bt.Indicator)]

        # now assign figures to tabs
        # 1. assign default tabs if no manual tab is assigned
        for figure in [x for x in datas if x.plottab is None]:
            figure.plottab = 'Plots' if self.is_tabs_single else 'Datas'

        for figure in [x for x in inds if x.plottab is None]:
            figure.plottab = 'Plots' if self.is_tabs_single else 'Indicators'

        for figure in [x for x in observers if x.plottab is None]:
            figure.plottab = 'Plots' if self.is_tabs_single else 'Observers'

        # 2. group panels by desired tabs
        # groupby expects the groups to be sorted or else will produce duplicated groups
        sorted_figs = list(itertools.chain(datas, inds, observers))

        # 3. filter tradingdomains
        if tradingdomain is not None:
            filtered = []
            for f in sorted_figs:
                lgs = f.get_tradingdomains()
                for lg in lgs:
                    if lg is True or lg == tradingdomain:
                        filtered.append(f)
            sorted_figs = filtered

        sorted_figs.sort(key=lambda x: x.plottab)
        tabgroups = itertools.groupby(sorted_figs, lambda x: x.plottab)

        panels = []

        def build_panel(objects, panel_title):
            if len(objects) == 0:
                return

            Bokeh._sort_plotobjects(objects)

            g = gridplot([[x.figure] for x in objects],
                         toolbar_options={'logo': None},
                         toolbar_location=self.p.scheme.toolbar_location,
                         sizing_mode=self.p.scheme.plot_sizing_mode,
                         )
            panels.append(Panel(title=panel_title, child=g))

        for tabname, figures in tabgroups:
            build_panel(list(figures), tabname)

        return panels
    # endregion

    def get_analyzer_panel(self, analyzers: List[bt.Analyzer]) -> Optional[Panel]:
        if len(analyzers) == 0:
            return None

        table_width = int(self.p.scheme.analyzer_tab_width / self.p.scheme.analyzer_tab_num_cols)

        acolumns = []
        for analyzer in analyzers:
            table_header, elements = self._tablegen.get_analyzers_tables(analyzer, table_width)

            acolumns.append(column([table_header] + elements))

        childs = gridplot(acolumns, ncols=self.p.scheme.analyzer_tab_num_cols, toolbar_options={'logo': None})
        return Panel(child=childs, title='Analyzers')

    def _output_stylesheet(self, template="basic.css.j2"):
        return generate_stylesheet(self.p.scheme, template)

    def _output_plot_file(self, model, idx, filename=None, template="basic.html.j2"):
        if filename is None:
            tmpdir = tempfile.gettempdir()
            filename = os.path.join(tmpdir, f"bt_bokeh_plot_{idx}.html")

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

    def build_strategy_data(self, strategy: bt.Strategy,
                            start: Optional[datetime.datetime] = None, end: Optional[datetime.datetime] = None,
                            num_back: Optional[int] = None,
                            startidx: int = 0):
        """startidx: index number to write into the dataframe for the index column"""
        strategydf = pd.DataFrame()

        start, end = Bokeh._get_start_end(strategy, start, end)

        strat_clk: array[float] = strategy.lines.datetime.plotrange(start, end)

        # if patches occured then we see duplicate entries in the strategie clock -> clean them
        strat_clk = np.unique(strat_clk)

        if num_back is None:
            num_back = len(strat_clk)

        strat_clk = strat_clk[-num_back:]

        # we use timezone of first data. we might see duplicated timestamps here
        dtline = [bt.num2date(x, strategy.datas[0]._tz) for x in strat_clk]

        # add an index line to use as x-axis (instead of datetime axis) to avoid datetime gaps (e.g. weekends)
        indices = list(range(startidx, startidx + len(dtline)))
        strategydf['index'] = indices

        strategydf['datetime'] = dtline

        for data in strategy.datas:
            source_id = FigureEnvelope._source_id(data)
            df_data = convert_to_pandas(strat_clk, data, start, end, source_id)

            strategydf = strategydf.join(df_data)

            df_colors = FigureEnvelope.build_color_lines(df_data, self.p.scheme, col_open=source_id + 'open', col_close=source_id + 'close', col_prefix=source_id)
            strategydf = strategydf.join(df_colors)

        for obj in itertools.chain(strategy.getindicators(), strategy.getobservers()):
            for lineidx in range(obj.size()):
                line = obj.lines[lineidx]
                source_id = FigureEnvelope._source_id(line)
                dataline = line.plotrange(start, end)

                line_clk = get_clock_line(obj).plotrange(start, end)
                dataline = convert_by_line_clock(dataline, line_clk, strat_clk)
                strategydf[source_id] = dataline

        # apply a proper index (should be identical to 'index' column)
        if strategydf.shape[0] > 0:
            strategydf.index = indices
        return strategydf

    #  region interface for backtrader
    def plot(self, obj: Union[bt.Strategy, bt.OptReturn], figid=0, numfigs=1, iplot=True, start=None, end=None, use=None, fill_data=True, tradingdomain=None, **kwargs):
        """Called by backtrader to plot either a strategy or an optimization result."""

        # prepare new FigurePage
        fp = FigurePage(obj)
        self.figurepages.append(fp)
        self._current_fig_idx = len(self.figurepages) - 1
        self._is_optreturn = isinstance(obj, bt.OptReturn)

        if isinstance(obj, bt.Strategy):
            # only configure plotting for regular backtesting (not for optimization)
            self._configure_plotting(obj)

        if numfigs > 1:
            raise Exception("numfigs must be 1")
        if use is not None:
            raise Exception("Different backends by 'use' not supported")

        self._iplot = iplot and 'ipykernel' in sys.modules

        if isinstance(obj, bt.Strategy):
            self._blueprint_strategy(obj, start, end, tradingdomain, **kwargs)
            if fill_data:
                df: pd.DataFrame = self.build_strategy_data(obj, start, end)

                new_cds = ColumnDataSource.from_df(df)
                append_cds(fp.cds, new_cds)
        elif isinstance(obj, bt.OptReturn):
            # for optresults we only plot analyzers!
            self._cur_figurepage.analyzers += [a for _, a in obj.analyzers.getitems()]
        else:
            raise Exception(f'Unsupported plot source object: {str(type(obj))}')

        return [self._cur_figurepage]

    def show(self):
        """Display a figure (called by backtrader)."""
         # as the plot() function only created the figures and the columndatasources with no data -> now we fill it
        for idx in range(len(self.figurepages)):
            model = self.generate_model(idx)

            if self.p.output_mode in ['show', 'save']:
                if self._iplot:
                    css = self._output_stylesheet()
                    display(HTML(css))
                    show(model)
                else:
                    filename = self._output_plot_file(model, idx, self.p.filename)
                    if self.p.output_mode == 'show':
                        view(filename)
            elif self.p.output_mode == 'memory':
                pass
            else:
                raise RuntimeError(f'Invalid parameter "output_mode" with value: {self.p.output_mode}')

        self._reset()

    def _reset(self):
        self.figurepages = []
        self._is_optreturn = False
    #  endregion
