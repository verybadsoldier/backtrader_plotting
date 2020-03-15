from array import array
import collections
import itertools
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

import backtrader as bt

import pandas as pd

from bokeh.models import Span
from bokeh.plotting import figure
from bokeh.models import HoverTool, CrosshairTool
from bokeh.models import LinearAxis, DataRange1d, Renderer
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.models import ColumnDataSource, FuncTickFormatter, DatetimeTickFormatter

from backtrader_plotting.bokeh import label_resolver
from backtrader_plotting.bokeh.label_resolver import plotobj2label
from backtrader_plotting.bokeh.utils import convert_color, sanitize_source_name, get_bar_width, convert_linestyle, get_indicator_data


class HoverContainer(metaclass=bt.MetaParams):
    """Class to store information about hover tooltips. Will be filled while Bokeh glyphs are created. After all figures are complete, hovers will be applied"""

    params = (('hover_tooltip_config', None),
              ('is_multidata', False)
              )

    def __init__(self):
        self._hover_tooltips = []

        self._config = []
        input_config = [] if len(self.p.hover_tooltip_config) == 0 else self.p.hover_tooltip_config.split(',')
        for c in input_config:
            if len(c) != 2:
                raise RuntimeError(f'Invalid hover config entry "{c}"')
            self._config.append((self._get_type(c[0]), self._get_type(c[1])))

    def add_hovertip(self, label: str, tmpl: str, src_obj=None) -> None:
        self._hover_tooltips.append((label, tmpl, src_obj))

    @staticmethod
    def _get_type(t):
        if t == 'd':
            return bt.AbstractDataBase
        elif t == 'i':
            return bt.Indicator
        elif t == 'o':
            return bt.Observer
        else:
            raise RuntimeError(f'Invalid hovertool config type: "{t}')

    def _apply_to_figure(self, fig, hovertool):
        # provide ordering by two groups
        tooltips_top = []
        tooltips_bottom = []
        for label, tmpl, src_obj in self._hover_tooltips:
            apply: bool = src_obj is fig.master  # apply to own
            foreign = False
            if not apply and (isinstance(src_obj, bt.Observer) or isinstance(src_obj, bt.Indicator)) and src_obj.plotinfo.subplot is False:
                # add objects that are on the same figure cause subplot is False (for Indicators and Observers)
                # if plotmaster is set then it will decide where to add, otherwise clock is used
                if src_obj.plotinfo.plotmaster is not None:
                    apply = src_obj.plotinfo.plotmaster is fig.master
                else:
                    apply = src_obj._clock is fig.master
            if not apply:
                for c in self._config:
                    if isinstance(src_obj, c[0]) and isinstance(fig.master, c[1]):
                        apply = True
                        foreign = True
                        break

            if apply:
                prefix = ''
                top = True
                # prefix with data name if we got multiple datas
                if self.p.is_multidata and foreign:
                    if isinstance(src_obj, bt.Indicator):
                        prefix = label_resolver.datatarget2label(src_obj.datas) + " - "
                    elif isinstance(src_obj, bt.AbstractDataBase):
                        prefix = label_resolver.datatarget2label([src_obj]) + " - "
                    top = False

                item = (prefix + label, tmpl)
                if top:
                    tooltips_top.append(item)
                else:
                    tooltips_bottom.append(item)

        # first apply all top hover then all bottoms
        for t in itertools.chain(tooltips_top, tooltips_bottom):
            hovertool.tooltips.append(t)

    def apply_hovertips(self, figures: List['FigureEnvelope']) -> None:
        """Add hovers to to all figures from the figures list"""
        for f in figures:
            for t in f.figure.tools:
                if not isinstance(t, HoverTool):
                    continue

                self._apply_to_figure(f, t)
                break


class FigureEnvelope(object):
    _tools = "pan,wheel_zoom,box_zoom,reset"

    _mrk_fncs = {'^': 'triangle',
                 'v': 'inverted_triangle',
                 'o': 'circle',
                 '<': 'circle_cross',
                 '>': 'circle_x',
                 '1': 'diamond',
                 '2': 'diamond_cross',
                 '3': 'hex',
                 '4': 'square',
                 '8': 'square_cross',
                 's': 'square_x',
                 'p': 'triangle',
                 '*': 'asterisk',
                 'h': 'hex',
                 'H': 'hex',
                 '+': 'asterisk',
                 'x': 'x',
                 'D': 'diamond_cross',
                 'd': 'diamond',
                 }

    def __init__(self, strategy: bt.Strategy, cds: ColumnDataSource, hoverc: HoverContainer, start, end, scheme, master, plotorder, is_multidata):
        self._strategy = strategy
        self._cds: ColumnDataSource = cds
        self._scheme = scheme
        self._start = start
        self._end = end
        self.figure: figure = None
        self._hover_line_set = False
        self._hover: Optional[HoverTool] = None
        self._hoverc = hoverc
        self._coloridx = collections.defaultdict(lambda: -1)
        self.master = master
        self.plottab = None
        self.plotorder = plotorder
        self.datas = []  # list of all datas that have been plotted to this figure
        self._is_multidata = is_multidata
        self._tradingdomain = None
        self._init_figure()

    @staticmethod
    def should_filter_by_tradingdomain(obj, tradingdomain):
        if tradingdomain is None:
            return True

        if isinstance(tradingdomain, str):
            tradingdomain = [tradingdomain]

        obj_lg = FigureEnvelope._resolve_tradingdomain(obj)
        return obj_lg is True or obj_lg in tradingdomain

    @staticmethod
    def _resolve_tradingdomain(obj) -> Union[bool, str]:
        if isinstance(obj, bt.AbstractDataBase):
            # data feeds are end points
            return obj._name
        elif isinstance(obj, bt.IndicatorBase):
            # lets find the data the indicator is based on
            data = get_indicator_data(obj)
            return FigureEnvelope._resolve_tradingdomain(data)
        elif isinstance(obj, bt.ObserverBase):
            if isinstance(obj._clock, bt.AbstractDataBase):
                return FigureEnvelope._resolve_tradingdomain(obj._clock)
            else:
                return True  # for wide observers we return True which means it belongs to all logic groups
        else:
            raise Exception('unsupported')

    def get_tradingdomains(self) -> List[str]:
        tradingdomains = []
        if self._tradingdomain is None:
            tradingdomains.append(self._resolve_tradingdomain(self.master))
        elif isinstance(self._tradingdomain, list):
            tradingdomains += self._tradingdomain
        elif isinstance(self._tradingdomain, str):
            tradingdomains.append(self._tradingdomain)
        else:
            raise Exception(f'Invalid type for tradingdomain: {type(self._tradingdomain)}')

        return tradingdomains

    def _set_single_hover_renderer(self, ren: Renderer):
        """Sets this figure's hover to a single renderer"""
        if self._hover_line_set:
            return

        self._hover.renderers = [ren]
        self._hover_line_set = True

    def _add_hover_renderer(self, ren: Renderer):
        """Adds another hover render target. Only has effect if not single renderer has been set before"""
        if self._hover_line_set:
            return

        if isinstance(self._hover.renderers, list):
            self._hover.renderers.append(ren)
        else:
            self._hover.renderers = [ren]

    def _nextcolor(self, key: object = None) -> int:
        self._coloridx[key] += 1
        return self._coloridx[key]

    def _color(self, key: object = None):
        return convert_color(self._scheme.color(self._coloridx[key]))

    def _init_figure(self):
        # plot height will be set later
        f = figure(tools=FigureEnvelope._tools, x_axis_type='linear', aspect_ratio=self._scheme.plotaspectratio)
        # TODO: backend webgl (output_backend="webgl") removed due to this bug:
        # https://github.com/bokeh/bokeh/issues/7568

        f.y_range.range_padding = self._scheme.y_range_padding

        f.border_fill_color = convert_color(self._scheme.border_fill)

        f.xaxis.axis_line_color = convert_color(self._scheme.axis_line_color)
        f.yaxis.axis_line_color = convert_color(self._scheme.axis_line_color)
        f.xaxis.minor_tick_line_color = convert_color(self._scheme.tick_line_color)
        f.yaxis.minor_tick_line_color = convert_color(self._scheme.tick_line_color)
        f.xaxis.major_tick_line_color = convert_color(self._scheme.tick_line_color)
        f.yaxis.major_tick_line_color = convert_color(self._scheme.tick_line_color)

        f.xaxis.major_label_text_color = convert_color(self._scheme.axis_label_text_color)
        f.yaxis.major_label_text_color = convert_color(self._scheme.axis_label_text_color)

        f.xgrid.grid_line_color = convert_color(self._scheme.grid_line_color)
        f.ygrid.grid_line_color = convert_color(self._scheme.grid_line_color)
        f.title.text_color = convert_color(self._scheme.plot_title_text_color)

        f.left[0].formatter.use_scientific = False
        f.background_fill_color = convert_color(self._scheme.background_fill)

        # mechanism for proper date axis without gaps, thanks!
        # https://groups.google.com/a/continuum.io/forum/#!topic/bokeh/t3HkalO4TGA
        f.xaxis.formatter = FuncTickFormatter(
            args=dict(
                axis=f.xaxis[0],
                formatter=DatetimeTickFormatter(days=[self._scheme.axis_tickformat_days],
                                                hourmin=[self._scheme.axis_tickformat_hourmin],
                                                hours=[self._scheme.axis_tickformat_hours],
                                                minsec=[self._scheme.axis_tickformat_minsec],
                                                minutes=[self._scheme.axis_tickformat_minutes],
                                                months=[self._scheme.axis_tickformat_months],
                                                seconds=[self._scheme.axis_tickformat_seconds],
                                                years=[self._scheme.axis_tickformat_years],
                                                ),
                source=self._cds,
            ),
            code="""
                // We override this axis' formatter's `doFormat` method
                // with one that maps index ticks to dates. Some of those dates
                // are undefined (e.g. those whose ticks fall out of defined data
                // range) and we must filter out and account for those, otherwise
                // the formatter computes invalid visible span and returns some
                // labels as 'ERR'.
                // Note, after this assignment statement, on next plot redrawing,
                // our override `doFormat` will be called directly
                // -- FunctionTickFormatter.doFormat(), i.e. _this_ code, no longer
                // executes.
                axis.formatter.doFormat = function (ticks) {
                    const dates = ticks.map(i => source.data.datetime[source.data.index.indexOf(i)]),
                          valid = t => t !== undefined,
                          labels = formatter.doFormat(dates.filter(valid));
                    let i = 0;
                    return dates.map(t => valid(t) ? labels[i++] : '');
                };
                
                // we do this manually only for the first time we are called
                const labels = axis.formatter.doFormat(ticks);
                return labels[index];
            """
            )

        ch = CrosshairTool(line_color=self._scheme.crosshair_line_color)
        f.tools.append(ch)

        h = HoverTool(tooltips=[('Time', f'@datetime{{{self._scheme.hovertool_timeformat}}}')],
                      mode="vline",
                      formatters={'@datetime': 'datetime'}
                      )
        f.tools.append(h)

        self._hover = h
        self.figure = f

    def plot(self, obj, master=None):
        if isinstance(obj, bt.AbstractDataBase):
            self.plot_data(obj)
        elif isinstance(obj, bt.indicator.Indicator):
            self.plot_indicator(obj, master)
        elif isinstance(obj, bt.observers.Observer):
            self.plot_observer(obj, master)
        else:
            raise Exception(f"Unsupported plot object: {type(obj)}")

        # first object can apply config
        if len(self.datas) == 0:
            aspectr = getattr(obj.plotinfo, 'plotaspectratio', None)
            if aspectr is not None:
                self.figure.aspect_ratio = aspectr

            tab = getattr(obj.plotinfo, 'plottab', None)
            if tab is not None:
                self.plottab = tab

            order = getattr(obj.plotinfo, 'plotorder', None)
            if order is not None:
                self.plotorder = order

            # just store the tradingdomain of the master for later reference
            tradingdomain = getattr(obj.plotinfo, 'tradingdomain', None)
            if tradingdomain is not None:
                self._tradingdomain = tradingdomain

        self.datas.append(obj)

    @staticmethod
    def build_color_lines(df: pd.DataFrame, scheme, col_open: str = 'open', col_close: str = 'close', col_prefix: str='') -> pd.DataFrame:
        # build color strings from scheme
        colorup = convert_color(scheme.barup)
        colordown = convert_color(scheme.bardown)
        colorup_wick = convert_color(scheme.barup_wick)
        colordown_wick = convert_color(scheme.bardown_wick)
        colorup_outline = convert_color(scheme.barup_outline)
        colordown_outline = convert_color(scheme.bardown_outline)
        volup = convert_color(scheme.volup)
        voldown = convert_color(scheme.voldown)

        # build binary series determining if up or down bar
        is_up: pd.DataFrame = df[col_close] >= df[col_open]

        # we use the open-line as a indicator for NaN values
        nan_ref = df[col_open]

        # TODO: we want to have NaN values in the color lines if the corresponding data is also NaN
        # find better way with less isnan usage

        color_df = pd.DataFrame(index=df.index)
        color_df[col_prefix + 'colors_bars'] = [np.nan if np.isnan(n) else colorup if x else colordown for x, n in zip(is_up, nan_ref)]
        color_df[col_prefix + 'colors_wicks'] = [np.nan if np.isnan(n) else colorup_wick if x else colordown_wick for x, n in zip(is_up, nan_ref)]
        color_df[col_prefix + 'colors_outline'] = [np.nan if np.isnan(n) else colorup_outline if x else colordown_outline for x, n in zip(is_up, nan_ref)]
        color_df[col_prefix + 'colors_volume'] = [np.nan if np.isnan(n) else volup if x else voldown for x, n in zip(is_up, nan_ref)]

        # convert to object since we want to hold str and NaN
        for c in color_df.columns:
            color_df[c] = color_df[c].astype(object)

        return color_df

    def _add_column(self, name, dtype):
        self._add_columns([(name, dtype)])

    def _add_columns(self, cols: List[Tuple[str, object]]):
        for name, dtype in cols:
            self._cds.add(np.array([], dtype=dtype), name)

    def plot_data(self, data: bt.AbstractDataBase):
        source_id = FigureEnvelope._source_id(data)
        title = sanitize_source_name(label_resolver.datatarget2label([data]))

        # append to title
        self._figure_append_title(title)

        self._add_columns([(source_id + x, object) for x in ['open', 'high', 'low', 'close']])
        self._add_columns([(source_id + x, str) for x in ['colors_bars', 'colors_wicks', 'colors_outline']])

        if self._scheme.style == 'line':
            if data.plotinfo.plotmaster is None:
                color = convert_color(self._scheme.loc)
            else:
                self._nextcolor(data.plotinfo.plotmaster)
                color = convert_color(self._color(data.plotinfo.plotmaster))

            renderer = self.figure.line('index', source_id + 'close', source=self._cds, line_color=color, legend_label=title)
            self._set_single_hover_renderer(renderer)

            self._hoverc.add_hovertip("Close", f"@{source_id}close", data)
        elif self._scheme.style == 'bar':
            self.figure.segment('index', source_id + 'high', 'index', source_id + 'low', source=self._cds, color=source_id + 'colors_wicks', legend_label=title)
            renderer = self.figure.vbar('index',
                                        get_bar_width(),
                                        source_id + 'open',
                                        source_id + 'close',
                                        source=self._cds,
                                        fill_color=source_id + 'colors_bars',
                                        line_color=source_id + 'colors_outline',
                                        legend_label=title,
                                        )

            self._set_single_hover_renderer(renderer)

            self._hoverc.add_hovertip("Open", f"@{source_id}open{{{self._scheme.number_format}}}", data)
            self._hoverc.add_hovertip("High", f"@{source_id}high{{{self._scheme.number_format}}}", data)
            self._hoverc.add_hovertip("Low", f"@{source_id}low{{{self._scheme.number_format}}}", data)
            self._hoverc.add_hovertip("Close", f"@{source_id}close{{{self._scheme.number_format}}}", data)
        else:
            raise Exception(f"Unsupported style '{self._scheme.style}'")

        # make sure the regular y-axis only scales to the normal data on 1st axis (not to e.g. volume data on 2nd axis)
        self.figure.y_range.renderers.append(renderer)

        if self._scheme.volume and self._scheme.voloverlay:
            self.plot_volume(data, self._scheme.voltrans, True)

    def plot_volume(self, data: bt.AbstractDataBase, alpha=1.0, extra_axis=False):
        """extra_axis displays a second axis (for overlay on data plotting)"""
        source_id = FigureEnvelope._source_id(data)

        self._add_columns([(source_id + 'volume', np.float64), (source_id + 'colors_volume', np.object)])
        kwargs = {'fill_alpha': alpha,
                  'line_alpha': alpha,
                  'name': 'Volume',
                  'legend_label': 'Volume'}

        ax_formatter = NumeralTickFormatter(format=self._scheme.number_format)

        if extra_axis:
            source_data_axis = 'axvol'

            self.figure.extra_y_ranges = {source_data_axis: DataRange1d(
                range_padding=1.0/self._scheme.volscaling,
                start=0,
            )}

            # use colorup
            ax_color = convert_color(self._scheme.volup)

            ax = LinearAxis(y_range_name=source_data_axis, formatter=ax_formatter,
                            axis_label_text_color=ax_color, axis_line_color=ax_color, major_label_text_color=ax_color,
                            major_tick_line_color=ax_color, minor_tick_line_color=ax_color)
            self.figure.add_layout(ax, 'left')
            kwargs['y_range_name'] = source_data_axis
        else:
            self.figure.yaxis.formatter = ax_formatter

        vbars = self.figure.vbar('index', get_bar_width(), f'{source_id}volume', 0, source=self._cds, fill_color=f'{source_id}colors_volume', line_color="black", **kwargs)

        # make sure the new axis only auto-scales to the volume data
        if extra_axis:
            self.figure.extra_y_ranges['axvol'].renderers = [vbars]

        self._hoverc.add_hovertip("Volume", f"@{source_id}volume{{({self._scheme.number_format})}}", data)

    def plot_observer(self, obj, master):
        self._plot_indicator_observer(obj, master)

    def plot_indicator(self, obj: Union[bt.Indicator, bt.Observer], master):
        self._plot_indicator_observer(obj, master)

    def _plot_indicator_observer(self, obj: Union[bt.Indicator, bt.Observer], master):
        pl = plotobj2label(obj)

        self._figure_append_title(pl)
        indlabel = obj.plotlabel()
        plotinfo = obj.plotinfo

        is_multiline = obj.size() > 1
        for lineidx in range(obj.size()):
            line = obj.lines[lineidx]
            source_id = FigureEnvelope._source_id(line)
            linealias = obj.lines._getlinealias(lineidx)

            lineplotinfo = getattr(obj.plotlines, '_%d' % lineidx, None)
            if not lineplotinfo:
                lineplotinfo = getattr(obj.plotlines, linealias, None)

            if not lineplotinfo:
                lineplotinfo = bt.AutoInfoClass()

            if lineplotinfo._get('_plotskip', False):
                continue

            marker = lineplotinfo._get("marker", None)
            method = lineplotinfo._get('_method', "line")

            color = getattr(lineplotinfo, "color", None)
            if color is None:
                if not lineplotinfo._get('_samecolor', False):
                    self._nextcolor()
                color = self._color()
            color = convert_color(color)

            kwglyphs = {'name': linealias}

            self._add_column(source_id, np.float64)

            # either all individual lines of are displayed in the legend or only the ind/obs as a whole
            label = indlabel
            if is_multiline and plotinfo.plotlinelabels:
                label += " " + (lineplotinfo._get("_name", "") or linealias)
            kwglyphs['legend_label'] = label

            if marker is not None:
                kwglyphs['size'] = lineplotinfo.markersize * 1.2
                kwglyphs['color'] = color
                kwglyphs['y'] = source_id

                if marker not in FigureEnvelope._mrk_fncs:
                    raise Exception(f"Sorry, unsupported marker: '{marker}'. Please report to GitHub.")
                glyph_fnc_name = FigureEnvelope._mrk_fncs[marker]
                glyph_fnc = getattr(self.figure, glyph_fnc_name)
            elif method == "bar":
                kwglyphs['bottom'] = 0
                kwglyphs['line_color'] = 'black'
                kwglyphs['fill_color'] = color
                kwglyphs['width'] = get_bar_width()
                kwglyphs['top'] = source_id

                glyph_fnc = self.figure.vbar
            elif method == "line":
                kwglyphs['line_width'] = 1
                kwglyphs['color'] = color
                kwglyphs['y'] = source_id

                linestyle = getattr(lineplotinfo, "ls", None)
                if linestyle is not None:
                    kwglyphs['line_dash'] = convert_linestyle(linestyle)

                glyph_fnc = self.figure.line
            else:
                raise Exception(f"Unknown plotting method '{method}'")

            renderer = glyph_fnc("index", source=self._cds, **kwglyphs)

            # make sure the regular y-axis only scales to the normal data (data + ind/obs) on 1st axis (not to e.g. volume data on 2nd axis)
            self.figure.y_range.renderers.append(renderer)

            # for markers add additional renderer so hover pops up for all of them
            if marker is None:
                self._set_single_hover_renderer(renderer)
            else:
                self._add_hover_renderer(renderer)

            hover_label_suffix = f" - {linealias}" if obj.size() > 1 else ""  # we need no suffix if there is just one line in the indicator anyway
            hover_label = indlabel + hover_label_suffix
            hover_data = f"@{source_id}{{{self._scheme.number_format}}}"
            self._hoverc.add_hovertip(hover_label, hover_data, obj)

        self._set_yticks(obj)
        self._plot_hlines(obj)

    def _set_yticks(self, obj):
        yticks = obj.plotinfo._get('plotyticks', [])
        if not yticks:
            yticks = obj.plotinfo._get('plotyhlines', [])

        if yticks:
            self.figure.yaxis.ticker = yticks

    def _plot_hlines(self, obj):
        hlines = obj.plotinfo._get('plothlines', [])
        if not hlines:
            hlines = obj.plotinfo._get('plotyhlines', [])

        # Horizontal Lines
        hline_color = convert_color(self._scheme.hlinescolor)
        for hline in hlines:
            span = Span(location=hline,
                        dimension='width',
                        line_color=hline_color,
                        line_dash=convert_linestyle(self._scheme.hlinesstyle),
                        line_width=self._scheme.hlineswidth)
            self.figure.renderers.append(span)

    def _figure_append_title(self, title):
        # append to title
        if len(self.figure.title.text) > 0:
            self.figure.title.text += " | "
        self.figure.title.text += title

    def _add_to_cds(self, data, name):
        if name in self._cds.column_names:
            return
        self._cds.add(data, name)

    @staticmethod
    def _source_id(source):
        return str(id(source))
