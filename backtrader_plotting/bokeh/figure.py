from array import array
import collections
import itertools
from typing import List, Optional, Union

import backtrader as bt

from bokeh.models import Span
from bokeh.plotting import figure
from bokeh.models import HoverTool, CrosshairTool
from bokeh.models import LinearAxis, DataRange1d, Renderer
from bokeh.models.formatters import NumeralTickFormatter
from bokeh.models import ColumnDataSource, FuncTickFormatter, DatetimeTickFormatter

from backtrader_plotting.bokeh import label_resolver
from backtrader_plotting.bokeh.label_resolver import plotobj2label
from backtrader_plotting.utils import resample_line, convert_to_pandas, nanfilt, get_data_obj
from backtrader_plotting.bokeh.utils import convert_color, sanitize_source_name, get_bar_width, convert_linestyle, adapt_yranges


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
            apply = src_obj is fig.master  # apply to own
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

    def apply_hovertips(self, figures: List['Figure']) -> None:
        """Add hovers to to all figures from the figures list"""
        for f in figures:
            for t in f.figure.tools:
                if not isinstance(t, HoverTool):
                    continue

                self._apply_to_figure(f, t)
                break


class Figure(object):
    _tools = "pan,wheel_zoom,box_zoom,reset"

    def __init__(self, strategy: bt.Strategy, cds: ColumnDataSource, hoverc: HoverContainer, start, end, scheme, master, plotorder, is_multidata):
        self._strategy = strategy
        self._cds: ColumnDataSource = cds
        self._scheme = scheme
        self._start = start
        self._end = end
        self.figure = None
        self._hover_line_set = False
        self._hover: Optional[HoverTool] = None
        self._hoverc = hoverc
        self._coloridx = collections.defaultdict(lambda: -1)
        self.master = master
        self.plottab = None
        self.plotorder = plotorder
        self.datas = []  # list of all datas that have been plotted to this figure
        self._is_multidata = is_multidata
        self._init_figure()

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
        f = figure(tools=Figure._tools, x_axis_type='linear', aspect_ratio=self._scheme.plot_aspect_ratio)
        # TODO: backend webgl (output_backend="webgl") removed due to this bug:
        # https://github.com/bokeh/bokeh/issues/7568

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
                formatter=DatetimeTickFormatter(minutes=[self._scheme.axis_tickformat_minutes],
                                                hourmin=[self._scheme.axis_tickformat_hourmin],
                                                hours=[self._scheme.axis_tickformat_hours],
                                                days=[self._scheme.axis_tickformat_days],
                                                months=[self._scheme.axis_tickformat_months],
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
                    const dates = ticks.map(i => source.data.datetime[i]),
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
                      formatters={'datetime': 'datetime'}
                      )
        f.tools.append(h)

        self._hover = h
        self.figure = f

    def plot(self, obj, strat_clk, master=None):
        if isinstance(obj, bt.AbstractDataBase):
            self.plot_data(obj, strat_clk)
        elif isinstance(obj, bt.indicator.Indicator):
            self.plot_indicator(obj, master, strat_clk)
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

        self.datas.append(obj)

    def plot_data(self, data: bt.AbstractDataBase, strat_clk: array = None):
        source_id = Figure._source_id(data)
        title = sanitize_source_name(label_resolver.datatarget2label([data]))

        # append to title
        self._figure_append_title(title)

        df = convert_to_pandas(strat_clk, data, self._start, self._end)

        # configure colors
        colorup = convert_color(self._scheme.barup)
        colordown = convert_color(self._scheme.bardown)
        colorup_wick = convert_color(self._scheme.barup_wick)
        colordown_wick = convert_color(self._scheme.bardown_wick)
        colorup_outline = convert_color(self._scheme.barup_outline)
        colordown_outline = convert_color(self._scheme.bardown_outline)
        is_up = df.close > df.open

        self._add_to_cds(df.open, source_id + 'open')
        self._add_to_cds(df.high, source_id + 'high')
        self._add_to_cds(df.low, source_id + 'low')
        self._add_to_cds(df.close, source_id + 'close')
        self._add_to_cds([colorup if x else colordown for x in is_up], source_id + 'colors_bars')
        self._add_to_cds([colorup_wick if x else colordown_wick for x in is_up], source_id + 'colors_wicks')
        self._add_to_cds([colorup_outline if x else colordown_outline for x in is_up], source_id + 'colors_outline')

        if self._scheme.style == 'line':
            if data.plotinfo.plotmaster is None:
                color = convert_color(self._scheme.loc)
            else:
                self._nextcolor(data.plotinfo.plotmaster)
                color = convert_color(self._color(data.plotinfo.plotmaster))

            renderer = self.figure.line('index', source_id + 'close', source=self._cds, line_color=color, legend=title)
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

        adapt_yranges(self.figure.y_range, df.low, df.high)

        # check if we have to plot volume overlay
        if self._scheme.volume and self._scheme.voloverlay:
            self.plot_volume(data, strat_clk, self._scheme.voltrans, True)

    def plot_volume(self, data: bt.AbstractDataBase, strat_clk: array, alpha, extra_axis=False):
        """extra_axis displays a second axis (for overlay on data plotting)"""
        source_id = Figure._source_id(data)

        df = convert_to_pandas(strat_clk, data, self._start, self._end)

        if len(nanfilt(df.volume)) == 0:
            return

        colorup = convert_color(self._scheme.volup)
        colordown = convert_color(self._scheme.voldown)

        is_up = df.close > df.open
        colors = [colorup if x else colordown for x in is_up]

        self._add_to_cds(df.volume, f'{source_id}volume')
        self._add_to_cds(colors, f'{source_id}volume_colors')

        kwargs = {'fill_alpha': alpha,
                  'line_alpha': alpha,
                  'name': 'Volume',
                  'legend_label': 'Volume'}

        ax_formatter = NumeralTickFormatter(format=self._scheme.number_format)

        if extra_axis:
            self.figure.extra_y_ranges = {'axvol': DataRange1d()}
            adapt_yranges(self.figure.extra_y_ranges['axvol'], df.volume)
            self.figure.extra_y_ranges['axvol'].end /= self._scheme.volscaling

            ax_color = colorup

            ax = LinearAxis(y_range_name="axvol", formatter=ax_formatter,
                            axis_label_text_color=ax_color, axis_line_color=ax_color, major_label_text_color=ax_color,
                            major_tick_line_color=ax_color, minor_tick_line_color=ax_color)
            self.figure.add_layout(ax, 'left')
            kwargs['y_range_name'] = "axvol"
        else:
            self.figure.yaxis.formatter = ax_formatter
            adapt_yranges(self.figure.y_range, df.volume)
            self.figure.y_range.end /= self._scheme.volscaling

        self.figure.vbar('index', get_bar_width(), f'{source_id}volume', 0, source=self._cds, fill_color=f'{source_id}volume_colors', line_color="black", **kwargs)

        self._hoverc.add_hovertip("Volume", f"@{source_id}volume{{({self._scheme.number_format})}}", data)

    def plot_observer(self, obj, master):
        self._plot_indicator_observer(obj, master)

    def plot_indicator(self, obj: Union[bt.Indicator, bt.Observer], master, strat_clk: array = None):
        self._plot_indicator_observer(obj, master, strat_clk)

    def _plot_indicator_observer(self, obj: Union[bt.Indicator, bt.Observer], master, strat_clk: array = None):
        pl = plotobj2label(obj)

        self._figure_append_title(pl)
        indlabel = obj.plotlabel()
        plotinfo = obj.plotinfo

        for lineidx in range(obj.size()):
            line = obj.lines[lineidx]
            source_id = Figure._source_id(line)
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

            dataline = line.plotrange(self._start, self._end)
            line_clk = get_data_obj(obj).lines.datetime.plotrange(self._start, self._end)
            dataline = resample_line(dataline, line_clk, strat_clk)
            self._add_to_cds(dataline, source_id)

            label = None
            if master is None or lineidx == 0 or plotinfo.plotlinelabels:
                label = indlabel
                if master is None or plotinfo.plotlinelabels:
                    label += " " + (lineplotinfo._get("_name", "") or linealias)
            kwglyphs['legend_label'] = label

            if marker is not None:
                kwglyphs['size'] = lineplotinfo.markersize * 1.2
                kwglyphs['color'] = color
                kwglyphs['y'] = source_id

                mrk_fncs = {'^': self.figure.triangle,
                            'v': self.figure.inverted_triangle,
                            'o': self.figure.circle,

                            '<': self.figure.circle_cross,
                            '>': self.figure.circle_x,
                            '1': self.figure.diamond,
                            '2': self.figure.diamond_cross,
                            '3': self.figure.hex,
                            '4': self.figure.square,
                            '8': self.figure.square_cross,
                            's': self.figure.square_x,
                            'p': self.figure.triangle,
                            '*': self.figure.asterisk,
                            'h': self.figure.hex,
                            'H': self.figure.hex,
                            '+': self.figure.asterisk,
                            'x': self.figure.x,
                            'D': self.figure.diamond_cross,
                            'd': self.figure.diamond,
                            }
                if marker not in mrk_fncs:
                    raise Exception(f"Sorry, unsupported marker: '{marker}'. Please report to GitHub.")
                glyph_fnc = mrk_fncs[marker]
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

            # for markers add additional renderer so hover pops up for all of them
            if marker is None:
                self._set_single_hover_renderer(renderer)
            else:
                self._add_hover_renderer(renderer)

            hover_label_suffix = f" - {linealias}" if obj.size() > 1 else ""  # we need no suffix if there is just one line in the indicator anyway
            hover_label = indlabel + hover_label_suffix
            hover_data = f"@{source_id}{{{self._scheme.number_format}}}"
            self._hoverc.add_hovertip(hover_label, hover_data, obj)

            # adapt y-axis if needed
            if master is None or getattr(master.plotinfo, 'plotylimited', False) is False:
                adapt_yranges(self.figure.y_range, dataline)

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
