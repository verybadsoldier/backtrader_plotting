from typing import List
import collections

import pandas
import backtrader as bt

from ._utils import convert_color, sanitize_source_name, get_bar_length_ms, convert_linestyle, adapt_yranges
from .._convert import convert_to_pandas, nanfilt

from bokeh.models import Span
from bokeh.plotting import figure
from bokeh.models import HoverTool, CrosshairTool
from bokeh.models import LinearAxis, DataRange1d, Renderer
from bokeh.models.formatters import NumeralTickFormatter


class HoverContainer(object):
    """Class to store information about hover tooltips. Will be filled while Bokeh glyphs are created. After all figures are complete, hovers will be applied"""
    def __init__(self):
        self._hover_tooltips = collections.defaultdict(list)

    def add_hovertip(self, label: str, tmpl:str, hover_target: object=None) -> None:
        """hover_target being None means all"""
        self._hover_tooltips[hover_target].append((label, tmpl))

    def apply_hovertips(self, figures: List) -> None:
        """Add hovers to to all figures from the figures list"""
        for f in figures:
            for t in f.figure.tools:
                if not isinstance(t, HoverTool):
                    continue
                if f.figure in self._hover_tooltips:
                    hv = self._hover_tooltips[f.figure]
                else:
                    hv = self._hover_tooltips[None]
                t.tooltips += hv


class Figure(object):
    _tools = "pan,wheel_zoom,box_zoom,reset"

    def __init__(self, cds, hoverc: HoverContainer, start, end, scheme, master_type, plotabove: bool):
        self._cds = cds
        self._hoverc = hoverc
        self._scheme = scheme
        self._start = start
        self._end = end
        self.figure = None
        self._hover = None
        self._coloridx = collections.defaultdict(lambda: -1)
        self._hover_line_set = False
        self.master_type = master_type
        self.plotabove = plotabove
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

        self._hover.renderers.append(ren)

    def _nextcolor(self, key: object=None) -> None:
        self._coloridx[key] += 1
        return self._coloridx[key]

    def _color(self, key: object=None):
        return self._scheme.color(self._coloridx[key])

    def _init_figure(self):
        # plot height will be set later
        f = figure(tools=Figure._tools, plot_width=self._scheme.plot_width, logo=None, sizing_mode='scale_width', x_axis_type='datetime', output_backend="webgl")
        f.border_fill_color = self._scheme.border_fill

        f.xaxis.axis_line_color = self._scheme.axis_line_color
        f.yaxis.axis_line_color = self._scheme.axis_line_color
        f.xaxis.minor_tick_line_color = self._scheme.tick_line_color
        f.yaxis.minor_tick_line_color = self._scheme.tick_line_color
        f.xaxis.major_tick_line_color = self._scheme.tick_line_color
        f.yaxis.major_tick_line_color = self._scheme.tick_line_color

        f.xaxis.major_label_text_color = self._scheme.axis_label_text_color
        f.yaxis.major_label_text_color = self._scheme.axis_label_text_color

        f.xgrid.grid_line_color = self._scheme.grid_line_color
        f.ygrid.grid_line_color = self._scheme.grid_line_color
        f.title.text_color = self._scheme.plot_title_text_color

        f.left[0].formatter.use_scientific = False
        f.background_fill_color = self._scheme.background_fill

        ch = CrosshairTool(line_color=self._scheme.crosshair_line_color)
        f.tools.append(ch)

        h = HoverTool(tooltips=[('Time', '@datetime{%x %X}')],
                      mode="vline",
                      formatters={'datetime': 'datetime'}
                      )
        f.tools.append(h)

        self._hover = h
        self.figure = f

    def plot(self, obj, master=None):
        if isinstance(obj, bt.feeds.DataBase):
            self.plot_data(obj, master)
            height_set = self._scheme.plot_height_data
        elif isinstance(obj, bt.indicator.Indicator):
            self.plot_indicator(obj, master)
            height_set = self._scheme.plot_height_indicator
        elif isinstance(obj, bt.observers.Observer):
            self.plot_indicator(obj, master)
            height_set = self._scheme.plot_height_observer

        # set height according to master type
        if master is None:
            self.figure.plot_height = height_set


    def plot_indicator(self, obj, master):
        self._figure_append_title(obj.plotlabel())
        indlabel = obj.plotlabel()
        plotinfo = obj.plotinfo
        data = obj._clock

        for lineidx in range(obj.size()):
            line = obj.lines[lineidx]
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

            kwglyphs = {'name': linealias }

            source_dataname = sanitize_source_name(indlabel + linealias)
            dataline = line.plotrange(self._start, self._end)
            self._cds.add(dataline, source_dataname)

            label = None
            if master is None or lineidx == 0 or plotinfo.plotlinelabels:
                label = indlabel
                if master is None or plotinfo.plotlinelabels:
                    label += " " + (lineplotinfo._get("_name", "") or linealias)
            kwglyphs['legend'] = label

            if marker is not None:
                kwglyphs['size'] = lineplotinfo.markersize * 1.2
                kwglyphs['color'] = color
                kwglyphs['y'] = source_dataname

                mrk_fncs = {'^': self.figure.triangle,
                            'v': self.figure.inverted_triangle,
                            'o': self.figure.circle,
                            }
                glyph_fnc = mrk_fncs[marker]
            elif method == "bar":
                kwglyphs['bottom'] = 0
                kwglyphs['line_color'] = 'black'
                kwglyphs['fill_color'] = color
                kwglyphs['width'] = get_bar_length_ms(data)
                kwglyphs['top'] = source_dataname

                glyph_fnc = self.figure.vbar
            elif method == "line":
                kwglyphs['line_width'] = 1
                kwglyphs['color'] = color
                kwglyphs['y'] = source_dataname

                linestyle = getattr(lineplotinfo, "ls", None)
                if linestyle is not None:
                    kwglyphs['line_dash'] = convert_linestyle(linestyle)

                glyph_fnc = self.figure.line
            else:
                raise Exception(f"Unknown plotting method '{method}'")

            renderer = glyph_fnc("datetime", source=self._cds, **kwglyphs)

            # for markers add additional renderer so hover pops up for all of them
            if marker is None:
                self._set_single_hover_renderer(renderer)
            else:
                self._add_hover_renderer(renderer)

            # add hovers
            hover_target = None
            is_obs = isinstance(obj, bt.Observer)
            if is_obs and master is None:
                hover_target = self.figure
            self._hoverc.add_hovertip(f"{indlabel} - {linealias}", f"@{source_dataname}{{0,0.000}}", hover_target)

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
            self.figure.title.text += " / "
        self.figure.title.text += title

    def plot_data(self, obj, master):
        src_prefix = sanitize_source_name(obj._name)

        df = convert_to_pandas(obj, self._start, self._end)

        # configure colors
        colorup = convert_color(self._scheme.barup)
        colordown = convert_color(self._scheme.bardown)
        colorup_wick = convert_color(self._scheme.barup_wick)
        colordown_wick = convert_color(self._scheme.bardown_wick)
        colorup_outline = convert_color(self._scheme.barup_outline)
        colordown_outline = convert_color(self._scheme.bardown_outline)
        is_up = df.close > df.open

        self._cds.add(df.open, src_prefix + 'open')
        self._cds.add(df.high, src_prefix + 'high')
        self._cds.add(df.low, src_prefix + 'low')
        self._cds.add(df.close, src_prefix + 'close')
        self._cds.add([colorup if x else colordown for x in is_up], src_prefix + 'colors_bars')
        self._cds.add([colorup_wick if x else colordown_wick for x in is_up], src_prefix + 'colors_wicks')
        self._cds.add([colorup_outline if x else colordown_outline for x in is_up], src_prefix + 'colors_outline')

        # append to title
        self._figure_append_title(src_prefix)

        if self._scheme.style == 'line':
            if obj.plotinfo.plotmaster is None:
                color = self._scheme.loc
            else:
                self._nextcolor(obj.plotinfo.plotmaster)
                color = self._color(obj.plotinfo.plotmaster)

            self.figure.line('datetime', src_prefix + 'close', source=self._cds, line_color=color, legend=obj._name, output_backend="webgl")
        elif self._scheme.style == 'bar':
            self.figure.segment('datetime', src_prefix + 'high', 'datetime', src_prefix + 'low', source=self._cds, color=src_prefix + 'colors_wicks', legend=obj._name)
            renderer = self.figure.vbar('datetime',
                                      get_bar_length_ms(obj) * 0.7,
                                      src_prefix + 'open',
                                      src_prefix + 'close',
                                      source=self._cds,
                                      fill_color=src_prefix + 'colors_bars',
                                      line_color=src_prefix + 'colors_outline')
            self._set_single_hover_renderer(renderer)

            self._hoverc.add_hovertip("Open", f"@{src_prefix}open{{0,0.000}}")
            self._hoverc.add_hovertip("High", f"@{src_prefix}high{{0,0.000}}")
            self._hoverc.add_hovertip("Low", f"@{src_prefix}low{{0,0.000}}")
            self._hoverc.add_hovertip("Close", f"@{src_prefix}close{{0,0.000}}")
        else:
            raise Exception(f"Unsupported style '{self._scheme.style}'")

        adapt_yranges(self.figure.y_range, df.low, df.high)

        # check if we have to plot volume overlay
        if self._scheme.volume and self._scheme.voloverlay:
            self.plot_volume(obj, self._scheme.voltrans, True)

    def plot_volume(self, obj, alpha, extra_axis=False):
        src_prefix = sanitize_source_name(obj._name)

        df = convert_to_pandas(obj, self._start, self._end)

        if len(nanfilt(df.volume)) == 0:
            return

        colorup = self._scheme.volup
        colordown = self._scheme.voldown

        is_up = df.close > df.open
        colors = [colorup if x else colordown for x in is_up]

        self._cds.add(df.volume, src_prefix + 'volume')
        self._cds.add(colors, src_prefix + 'volume_colors')

        kwargs = {'fill_alpha': alpha,
                  'line_alpha': alpha,
                  'name': 'Volume',
                  'legend': 'Volume'}

        ax_formatter = NumeralTickFormatter(format='0.000 a')

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

        self.figure.vbar('datetime', get_bar_length_ms(obj) * 0.7, src_prefix + 'volume', 0, source=self._cds, fill_color=src_prefix + 'volume_colors', line_color="black", **kwargs)
        self._hoverc.add_hovertip("Volume", f"@{src_prefix}volume{{(0.00 a)}}")
