tableau20 = [
    'steelblue',  # 0
    'lightsteelblue',  # 1
    'darkorange',  # 2
    'peachpuff',  # 3
    'green',  # 4
    'lightgreen',  # 5
    'crimson',  # 6
    'lightcoral',  # 7
    'mediumpurple',  # 8
    'thistle',  # 9
    'saddlebrown',  # 10
    'rosybrown',  # 11
    'orchid',  # 12
    'lightpink',  # 13
    'gray',  # 14
    'lightgray',  # 15
    'olive',  # 16
    'palegoldenrod',  # 17
    'mediumturquoise',  # 18
    'paleturquoise',  # 19
]

tableau10 = [
    'blue',  # 'steelblue',  # 0
    'darkorange',  # 1
    'green',  # 2
    'crimson',  # 3
    'mediumpurple',  # 4
    'saddlebrown',  # 5
    'orchid',  # 6
    'gray',  # 7
    'olive',  # 8
    'mediumturquoise',  # 9
]

tableau10_light = [
    'lightsteelblue',  # 0
    'peachpuff',  # 1
    'lightgreen',  # 2
    'lightcoral',  # 3
    'thistle',  # 4
    'rosybrown',  # 5
    'lightpink',  # 6
    'lightgray',  # 7
    'palegoldenrod',  # 8
    'paleturquoise',  # 9
]

tab10_index = [3, 0, 2, 1, 2, 4, 5, 6, 7, 8, 9]


class PlotScheme(object):
    def __init__(self):
        # to have a tight packing on the chart wether only the x axis or also
        # the y axis have (see matplotlib)
        self.ytight = False

        # y-margin (top/bottom) for the subcharts. This will not overrule the
        # option plotinfo.plotymargin
        self.yadjust = 0.0
        # Each new line is in z-order below the previous one. change it False
        # to have lines paint above the previous line
        self.zdown = True
        # Rotation of the date labes on the x axis
        self.tickrotation = 15

        # How many "subparts" takes a major chart (datas) in the overall chart
        # This is proportional to the total number of subcharts
        self.rowsmajor = 5

        # How many "subparts" takes a minor chart (indicators/observers) in the
        # overall chart. This is proportional to the total number of subcharts
        # Together with rowsmajor, this defines a proportion ratio betwen data
        # charts and indicators/observers charts
        self.rowsminor = 1

        # Distance in between subcharts
        self.plotdist = 0.0

        # Have a grid in the background of all charts
        self.grid = True

        # Default plotstyle for the OHLC bars which (line -> line on close)
        # Other options: 'bar' and 'candle'
        self.style = 'line'

        # Default color for the 'line on close' plot
        self.loc = 'black'
        # Default color for a bullish bar/candle (0.75 -> intensity of gray)
        self.barup = '0.75'
        # Default color for a bearish bar/candle
        self.bardown = 'red'
        # Level of transparency to apply to bars/cancles (NOT USED)
        self.bartrans = 1.0

        # Wether the candlesticks have to be filled or be transparent
        self.barupfill = True
        self.bardownfill = True

        # Opacity for the filled candlesticks (1.0 opaque - 0.0 transparent)
        self.baralpha = 1.0

        # Alpha blending for fill areas between lines (_fill_gt and _fill_lt)
        self.fillalpha = 0.20

        # Wether to plot volume or not. Note: if the data in question has no
        # volume values, volume plotting will be skipped even if this is True
        self.volume = True

        # Wether to overlay the volume on the data or use a separate subchart
        self.voloverlay = True
        # Scaling of the volume to the data when plotting as overlay
        self.volscaling = 0.33
        # Pushing overlay volume up for better visibiliy. Experimentation
        # needed if the volume and data overlap too much
        self.volpushup = 0.00

        # Default colour for the volume of a bullish day
        self.volup = '#aaaaaa'  # 0.66 of gray
        # Default colour for the volume of a bearish day
        self.voldown = '#cc6073'  # (204, 96, 115)
        # Transparency to apply to the volume when overlaying
        self.voltrans = 0.50

        # Transparency for text labels (NOT USED CURRENTLY)
        self.subtxttrans = 0.66
        # Default font text size for labels on the chart
        self.subtxtsize = 9

        # Transparency for the legend (NOT USED CURRENTLY)
        self.legendtrans = 0.25
        # Wether indicators have a leged displaey in their charts
        self.legendind = True
        # Location of the legend for indicators (see matplotlib)
        self.legendindloc = 'upper left'

        # Location of the legend for datafeeds (see matplotlib)
        self.legenddataloc = 'upper left'

        # Plot the last value of a line after the Object name
        self.linevalues = True

        # Plot a tag at the end of each line with the last value
        self.valuetags = True

        # Default color for horizontal lines (see plotinfo.plothlines)
        self.hlinescolor = '0.66'  # shade of gray
        # Default style for horizontal lines
        self.hlinesstyle = '--'
        # Default width for horizontal lines
        self.hlineswidth = 1.0

        # Default color scheme: Tableau 10
        self.lcolors = tableau10

        # strftime Format string for the display of ticks on the x axis
        self.fmt_x_ticks = None

        # strftime Format string for the display of data points values
        self.fmt_x_data = None

    def color(self, idx):
        colidx = tab10_index[idx % len(tab10_index)]
        return self.lcolors[colidx]


class Scheme(PlotScheme):
    def __init__(self, **kwargs):
        super().__init__()
        self._set_params()
        self._set_args(**kwargs)

    def _set_params(self):
        """
        hover tooltips of datas will contain all other datas and all indicators/observers
        if set to False then tooltips of datas will only contain the current data and indicators/observers related to that data
        """
        self.tabs = 'single'
        self.show_headline = True
        self.hover_tooltip_config = ''

        self.barup_wick = self.barup
        self.bardown_wick = self.bardown

        self.barup_outline = self.barup
        self.bardown_outline = self.bardown

        self.crosshair_line_color = '#999999'

        self.legend_background_color = '#3C3F41'
        self.legend_text_color = 'lightgrey'
        self.legend_location = 'top_left'
        self.legend_orientation = 'horizontal'

        self.loc = 'lightgray'
        self.background_fill = '#222222'
        self.body_background_color = "white"
        self.border_fill = "#3C3F41"
        self.legend_click = "hide"  # or "mute"
        self.axis_line_color = 'darkgrey'
        self.tick_line_color = self.axis_line_color
        self.grid_line_color = '#444444'
        self.axis_text_color = 'lightgrey'
        self.plot_title_text_color = 'darkgrey'
        self.axis_label_text_color = 'darkgrey'
        self.tag_pre_background_color = 'lightgrey'
        self.tag_pre_text_color = 'black'

        self.xaxis_pos = "all"  # "all" or "bottom"

        self.table_color_even = '#404040'
        self.table_color_odd = '#333333'
        self.table_header_color = '#7a7a7a'

        self.analyzer_tab_width = 1860
        self.analyzer_tab_num_cols = 2
        self.plotaspectratio = 3.0
        self.plot_sizing_mode = "scale_width"

        self.toolbar_location = "right"

        self.tooltip_background_color = '#4C4F51'
        self.tooltip_text_label_color = '#848EFF'
        self.tooltip_text_value_color = '#aaaaaa'

        self.tab_active_background_color = '#333333'
        self.tab_active_color = '#4C4F51'

        self.text_color = 'lightgrey'

        # https://docs.bokeh.org/en/latest/docs/reference/models/formatters.html#bokeh.models.formatters.DatetimeTickFormatter
        self.hovertool_timeformat = '%F %R'

        self.number_format = '0,0.000'
        self.number_format_volume = '0.00 a'

        # https://docs.bokeh.org/en/latest/docs/reference/models/formatters.html
        self.axis_tickformat_days = "%d %b %R"
        self.axis_tickformat_hourmin = "%H:%M:%S"
        self.axis_tickformat_hours = "%d %b %R"
        self.axis_tickformat_minsec = "%H:%M:%S"
        self.axis_tickformat_minutes = "%H:%M"
        self.axis_tickformat_months = "%d/%m/%y"
        self.axis_tickformat_seconds = "%H:%M:%S"
        self.axis_tickformat_years = "%Y %b"

        self.y_range_padding = 0.5  # used to add padding on the y-axis for all data except volume

    def _set_args(self, **kwargs):
        for k, v in kwargs.items():
            if not hasattr(self, k):
                raise Exception(f'Invalid scheme parameter "{k}')
            setattr(self, k, v)
