from backtrader.plot.scheme import PlotScheme as _BtPlotScheme
from enum import Enum


class PlotScheme(_BtPlotScheme):
    Mode = Enum('Mode', 'Single Tabs')

    def __init__(self):
        super().__init__()

        self.barup_wick = self.barup
        self.bardown_wick = self.bardown

        self.barup_outline = self.barup
        self.bardown_outline = self.bardown

        self.crosshair_line_color = '#999999'

        self.legend_background_color = '#3C3F41'
        self.legend_text_color = 'lightgrey'

        self.background_fill = '#222222'
        self.body_fill = "white"
        self.border_fill = "#3C3F41"
        self.legend_click = "hide"  # or "mute"
        self.axis_line_color = 'darkgrey'
        self.tick_line_color = self.axis_line_color
        self.grid_line_color = '#444444'
        self.axis_text_color = 'lightgrey'
        self.plot_title_text_color = 'darkgrey'
        self.axis_label_text_color = 'darkgrey'

        self.xaxis_pos = "all"  # "all" or "bottom"

        self.table_color_even = "#404040"
        self.table_color_odd = "#333333"
        self.table_header_color = "#7a7a7a"

        self.plot_width = 1800
        self.table_width = 950
        self.table_height = 300
        self.plot_height_data = 800
        self.plot_height_indicator = 400
        self.plot_height_observer = 400
        self.plot_mode = PlotScheme.Mode.Tabs

        self.tooltip_background_color = '#4C4F51'
        self.tooltip_text_label_color = "#848EFF"
        self.tooltip_text_value_color = "#aaaaaa"

class TradimoScheme(PlotScheme):
    def __init__(self):
        super().__init__()

        self.barup = "#265371"
        self.bardown = "#FC5D45"

        self.barup_wick = self.barup
        self.bardown_wick = self.bardown

        self.barup_outline = self.barup
        self.bardown_outline = self.bardown

        self.background_fill = "white"
