from jinja2 import Environment, PackageLoader

import matplotlib.colors

import backtrader as bt
from backtrader_plotting.utils import nanfilt

from bokeh.models import ColumnDataSource


def convert_color(color):
    """if color is a float value then it is interpreted as a shade of grey and converted to the corresponding html color code"""
    try:
        val = round(float(color) * 255.0)
        hex_string = '#{0:02x}{0:02x}{0:02x}'.format(val)
        return hex_string
    except ValueError:
        return matplotlib.colors.to_hex(color)


def sanitize_source_name(name: str) -> str:
    """removes illegal characters from source name to make it compatible with Bokeh"""
    forbidden_chars = ' (),.-/*:'
    for fc in forbidden_chars:
        name = name.replace(fc, '_')
    return name


def get_bar_width() -> float:
    return 0.5


_style_mpl2bokeh = {
    '-': 'solid',
    '--': 'dashed',
    ':': 'dotted',
    '.-': 'dotdash',
    '-.': 'dashdot',
}


def convert_linestyle(style: str) -> str:
    """Converts a backtrader/matplotlib style string to bokeh style string"""
    return _style_mpl2bokeh[style]


def adapt_yranges(y_range, data, padding_factor=200.0):
    nnan_data = nanfilt(data)
    dmin = min(nnan_data, default=None)
    dmax = max(nnan_data, default=None)

    if dmin is None or dmax is None:
        return

    diff = ((dmax - dmin) or dmin) * padding_factor
    dmin -= diff
    dmax += diff

    if y_range.start is not None:
        dmin = min(dmin, y_range.start)
    y_range.start = dmin

    if y_range.end is not None:
        dmax = max(dmax, y_range.end)
    y_range.end = dmax


def generate_stylesheet(scheme, template="basic.css.j2") -> str:
    env = Environment(loader=PackageLoader('backtrader_plotting.bokeh', 'templates'))
    templ = env.get_template(template)

    css = templ.render(dict(
                             datatable_row_color_even=scheme.table_color_even,
                             datatable_row_color_odd=scheme.table_color_odd,
                             datatable_header_color=scheme.table_header_color,
                             tab_active_background_color=scheme.tab_active_background_color,
                             tab_active_color=scheme.tab_active_color,

                             tooltip_background_color=scheme.tooltip_background_color,
                             tooltip_text_color_label=scheme.tooltip_text_label_color,
                             tooltip_text_color_value=scheme.tooltip_text_value_color,
                             body_background_color=scheme.body_background_color,
                             tag_pre_background_color=scheme.tag_pre_background_color,
                             headline_color=scheme.plot_title_text_color,
                             text_color=scheme.text_color,
                           )
                       )
    return css


def append_cds(base_cds: ColumnDataSource, new_cds: ColumnDataSource):
    updates = []
    for c in new_cds.keys():
        if c not in base_cds.column_names:
            continue
        updates.append((c, new_cds[c]))
    base_cds.data.update(updates)


def get_indicator_data(indicator: bt.Indicator):
    """The indicator might have been created using a specific line (like SMA(data.lines.close)). In this case
    a LineSeriesStub has been created for which we have to resolve the original data"""
    data = indicator.data
    if isinstance(data, bt.LineSeriesStub):
        return data._owner
    else:
        return data
