from jinja2 import Environment, PackageLoader

import matplotlib.colors
import numpy as np
import pandas as pd

from bokeh.models import ColumnDataSource


_style_mpl2bokeh = {
    '-': 'solid',
    '--': 'dashed',
    ':': 'dotted',
    '.-': 'dotdash',
    '-.': 'dashdot',
}


def convert_color(color):
    """if color is a float value then it is interpreted as a shade of grey and converted to the corresponding html color code"""
    try:
        val = round(float(color) * 255.0)
        hex_string = '#{0:02x}{0:02x}{0:02x}'.format(val)
        return hex_string
    except ValueError:
        return matplotlib.colors.to_hex(color)


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


def sanitize_source_name(name: str) -> str:
    """removes illegal characters from source name to make it compatible with Bokeh"""
    forbidden_chars = ' (),.-/*:'
    for fc in forbidden_chars:
        name = name.replace(fc, '_')
    return name


def get_bar_width() -> float:
    return 0.5


def convert_linestyle(style: str) -> str:
    """Converts a backtrader/matplotlib style string to bokeh style string"""
    return _style_mpl2bokeh[style]


def adapt_yranges(y_range, data, padding_factor=200.0):
    nnan_data = [x for x in data if not x != x]
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
    env = Environment(loader=PackageLoader('btplotting', 'templates'))
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
                             tag_pre_text_color=scheme.tag_pre_text_color,
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
