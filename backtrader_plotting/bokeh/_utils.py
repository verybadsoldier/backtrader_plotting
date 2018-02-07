import backtrader
from backtrader_archive.data.convert import nanfilt
import matplotlib.colors


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


def get_bar_length_ms(data) -> float:
    """Calculates the width of one bar depending on current timeframe and compression. In milliseconds"""
    if data._timeframe >= backtrader.TimeFrame.MicroSeconds:
        width_ms = 0.001
    if data._timeframe >= backtrader.TimeFrame.Seconds:
        width_ms *= 1e6
    if data._timeframe >= backtrader.TimeFrame.Minutes:
        width_ms *= 60
    if data._timeframe >= backtrader.TimeFrame.Days:
        width_ms *= 60 * 24
    if data._timeframe >= backtrader.TimeFrame.Weeks:
        width_ms *= 7
    if data._timeframe >= backtrader.TimeFrame.Months:
        width_ms *= 30
    if data._timeframe >= backtrader.TimeFrame.Years:
        width_ms *= 12

    width_ms *= data._compression
    return width_ms


_style_mpl2bokeh = {
    '-': 'solid',
    '--': 'dashed',
    ':': 'dotted',
    '.-': 'dotdash',
    '-.': 'dashdot',
}


def convert_linestyle(style: str) -> str:
    """Converts a backtrader/matplotlib style string to bokeh style string"""
    return _style_mpl2bokeh[style];


def adapt_yranges(y_range, data_min, data_max=None):
    dmin = min(nanfilt(data_min), default=None)
    dmax = max(nanfilt(data_max if data_max is not None else data_min), default=None)

    if dmin is None or dmax is None:
        return

    diff = ((dmax - dmin) or dmin) * 0.1
    dmin -= diff
    dmax += diff

    if y_range.start is not None:
        dmin = min(dmin, y_range.start)
    y_range.start = dmin

    if y_range.end is not None:
        dmax = max(dmax, y_range.end)
    y_range.end = dmax