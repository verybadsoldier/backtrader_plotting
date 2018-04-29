import backtrader as bt
from typing import Dict, Optional, List
import math
from datetime import datetime
import pandas
import logging


_logger = logging.getLogger(__name__)

def get_nondefault_params(params: object) -> Dict[str, object]:
    return {key: params._get(key) for key in params._getkeys() if not params.isdefault(key)}


def get_strategy_label(strategycls: bt.MetaStrategy, params: Optional[bt.AutoInfoClass]) -> str:
    label = strategycls.__name__
    user_params = get_nondefault_params(params)
    plabs = [f"{x}: {y:.2f}" for x, y in user_params.items()]
    plabs = '/'.join(plabs)
    return f'{label} [{plabs}]'


def nanfilt(x: List) -> List:
    """filters all NaN values from a list"""
    return [value for value in x if not math.isnan(value)]


def resample_line(line, line_clk, new_clk):
    """Resamples data line to a new clock. Missing values will be filled with NaN."""
    if new_clk is None:
        return line

    new_line = []
    next_idx = len(line_clk) - 1
    for sc in new_clk:
        for i in range(next_idx, 0, -1):
            v = line_clk[-i]
            if sc == v:
                # exact hit
                new_line.append(line[-i])
                next_idx = i
                break
        else:
            new_line.append(float('nan'))
    return new_line


def convert_to_pandas(strat_clk, obj: bt.LineSeries, start: datetime=None, end: datetime=None, name_prefix: str="") -> pandas.DataFrame:
    df = pandas.DataFrame()
    for lineidx in range(obj.size()):
        line = obj.lines[lineidx]
        linealias = obj.lines._getlinealias(lineidx)
        if linealias == 'datetime':
            continue
        data = line.plotrange(start, end)

        ndata = resample_line(data, obj.lines.datetime.plotrange(start, end), strat_clk)
        logging.info(f"Filled_line: {linealias}: {str(ndata)}")

        df[name_prefix + linealias] = ndata

    df[name_prefix + 'datetime'] = [bt.num2date(x) for x in strat_clk]
    return df

