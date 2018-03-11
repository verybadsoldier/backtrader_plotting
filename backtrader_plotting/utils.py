import backtrader as bt
from typing import Dict, Optional, List
import math
from datetime import datetime
import pandas


def get_nondefault_params(params: object) -> Dict[str, object]:
    return {key: params._get(key) for key in params._getkeys() if not params.isdefault(key)}


def get_strategy_label(strategycls: bt.MetaStrategy, params: Optional[bt.AutoInfoClass]) -> str:
    label = strategycls.__name__
    user_params = get_nondefault_params(params)
    plabs = [f"{x}: {y}" for x, y in user_params.items()]
    plabs = '/'.join(plabs)
    return f'{label} [{plabs}]'


def nanfilt(x: List) -> List:
    """filters all NaN values from a list"""
    return [value for value in x if not math.isnan(value)]


def convert_to_pandas(obj: bt.LineSeries, start: datetime=None, end: datetime=None, name_prefix: str="") -> pandas.DataFrame:
    df = pandas.DataFrame()
    for lineidx in range(obj.size()):
        line = obj.lines[lineidx]
        linealias = obj.lines._getlinealias(lineidx)
        data = line.plotrange(start, end)

        if linealias == 'datetime':
            data = [bt.num2date(x) for x in data]
        df[name_prefix + linealias] = data
    return df

