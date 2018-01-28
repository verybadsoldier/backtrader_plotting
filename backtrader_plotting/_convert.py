from datetime import datetime
import backtrader
import pandas
import math
from typing import List


def line_to_list(lineobj: List) -> List:
    l = []
    for i in range(0, len(lineobj)):
        l.append(lineobj[-i])
    return l


def convert_to_pandas(obj: backtrader.LineSeries, start: datetime=None, end: datetime=None, name_prefix: str="") -> pandas.DataFrame:
    df = pandas.DataFrame()
    for lineidx in range(obj.size()):
        line = obj.lines[lineidx]
        linealias = obj.lines._getlinealias(lineidx)
        data = line.plotrange(start, end)

        if linealias == 'datetime':
            data = [backtrader.num2date(x) for x in data]
        df[name_prefix + linealias] = data
    return df


def nanfilt(x: List) -> List:
    """filters all NaN values from a list"""
    return [value for value in x if not math.isnan(value)]
