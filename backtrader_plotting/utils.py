from datetime import datetime
import logging
import math
from typing import Dict, Optional, List, Union

import backtrader as bt

import numpy as np
import pandas as pd
import itertools


_logger = logging.getLogger(__name__)


def paramval2str(name, value):
    if value is None:  # catch None value early here!
        return str(value)
    elif name == "timeframe":
        return bt.TimeFrame.getname(value, 1)
    elif isinstance(value, float):
        return f"{value:.2f}"
    elif isinstance(value, (list,tuple)):
        return ','.join(value)
    elif isinstance(value, type):
        return value.__name__
    else:
        return str(value)


def get_nondefault_params(params: object) -> Dict[str, object]:
    return {key: params._get(key) for key in params._getkeys() if not params.isdefault(key)}


def get_params(params: bt.AutoInfoClass):
    return {key: params._get(key) for key in params._getkeys()}


def get_params_str(params: Optional[bt.AutoInfoClass]) -> str:
    user_params = get_nondefault_params(params)
    plabs = [f"{x}: {paramval2str(x, y)}" for x, y in user_params.items()]
    plabs = '/'.join(plabs)
    return plabs


def nanfilt(x: List) -> List:
    """filters all NaN values from a list"""
    return [value for value in x if not math.isnan(value)]


def convert_to_master_clock(line, line_clk, master_clock, fill_by_prev=False):
    """Takes a clock and generates an appropriate line with a value for each entry in clock. Values are taken from another line if the
    clock value in question is found in its line_clk. Otherwise NaN is used"""
    if master_clock is None:
        return line

    # sometimes the clock has more data than the data line
    # not sure when this is the case
    # i think both are left aligned so for latest clock values there is no data
    clk_offset = len(line_clk) - len(line)
    new_line = []  # hold the data for the resampled line
    next_start_idx = 0
    prev_lvalue = np.nan  # conserve previous line value for fill_by_prev
    # iterate each master_clock entry and try to find a matching clock in the source line
    for mc in master_clock:
        found = False
        for i in range(next_start_idx, len(line_clk)):
            lc = line_clk[i]
            line_idx = i - clk_offset

            # data line might be shorter so we don't have data
            lvalue = np.nan if line_idx < 0 else line[line_idx]

            if mc == lc:
                new_line.append(lvalue)
                next_start_idx = i  # start searching from *this* index to have a chance to catch it as prev value next round
                found = True
                break
            elif lc > mc:
                # no need to keep searching...
                break
            else:
                prev_lvalue = lvalue

        if not found:
            if fill_by_prev:
                fill_v = prev_lvalue  # fill missing values with prev value
            else:
                fill_v = float('nan')  # fill with NaN, Bokeh wont plot
            new_line.append(fill_v)
    return new_line


def convert_to_pandas(master_clock, obj: bt.LineSeries, start: datetime = None, end: datetime = None, name_prefix: str = "", num_back=None) -> pd.DataFrame:
    lines_clk = obj.lines.datetime.plotrange(start, end)

    df = pd.DataFrame()
    # iterate all lines
    for lineidx in range(obj.size()):
        line = obj.lines[lineidx]
        linealias = obj.lines._getlinealias(lineidx)
        if linealias == 'datetime':
            continue

        # get data limited to time range
        data = line.plotrange(start, end)

        ndata = convert_to_master_clock(data, lines_clk, master_clock)

        df[name_prefix + linealias] = ndata

    df[name_prefix + 'datetime'] = [bt.num2date(x) for x in master_clock]

    return df


def get_clock_line(obj: Union[bt.ObserverBase, bt.IndicatorBase, bt.StrategyBase]):
    """Find the corresponding clock for an object. A clock is a datetime line that holds timestamps for the line in question."""
    if isinstance(obj, (bt.ObserverBase, bt.IndicatorBase)):
        return get_clock_line(obj._clock)
    elif isinstance(obj, (bt.StrategyBase, bt.AbstractDataBase)):
        clk = obj
    elif isinstance(obj, bt.LineSeriesStub):
        # indicators can be created to run on a single line (instead of e.g. a data object)
        # in that case we grab the owner of that line to find the corresponding clok
        return get_clock_line(obj._owner)
    elif isinstance(obj, bt.LineActions):
        # used for line actions like "macd > data[0]"
        return get_clock_line(obj._owner)
    else:
        raise Exception(f'Unsupported object type passed: {obj.__class__}')
    return clk.lines.datetime


def find_by_plotid(strategy: bt.Strategy, plotid):
    objs = itertools.chain(strategy.datas, strategy.getindicators(), strategy.getobservers())
    founds = []
    for obj in objs:
        if getattr(obj.plotinfo, 'plotid', None) == plotid:
            founds.append(obj)

    num_results = len(founds)
    if num_results == 0:
        return None
    elif num_results == 1:
        return founds[0]
    else:
        raise RuntimeError(f'Found multiple objects with plotid "{plotid}"')
