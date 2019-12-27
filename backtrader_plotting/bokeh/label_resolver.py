from typing import Optional

import backtrader as bt

from backtrader_plotting.utils import get_params_str


def _datafeed2name(data: bt.AbstractDataBase):
    """Convert a datafeed to a readable string. If a name was provided manually then use that."""

    # try some popular attributes that might carry string represantations
    # _name: user assigned value upon instantiation
    # shortname: used by some datafeeds
    # _dataname: underlying bt dataname (should always be available as last resort)
    prim_names = ['_name', 'shortname', '_dataname']
    for n in prim_names:
        val = getattr(data, n, "")
        if val is None:
            continue
        val = str(val)

        if len(val) > 0:
            return val

    return "Unidentified"


def strategy2label(strategycls: bt.MetaStrategy, params: Optional[bt.AutoInfoClass]) -> str:
    label = strategycls.__name__
    parameter_labels = get_params_str(params)
    if len(parameter_labels) > 0:
        label += f' [{parameter_labels}'
    return label


def plotobj2label(obj):
    if isinstance(obj, bt.Indicator):
        return f'{_indicator2label(obj)}@{_indicator2fullid(obj)}'
    elif isinstance(obj, bt.Observer):
        return f'{_observer2label(obj)}@{strategy2label(type(obj._owner), obj._owner.params)}'


def _indicator2label(ind: bt.Indicator):
    return ind.plotlabel()


def _observer2label(obs: bt.Observer):
    return obs.plotlabel()


def _indicator2fullid(ind: bt.Indicator) -> str:
    """Returns a string listing allZ involved data feeds. Empty string if there is only a single feed in the mix"""
    names = []
    for x in ind.datas:
        if isinstance(x, bt.AbstractDataBase):
            return _datafeed2name(x)
        elif isinstance(x, bt.Indicator):
            names.append(_indicator2fullid(x))
    return f"({','.join(names)})"
