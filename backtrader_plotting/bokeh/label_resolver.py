from typing import List, Optional, Union

import backtrader as bt

from backtrader_plotting.utils import get_params_str
import operator


def datatarget2label(datas: List[Union[bt.AbstractDataBase, bt.Indicator]]):
    """Convert datas (usually a datafeed but might also be an indicator if one indicator operates on another indicator) to a readable string.
    If a name was provided manually then use that."""

    # try some popular attributes that might carry string represantations
    # _name: user assigned value upon instantiation
    # shortname: used by some datafeeds
    # _dataname: underlying bt dataname (should always be available as last resort)
    prim_names = ['_name', 'shortname', '_dataname']
    labels = []
    for d in datas:
        if isinstance(d, bt.Indicator):
            labels.append(obj2label(d))
        elif isinstance(d, bt.AbstractDataBase):
            for n in prim_names:
                val = getattr(d, n, "")
                if val is None:
                    continue
                val = str(val)

                if len(val) > 0:
                    labels.append(val)
                    break
        else:
            raise RuntimeError(f'Unexpected data type: {d.__class__}')

    if len(labels) == 0:
        return "Unidentified"
    return ','.join(labels)

"""
def strategy2label(strategycls: bt.MetaStrategy, params: Optional[bt.AutoInfoClass]) -> str:
    label = strategycls.__name__
    parameter_labels = get_params_str(params)
    if len(parameter_labels) > 0:
        label += f' [{parameter_labels}'
    return label
    
    
def strategy2shortname(strategy: bt.Strategy) -> str:
    return strategy.plotinfo.plotname or strategy.__class__.__name__
"""


def plotobj2label(obj):
    if isinstance(obj, (bt.Indicator, bt.LinesOperation)):
        return f'{obj2label(obj)}@{get_ind_depends_str(obj)}'
    elif isinstance(obj, bt.Observer):
        return f'{obj2label(obj)}'
    elif isinstance(obj, bt.AbstractDataBase):
        return obj.__class__.__name__
    else:
        raise RuntimeError(f'Unsupported type: {obj.__class__.__name__}')


def _operator2string(op) -> str:
    opmap = {
        operator.sub:'-',
        operator.add: '+',
        operator.mul: '*',
        operator.truediv: '/',
        operator.gt: '>',
        operator.ge: '>=',
        operator.lt: '<',
        operator.le: '<=',
        operator.eq: '=',
        operator.mod: '%',
        operator.ne: '!=',
    }

    return opmap[op] if op in opmap else '?'


def obj2label(ind: Union[bt.Indicator, bt.LinesOperation]):
    if isinstance(ind, bt.LinesOperation):
        # for LinesOperations we try to convert the operation to a string (like '+', '-' etc.)
        for cur_ind in [x for x in ind._owner.getindicators() if isinstance(x, bt.LinesOperation)]:
            op = _operator2string(cur_ind.operation)
            if cur_ind is ind:
                return f'LineOp{op}'
        raise RuntimeError('Line not found')
    else:
        return ind.plotlabel()


def get_line_alias(owner, line):
    for idx, curline in enumerate(owner.lines):
        if curline is not line:
            continue
        return owner._getlinealias(idx)
    raise RuntimeError('Could not find line in owner lines')


def line2str(line, owner):
    if isinstance(line, bt.LinesOperation):
        # (2)
        return plotobj2label(line)
    elif isinstance(line, bt.linebuffer._LineDelay):
        # (2)
        return plotobj2label(line)
    else:
        # (1) we try to find the line name by reading the owner
        alias = get_line_alias(owner, line)  # we basically assume to only hae 1 line here
        return alias + '@' + datatarget2label([owner])


def get_ind_depends_str(ind: Union[bt.Indicator, bt.LineActions]) -> str:
    """Returns a string listing all involved data feeds. Empty string if there is only a single feed in the mix"""
    names = []
    if isinstance(ind, bt.LinesOperation):
        # datas are the lines involved in the LinesOperation
        for line in ind._datas:
            name = line2str(line, line._owner) + '@' + datatarget2label([line._owner])
            names.append(name)
    else:
        for x in ind.datas:
            if isinstance(x, bt.AbstractDataBase):
                names.append(datatarget2label([x]))
            elif isinstance(x, bt.LineSeriesStub):
                # this happens in the following situations:
                # 1. an indicator was created on a specific data line (e.g. SMA(data.high)
                # 2. an indicator was created on an existing LineOperation
                line0 = x.lines[0]
                names.append(line2str(line0, x._owner))
            elif isinstance(x, bt.Indicator):
                names.append(obj2label(x))
    return f"({','.join(names)})"
