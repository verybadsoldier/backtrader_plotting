import backtrader as bt
from typing import Dict


def get_nondefault_params(obj: bt.LineBuffer) -> Dict[str, object]:
    return {key: obj._get(key) for key in obj._getkeys() if not obj.isdefault(key)}


def get_strategy_label(strat: bt.Strategy) -> str:
    label = type(strat).__name__
    user_params = get_nondefault_params(strat.params)
    plabs = [f"{x}: {y}" for x, y in user_params.items()]
    plabs = '/'.join(plabs)
    return f'{label} [{plabs}]'


def get_strategy_count(cerebro: bt.Cerebro):
    return len(cerebro.strats)
