import backtrader.analyzers


def inject_humanreadable():
    _labels = {
        backtrader.analyzers.sharpe.SharpeRatio: {
            'sharperatio': 'Sharp-Ratio'
        },
        backtrader.analyzers.DrawDown: {
            'len': 'Length',
            'drawdown': 'Draw-Down',
            'moneydown': 'Money-Down',
            'max': 'Maximum',
        },
    }

    for cls, labdict in _labels.items():
        curlab = getattr(cls, 'human_labels', None)
        if curlab is None:
            setattr(cls, 'human_labels', labdict)
