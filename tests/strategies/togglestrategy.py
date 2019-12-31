import backtrader as bt


class ToggleStrategy(bt.Strategy):
    params = (
        ('modbuy', 12),
        ('modsell', 19),
    )

    def __init__(self):
        macd = bt.indicators.MACD(subplot=True)

    def next(self):
        pos = len(self.data)
        if pos % self.p.modbuy:
            self.buy(self.datas[0], size=None)

        if pos % self.p.modsell:
            self.sell(self.datas[0], size=None)
