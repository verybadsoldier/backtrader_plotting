import backtrader as bt


class ToggleStrategy(bt.Strategy):
    params = (
        ('modbuy', 23),
        ('modsell', 54),
    )

    def next(self):
        pos = len(self.data)
        if pos % self.p.modbuy == 0:
            self.buy(self.datas[0], size=None)

        if pos % self.p.modsell == 0:
            self.sell(self.datas[0], size=None)
