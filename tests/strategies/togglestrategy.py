import backtrader as bt


class ToggleStrategy(bt.Strategy):
    def __init__(self):
        macd = bt.indicators.MACD(subplot=True)

    def next(self):
        pos = len(self.data)
        if pos % 12:
            self.buy(self.datas[0], size=None)

        if pos % 19:
            self.sell(self.datas[0], size=None)
