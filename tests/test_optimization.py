import os

import pandas as pd

import backtrader as bt

from backtrader_plotting import Bokeh, OptBrowser
from backtrader_plotting.schemes import Tradimo


class MyStrategy(bt.Strategy):
    params = (
        ('buydate', 21),
        ('holdtime', 20),
    )

    def __init__(self):
        sma1 = bt.indicators.SMA(period=11, subplot=True)
        bt.indicators.SMA(period=17, plotmaster=sma1)
        bt.indicators.RSI()

    def next(self):
        pos = len(self.data)
        if pos == self.p.buydate:
            self.buy(self.datas[0], size=None)

        if pos == self.p.buydate + self.p.holdtime:
            self.sell(self.datas[0], size=None)


def test_optimization_pandas():
    cerebro = bt.Cerebro(maxcpus=1)

    filedir = os.path.dirname(os.path.abspath(__file__))

    df = pd.read_csv(os.path.join(filedir, "datas/NQ.csv"), index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    data = bt.feeds.PandasData(dataname=df, name='NQ', timeframe=bt.TimeFrame.Minutes)
    cerebro.adddata(data)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)

    cerebro.optstrategy(MyStrategy, buydate=range(40, 180, 30))

    result = cerebro.run(optreturn=True)

    b = Bokeh(style='bar', scheme=Tradimo())
    browser = OptBrowser(b, result)
    browser.start()


if __name__ == '__main__':
    test_optimization_pandas()
