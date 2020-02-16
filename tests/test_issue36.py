import os

import pandas as pd
import backtrader as bt
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo


class MyStrategy(bt.Strategy):
    def next(self):
        pos = len(self.data)
        if pos == 45 or pos == 145:
            self.buy(self.datas[0], size=None)

        if pos == 116 or pos == 215:
            self.sell(self.datas[0], size=None)


def test_github_issue36():
    cerebro = bt.Cerebro()

    cerebro.addstrategy(MyStrategy)

    filedir = os.path.dirname(os.path.abspath(__file__))

    df = pd.read_csv(os.path.join(filedir, "datas/NQ.csv"), index_col=0)
    df.index = pd.DatetimeIndex(df.index)
    data = bt.feeds.PandasData(dataname=df, name='NQ', timeframe=bt.TimeFrame.Minutes)
    cerebro.adddata(data)
    cerebro.resampledata(data, name='NQ_5min', timeframe=bt.TimeFrame.Minutes, compression=5)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio)
    cerebro.run()

    b = Bokeh(filename='chart.html', style='bar', plot_mode="single", scheme=Tradimo(), output_mode='memory')
    cerebro.plot(b)


if __name__ == "__main__":
    test_github_issue36()
