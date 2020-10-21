# -*- coding:utf-8 -*-
import backtrader as bt
import datetime
from backtrader_plotting import Bokeh

from testcommon import getdatadir


class MyStrategy(bt.Strategy):
    def __init__(self):
        sma = bt.indicators.SimpleMovingAverage(period=20, subplot=True)
        sma2 = bt.indicators.SimpleMovingAverage(period=5, subplot=True, plotmaster=sma)

    def next(self):
        pos = len(self.data)
        if pos == 45 or pos == 145:
            self.buy(self.datas[0], size=None)

        if pos == 116 or pos == 215:
            self.sell(self.datas[0], size=None)


def test_github_issue44():
    cerebro = bt.Cerebro()
    data = bt.feeds.YahooFinanceCSVData(
        dataname=getdatadir("20170319-20200319-0388.HK.csv"),
        fromdata=datetime.datetime(2020, 2, 19, 0, 0, 0,),
        todata=datetime.datetime(2020, 3, 19, 0, 0, 0),
        reverse=False
    )

    cerebro.addobserver(bt.observers.Benchmark, data=data, timeframe=bt.TimeFrame.NoTimeFrame)
    cerebro.addstrategy(MyStrategy)
    cerebro.adddata(data)
    cerebro.addsizer(bt.sizers.AllInSizer)
    cerebro.broker.setcash(100000)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    import json
    strats = cerebro.run()[0]
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    bo = Bokeh(style='bar', plot_mode='single', output_mode='memory')
    cerebro.plot(bo)


if __name__ == '__main__':
    test_github_issue44()
