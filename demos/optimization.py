import datetime

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


if __name__ == '__main__':
    cerebro = bt.Cerebro(maxcpus=1)

    data = bt.feeds.YahooFinanceCSVData(
        dataname="datas/orcl-1995-2014.txt",
        fromdate=datetime.datetime(2000, 1, 1),
        todate=datetime.datetime(2001, 2, 28),
        reverse=False,
        swapcloses=True,
    )
    cerebro.adddata(data)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)

    cerebro.optstrategy(MyStrategy, buydate=range(40, 180, 30))

    result = cerebro.run(optreturn=True)

    b = Bokeh(style='bar', scheme=Tradimo())
    browser = OptBrowser(b, result)
    browser.start()
