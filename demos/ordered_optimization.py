import datetime
import backtrader as bt
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo
from backtrader_plotting.bttypes import OrderedOptResult


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
    )
    cerebro.adddata(data)
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)

    cerebro.optstrategy(MyStrategy, buydate=range(40, 180, 30))

    optres = cerebro.run(optreturn=False)

    def benchmark(optresults):
        a = [x.analyzers.tradeanalyzer.get_analysis() for x in optresults]
        return sum([x.pnl.gross.total if 'pnl' in x else 0 for x in a])

    result = [OrderedOptResult.BenchmarkedResult(benchmark(x), x) for x in optres]
    ordered_result = sorted(result, key=lambda x: x.benchmark, reverse=True)

    ordered_result = OrderedOptResult("Profit & Losss", ordered_result)

    b = Bokeh(style='bar', plot_mode="single", scheme=Tradimo())
    b.plot_result(ordered_result)
