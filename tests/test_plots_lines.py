import datetime
import backtrader as bt
import backtrader_plotting
from backtrader_plotting.schemes import Tradimo

from testcommon import getdatadir


def _run_cerebro(stratcls):
    cerebro = bt.Cerebro()

    cerebro.addstrategy(stratcls)

    datas = [
        'nvda-1999-2014.txt',
        'orcl-1995-2014.txt',
    ]

    for dname in datas:
        data = bt.feeds.YahooFinanceCSVData(
            dataname=getdatadir(dname),
            fromdate=datetime.datetime(2000, 1, 1),
            todate=datetime.datetime(2001, 2, 28),
            reverse=False,
        )
        cerebro.adddata(data)

    cerebro.addanalyzer(bt.analyzers.SharpeRatio)
    cerebro.run()
    b = backtrader_plotting.Bokeh(style='bar', scheme=Tradimo(), output_mode='memory')
    return cerebro.plot(b)


def test_plot_linedelay():
    class LineDelayStrat(bt.Strategy):
        def __init__(self):
            bt.indicators.SimpleMovingAverage(self.data(-5), period=20, subplot=True)

        def next(self):
            pos = len(self.data)
            if pos == 45 or pos == 145:
                self.buy(self.datas[0], size=None)

            if pos == 116 or pos == 215:
                self.sell(self.datas[0], size=None)

    figs = _run_cerebro(LineDelayStrat)

    assert isinstance(figs[0][0], backtrader_plotting.bokeh.bokeh.FigurePage)
    assert len(figs[0][0].figures) == 5
    assert len(figs[0][0].analyzers) == 1


def test_plot_double_operator():
    class LineDelayStrat(bt.Strategy):
        def __init__(self):
            d = self.data0.close - self.data0.high
            d + self.data1.high

        def next(self):
            pos = len(self.data)
            if pos == 45 or pos == 145:
                self.buy(self.datas[0], size=None)

            if pos == 116 or pos == 215:
                self.sell(self.datas[0], size=None)

    figs = _run_cerebro(LineDelayStrat)

    assert isinstance(figs[0][0], backtrader_plotting.bokeh.bokeh.FigurePage)
    assert len(figs[0][0].figures) == 4  # the two LineOps won't be plotted according to original backtrader plotting
    assert len(figs[0][0].analyzers) == 1


def test_coupled_in_lines():
    class StratLineOpData(bt.Strategy):
        def __init__(self):
            # data0 is a daily data
            sma0 = bt.indicators.SMA(self.data0, period=15)  # 15 days sma
            # data1 is a weekly data
            sma1 = bt.indicators.SMA(self.data1, period=5)  # 5 weeks sma

            self.buysig = sma0 > sma1()

    figs = _run_cerebro(StratLineOpData)

    assert isinstance(figs[0][0], backtrader_plotting.bokeh.bokeh.FigurePage)
    assert len(figs[0][0].figures) == 5
    assert len(figs[0][0].analyzers) == 1


def test_plot_lineseriesstub():
    class LineSeriesStubStrat(bt.Strategy):
        def __init__(self):
            di = bt.ind.DI(period=10)
            ddif = di.plusDI - di.minusDI
            self.crossLong = bt.ind.CrossOver(ddif, self.data1)

        def next(self):
            pos = len(self.data)
            if pos == 45 or pos == 145:
                self.buy(self.datas[0], size=None)

            if pos == 116 or pos == 215:
                self.sell(self.datas[0], size=None)

    figs = _run_cerebro(LineSeriesStubStrat)

    assert isinstance(figs[0][0], backtrader_plotting.bokeh.bokeh.FigurePage)
    assert len(figs[0][0].figures) == 6
    assert len(figs[0][0].analyzers) == 1


if __name__ == '__main__':
    test_plot_lineactions()
