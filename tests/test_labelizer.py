import backtrader as bt
import datetime
import backtrader_plotting.bokeh.labelizer as labelizer

from testcommon import getdatadir


def _run_test_cerebro(stratcls, expected_results):
    if not isinstance(expected_results, list):
        expected_results = [expected_results]

    c = bt.Cerebro()
    c.addstrategy(stratcls)
    data = bt.feeds.YahooFinanceCSVData(
        dataname=getdatadir("orcl-1995-2014.txt"),
        fromdate=datetime.datetime(2000, 1, 1),
        todate=datetime.datetime(2001, 1, 2),
        reverse=False,
    )
    c.adddata(data)

    data = bt.feeds.YahooFinanceCSVData(
        dataname=getdatadir("orcl-1995-2014.txt"),
        fromdate=datetime.datetime(2000, 1, 1),
        todate=datetime.datetime(2001, 1, 2),
        reverse=False,
    )
    c.adddata(data)

    c.run()

    strategy = c.runstrats[0][0]
    for idx, ind in enumerate(strategy.getindicators()):
        label = labelizer.label(ind)
        assert(label == expected_results[idx])


def test_simple():
    """A simple indicator"""
    class StratSimple(bt.Strategy):
        def __init__(self):
            # simple indicator
            bt.indicators.SimpleMovingAverage(period=20, subplot=True)

    _run_test_cerebro(StratSimple, 'SimpleMovingAverage (20)@(orcl-1995-2014)')


def test_simple_2lines():
    """An indicator based on two input lines"""
    class StratSimple(bt.Strategy):
        def __init__(self):
            self.crossLong = bt.ind.CrossOver(self.data0, self.data1)

    _run_test_cerebro(StratSimple, 'CrossOver@(orcl-1995-2014,orcl-1995-2014)')


def test_simple_2lines_spec():
    """An indicator based on two input lines. One line is a specific line of a feed"""
    class StratSimple(bt.Strategy):
        def __init__(self):
            bt.ind.CrossOver(self.data.high, self.data1)

    _run_test_cerebro(StratSimple, ['CrossOver@(orcl-1995-2014^high,orcl-1995-2014)'])


def test_specific_line():
    """An inddicator based on  specifiy line"""
    class StratSpecific(bt.Strategy):
        def __init__(self):
            bt.indicators.SimpleMovingAverage(self.data.high, period=20, subplot=True)

    _run_test_cerebro(StratSpecific, 'SimpleMovingAverage (20)@(orcl-1995-2014^high)')


def test_simple_2lines():
    class StratSimple2Lines(bt.Strategy):
        def __init__(self):
            # another simple indicator (with 2 lines)
            self.di = bt.ind.DI(period=10)

    _run_test_cerebro(StratSimple2Lines, 'DI (10)@(orcl-1995-2014)')


def test_lineop():
    """A line operation on 2 lines of an indicator"""
    class StratLineOp(bt.Strategy):
        def __init__(self):
            di = bt.ind.DI(period=10)
            di.plusDI - di.minusDI

    exp_results = [
        'DI (10)@(orcl-1995-2014)',
        'LineOp-@(DI (10)^plusDI@(orcl-1995-2014),DI (10)^minusDI@(orcl-1995-2014))',
    ]
    _run_test_cerebro(StratLineOp, exp_results)


def test_ind_on_lineop():
    """An indicator based on 2 lines. One line is a line operation on two result lines on another indicator"""
    class StratIndOnLineOp(bt.Strategy):
        def __init__(self):
            di = bt.ind.DI(period=10)
            ddif = di.plusDI - di.minusDI

            bt.ind.CrossOver(ddif, self.data1)

    exp_results = [
        'DI (10)@(orcl-1995-2014)',
        'LineOp-@(DI (10)^plusDI@(orcl-1995-2014),DI (10)^minusDI@(orcl-1995-2014))',
        'CrossOver@(LineOp-@(DI (10)^plusDI@(orcl-1995-2014),DI (10)^minusDI@(orcl-1995-2014)),orcl-1995-2014)',
    ]
    _run_test_cerebro(StratIndOnLineOp, exp_results)


def test_data_lineop2():
    """Two connected lineops on data"""
    class StratIndOnLineOp(bt.Strategy):
        def __init__(self):
            self.data.high - self.data.close * 2

    exp_results = [
        'LineOp*@(orcl-1995-2014^close,2)',
        'LineOp-@(orcl-1995-2014^high,LineOp*@(orcl-1995-2014^close,2))',
    ]
    _run_test_cerebro(StratIndOnLineOp, exp_results)


def test_lineop_data_lines():
    """A line operation on data lines"""
    class StratLineOpData(bt.Strategy):
        def __init__(self):
            self.data.high - self.data.close

    _run_test_cerebro(StratLineOpData, 'LineOp-@(orcl-1995-2014^high,orcl-1995-2014^close)')


def test_coupled_in_lines():
    class StratLineOpData(bt.Strategy):
        def __init__(self):
            # data0 is a daily data
            sma0 = bt.indicators.SMA(self.data0, period=15)  # 15 days sma
            # data1 is a weekly data
            sma1 = bt.indicators.SMA(self.data1, period=5)  # 5 weeks sma

            self.buysig = sma0 > sma1()

    _run_test_cerebro(StratLineOpData, ['SMA (15)@(orcl-1995-2014)',
                                        'SMA (5)@(orcl-1995-2014)',
                                        'Coupler@(SMA (5)@(orcl-1995-2014))',
                                        'LineOp>@(SMA (15)^sma@(orcl-1995-2014),SMA (5)@(orcl-1995-2014)^sma)'])


def test_delayed_data():
    """A line operation on data lines"""
    class StratLineOpData(bt.Strategy):
        def __init__(self):
            bt.indicators.SimpleMovingAverage(self.data.high(-7), period=20, subplot=True)

    _run_test_cerebro(StratLineOpData, ['orcl-1995-2014^high(-7)', 'SimpleMovingAverage (20)@(orcl-1995-2014^high(-7))'])
