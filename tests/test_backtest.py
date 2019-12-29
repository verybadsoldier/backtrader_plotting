import backtrader as bt
import datetime
import pytest

import backtrader_plotting.bokeh.bokeh
from backtrader_plotting import Bokeh
from tests.strategies.togglestrategy import ToggleStrategy
from tests.asserts.asserts import assert_num_tabs, assert_num_figures

# disable for debuggin
_inmemory = False


@pytest.fixture
def cerebro() -> bt.Cerebro:
    cerebro = bt.Cerebro()

    datapath = 'datas/orcl-1995-2014.txt'
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        fromdate=datetime.datetime(2000, 1, 1),
        todate=datetime.datetime(2000, 12, 31),
        reverse=False)
    cerebro.adddata(data)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio)
    cerebro.addanalyzer(bt.analyzers.TimeDrawDown)

    return cerebro


@pytest.fixture()
def cerebro_no_optreturn() -> bt.Cerebro:
    cerebro = bt.Cerebro(optreturn=False)

    datapath = 'datas/orcl-1995-2014.txt'
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        fromdate=datetime.datetime(2000, 1, 1),
        todate=datetime.datetime(2000, 12, 31),
        reverse=False)
    cerebro.adddata(data)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio)
    cerebro.addanalyzer(bt.analyzers.TimeDrawDown)

    return cerebro


def test_std_backtest_volume_subplot(cerebro: bt.Cerebro):
    cerebro.addstrategy(bt.strategies.MA_CrossOver)
    cerebro.run()

    s = backtrader_plotting.schemes.Blackly()
    s.voloverlay = False
    b = Bokeh(style='bar', scheme=s)
    figs = cerebro.plot(b)

    assert isinstance(figs[0][0], backtrader_plotting.bokeh.bokeh.FigurePage)
    assert_num_figures(figs, 6)


def test_std_backtest(cerebro: bt.Cerebro):
    cerebro.addstrategy(bt.strategies.MA_CrossOver)
    cerebro.run()

    s = backtrader_plotting.schemes.Blackly()
    b = Bokeh(style='bar', scheme=s)
    figs = cerebro.plot(b)

    assert_num_tabs(figs, 4)
    assert_num_figures(figs, 5)


def test_std_backtest_ind_subplot(cerebro: bt.Cerebro):
    cerebro.addstrategy(bt.strategies.MA_CrossOver)
    cerebro.run()

    plotconfig = {
        '#:i-0': {
            'subplot': True,
        }
    }

    s = backtrader_plotting.schemes.Blackly()
    b = Bokeh(style='bar', scheme=s, plotconfig=plotconfig)

    figs = cerebro.plot(b)

    assert_num_tabs(figs, 4)
    assert_num_figures(figs, 6)


def test_backtest_2strats(cerebro: bt.Cerebro):
    cerebro.addstrategy(bt.strategies.MA_CrossOver)
    cerebro.addstrategy(ToggleStrategy)
    cerebro.run()

    b = Bokeh(style='bar', inmemory=_inmemory)
    figs = cerebro.plot(b)

    assert_num_tabs(figs, 7)
    assert_num_figures(figs, 8)


def test_optimize(cerebro: bt.Cerebro):
    cerebro.optstrategy(bt.strategies.MA_CrossOver, slow=[5, 10, 20], fast=[5, 10, 20])
    res = cerebro.run()

    b = Bokeh(cerebro, style='bar', inmemory=_inmemory)
    model = b.generate_optresult_model(res)

    def count_children(obj):
        numo = 1
        if hasattr(obj, "children"):
            numo = count_children(obj.children)
        if hasattr(obj, '__len__'):
            numo += len(obj)
        return numo

    num = count_children(model)

    assert num == 3


def test_optimize_no_opreturn(cerebro_no_optreturn: bt.Cerebro):
    cerebro_no_optreturn.optstrategy(bt.strategies.MA_CrossOver, slow=[5, 10, 20], fast=[5, 10, 20])
    res = cerebro_no_optreturn.run()

    b = Bokeh(cerebro_no_optreturn, style='bar', inmemory=_inmemory)
    model = b.generate_optresult_model(res)

    def count_children(obj):
        numo = 1
        if hasattr(obj, "children"):
            numo = count_children(obj.children)
        if hasattr(obj, '__len__'):
            numo += len(obj)
        return numo

    num = count_children(model)

    assert num == 3


def test_ordered_optimize(cerebro: bt.Cerebro):
    from backtrader_plotting import Bokeh
    from backtrader_plotting.bttypes import OrderedOptResult

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
    cerebro.optstrategy(bt.strategies.MA_CrossOver, slow=[5, 10, 20], fast=[5, 10, 20])
    res = cerebro.run()

    def benchmark(optresults):
        a = [x.analyzers.tradeanalyzer.get_analysis() for x in optresults]
        return sum([x.pnl.gross.total if 'pnl' in x else 0 for x in a])

    result = [OrderedOptResult.BenchmarkedResult(benchmark(x), x) for x in res]
    ordered_result = sorted(result, key=lambda x: x.benchmark, reverse=True)

    ordered_result = OrderedOptResult("Profit & Losss", ordered_result)

    b = Bokeh(style='bar')
    model = b.generate_optresult_model(ordered_result)

    def count_children(obj):
        numo = 1
        if hasattr(obj, "children"):
            numo = count_children(obj.children)
        if hasattr(obj, '__len__'):
            numo += len(obj)
        return numo

    num = count_children(model)

    assert num == 3
