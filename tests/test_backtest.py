import backtrader as bt
import datetime
import pytest

import backtrader_plotting.bokeh.bokeh
from backtrader_plotting import Bokeh
from backtrader_plotting.bokeh.optbrowser import OptBrowser
from backtrader_plotting.bttypes import OrderedOptResult

from tests.strategies.togglestrategy import ToggleStrategy
from tests.asserts.asserts import assert_num_tabs, assert_num_figures

# set to 'show' for debugging
_output_mode = 'show'


@pytest.fixture
def cerebro() -> bt.Cerebro:
    cerebro = bt.Cerebro()

    datapath = 'datas/orcl-1995-2014.txt'
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        fromdate=datetime.datetime(1998, 1, 1),
        todate=datetime.datetime(2000, 12, 31),
        reverse=False,
        swapcloses=True,
    )
    cerebro.adddata(data)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, compression=2)
    cerebro.addanalyzer(bt.analyzers.TimeDrawDown)

    return cerebro


@pytest.fixture()
def cerebro_no_optreturn() -> bt.Cerebro:
    cerebro = bt.Cerebro(optreturn=False)

    datapath = 'datas/orcl-1995-2014.txt'
    data = bt.feeds.YahooFinanceCSVData(
        dataname=datapath,
        fromdate=datetime.datetime(1998, 1, 1),
        todate=datetime.datetime(2000, 12, 31),
        reverse=False,
        swapcloses=True,
    )
    cerebro.adddata(data)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, compression=2)
    cerebro.addanalyzer(bt.analyzers.TimeDrawDown)

    return cerebro


def test_std_backtest_volume_subplot(cerebro: bt.Cerebro):
    cerebro.addstrategy(bt.strategies.MA_CrossOver)
    cerebro.run()

    s = backtrader_plotting.schemes.Blackly()
    s.voloverlay = False
    b = Bokeh(style='bar', scheme=s, output_mode=_output_mode)
    figs = cerebro.plot(b)

    assert len(figs) == 1
    assert_num_tabs(figs, 3)
    assert_num_figures(figs, 5)


def test_std_backtest(cerebro: bt.Cerebro):
    cerebro.addstrategy(bt.strategies.MA_CrossOver)
    cerebro.run()

    s = backtrader_plotting.schemes.Blackly()
    b = Bokeh(style='bar', scheme=s, output_mode=_output_mode)
    figs = cerebro.plot(b)

    assert len(figs) == 1
    assert_num_tabs(figs, 3)
    assert_num_figures(figs, 4)


def test_std_backtest_tabs_multi(cerebro: bt.Cerebro):
    cerebro.addstrategy(bt.strategies.MA_CrossOver)
    cerebro.run()

    s = backtrader_plotting.schemes.Blackly()
    b = Bokeh(style='bar', tabs='multi', scheme=s, output_mode=_output_mode)
    figs = cerebro.plot(b)

    assert len(figs) == 1
    assert_num_tabs(figs, 5)
    assert_num_figures(figs, 4)


def test_std_backtest_ind_subplot(cerebro: bt.Cerebro):
    cerebro.addstrategy(bt.strategies.MA_CrossOver)
    cerebro.run()

    plotconfig = {
        '#:i-0': {
            'subplot': True,
        }
    }

    s = backtrader_plotting.schemes.Blackly()
    b = Bokeh(style='bar', scheme=s, output_mode=_output_mode, plotconfig=plotconfig)

    figs = cerebro.plot(b)

    assert_num_tabs(figs, 3)
    assert_num_figures(figs, 5)


def test_backtest_2strats(cerebro: bt.Cerebro):
    cerebro.addstrategy(bt.strategies.MA_CrossOver)
    cerebro.addstrategy(ToggleStrategy)
    cerebro.run()

    b = Bokeh(style='bar', output_mode=_output_mode)

    figs = cerebro.plot(b)

    assert len(figs) == 2
    assert_num_tabs(figs, 3, 3)
    assert_num_figures(figs, 4, 4)


def test_optimize(cerebro: bt.Cerebro):
    cerebro.optstrategy(bt.strategies.MA_CrossOver, slow=[5, 10, 20], fast=[5, 10, 20])
    res = cerebro.run(optreturn=True)

    b = Bokeh(style='bar', output_mode=_output_mode)

    browser = OptBrowser(b, res)
    model = browser._build_optresult_model()
    # browser.start()

    def count_children(obj):
        numo = 1
        if hasattr(obj, "children"):
            numo = count_children(obj.children)
        if hasattr(obj, '__len__'):
            numo += len(obj)
        return numo

    num = count_children(model)

    assert num == 3


def test_optimize_2strat(cerebro: bt.Cerebro):
    cerebro.optstrategy(bt.strategies.MA_CrossOver, slow=[5, 10, 20], fast=[5, 10, 20])
    cerebro.optstrategy(ToggleStrategy, modbuy=[12, 15], modsell=[17, 19])
    res = cerebro.run()

    b = Bokeh(style='bar', output_mode=_output_mode)

    browser = OptBrowser(b, res)

    with pytest.raises(RuntimeError):
        browser._build_optresult_model()
    # browser.start()


def test_optimize_no_optreturn(cerebro_no_optreturn: bt.Cerebro):
    cerebro_no_optreturn.optstrategy(bt.strategies.MA_CrossOver, slow=[5, 10, 20], fast=[5, 10, 20])
    res = cerebro_no_optreturn.run()

    s = backtrader_plotting.schemes.Blackly()
    b = Bokeh(style='bar', output_mode=_output_mode, scheme=s)

    browser = OptBrowser(b, res)
    model = browser._build_optresult_model()
    #browser.start()

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
    cerebro.optstrategy(bt.strategies.MA_CrossOver, slow=[20], fast=[5, 10, 20])
    res = cerebro.run(optreturn=True)

    ordered_result = OrderedOptResult(res, "Profit & Losss", lambda analysises: sum([x.pnl.gross.total if 'pnl' in x else 0 for x in analysises]))

    b = Bokeh(style='bar', output_mode=_output_mode)

    browser = OptBrowser(b, ordered_result)
    model = browser._build_optresult_model()
    # browser.start()

    def count_children(obj):
        numo = 1
        if hasattr(obj, "children"):
            numo = count_children(obj.children)
        if hasattr(obj, '__len__'):
            numo += len(obj)
        return numo

    num = count_children(model)

    assert num == 3
