import backtrader as bt
import datetime
import pytest
import threading
import time
import tornado.ioloop


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
    return cerebro


def test_backtest(cerebro: bt.Cerebro):
    import backtrader_plotting.bokeh.bokeh
    from backtrader_plotting import Bokeh

    cerebro.addstrategy(bt.strategies.MA_CrossOver)

    cerebro.run()

    b = Bokeh(style='bar', plot_mode='single')
    figs = cerebro.plot(b)

    assert isinstance(figs[0][0], backtrader_plotting.bokeh.bokeh.FigurePage)
    assert len(figs[0][0].figures) == 4


def test_optimize(cerebro: bt.Cerebro):
    from backtrader_plotting import Bokeh

    cerebro.optstrategy(bt.strategies.MA_CrossOver, slow=[5, 10, 20], fast=[5, 10, 20])
    res = cerebro.run()

    b = Bokeh(style='bar', plot_mode='single')
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
