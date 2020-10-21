from backtrader_plotting import Bokeh
import datetime
import backtrader as bt
from backtrader_plotting.schemes import Tradimo

from testcommon import getdatadir


class BokehTest(bt.Strategy):

    def __init__(self):
        self.rsi = bt.indicators.RSI_SMA(self.data.close, period=14, safediv=True)

        # we set it manually (vanilla backtrader doesn't know about plotid so we can't set regularly in constructor)
        self.rsi.plotinfo.plotid = 'rsi'


def test_github_issue37_plotaspectratio():
    cerebro = bt.Cerebro()

    data = bt.feeds.YahooFinanceCSVData(
        dataname=getdatadir("orcl-1995-2014.txt"),
        fromdate=datetime.datetime(2000, 1, 1),
        todate=datetime.datetime(2001, 2, 28),
        reverse=False,
    )

    cerebro.adddata(data)
    cerebro.broker.setcash(10000)
    cerebro.coc = True
    cerebro.broker.setcommission(commission=0.00075)
    cerebro.addstrategy(BokehTest)
    cerebro.run()

    plotconfig = {
            'id:rsi': dict(
                plotaspectratio=10,
            ),
        }

    b = Bokeh(style='bar', plot_mode='single', scheme=Tradimo(), plotconfig=plotconfig, output_mode='memory')
    output = cerebro.plot(b)

    assert output[0][0].figures[1].bfigure.aspect_ratio == 10
