import datetime  # For datetime objects

# Import the backtrader platform
import backtrader as bt

import backtrader_plotting

from testcommon import getdatadir


# Create a Stratey
class MACDStrategy(bt.Strategy):
    params = (
        # Standard MACD Parameters
        ('macd1', 12),
        ('macd2', 26),
        ('macdsig', 9),
    )

    def __init__(self):
        self.macd = bt.indicators.MACD(self.data,
                                       period_me1=self.p.macd1,
                                       period_me2=self.p.macd2,
                                       period_signal=self.p.macdsig)
        # backtrader.LinePlotterIndicator(macd, name='MACD')
        # Cross of macd.macd and macd.signal
        self.mcross = bt.indicators.CrossOver(self.macd.macd, self.macd.signal)

    # backtrader.LinePlotterIndicator(mcross, name='MACDCross')

    def start(self):
        self.order = None  # sentinel to avoid operrations on pending order

    def log(self, txt, dt=None):
        """ Logging function for this strategy
        """
        dt = dt or self.datas[0].datetime.date(0)
        time = self.datas[0].datetime.time()
        print('%s,%s' % (dt.isoformat(), txt))

    def next(self):
        if self.order:
            return  # pending order execution

        if not self.position:  # not in the market
            if self.mcross[0] > 0.0 and self.macd.lines.signal[0] > 0 and self.macd.lines.macd[0] > 0:
                self.order = self.buy()
                self.log('BUY CREATED, %.2f' % self.data[0])
        # else:
        # 	if self.mcross[0] > 0.0 and self.macd.lines.signal[0] < 0 and self.macd.lines.macd[0] < 0:
        # 		self.order = self.buy()
        # 		self.log('BUY CREATED, %.2f' % self.data[0])

        else:  # in the market
            if self.mcross[0] < 0.0 and self.macd.lines.signal[0] < 0 and self.macd.lines.macd[0] < 0:
                self.sell()  # stop met - get out
                self.log('BUY CREATED, %.2f' % self.data[0])
        # else:
        # 	if self.mcross[0] < 0.0 and self.macd.lines.signal[0] > 0 and self.macd.lines.macd[0] > 0:
        # 		self.sell()  # stop met - get out
        # 		self.log('BUY CREATED, %.2f' % self.data[0])


def bokeh_plot(data):
    # Create a cerebro entity
    cerebro = bt.Cerebro()

    # Add a strategy
    cerebro.addstrategy(MACDStrategy)

    # Add the Data Feed to Cerebro
    cerebro.adddata(data)

    # Set our desired cash start
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0002)

    '''
    -----------------strategy & sizer------------------------------
    '''
    cerebro.addsizer(bt.sizers.PercentSizer, percents=98)

    cerebro.addobserver(bt.observers.FundShares)
    # cerebro.addanalyzer(bt.analyzers.TimeReturn, timeframe=bt.TimeFrame.Years)
    # cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    # Print out the starting conditions
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # Run over everything
    cerebro.run()

    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())
    # cerebro.plot(style='bar')
    b = backtrader_plotting.Bokeh(style='bar', scheme=backtrader_plotting.schemes.Tradimo(), output_mode='memory')
    figs = cerebro.plot(b)

    assert isinstance(figs[0][0], backtrader_plotting.bokeh.bokeh.FigurePage)
    assert len(figs[0][0].figures) == 6


def test_github_issue10():
    data = bt.feeds.YahooFinanceCSVData(
        dataname=getdatadir("orcl-1995-2014.txt"),
        fromdate=datetime.datetime(2000, 1, 1),
        todate=datetime.datetime(2001, 2, 28),
    )

    bokeh_plot(data)


if __name__ == '__main__':
    test_github_issue10()
