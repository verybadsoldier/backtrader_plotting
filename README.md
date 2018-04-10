# backtrader_plotting
Plotting addon for backtrader to support Bokeh (and maybe more).

This is in an early development state. Expect bugs, heavy restructuring and commit reordering. 
Currently this is Python >3.6 only.

Feel free to test it and play with it. I am happy about feedback, critics and ideas:
https://community.backtrader.com/topic/813/bokeh-integration-interactive-webbrowser-plotting

## Demos
https://verybadsoldier.github.io/backtrader_plotting/

## Installation

Clone the repository:

`git clone https://github.com/verybadsoldier/backtrader_plotting.git`

Switch to branch `develop`:

`git checkout develop`

It is recommended to install the package in development mode so there will be only a link created in the python directory to your cloned repository. So updating your checkout directory will immediately take effect:

`python setup.py install develop`

## Quickstart

```python
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo, PlotMode

<your backtrader code>

b = Bokeh(style='bar', plot_mode=PlotMode.Single, scheme=Tradimo())
cerebro.plot(b)
```

## Minimal Example
```python
import datetime
import backtrader as bt
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import PlotMode


class TestStrategy(bt.Strategy):
    params = (
        ('buydate', 21),
        ('holdtime', 6),
    )

    def next(self):
        if len(self.data) == self.p.buydate:
            self.buy(self.datas[0], size=None)

        if len(self.data) == self.p.buydate + self.p.holdtime:
            self.sell(self.datas[0], size=None)


if __name__ == '__main__':
    cerebro = bt.Cerebro()

    cerebro.addstrategy(TestStrategy, buydate=3)

    data = bt.feeds.YahooFinanceCSVData(
        dataname="data/orcl-1995-2014.txt",
        # Do not pass values before this date
        fromdate=datetime.datetime(2000, 1, 1),
        # Do not pass values after this date
        todate=datetime.datetime(2001, 2, 28),
        reverse=False,
        )
    cerebro.adddata(data)

    cerebro.run()

    b = Bokeh(style='bar', plot_mode=PlotMode.Single)
    cerebro.plot(b)
```

## Plotting Optimization Results
Another way to use this package is to invoke `Bokeh.plot_result`. This function also supports passing a `cerebro` result object as it is generated in optimization:

```
    ...
    cerebro.optstrategy(TestStrategy, buydate=range(1, 10, 1))
    cerebro.addanalyzer(bt.analyzers.SharpeRatio)
    ...
    res = cerebro.run()
    bo = Bokeh()
    bo.plot_result(res)
```

This will start a Bokeh application (standalone webserver) displaying all optimization results.

*NOTE:* When using this feature with `optreturn=True` then your package of `backtrader` has to contain this change of mine:
https://github.com/verybadsoldier/backtrader/commit/f03a0ed115338ed8f074a942f6520b31c630bcfb
