# backtrader_plotting
Library to add extended plotting capabilities to `backtrader` (https://www.backtrader.com/). Currently the only available backend is `Bokeh` (https://bokeh.org/).

## Features
* Interactive plots
* Live trading support (needs custom `backtrader` - refer to the Wiki for further information)
* Data replaying support
* Interactive `backtrader` optimization result browser (only supported for single-strategy runs)
* Customziable tab panels
* Highly configurable
* Different skinnable themes
* Easy to use

Needs Python >= 3.6.

## Demos
Find runnable function demonstrations in folder `demos` or try the live demos here:
https://verybadsoldier.github.io/backtrader_plotting/

## Installation
`pip install backtrader_plotting`

## Quickstart

```python
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo

<your backtrader code>

b = Bokeh(style='bar', plot_mode='single', scheme=Tradimo())
cerebro.plot(b)
```

## Minimal Example
```python
import datetime

import backtrader as bt

from backtrader_plotting import Bokeh


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
        dataname="datas/orcl-1995-2014.txt",
        # Do not pass values before this date
        fromdate=datetime.datetime(2000, 1, 1),
        # Do not pass values after this date
        todate=datetime.datetime(2001, 2, 28),
        reverse=False,
        )
    cerebro.adddata(data)

    cerebro.run()

    b = Bokeh(style='bar', plot_mode='single')
    cerebro.plot(b)
```

## Plotting Optimization Results
Another way to use this package is to use the `OptBrowser` to browse a `backtrader` optimization result:

```python
...
cerebro.optstrategy(TestStrategy, buydate=range(1, 10, 1))
cerebro.addanalyzer(bt.analyzers.SharpeRatio)
...
res = cerebro.run()
bo = Bokeh()
browser = OptBrowser(bo, result)
browser.start()
```

This will start a Bokeh application (standalone webserver) displaying all optimization results. Different results can be selected and viewed.

It is possible possible to add further user-provided columns.
When dealing with huge amounts of optimization results the number of results can be limited and the remaining results can be sorted by a user-provided function to allow for simple selection of the best results.

## Documentation
Please refert to the Wiki for further documentation: https://github.com/verybadsoldier/backtrader_plotting/wiki
