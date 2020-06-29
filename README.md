# backtrader_plotting
Library to add extended plotting capabilities to `backtrader` (https://www.backtrader.com/). Currently the only available backend is `Bokeh` (https://bokeh.org/).

**This fork has some changes compared to backtrader_plotting:**

* No need for custom backtrader
* Different naming / structure
* No OptBrowser
* no info logging, using debug instead
* Live plotting is done using an analyzer
* Some functionality removed

## Features
* Interactive plots
* Highly configurable
* Different skinnable themes
* Easy to use

Needs Python >= 3.6.

## Demos
https://verybadsoldier.github.io/backtrader_plotting/

## Installation
`pip install git+https://github.com/happydasch/btplotting`

## Quickstart

```python
from btplotting import BacktraderPlotting
from btplotting.schemes import Tradimo

<your backtrader code>

p = BacktraderPlotting(style='bar', plot_mode='single', scheme=Tradimo())
cerebro.plot(p)
```

## Minimal Example
```python
import datetime

import backtrader as bt

from btplotting import BacktraderPlotting


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

    p = BacktraderPlotting(style='bar', plot_mode='single')
    cerebro.plot(p)
```

## Live Plotting Example

```python
import datetime

import backtrader as bt

from btplotting import LivePlot


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
    cerebro.addanalyzer(
        LivePlot,
        scheme=Tradimo(volume=False),
        lookback=250)
    cerebro.run()

```
