# backtrader_plotting
Plotting addon for backtrader to support Bokeh (and maybe more).

This is in an early development state. Expect bugs, heavy restructuring and commit reordering.

Feel free to test it and play with it. I am happy about feedback, critics and ideas:
https://community.backtrader.com/topic/813/bokeh-integration-interactive-webbrowser-plotting

## Demos
https://verybadsoldier.github.io/backtrader_plotting/

## Quickstart

```
from backtrader_plotting import Bokeh
from backtrader_plotting.schemes import Tradimo, PlotMode

b = Bokeh(style='bar', plot_mode=PlotMode.Single, scheme=Tradimo())
cerebro.plot(b)
```
