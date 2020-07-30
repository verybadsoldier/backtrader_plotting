import datetime
import logging

import backtrader as bt

from backtrader_plotting.schemes import Blackly
try:
    from backtrader_plotting.bokeh.live.plotlistener import PlotListener
except AttributeError:
    raise Exception('Plotting in live mode only supported when using modified backtrader package from this branch https://github.com/verybadsoldier/backtrader/tree/development')

_logger = logging.getLogger(__name__)


class LiveDemoStrategy(bt.Strategy):
    params = (
        ('modbuy', 2),
        ('modsell', 3),
    )

    def __init__(self):
        pass
        #self._sma = bt.indicators.SMA(self.data0.close, subplot=True)
        #self._sma2 = bt.indicators.SMA(self.data1.close, subplot=True)

    def next(self):
        pos = len(self.data)
        if pos % self.p.modbuy == 0:
            if self.broker.getposition(self.datas[0]).size == 0:
                self.buy(self.datas[0], size=None)

        if pos % self.p.modsell == 0:
            if self.broker.getposition(self.datas[0]).size > 0:
                self.sell(self.datas[0], size=None)


def run_cerebro_live(strategycls,
                     data_timeframes,
                     data_compressions,
                     resample_timeframes,
                     resample_compressions,
                     runtime_secondss=27,
                     starting_values=200,
                     tick_intervals=datetime.timedelta(seconds=11),
                     num_gen_barss=None,
                     start_delays=None,
                     num_data=1,
                     fnc_name='resampledata',
                     ):
    def _listify(val):
        return [val] * num_data if not isinstance(val, list) else val

    data_timeframes = _listify(data_timeframes)
    data_compressions = _listify(data_compressions)
    resample_timeframes = _listify(resample_timeframes)
    resample_compressions = _listify(resample_compressions)
    runtimes_secondss = _listify(runtime_secondss)
    starting_values = _listify(starting_values)
    tick_intervals = _listify(tick_intervals)
    num_gen_barss = _listify(num_gen_barss)
    start_delays = _listify(start_delays)

    _logger.info("Constructing Cerebro")
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategycls)

    cerebro.addlistener(bt.listeners.RecorderListener)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)
    cerebro.addlistener(PlotListener, volume=False, scheme=Blackly(hovertool_timeformat='%F %R:%S', tabs='multi'), lookback=120)

    for i in range(0, num_data):
        num_gen_bars = 0 if num_gen_barss[i] is None else num_gen_barss[i]
        data_timeframe = 0 if data_timeframes[i] is None else data_timeframes[i]
        data_compression = 0 if data_compressions[i] is None else data_compressions[i]
        runtime_seconds = 0 if runtimes_secondss[i] is None else runtimes_secondss[i]
        starting_value = 0 if starting_values[i] is None else starting_values[i]
        tick_interval = 0 if tick_intervals[i] is None else tick_intervals[i]
        start_delay = 0 if start_delays[i] is None else start_delays[i]
        resample_timeframe = 0 if resample_timeframes[i] is None else resample_timeframes[i]
        resample_compression = 0 if resample_compressions[i] is None else resample_compressions[i]

        data = bt.feeds.FakeFeed(timeframe=data_timeframe,
                                 compression=data_compression,
                                 run_duration=datetime.timedelta(seconds=runtime_seconds),
                                 starting_value=starting_value,
                                 tick_interval=tick_interval,
                                 live=True,
                                 num_gen_bars=num_gen_bars,
                                 start_delay=start_delay,
                                 name=f'data{i}',
                                 )

        fnc = getattr(cerebro, fnc_name, None)
        fnc(data, timeframe=resample_timeframe, compression=resample_compression)

    # return the recorded bars attribute from the first strategy
    res = cerebro.run()
    return cerebro, res[0]
