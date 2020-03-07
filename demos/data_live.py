#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

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


def _get_trading_calendar(open_hour, close_hour, close_minute):
    cal = bt.TradingCalendar(open=datetime.time(hour=open_hour), close=datetime.time(hour=close_hour, minute=close_minute))
    return cal


def _run_resampler(data_timeframe,
                   data_compression,
                   resample_timeframe,
                   resample_compression,
                   runtime_seconds=27,
                   starting_value=200,
                   tick_interval=datetime.timedelta(seconds=11),
                   num_gen_bars=None,
                   start_delays=None,
                   num_data=1,
                   ) -> bt.Strategy:
    _logger.info("Constructing Cerebro")
    cerebro = bt.Cerebro()
    cerebro.addstrategy(LiveDemoStrategy)

    cerebro.addlistener(bt.listeners.RecorderListener)

    cerebro.addlistener(PlotListener, volume=False, scheme=Blackly(hovertool_timeformat='%F %R:%S'), lookback=120)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)

    for i in range(0, num_data):
        start_delay = 0
        if start_delays is not None and i <= len(start_delays) and start_delays[i] is not None:
            start_delay = start_delays[i]

        num_gen_bar = 0
        if num_gen_bars is not None and i <= len(num_gen_bars) and num_gen_bars[i] is not None:
            num_gen_bar = num_gen_bars[i]

        data = bt.feeds.FakeFeed(timeframe=data_timeframe,
                                 compression=data_compression,
                                 run_duration=datetime.timedelta(seconds=runtime_seconds),
                                 starting_value=starting_value,
                                 tick_interval=tick_interval,
                                 live=True,
                                 num_gen_bars=num_gen_bar,
                                 start_delay=start_delay,
                                 name=f'data{i}',
                                 )

        cerebro.resampledata(data, timeframe=resample_timeframe, compression=resample_compression)

    # return the recorded bars attribute from the first strategy
    res = cerebro.run()
    return cerebro, res[0]


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(name)s:%(levelname)s:%(message)s', level=logging.INFO)
    cerebro, strat = _run_resampler(data_timeframe=bt.TimeFrame.Ticks,
                                    data_compression=1,
                                    resample_timeframe=bt.TimeFrame.Seconds,
                                    resample_compression=10,
                                    runtime_seconds=60000,
                                    tick_interval=datetime.timedelta(seconds=1),
                                    start_delays=[None, 30],
                                    num_gen_bars=[0, 10],
                                    num_data=2,
                                    )
