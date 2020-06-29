#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime

import backtrader as bt
import logging
from freezegun import freeze_time

from btplotting import LivePlot
from btplotting.schemes import Blackly
from btplotting.analyzers import Recorder
from btplotting.feeds import FakeFeed

_logger = logging.getLogger(__name__)


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
                   num_gen_bars=20,
                   ) -> bt.Strategy:
    _logger.info("Constructing Cerebro")
    cerebro = bt.Cerebro()
    cerebro.addstrategy(bt.strategies.NullStrategy)

    cerebro.addanalyzer(Recorder)
    cerebro.addanalyzer(LivePlot, volume=False, scheme=Blackly(hovertool_timeformat='%F %R:%S'), lookback=120)

    data = FakeFeed(timeframe=data_timeframe,
                             compression=data_compression,
                             run_duration=datetime.timedelta(seconds=runtime_seconds),
                             starting_value=starting_value,
                             tick_interval=tick_interval,
                             live=True,
                             num_gen_bars=num_gen_bars,
                             start_delay=0,
                             name='data0',
                             )

    cerebro.resampledata(data, timeframe=resample_timeframe, compression=resample_compression)

    data2 = FakeFeed(timeframe=data_timeframe,
                              compression=data_compression,
                              run_duration=datetime.timedelta(seconds=runtime_seconds),
                              starting_value=starting_value,
                              tick_interval=tick_interval,
                              live=True,
                              num_gen_bars=num_gen_bars,
                              start_delay=80,
                              name='data1',
                              )

    cerebro.resampledata(data2, timeframe=resample_timeframe, compression=resample_compression)

    # return the recorded bars attribute from the first strategy
    res = cerebro.run()
    return cerebro, res[0]


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(name)s:%(levelname)s:%(message)s', level=logging.DEBUG)
    cerebro, strat = _run_resampler(data_timeframe=bt.TimeFrame.Ticks,
                                    data_compression=1,
                                    resample_timeframe=bt.TimeFrame.Minutes,
                                    resample_compression=1,
                                    runtime_seconds=190,
                                    )
