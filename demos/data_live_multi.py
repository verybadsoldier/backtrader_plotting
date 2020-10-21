#!/usr/bin/env python
# -*- coding: utf-8; py-indent-offset:4 -*-
from __future__ import (absolute_import, division, print_function,
                        unicode_literals)

import datetime
import logging

from utils import run_cerebro_live, LiveDemoStrategy

import backtrader as bt


if __name__ == '__main__':
    logging.basicConfig(format='%(asctime)s %(name)s:%(levelname)s:%(message)s', level=logging.DEBUG)
    cerebro, strat = run_cerebro_live(LiveDemoStrategy,
                                      data_timeframes=bt.TimeFrame.Ticks,
                                      data_compressions=1,
                                      resample_timeframes=bt.TimeFrame.Seconds,
                                      resample_compressions=[5, 10],
                                      runtime_secondss=60000,
                                      tick_intervals=datetime.timedelta(seconds=1),
                                      start_delays=[None, None],
                                      num_gen_barss=[0, 10],
                                      num_data=2,
                                      fnc_name='resampledata',
                                      )
