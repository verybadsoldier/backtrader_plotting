import datetime
import logging

import backtrader as bt

_logger = logging.getLogger(__name__)


def run_cerebro(strategy: bt.Strategy,
                data_timeframe,
                data_compression,
                resample_timeframe,
                resample_compression,
                runtime_seconds=27,
                starting_value=200,
                tick_interval=datetime.timedelta(seconds=11),
                num_gen_bars=None,
                start_delays=None,
                num_data=1,
                ):
    _logger.info("Constructing Cerebro")
    cerebro = bt.Cerebro()
    cerebro.addstrategy(strategy)

    cerebro.addlistener(bt.listeners.RecorderListener)

    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer)

    if not isinstance(resample_compression, list):
        resample_compression = [resample_compression]

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
                                 live=False,
                                 num_gen_bars=num_gen_bar,
                                 start_delay=start_delay,
                                 name=f'data{i}',
                                 )

        cerebro.resampledata(data, timeframe=resample_timeframe, compression=resample_compression[i])

    # return the recorded bars attribute from the first strategy
    res = cerebro.run()
    return cerebro, res[0]
