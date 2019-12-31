import backtrader as bt
from typing import Union, List


BtResult = List[bt.Strategy]  # backtrader backtest result
OptResult = List[List[bt.OptReturn]]  # backtrader optresult


class OrderedOptResult:
    """Class to store an optresult which has been evaluated by a benchmark. The benchmark has a name (`benchmark_label`)."""
    class BenchmarkedResult:
        def __init__(self, benchmark_value, optresult: List[List[Union[bt.OptReturn]]]):
            self.benchmark_value: int = benchmark_value
            self.optresult = optresult

        def __getitem__(self, key):
            return self.optresult[key]

    def __init__(self, optresult: List, benchmark_label: str, benchmark_fnc):
        """benchmark_fnc is a callable that accepts a list of analysises (one for each involved strategy) and returns a value (usually numerical)"""
        def benchmark(optresults):
            analysises = [x.analyzers.tradeanalyzer.get_analysis() for x in optresults]
            return benchmark_fnc(analysises)

        # calculate benchmark score for every optresult
        result: List[OrderedOptResult.BenchmarkedResult] = [OrderedOptResult.BenchmarkedResult(benchmark(x), x) for x in optresult]

        self.benchmark_label: str = benchmark_label
        self.benchmarked_results: List[OrderedOptResult.BenchmarkedResult] = sorted(result, key=lambda x: x.benchmark_value, reverse=True)

    def __len__(self):
        return len(self.benchmarked_results)

    def __getitem__(self, key):
        return self.benchmarked_results[key]

    @property
    def num_strategies(self):
        if len(self) == 0:
            return None
        return len(self.benchmarked_results[0].optresult)


def is_btresult(result: Union[BtResult, OptResult, OrderedOptResult]):
    return isinstance(result, List) and isinstance(result[0], bt.Strategy) and len(result) > 0


def is_optresult(result: Union[BtResult, OptResult, OrderedOptResult]):
    return isinstance(result, List) and \
           isinstance(result[0], List) and \
           len(result[0]) > 0 and \
           isinstance(result[0][0], (bt.OptReturn, bt.Strategy)) and \
           len(result) > 0


def is_ordered_optresult(result: Union[BtResult, OptResult, OrderedOptResult]):
    return isinstance(result, OrderedOptResult)
