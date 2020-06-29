import copy
import logging
from typing import Optional
import backtrader as bt

_logger = logging.getLogger(__name__)


class Recorder(bt.Analyzer):
    def __init__(self):
        self.nexts = []

    @staticmethod
    def print_line_snapshot(name, snapshot):
        line = snapshot['array']
        if name == 'datetime':
            line = [bt.num2date(x) for x in line]
        _logger.debug(f"Line '{name:20}' idx: {snapshot['idx']} - lencount: {snapshot['lencount']} - {list(reversed(line))}")

    @staticmethod
    def print_next(idx, next):
        _logger.debug(f'--- Next: {next["prenext"]} - #{idx}')
        __class__.print_line_snapshot('datetime', next['strategy']['datetime'])

        for di, data in enumerate(next['datas']):
            _logger.debug(f'\t--- Data {di}')
            for k, v in data[1].items():
                __class__.print_line_snapshot(k, v)

        for oi, obs in enumerate(next['observers']):
            _logger.debug(f'\t--- Obvserver {oi}')
            for k, v in obs[1].items():
                __class__.print_line_snapshot(k, v)

    @staticmethod
    def print_nexts(nexts):
        for i, n in enumerate(nexts):
            __class__.print_next(i, n)

    @staticmethod
    def _copy_lines(data):
        lines = {}

        for lineidx in range(data.lines.size()):
            line = data.lines[lineidx]
            linealias = data.lines._getlinealias(lineidx)
            lines[linealias] = {'idx': line.idx, 'lencount': line.lencount, 'array': copy.deepcopy(line.array)}

        return lines

    def _record_data(self, strat, is_prenext=False):
        curbars = []
        for i, d in enumerate(strat.datas):
            curbars.append((d._name, self._copy_lines(d)))

        oblines = []
        for obs in strat.getobservers():
            oblines.append((obs.__class__, self._copy_lines(obs)))

        self.nexts.append({'prenext': is_prenext, 'strategy': self._copy_lines(strat), 'datas': curbars, 'observers': oblines})

        _logger.debug(f"------------------- next")
        self.print_next(len(strat), self.nexts[-1])
        _logger.debug(f"------------------- next-end")

    def next(self):
        for s in self.strategy.env.runningstrats:
            minper = s._getminperstatus()
            if minper > 0:
                continue
            self._record_data(s)
