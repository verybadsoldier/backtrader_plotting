import backtrader.analyzers
import logging
from .bokeh.datatable import ColummDataType

_logger = logging.getLogger(__name__)


def _get_rets_table_sharperatio(self):
    cols = [['Name', ColummDataType.STRING], ['Value', ColummDataType.FLOAT]]
    cols[0].append('Sharpe-Ratio')
    cols[1].append(self.rets['sharperatio'])
    return "Sharpe-Ratio", [cols]


def _get_rets_table_drawdown(self):
    cols1 = [['Feature', ColummDataType.STRING], ['Value', ColummDataType.FLOAT], ['Maximum', ColummDataType.FLOAT]]

    cols1[0].append('Length')
    cols1[1].append(self.rets['len'])
    cols1[2].append(self.rets['max']['len'])

    cols1[0].append('Moneydown')
    cols1[1].append(self.rets['moneydown'])
    cols1[2].append(self.rets['max']['moneydown'])

    cols1[0].append('Drawdown')
    cols1[1].append(self.rets['drawdown'])
    cols1[2].append(self.rets['max']['drawdown'])

    return "Drawdown", [cols1]


def _get_rets_table_transaction(self):
    cols = [['Time', ColummDataType.DATETIME],
            # ['Data ID', ColummDataType.INT],
            ['Size', ColummDataType.INT],
            ['Price', ColummDataType.FLOAT],
            ['Instrument', ColummDataType.STRING],
            ['Total Price', ColummDataType.FLOAT]
            ]

    # size, price, i, dname, -size * price
    for k, v in self.rets.items():
        cols[0].append(k)
        # cols[1].append(v[0][2])
        cols[1].append(v[0][0])
        cols[2].append(v[0][1])
        cols[3].append(v[0][3])
        cols[4].append(v[0][4])

    return "Transactions", [cols]


def _get_rets_table_tradeanalyzer(self):
    def gdef(obj, attr, d):
        return obj[attr] if attr in obj else d

    tables = []

    tab1 = [['', ColummDataType.STRING], ['Total', ColummDataType.INT], ['Open', ColummDataType.INT], ['Closed', ColummDataType.INT]]
    tab1[0].append('Number of Trades')
    tab1[1].append(self.rets['total']['total'])
    tab1[2].append(self.rets['total']['open'])
    tab1[3].append(gdef(self.rets['total'], 'closed', 0))
    tables.append(tab1)

    if 'streak' in self.rets and 'pnl' in self.rets:
        tab2 = [['Streak', ColummDataType.STRING], ['Current', ColummDataType.INT], ['Longest', ColummDataType.INT]]
        tab2[0].append('Won')
        tab2[1].append(self.rets['streak']['won']['current'])
        tab2[2].append(self.rets['streak']['won']['longest'])

        tab2[0].append('Lost')
        tab2[1].append(self.rets['streak']['lost']['current'])
        tab2[2].append(self.rets['streak']['lost']['longest'])

        tables.append(tab2)

        tab3 = [['Profit & Loss', ColummDataType.STRING], ['Total', ColummDataType.FLOAT], ['Average', ColummDataType.FLOAT]]
        tab3[0].append('Gross')
        tab3[1].append(self.rets['pnl']['gross']['total'])
        tab3[2].append(self.rets['pnl']['gross']['average'])

        tab3[0].append('Net')
        tab3[1].append(self.rets['pnl']['net']['total'])
        tab3[2].append(self.rets['pnl']['net']['average'])

        tab3[0].append('Short')
        tab3[1].append(self.rets['short']['pnl']['total'])
        tab3[2].append(self.rets['short']['pnl']['average'])

        tab3[0].append('Long')
        tab3[1].append(self.rets['long']['pnl']['total'])
        tab3[2].append(self.rets['long']['pnl']['average'])

        tab3[0].append('Won / Short')
        tab3[1].append(self.rets['short']['pnl']['won']['total'])
        tab3[2].append(self.rets['short']['pnl']['won']['average'])

        tab3[0].append('Lost / Short')
        tab3[1].append(self.rets['short']['pnl']['lost']['total'])
        tab3[2].append(self.rets['short']['pnl']['lost']['average'])

        tab3[0].append('Won / Long')
        tab3[1].append(self.rets['long']['pnl']['won']['total'])
        tab3[2].append(self.rets['long']['pnl']['won']['average'])

        tab3[0].append('Lost / Long')
        tab3[1].append(self.rets['long']['pnl']['lost']['total'])
        tab3[2].append(self.rets['long']['pnl']['lost']['average'])

        tables.append(tab3)

        tab4 = [['Long', ColummDataType.STRING], ['Gross', ColummDataType.FLOAT], ['Net', ColummDataType.FLOAT]]
        tab4[0].append('Longest')
        tab4[1].append(self.rets['streak']['won']['longest'])
        tab4[2].append(self.rets['streak']['lost']['longest'])
        tables.append(tab4)

        tab5 = [['Trades', ColummDataType.STRING], ['Total', ColummDataType.INT], ['Won', ColummDataType.INT], ['Lost', ColummDataType.INT]]
        tab5[0].append('Long')
        tab5[1].append(self.rets['long']['total'])
        tab5[2].append(self.rets['long']['won'])
        tab5[3].append(self.rets['long']['lost'])

        tab5[0].append('Short')
        tab5[1].append(self.rets['short']['total'])
        tab5[2].append(self.rets['short']['won'])
        tab5[3].append(self.rets['short']['lost'])

        tab5[0].append('All')
        tab5[1].append(self.rets['won']['total'] + self.rets['lost']['total'])
        tab5[2].append(self.rets['won']['total'])
        tab5[3].append(self.rets['lost']['total'])

        tables.append(tab5)

        tab_len = [['Trade Length', ColummDataType.STRING], ['Total', ColummDataType.INT], ['Min', ColummDataType.INT], ['Max', ColummDataType.INT], ['Average', ColummDataType.FLOAT]]
        tab_len[0].append('Won')
        tab_len[1].append(self.rets['len']['won']['total'])
        tab_len[2].append(self.rets['len']['won']['min'])
        tab_len[3].append(self.rets['len']['won']['max'])
        tab_len[4].append(self.rets['len']['won']['average'])

        tab_len[0].append('Lost')
        tab_len[1].append(self.rets['len']['lost']['total'])
        tab_len[2].append(99999999999999999)  # bug missing: self.rets['len']['lost']['min'])
        tab_len[3].append(self.rets['len']['lost']['max'])
        tab_len[4].append(self.rets['len']['lost']['average'])

        tab_len[0].append('Long')
        tab_len[1].append(self.rets['len']['long']['total'])
        tab_len[2].append(self.rets['len']['long']['min'])
        tab_len[3].append(self.rets['len']['long']['max'])
        tab_len[4].append(self.rets['len']['long']['average'])

        tab_len[0].append('Short')
        tab_len[1].append(self.rets['len']['short']['total'])
        tab_len[2].append(self.rets['len']['short']['min'])
        tab_len[3].append(self.rets['len']['short']['max'])
        tab_len[4].append(self.rets['len']['short']['average'])

        tab_len[0].append('Won / Long')
        tab_len[1].append(self.rets['len']['long']['won']['total'])
        tab_len[2].append(self.rets['len']['long']['won']['min'])
        tab_len[3].append(self.rets['len']['long']['won']['max'])
        tab_len[4].append(self.rets['len']['long']['won']['average'])

        tab_len[0].append('Won / Short')
        tab_len[1].append(self.rets['len']['short']['won']['total'])
        tab_len[2].append(self.rets['len']['short']['won']['min'])
        tab_len[3].append(self.rets['len']['short']['won']['max'])
        tab_len[4].append(self.rets['len']['short']['won']['average'])

        tab_len[0].append('Lost / Long')
        tab_len[1].append(self.rets['len']['long']['lost']['total'])
        tab_len[2].append(self.rets['len']['long']['lost']['min'])
        tab_len[3].append(self.rets['len']['long']['lost']['max'])
        tab_len[4].append(self.rets['len']['long']['lost']['average'])

        tab_len[0].append('Lost / Short')
        tab_len[1].append(self.rets['len']['short']['lost']['total'])
        tab_len[2].append(self.rets['len']['short']['lost']['min'])
        tab_len[3].append(self.rets['len']['short']['lost']['max'])
        tab_len[4].append(self.rets['len']['short']['lost']['average'])

        tables.append(tab_len)

    return "Transaction Analyzer", tables


def inject_humanreadable():
    """Injects function 'get_rets_table' to some well-known Analyzer classes."""
    _labels = {
        backtrader.analyzers.sharpe.SharpeRatio: _get_rets_table_sharperatio,
        backtrader.analyzers.DrawDown: _get_rets_table_drawdown,
        backtrader.analyzers.TradeAnalyzer: _get_rets_table_tradeanalyzer,
        backtrader.analyzers.Transactions: _get_rets_table_transaction,
    }

    for cls, labdict in _labels.items():
        curlab = getattr(cls, 'get_rets_table', None)
        if curlab is not None:
            _logger.warning(f"Analyzer class '{cls.__name__}' already contains a function 'get_rets_table'. Not overriding.")
            continue
        setattr(cls, 'get_rets_table', labdict)
