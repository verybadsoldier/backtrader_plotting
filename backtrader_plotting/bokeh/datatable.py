import backtrader as bt
from collections import OrderedDict
from bokeh.models import ColumnDataSource, Paragraph, TableColumn, DataTable, DateFormatter, NumberFormatter, StringFormatter, Widget
from typing import List
from enum import Enum


class ColummDataType(Enum):
    DATETIME = 1
    FLOAT = 2
    INT = 3
    PERCENTAGE = 4
    STRING = 5


class TableGenerator(object):
    def __init__(self, scheme):
        self._scheme = scheme

    @staticmethod
    def _get_analyzer_table_generic(analyzer: bt.analyzers.Analyzer) -> List[List[object]]:
        """Returns two columns labeled 'Performance' and 'Value'"""
        table = [['Performance', ColummDataType.STRING], ['Value', ColummDataType.STRING]]

        def add_to_table(item: object, baselabel: str=""):
            for ak, av in item.items():
                label = f"{baselabel} - {ak}" if len(baselabel) > 0 else ak
                if isinstance(av, (bt.AutoOrderedDict, OrderedDict)):
                    add_to_table(av, label)
                else:
                    table[0].append(label)
                    table[1].append(av)

        add_to_table(analyzer.rets)
        return type(analyzer).__name__, [table]

    @staticmethod
    def _get_formatter(ctype: ColummDataType):
        if ctype == ColummDataType.FLOAT:
            return NumberFormatter(format="0.000")
        elif ctype == ColummDataType.INT:
            return NumberFormatter()
        elif ctype == ColummDataType.DATETIME:
            return DateFormatter(format="%c")
        elif ctype == ColummDataType.STRING:
            return StringFormatter()
        elif ctype == ColummDataType.PERCENTAGE:
            return NumberFormatter(format="0.000 %")
        else:
            raise Exception(f"Unsupported ColumnDataType: '{ctype}'")

    def get_analyzers_tables(self, analyzer: bt.analyzers.Analyzer) -> (Paragraph, List[DataTable]):
        acls = type(analyzer)

        if hasattr(acls, 'get_rets_table'):
            title, table_columns_list = analyzer.get_rets_table()
        else:
            # Analyzer does not provide a table function. Use our generic one
            title, table_columns_list = TableGenerator._get_analyzer_table_generic(analyzer)

        elems: List[DataTable] = []
        for table_columns in table_columns_list:
            cds = ColumnDataSource()
            columns = []
            for i, c in enumerate(table_columns):
                col_name = f'col{i}'
                cds.add(c[2:], col_name)
                columns.append(TableColumn(field=col_name, title=c[0], formatter=TableGenerator._get_formatter(c[1])))
            column_height = len(table_columns[0]) * 25
            elems.append(DataTable(source=cds, columns=columns, width=self._scheme.table_width, height=column_height, row_headers=False))
        return Paragraph(text=title, width=200, style={'font-size': 'large'}), elems
