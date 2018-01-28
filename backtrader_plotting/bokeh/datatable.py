import backtrader as bt
from collections import OrderedDict
from bokeh.models import ColumnDataSource, Paragraph, TableColumn, DataTable


class TableGenerator(object):
    def __init__(self, scheme):
        self._scheme = scheme

    def get_analyzers_tables(self, name, analyzer):
        acls = type(analyzer)
        hrlables = getattr(acls, 'human_labels', {})
        cds = ColumnDataSource()
        labels = []
        values = []

        def add_to_table(item, baselabel=""):
            for ak, av in item.items():
                if ak in hrlables:
                    ak = hrlables[ak]
                label = f"{baselabel} - {ak}" if len(baselabel) > 0 else ak
                if isinstance(av, (bt.AutoOrderedDict, OrderedDict)):
                    add_to_table(av, label)
                else:
                    labels.append(label)
                    values.append(av)

        title = Paragraph(text=name, width=200, style={'font-size': 'large'})
        add_to_table(analyzer.rets)

        cds.add(labels, 'labels')
        cds.add(values, 'features')

        columns = [
            TableColumn(field="labels", title="Performance"),
            TableColumn(field="features", title="Value"),
        ]
        return [title, DataTable(source=cds, columns=columns, width=self._scheme.table_width, height=self._scheme.table_height, row_headers=False)]
