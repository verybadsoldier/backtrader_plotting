import backtrader as bt
from collections import OrderedDict
from bokeh.models import ColumnDataSource, Paragraph, TableColumn, DataTable


def get_analyzers_tables(analyzer):
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
    cur_row.append([title, DataTable(source=cds, columns=columns, width=self.p.scheme.table_width, height=self.p.scheme.table_height, row_headers=False)])