import inspect

import markdown2

import backtrader as bt

from backtrader_plotting.utils import get_params, paramval2str


def _get_parameter_table(params) -> str:
    params = get_params(params)

    if len(params) == 0:
        params[''] = ''
    else:
        for k, v in params.items():
            params[k] = paramval2str(k, v)

    return _get_table(['Parameter', 'Value'], params)


def _get_table(header, data) -> str:
    table = f'|{header[0]}|{header[1]}|\n'
    table += f'|--|--|\n'
    for k, v in data.items():
        table += f'|{k}|{v}|\n'
    return table


def _get_datas(cerbebro: bt.cerebro) -> str:
    md = '\n# Data Feeds\n'

    for data in cerbebro.datas:
        md += f'## {data.__class__.__name__}\n'

        data = {
            'DataName:': str(data._dataname).replace("|", "\\|"),
            'Timezone:': data._tz,
            'Time From:': bt.num2date(data.fromdate),
            'Time To:': bt.num2date(data.todate),
            'Number of bars:': len(data),
            'Bar Length:': f"{data._compression} {bt.TimeFrame.getname(data._timeframe, data._compression)}",
        }
        md += _get_table(['Property', 'Value'], data)

    return md


def _get_strategies(cerbebro: bt.cerebro) -> str:
    md = '\n# Strategies\n'

    for s in cerbebro.runningstrats:
        md += f'## {s.__class__.__name__}\n'
        md += _get_parameter_table(s.params)

        md += '### Indicators:\n\n'
        for i in s.getindicators():
            md += f'{i.__class__.__name__}\n\n'
            md += _get_parameter_table(i.params)

        md += 'Source Code:\n'
        md += f'\n```\n{inspect.getsource(s.__class__)}\n```\n\n'

    return md


def _get_analyzers(cerbebro: bt.cerebro) -> str:
    md = '\n# Analyzers\n'

    strat = cerbebro.runningstrats[0]

    for a in strat.analyzers:
        md += f'## {a.__class__.__name__}\n'
        md += _get_parameter_table(a.params)

    return md


def _get_observers(cerbebro: bt.cerebro) -> str:
    md = '\n# Observers\n'

    strat = cerbebro.runningstrats[0]

    for o in strat.observers:
        md += f'## {o.__class__.__name__}\n'
        md += _get_parameter_table(o.params)

    return md


def get_metadata_div(cerebro: bt.cerebro) -> str:
    md = ""

    md += _get_datas(cerebro)
    md += '* * *'
    md += _get_analyzers(cerebro)
    md += '* * *'
    md += _get_observers(cerebro)
    md += '* * *'
    md += _get_strategies(cerebro)
    md += '* * *'

    css_classes = {'table': 'metaDataTable'}

    html = markdown2.markdown(md, extras={
        'fenced-code-blocks': None,
        'tables': None,
        'html-classes': css_classes
    })
    return html
