import plotly as py
from plotly import tools
import plotly.graph_objs as go

import backtrader as bt


class Plotly(object):
    def __init__(self):
        self._traces = []

    def plot(self, strategy, figid, numfigs, iplot, start, end, use):
        if not strategy.datas:
            return

        if not len(strategy):
            return

        for data in strategy.datas:
            if not data.plotinfo.plot:
                continue

            o = data.open.plotrange(start, end)
            h = data.high.plotrange(start, end)
            l = data.low.plotrange(start, end)
            c = data.close.plotrange(start, end)

            dt = [bt.num2date(x) for x in strategy.data.lines.datetime.plotrange(start, end)]

            self._traces.append(go.Candlestick(x=dt,
                                               open=o,
                                               high=h,
                                               low=l,
                                               close=c,
                                               name=data._name))

        next_plot = 1
        for x in strategy.getindicators():
            if not hasattr(x, 'plotinfo'):
                # no plotting support - so far LineSingle derived classes
                continue

            if not x.plotinfo.plot or x.plotinfo.plotskip:
                continue

            if x.plotinfo.subplot:
                next_plot += 1

            for lineidx in range(x.size()):
                line = x.lines[lineidx]
                linealias = x.lines._getlinealias(lineidx)

                ind_trace = go.Scatter(x=dt,
                                       y=line.plotrange(start, end),
                                       name=linealias,
                                       )

                if x.plotinfo.subplot:
                    ind_trace['xaxis'] = f'x{next_plot}'
                    ind_trace['yaxis'] = f'y{next_plot}'
                #self._traces.append(ind_trace)



    def show(self):
        num_subplots = len(self._traces)
        fig = tools.make_subplots(rows=num_subplots, cols=1)

        layout = go.Layout(
        )
        figure = go.Figure(data=self._traces, layout=layout)

        config = {'scrollZoom': True}
        py.offline.plot(figure, config=config, show_link=False)

    def savefig(self, fig, filename, width, height, dpi, tight):
        pass
