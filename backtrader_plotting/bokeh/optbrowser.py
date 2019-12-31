import inspect
import sys
from typing import List, Dict, Callable, Union

import backtrader as bt

from bokeh.models import ColumnDataSource, Model
from bokeh.layouts import column
from bokeh.server.server import Server
from bokeh.document import Document
from bokeh.application import Application
from bokeh.application.handlers.function import FunctionHandler
from bokeh.models.widgets import DataTable, TableColumn, NumberFormatter, StringFormatter, Div

from backtrader_plotting import bttypes
from backtrader_plotting import Bokeh
from backtrader_plotting.bttypes import is_ordered_optresult
from backtrader_plotting.bokeh import utils

from jinja2 import Environment, PackageLoader

from .utils import get_limited_optresult


class OptBrowser:
    def __init__(self, bokeh: Bokeh, optresults: bttypes.OptResult, usercolumns: Dict[str, Callable] = {}, num_result_limit=None):
        self._usercolumns = usercolumns
        self._num_result_limit = num_result_limit
        self._optresults = optresults
        self._bokeh: Bokeh = bokeh

    def start(self, ioloop=None):
        """Serves an optimization resulst as a Bokeh application running on a web server"""
        def make_document(doc: Document):
            doc.title = "Backtrader Optimization Result"

            env = Environment(loader=PackageLoader('backtrader_plotting.bokeh', 'templates'))
            doc.template = env.get_template("basic.html.j2")

            doc.template_variables['stylesheet'] = utils.generate_stylesheet(self._bokeh.params.scheme)

            model = self._build_optresult_model()
            doc.add_root(model)

        self._run_server(make_document, ioloop=ioloop)

    def _build_optresult_selector(self, optresults) -> DataTable:
        cds = ColumnDataSource()
        tab_columns = []

        col_formatter_num = NumberFormatter(format='0.000')
        col_formatter_str = StringFormatter()

        # extract the actual optresult if it is wrapped into orderedresults
        plain_result = optresults if bttypes.is_optresult(optresults) else [x.optresult for x in optresults.benchmarked_results]
        if bttypes.is_ordered_optresult(optresults):
            benchmarks = [x.benchmark_value for x in get_limited_optresult(optresults.benchmarked_results, self._num_result_limit)]
            cds.add(benchmarks, "benchmark")
            tab_columns.append(TableColumn(field='benchmark', title=optresults.benchmark_label, sortable=False, formatter=col_formatter_num))

        # take the first result as example and take the first strategy (we don't support more)
        optresult = plain_result[0][0]

        # iterate all parameters of this strategy
        for name, val in optresult.params._getitems():
            # get value for the current param for all results
            param_vals = []
            formatter = col_formatter_num
            for opt in get_limited_optresult(plain_result, self._num_result_limit):
                param = opt[0].params._get(name)
                if inspect.isclass(param):
                    paramstr = param.__name__
                else:
                    paramstr = param
                param_vals.append(paramstr)

            if len(param_vals) > 0 and isinstance(param_vals[0], str):
                formatter = col_formatter_str
            tab_columns.append(TableColumn(field=name, title=f'{name}', sortable=False, formatter=formatter))

            cds.add(param_vals, f"{name}")

        for k, v in self._usercolumns.items():
            ll = [str(v(x)) for x in get_limited_optresult(self._optresults, self._num_result_limit)]
            cds.add(ll, k)
            tab_columns.append(TableColumn(field=k, title=k, sortable=False, formatter=col_formatter_str))

        # TODO: currently table size is hardcoded
        selector = DataTable(source=cds, columns=tab_columns, width=1600, height=150)
        return selector, cds

    def _build_optresult_model(self) -> Model:
        """Generates and returns an interactive model for an OptResult or an OrderedOptResult"""

        # we have list of results, each result contains the result for one strategy. we don't support having more than one strategy!
        if is_ordered_optresult(self._optresults) and self._optresults.num_strategies > 1:
            raise RuntimeError("You passed an optimization result based on more than one strategy which is not supported!")
        elif not is_ordered_optresult(self._optresults) and len(self._optresults) > 0 and len(self._optresults[0]) > 1:
            raise RuntimeError("You passed on optimization result based on more than one strategy which is not supported!")

        selector, selector_cds = self._build_optresult_selector(self._optresults)

        def _get_model(idx: int):
            return self._bokeh.plot_and_generate_optmodel(self._optresults[idx][0])

        #  first zero is because we show the first opt result by default and second zero cause we support only 1 strategy
        model = column([selector, _get_model(0)])
        model.background = self._bokeh.params.scheme.background_fill

        def update(_name, _old, new):
            if len(new) == 0:
                return

            stratidx = new[0]
            model.children[-1] = _get_model(stratidx)

        selector_cds.selected.on_change('indices', update)

        return model

    @staticmethod
    def _run_server(fnc_make_document, iplot=True, notebook_url="localhost:8889", port=80, ioloop=None):
        """Runs a Bokeh webserver application. Documents will be created using fnc_make_document"""
        handler = FunctionHandler(fnc_make_document)
        app = Application(handler)
        if iplot and 'ipykernel' in sys.modules:
            show(app, notebook_url=notebook_url)
        else:
            apps = {'/': app}

            print("Open your browser here: http://localhost")
            server = Server(apps, port=port, io_loop=ioloop)
            server.run_until_shutdown()
