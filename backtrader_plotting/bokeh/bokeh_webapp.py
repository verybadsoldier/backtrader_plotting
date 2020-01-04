import sys

from bokeh.application.handlers.function import FunctionHandler
from bokeh.application import Application
from bokeh.document import Document
from bokeh.server.server import Server

from jinja2 import Environment, PackageLoader

from backtrader_plotting.bokeh import utils


class BokehWebapp:
    def __init__(self, title, html_template, scheme, model_fnc):
        self._title = title
        self._html_template = html_template
        self._scheme = scheme
        self._model_fnc = model_fnc

    def start(self, ioloop=None):
        """Serves a backtrader result as a Bokeh application running on a web server"""
        def make_document(doc: Document):
            doc.title = self._title

            env = Environment(loader=PackageLoader('backtrader_plotting.bokeh', 'templates'))
            doc.template = env.get_template(self._html_template)

            doc.template_variables['stylesheet'] = utils.generate_stylesheet(self._scheme)

            model = self._model_fnc()
            doc.add_root(model)

        self._run_server(make_document, ioloop=ioloop)

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

