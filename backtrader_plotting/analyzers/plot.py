import asyncio
from collections import defaultdict
import logging
from threading import Lock
from typing import Dict
import threading

import numpy as np

import backtrader as bt

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
from bokeh.document import Document

from ..app import BacktraderPlotting
from ..webapp import Webapp
from ..schemes import Blackly, Scheme
from ..liveclient import LiveClient

import pandas

import tornado.ioloop

_logger = logging.getLogger(__name__)


class LivePlot(bt.Analyzer):
    params = (
        ('scheme', Blackly()),
        ('style', 'bar'),
        ('lookback', 23),
        ('http_port', 80),
        ('title', None),
    )

    def __init__(self, **kwargs):
        title = self.p.title
        if title is None:
            title = "Live %s" % type(self.strategy).__name__
        self._webapp = Webapp(
            title,
            'basic.html.j2',
            self.p.scheme,
            self._app_cb_build_root_model,
            on_session_destroyed=self._on_session_destroyed,
            port=self.p.http_port)
        self._lock = Lock()
        self._datastore = None
        self._clients: Dict[str, LiveClient] = {}
        self._app_kwargs = kwargs
        self._app = self._create_app()
        self._patch_pkgs = defaultdict(lambda: [])
        self._lastlen = 0

    def _create_app(self):
        return BacktraderPlotting(
            style=self.p.style,
            scheme=self.p.scheme,
            **self._app_kwargs)

    def _on_session_destroyed(self, session_context):
        with self._lock:
            del self._clients[session_context.id]

    def start(self):
        _logger.debug("Starting PlotListener...")

        self._datastore = self._app.build_strategy_data(self.strategy)
        t = threading.Thread(target=self._t_server)
        t.daemon = True
        t.start()

    def stop(self):
        pass

    def _t_server(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = tornado.ioloop.IOLoop.current()
        self._webapp.start(loop)

    def _app_cb_build_root_model(self, doc: Document):
        client = LiveClient(doc,
                            self._app_cb_push_adds,
                            self._create_app,
                            self._app_cb_push_adds,
                            self.strategy,
                            lookback=self.p.lookback)
        with self._lock:
            self._clients[doc.session_context.id] = client
        self._app_cb_push_adds(doc)
        return client.model

    def _app_cb_push_adds(self, bootstrap_document=None):
        if bootstrap_document is None:
            document = curdoc()
        else:
            document = bootstrap_document
        with self._lock:
            client = self._clients[document.session_context.id]
            updatepkg_df: pandas.DataFrame = self._datastore[self._datastore['index'] > client.last_data_index]
            # skip if we don't have new data
            if updatepkg_df.shape[0] == 0:
                return
            updatepkg = ColumnDataSource.from_df(updatepkg_df)
            client.push_adds(updatepkg, new_last_index=updatepkg_df['index'].iloc[-1])

    def _app_cb_push_patches(self):
        document = curdoc()
        session_id = document.session_context.id
        with self._lock:
            client: LiveClient = self._clients[session_id]
            patch_pkgs = self._patch_pkgs[session_id]
            self._patch_pkgs[session_id] = []
            client.push_patches(patch_pkgs)

    def next(self):
        # treat as update of old data if strategy datetime is duplicated and we have already data stored
        is_update = self._datastore.shape[0] > 0 and len(self.strategy.datas[0]) == self._lastlen
        self._lastlen = len(self.strategy.datas[0])

        if is_update:
            with self._lock:
                fulldata = self._app.build_strategy_data(self.strategy, num_back=1)

                # assuming, that the last entry is being patched
                index = self._datastore['index'].iloc[-1]
                odt = self._datastore['datetime'].iloc[-1]
                dt = fulldata['datetime'][0]

                for c in fulldata:
                    if c in ['datetime', 'index']:
                        continue
                    od = self._datastore[c].iloc[-1]
                    d = fulldata[c][0]
                    if d == d and d != od:
                        self._datastore.at[index, c] = d  # update data in datastore
                        for sess_id in self._clients.keys():
                            self._patch_pkgs[sess_id].append((c, odt, d))
                self._datastore.at[index, 'datetime'] = dt
                for sess_id in self._clients.keys():
                    self._patch_pkgs[sess_id].append(('datetime', odt, dt))
                for client in self._clients.values():
                    client.document.add_next_tick_callback(self._app_cb_push_patches)
        else:
            with self._lock:
                nextidx = 0 if self._datastore.shape[0] == 0 else int(self._datastore['index'].iloc[-1]) + 1

                num_back = 1 if self._datastore.shape[0] > 0 else None  # fetch all on first call
                new_frame = self._app.build_strategy_data(self.strategy, num_back=num_back, startidx=nextidx)

                # i have seen an empty line in the past. let's catch it here
                assert new_frame['datetime'].iloc[0] != np.datetime64('NaT')

                # append data and remove old data
                self._datastore = self._datastore.append(new_frame)
                self._datastore = self._datastore.tail(self.p.lookback)

                for client in self._clients.values():
                    doc = client.document
                    try:
                        doc.remove_next_tick_callback(self._app_cb_push_adds)
                    except ValueError:
                        # there was no callback to remove
                        pass
                    doc.add_next_tick_callback(self._app_cb_push_adds)
