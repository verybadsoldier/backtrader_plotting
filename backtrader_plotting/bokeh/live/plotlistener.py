import asyncio
import logging
from collections import defaultdict
from threading import Lock
from typing import Dict, Optional
import numpy as np
import threading

from attr import dataclass

import backtrader as bt

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, DataRange1d, Range1d
from bokeh.document import Document

from backtrader_plotting.bokeh.bokeh_webapp import BokehWebapp
from backtrader_plotting.schemes import Blackly
from backtrader_plotting import Bokeh
from backtrader_plotting.bokeh.live.liveclient import LiveClient

import pandas

import tornado.ioloop

_logger = logging.getLogger(__name__)


class PlotListener(bt.ListenerBase):
    params = (
        ('scheme', Blackly()),
        ('style', 'bar'),
        ('lookback', 23),
        ('strategyidx', 0)
    )

    def __init__(self, **kwargs):
        self._cerebro: Optional[bt.Cerebro] = None
        self._webapp = BokehWebapp('Live', "basic.html.j2", self.p.scheme, self._build_root_model, on_session_destroyed=self._on_session_destroyed)
        self._lock = Lock()
        self._datastore = None
        self._clients: Dict[Document, LiveClient] = {}
        self._bokeh_kwargs = kwargs
        self._bokeh = self._create_bokeh()
        self._patch_pkgs = {}

    def _create_bokeh(self):
        return Bokeh(style=self.p.style, **self._bokeh_kwargs)

    def _on_session_destroyed(self, session_context):
        with self._lock:
            doc = curdoc()
            del self._clients[doc]

    def _build_root_model(self, doc: Document):
        client = LiveClient(self._push_adds, self._create_bokeh, self._cerebro.runningstrats[self.p.strategyidx], lookback=self.p.lookback)
        with self._lock:
            self._clients[doc] = client

        self._push_adds(doc)

        return client.model

    def start(self, cerebro):
        self._cerebro = cerebro

        self._datastore = self._bokeh.build_strategy_data(self._cerebro.runningstrats[self.p.strategyidx])

        t = threading.Thread(target=self._t_bokeh_server)
        t.daemon = True
        t.start()

    def stop(self):
        pass

    def _t_bokeh_server(self):
        asyncio.set_event_loop(asyncio.new_event_loop())
        loop = tornado.ioloop.IOLoop.current()
        self._webapp.start(loop)

    def _push_adds(self, bootstrap_document=None):
        if bootstrap_document is None:
            document = curdoc()
        else:
            document = bootstrap_document

        with self._lock:
            client = self._clients[document]
            updatepkg_df: pandas.DataFrame = self._datastore[self._datastore['index'] > client.last_index]

            # skip if we don't have new data
            if updatepkg_df.shape[0] == 0:
                return

            updatepkg = ColumnDataSource.from_df(updatepkg_df)

            client.push_adds(updatepkg, last_index=updatepkg_df['index'].iloc[-1])

    def _push_patches(self):
        document = curdoc()
        with self._lock:
            client: LiveClient = self._clients[document]

            patch_pkgs = self._patch_pkgs[document]
            self._patch_pkgs[document] = []
            client.push_patches(patch_pkgs)

    def next(self, doc=None):
        strategy = self._cerebro.runningstrats[self.p.strategyidx]

        # treat as update of old data if strategy datetime is duplicated and we have already data stored
        is_update = len(strategy) > 1 and strategy.datetime[0] == strategy.datetime[-1] and self._datastore.shape[0] > 0

        if is_update:
            with self._lock:
                fulldata = self._bokeh.build_strategy_data(strategy)

                new_count = fulldata.isnull().sum()
                cur_count = self._datastore.isnull().sum()

                patched_cols = new_count != cur_count

                patchcols = fulldata[fulldata.columns[patched_cols]]
                for columnName, columnData in patchcols.iteritems():
                    # compare all values in this column
                    for i, d in enumerate(columnData):
                        od = self._datastore[columnName].iloc[i]
                        # if value is different then put to patch package
                        # either it WAS NaN and it's not anymore
                        # or both not NaN but different now
                        # and don't could it as True when both are NaN
                        if not (pandas.isna(d) and pandas.isna(od)) and ((pandas.isna(od) and not pandas.isna(d)) or d != od):
                            self._datastore[columnName].iloc[i] = d  # update data in datastore
                            for doc in self._clients.keys():
                                if doc not in self._patch_pkgs:
                                    self._patch_pkgs[doc] = []
                                self._patch_pkgs[doc].append((columnName, self._datastore['datetime'].iloc[i].to_datetime64(), fulldata[columnName][i]))

                for doc in self._clients.keys():
                    doc.add_next_tick_callback(self._push_patches)
        else:
            with self._lock:
                nextidx = 0 if self._datastore.shape[0] == 0 else int(self._datastore['index'].iloc[-1]) + 1

            num_back = 1 if self._datastore.shape[0] > 0 else None  # fetch all on first call
            new_frame = self._bokeh.build_strategy_data(strategy, num_back=num_back, startidx=nextidx)

            with self._lock:
                self._datastore = self._datastore.append(new_frame)
                self._datastore = self._datastore.tail(self.p.lookback)

                for doc in self._clients.keys() if doc is None else [doc]:
                    try:
                        doc.remove_next_tick_callback(self._push_adds)
                    except ValueError:
                        # there was no callback to remove
                        pass
                    doc.add_next_tick_callback(self._push_adds)
