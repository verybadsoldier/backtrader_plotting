import asyncio
from collections import defaultdict
import logging
from threading import Lock
from typing import Dict, Optional
import threading

import numpy as np

import backtrader as bt

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource
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
        ('strategyidx', 0),
        ('http_port', 80),
        ('title', 'Live'),
    )

    def __init__(self, **kwargs):
        self._cerebro: Optional[bt.Cerebro] = None
        self._webapp = BokehWebapp(self.p.title,
                                   'basic.html.j2',
                                   self.p.scheme,
                                   self._bokeh_cb_build_root_model,
                                   on_session_destroyed=self._on_session_destroyed,
                                   port=self.p.http_port)
        self._lock = Lock()
        self._datastore = None
        self._clients: Dict[str, LiveClient] = {}
        self._bokeh_kwargs = kwargs
        self._bokeh = self._create_bokeh()
        self._patch_pkgs = defaultdict(lambda: [])

    def _create_bokeh(self):
        return Bokeh(style=self.p.style, scheme=self.p.scheme, **self._bokeh_kwargs)  # get a copy of the scheme so we can modify it per client

    def _on_session_destroyed(self, session_context):
        with self._lock:
            del self._clients[session_context.id]

    def _bokeh_cb_build_root_model(self, doc: Document):
        client = LiveClient(doc,
                            self._bokeh_cb_push_adds,
                            self._create_bokeh,
                            self._bokeh_cb_push_adds,
                            self._cerebro.runningstrats[self.p.strategyidx],
                            lookback=self.p.lookback)

        with self._lock:
            self._clients[doc.session_context.id] = client

        self._bokeh_cb_push_adds(doc)

        return client.model

    def start(self, cerebro):
        _logger.info("Starting PlotListener...")

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

    def _bokeh_cb_push_adds(self, bootstrap_document=None):
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

    def _bokeh_cb_push_patches(self):
        document = curdoc()
        session_id = document.session_context.id
        with self._lock:
            client: LiveClient = self._clients[session_id]

            patch_pkgs = self._patch_pkgs[session_id]
            self._patch_pkgs[session_id] = []
            client.push_patches(patch_pkgs)

    def next(self):
        strategy = self._cerebro.runningstrats[self.p.strategyidx]

        # treat as update of old data if strategy datetime is duplicated and we have already data stored
        is_update = len(strategy) > 1 and strategy.datetime[0] == strategy.datetime[-1] and self._datastore.shape[0] > 0

        if is_update:
            with self._lock:
                fulldata = self._bokeh.build_strategy_data(strategy)

                # generate series with number of missing values per column
                new_count = fulldata.isnull().sum()
                cur_count = self._datastore.isnull().sum()

                # boolean series that indicates which column is missing data
                patched_cols = new_count != cur_count

                # get dataframe with only those columns that added data
                patchcols = fulldata[fulldata.columns[patched_cols]]
                for column_name in patchcols.columns:
                    for index, row in self._datastore.iterrows():
                        # compare all values in this column
                        od = row[column_name]
                        odt = row['datetime']
                        d = fulldata[column_name][index]
                        dt = fulldata['datetime'][index]

                        assert odt == dt

                        # if value is different then put to patch package
                        # either it WAS NaN and it's not anymore
                        # or both not NaN but different now
                        # and don't count it as True when both are NaN
                        if not (pandas.isna(d) and pandas.isna(od)) and ((pandas.isna(od) and not pandas.isna(d)) or d != od):
                            self._datastore.at[index, column_name] = d  # update data in datastore
                            for sess_id in self._clients.keys():
                                self._patch_pkgs[sess_id].append((column_name, odt, d))

                for client in self._clients.values():
                    client.document.add_next_tick_callback(self._bokeh_cb_push_patches)
        else:
            with self._lock:
                nextidx = 0 if self._datastore.shape[0] == 0 else int(self._datastore['index'].iloc[-1]) + 1

                num_back = 1 if self._datastore.shape[0] > 0 else None  # fetch all on first call
                new_frame = self._bokeh.build_strategy_data(strategy, num_back=num_back, startidx=nextidx)

                # i have seen an empty line in the past. let's catch it here
                assert new_frame['datetime'].iloc[0] != np.datetime64('NaT')

                # append data and remove old data
                self._datastore = self._datastore.append(new_frame)
                self._datastore = self._datastore.tail(self.p.lookback)

                for client in self._clients.values():
                    doc = client.document
                    try:
                        doc.remove_next_tick_callback(self._bokeh_cb_push_adds)
                    except ValueError:
                        # there was no callback to remove
                        pass
                    doc.add_next_tick_callback(self._bokeh_cb_push_adds)
