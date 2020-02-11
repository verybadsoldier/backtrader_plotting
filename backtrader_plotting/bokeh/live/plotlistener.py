import asyncio
from threading import Lock
from typing import Dict, Optional
import threading

from attr import dataclass

import backtrader as bt

from bokeh.io import curdoc
from bokeh.models import ColumnDataSource, DataRange1d, Range1d
from bokeh.document import Document

from backtrader_plotting.bokeh.bokeh_webapp import BokehWebapp
from backtrader_plotting.schemes import Blackly
from backtrader_plotting import Bokeh

import pandas

import tornado.ioloop


class PlotListener(bt.ListenerBase):
    params = (
        ('scheme', Blackly()),
        ('style', 'bar'),
        ('lookback', 23),
        ('strategyidx', 0)
    )

    @dataclass
    class DocDataState:
        bokeh: Bokeh = None
        last_index: int = -1

    def __init__(self, **kwargs):
        self._cerebro: Optional[bt.Cerebro] = None
        self._webapp = BokehWebapp('Live', "basic.html.j2", self.p.scheme, self._build_root_model, on_session_destroyed=self._on_session_destroyed)
        self._lock = Lock()
        self._datastore = None
        self._docstates: Dict[Document, PlotListener.DocDataState] = {}
        self._bokeh_kwargs = kwargs
        self._bokeh = self._create_bokeh()

    def _create_bokeh(self):
        return Bokeh(style=self.p.style, **self._bokeh_kwargs)

    def _on_session_destroyed(self, session_context):
        with self._lock:
            doc = curdoc()
            del self._docstates[doc]

    def _build_root_model(self, doc: Document):
        bokeh = self._create_bokeh()
        bokeh.plot(self._cerebro.runningstrats[self.p.strategyidx], fill_data=False)
        model = bokeh.generate_model()

        with self._lock:
            self._docstates[doc] = PlotListener.DocDataState(bokeh=bokeh, last_index=-1)

        self._push_updates(doc)
        return model

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

    def _push_updates(self, bootstrap_document=None):
        if bootstrap_document is None:
            document = curdoc()
        else:
            document = bootstrap_document

        with self._lock:
            state: PlotListener.DocDataState = self._docstates[document]
            last_index = state.last_index

            updatepkg_df: pandas.DataFrame = self._datastore[self._datastore['index'] > last_index]

            # skip if we don't have new data
            if updatepkg_df.shape[0] == 0:
                return

            state.last_index = updatepkg_df['index'].iloc[-1]  # update last index
            updatepkg = ColumnDataSource.from_df(updatepkg_df)

            cds: ColumnDataSource = state.bokeh.get_figurepage().cds
            sendpkg = {}
            for c in updatepkg.keys():
                if c in cds.data:
                    sendpkg[c] = updatepkg[c]

            cds.stream(sendpkg, self.p.lookback)

            i = 6
            i += 3

    def next(self, doc=None):
        with self._lock:
            nextidx = 0 if self._datastore.shape[0] == 0 else int(self._datastore['index'].iloc[-1]) + 1
        new_frame = self._bokeh.build_strategy_data(self._cerebro.runningstrats[self.p.strategyidx], num_back=1, startidx=nextidx)

        with self._lock:
            self._datastore = self._datastore.append(new_frame)
            self._datastore = self._datastore.tail(self.p.lookback)

            for doc in self._docstates.keys() if doc is None else [doc]:
                try:
                    doc.remove_next_tick_callback(self._push_updates)
                except ValueError:
                    # there was no callback to remove
                    pass
                doc.add_next_tick_callback(self._push_updates)
