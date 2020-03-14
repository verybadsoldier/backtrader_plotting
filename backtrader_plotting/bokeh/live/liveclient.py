from collections import defaultdict
from copy import copy
import logging

import backtrader as bt

from bokeh.models.widgets import Panel, Tabs, Slider, Button
from bokeh.layouts import column, gridplot, row
from bokeh.io import curdoc
from bokeh.models.widgets import CheckboxGroup, Div, Select
from bokeh.document import Document

from backtrader_plotting.html import metadata
from backtrader_plotting.bokeh.bokeh import FigurePage

_logger = logging.getLogger(__name__)


class LiveClient:
    def __init__(self, doc: Document, push_fnc, bokeh_fac: callable, push_data_fnc:callable, strategy: bt.Strategy, figurepage_idx: int = 0, lookback: int = 20):
        self._slider_aspectratio = None
        self._push_data_fnc = push_data_fnc
        self._push_fnc = push_fnc
        self._figurepage_idx = figurepage_idx
        self.last_data_index = -1
        self._lookback = lookback
        self._strategy = strategy
        self._current_group = None
        self.document = doc

        self._bokeh_fac = bokeh_fac
        self._bokeh = None

        bokeh = self._bokeh_fac()  # temporary bokeh object to get tradingdomains and scheme
        self._scheme = copy(bokeh.p.scheme)  # preserve original scheme as originally provided by the user

        tradingdomains = bokeh.list_tradingdomains(strategy)
        self._current_group = tradingdomains[0]
        self._select_tradingdomain = Select(value=self._current_group, options=tradingdomains)
        self._select_tradingdomain.on_change('value', self._on_select_group)

        btn_refresh_analyzers = Button(label='Refresh Analyzers', width=100)
        btn_refresh_analyzers.on_click(self._on_click_refresh_analyzers)

        td_label = Div(text="Trading Domain:", margin=(9, 5, 15, 5))
        controls = row(children=[td_label, self._select_tradingdomain, btn_refresh_analyzers])
        self.model = column(children=[controls, Tabs(tabs=[])], sizing_mode=self._scheme.plot_sizing_mode)

        # append meta tab
        meta = Div(text=metadata.get_metadata_div(strategy))
        self._panel_metadata = Panel(child=meta, title="Meta")

        self._refreshmodel()

    def _refreshmodel(self):
        self._bokeh = self._bokeh_fac()
        self._bokeh.p.scheme = self._scheme  # replace the original scheme with a possibly user customized scheme

        self._bokeh.plot(self._strategy, tradingdomain=self._current_group, fill_data=False)

        self._figurepage: FigurePage = self._bokeh.figurepages[self._figurepage_idx]

        panels = self._bokeh.generate_model_tabs(self._figurepage)

        # now append analyzer tab(s)
        analyzers = self._figurepage.analyzers
        panel_analyzer = self._bokeh.get_analyzer_panel(analyzers)
        if panel_analyzer is not None:
            panels.append(panel_analyzer)

        panels.append(self._panel_metadata)

        # append config panel
        panels.append(self._get_config_panel())
        self.model.children[1].tabs = panels

        self.last_data_index = -1

    def _on_click_refresh_analyzers(self):
        panel = self._bokeh.get_analyzer_panel(self._figurepage.analyzers)
        self.model.children[1].tabs[1] = panel

    def on_button_save_config(self):
        self._scheme.plotaspectratio = self._slider_aspectratio.value
        for f in self._bokeh.figurepages[0].figure_envs:
            f.figure.aspect_ratio = self._slider_aspectratio.value

    def _get_config_panel(self):
        def on_change_checkbox(vals):
            for i, f in enumerate(self._bokeh.figurepages[0].figure_envs):
                if i > 1:
                    continue
                f.figure.visible = i in vals

        self._slider_aspectratio = Slider(value=self._scheme.plotaspectratio, start=0.1, end=10.0, step=0.1)

        button = Button(label="Save", button_type="success")
        button.on_click(self.on_button_save_config)

        r1 = row(children=[Div(text='Aspect Ratio', margin=(15, 10, 0, 10)), self._slider_aspectratio])

        return Panel(child=column(children=[r1, button]), title='Config')

    def _on_select_group(self, a, old, new):
        _logger.info(f"Switching logic group to {new}...")
        self._current_group = new
        doc = curdoc()
        doc.hold()
        self._refreshmodel()
        doc.unhold()

        self._push_data_fnc(doc)

        _logger.info(f"Switching logic group finished")

    def push_patches(self, patch_pkgs):
        cds = self._figurepage.cds

        dt_idx_map = {d: idx for idx, d in enumerate(cds.data['datetime'])}

        patch_dict = defaultdict(list)
        for pp in patch_pkgs:
            colname, dt, val = pp
            if colname not in cds.data:
                continue
            idx = dt_idx_map[dt.to_datetime64()]
            patch_dict[colname].append((idx, val))
        _logger.info(f"Sending patch dict: {patch_dict}")

        cds.patch(patch_dict)

    def push_adds(self, updatepkg: dict, new_last_index: int):
        self.last_data_index = new_last_index

        cds = self._figurepage.cds

        sendpkg = {}
        for c in updatepkg.keys():
            if c in cds.data:
                sendpkg[c] = updatepkg[c]

        _logger.info(f'Sending stream package: {sendpkg}')
        cds.stream(sendpkg, self._lookback)
