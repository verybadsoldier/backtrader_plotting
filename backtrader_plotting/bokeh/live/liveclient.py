from collections import defaultdict
import logging

import backtrader as bt

from bokeh.models.widgets import Panel, Tabs, Slider, Button
from bokeh.layouts import column, gridplot, row
from bokeh.io import curdoc
from bokeh.models.widgets import CheckboxGroup, Div, Select
from bokeh.models import GridBox, DataRange1d

from backtrader_plotting.html import metadata
from backtrader_plotting.bokeh.bokeh import FigurePage

_logger = logging.getLogger(__name__)


class LiveClient:
    def __init__(self, push_fnc, bokeh_fac: callable, push_data_fnc:callable, strategy: bt.Strategy, figurepage_idx: int = 0, lookback: int = 20):
        self._slider = None
        self._push_data_fnc = push_data_fnc
        self._push_fnc = push_fnc
        self._figurepage_idx = figurepage_idx
        self.last_index = -1
        self._lookback = lookback
        self._strategy = strategy
        self._current_group = None

        self._bokeh_fac = bokeh_fac
        self._bokeh = None

        bokeh = self._bokeh_fac()  # temporary bokeh object to get logicgroups and scheme
        logicgroups = bokeh.list_logicgroups(strategy)
        self._current_group = logicgroups[0]
        self._select_logicgroup = Select(value=self._current_group, options=logicgroups)
        self._select_logicgroup.on_change('value', self._on_select_group)

        btn_refresh_analyzers = Button(label='Refresh Analyzers')
        btn_refresh_analyzers.on_click(self._on_click_refresh_analyzers)

        controls = row(children=[self._select_logicgroup, btn_refresh_analyzers])
        self.model = column(children=[controls, Tabs(tabs=[])], sizing_mode=bokeh.p.scheme.plot_sizing_mode)

        # append meta tab
        meta = Div(text=metadata.get_metadata_div(strategy))
        self._panel_metadata = Panel(child=meta, title="Meta")

        self._refreshmodel()

    def _refreshmodel(self):
        self._bokeh = self._bokeh_fac()

        self._bokeh.plot(self._strategy, logicgroup=self._current_group, fill_data=False)

        self._figurepage: FigurePage = self._bokeh._figurepages[self._figurepage_idx]

        panels = self._bokeh._generate_model_tabs(self._figurepage)

        # now append analyzer tab(s)
        analyzers = self._figurepage.analyzers
        panel_analyzer = self._bokeh._get_analyzer_panel(analyzers)
        if panel_analyzer is not None:
            panels.append(panel_analyzer)

        panels.append(self._panel_metadata)

        # append config panel
        panels.append(self._get_config_panel())
        self.model.children[1].tabs = panels

        self.last_index = -1

    def _on_click_refresh_analyzers(self):
        panel = self._bokeh._get_analyzer_panel(self._figurepage.analyzers)
        self.model.children[1].tabs[1] = panel

    def on_commitbtn(self):
        for f in self._bokeh._figurepages[0].figure_envs:
            f.figure.aspect_ratio = self._slider.value

        c1 = self._model.tabs[0].child.children[0].children[0]
        c2 = self._model.tabs[0].child.children[0].children[1]

#        self._model.tabs[0].child.children[0].children[0] = c2
#        self._model.tabs[0].child.children[0].children[1] = c1

        self._model[1].tabs[0].child.children[0] = gridplot([[c1], [c2]],
                                                         toolbar_options={'logo': None},
                                                         toolbar_location=self.p.scheme.toolbar_location,
                                                         sizing_mode=self.p.scheme.plot_sizing_mode)

    def _get_config_panel(self):
        def on_change_checkbox(vals):
            for i, f in enumerate(self._bokeh._figurepages[0].figure_envs):
                if i > 1:
                    continue
                f.figure.visible = i in vals

        names = []
        for d in self._bokeh._figurepages[0].strategy.datas:
            names.append(d._name)
        checkbox_group = CheckboxGroup(labels=names, active=[1 for _ in range(len(names))])
        checkbox_group.on_click(on_change_checkbox)

        self._slider = Slider(value=3.0, start=0.1, end=10.0, step=0.1)
        #rs.on_click(on_change_checkbox)

        button = Button(label="Save", button_type="success")
        button.on_click(self.on_commitbtn)

        childs = gridplot([[checkbox_group, self._slider, button]], toolbar_options={'logo': None})
        return Panel(child=childs, title='Config')

    def _update_model(self, logicgroup):
        pass

    def _on_select_group(self, a, old, new):
        _logger.info(f"Switching logic group to {new}...")
        self._current_group = new
        doc = curdoc()
        doc.hold()
        self._refreshmodel()
        doc.unhold()

        self._push_data_fnc(doc)

        _logger.info(f"Switching logic group finished")

    def _on_select_group2(self, a, old, new):
        _logger.info(f"Switching logic group to {new}...")
        doc = curdoc()
        doc.hold()

        self._current_group = new
        panels = self._bokeh._generate_model_tabs(self._figurepage, self._current_group)

        self.model.children[1].tabs[0] = panels[0]

        doc.unhold()
        _logger.info(f"Switching logic group finished")

    def _update_visibility(self, new_group):
        for f in self._figurepage.figure_envs:
            logicgroups = f.get_logicgroups()
            f.figure.visible = new_group in logicgroups or new_group == self._ALL_GROUP_STR or True in logicgroups

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
        self.last_index = new_last_index

        cds = self._figurepage.cds

        sendpkg = {}
        for c in updatepkg.keys():
            if c in cds.data:
                sendpkg[c] = updatepkg[c]

        _logger.info(f'Sending stream package: {sendpkg}')
        cds.stream(sendpkg, self._lookback)
