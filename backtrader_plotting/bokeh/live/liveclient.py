from collections import defaultdict
import logging
from threading import Lock

import backtrader as bt

from bokeh.models.widgets import Panel, Tabs, Slider, Button
from bokeh.layouts import column, gridplot, row
from bokeh.io import curdoc
from bokeh.models.widgets import CheckboxGroup, Div, Select

from backtrader_plotting.html import metadata
from backtrader_plotting.bokeh.bokeh import FigurePage

_logger = logging.getLogger(__name__)


class LiveClient:
    def __init__(self, push_fnc, bokeh_fac: callable, strategy: bt.Strategy, figurepage_idx: int = 0, lookback: int = 20):
        self._slider = None
        self._push_fnc = push_fnc
        self.last_index = -1
        self._lookback = lookback

        self._lock = Lock()
        self._bokeh = bokeh_fac()
        self._bokeh.plot(strategy, fill_data=False)

        self._figurepage: FigurePage = self._bokeh._figurepages[figurepage_idx]
        groups = [x for x in self._figurepage.get_logicgroups() if not isinstance(x, bool)]
        self._current_group = groups[0]

        panels = self._bokeh._generate_model_tabs(self._figurepage, self._current_group)

        self._bokeh._update_cds(self._figurepage)

        # now append analyzer tab(s)
        analyzers = self._figurepage.analyzers
        panel_analyzer = self._bokeh._get_analyzer_panel(analyzers)
        if panel_analyzer is not None:
            panels.append(panel_analyzer)

        # append meta tab
        assert self._figurepage.strategy is not None
        meta = Div(text=metadata.get_metadata_div(self._figurepage.strategy))
        metapanel = Panel(child=meta, title="Meta")
        panels.append(metapanel)

        # append config panel
        panels.append(self._get_config_panel())


        self._current_group = groups[0]
        s = Select(value=self._current_group, options=groups)
        s.on_change('value', self._on_select_group)

        btn_refresh_analyzers = Button(label='Refresh Analyzers')
        btn_refresh_analyzers.on_click(self._on_click_refresh_analyzers)

        controls = row(children=[s, btn_refresh_analyzers])
        self.model = column(children=[controls, Tabs(tabs=panels)], sizing_mode=self._bokeh.p.scheme.plot_sizing_mode)

        # self._update_visibility(self._current_group)

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
        panels = self._bokeh._generate_model_tabs(self._figurepage, self._current_group)
        self.model.children[1].tabs[0] = panels[0]

        doc = curdoc()
        doc.hold('combine')
        self._bokeh._update_cds(self._figurepage)
        doc.unhold()

        self.last_index = -1
        self._push_fnc()
        # self._update_visibility(new)
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
            idx = dt_idx_map[dt]
            patch_dict[colname].append((idx, val))
        _logger.info(f"Sending patch dict: {patch_dict}")

        cds.patch(patch_dict)

    def push_adds(self, updatepkg: dict, last_index: int):
        self.last_index = last_index

        cds = self._figurepage.cds

        sendpkg = {}
        for c in updatepkg.keys():
            if c in cds.data:
                sendpkg[c] = updatepkg[c]

        cds.stream(sendpkg, self._lookback)
