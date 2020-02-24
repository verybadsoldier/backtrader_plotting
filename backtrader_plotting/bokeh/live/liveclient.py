from bokeh.models import ColumnDataSource, Model
from bokeh.models.widgets import Panel, Tabs, Slider, Button
from bokeh.layouts import column, gridplot, row
from bokeh.models.widgets import CheckboxGroup, Div, Select

from backtrader_plotting.html import metadata
from backtrader_plotting.bokeh.bokeh import Bokeh, FigurePage


class LiveClient:
    _ALL_GROUP_STR = '*All*'

    def __init__(self, bokeh: Bokeh, figurepage_idx:int = 0):
        self._bokeh = bokeh
        self._slider = None

        self._bokeh = bokeh

        self._current_group = None

        self._figurepage: FigurePage = self._bokeh._figurepages[figurepage_idx]
        panels = self._bokeh._generate_model_tabs(self._figurepage)

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

        groups = [self._ALL_GROUP_STR] + self._figurepage.get_logicgroups()
        self._current_group = groups[0]
        s = Select(value=self._current_group, options=groups)
        s.on_change('value', self._on_select_group)

        controls = row(children=[s])
        self.model = column(children=[controls, Tabs(tabs=panels)])

        self._update_visibility(self._current_group)

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

    def _on_select_group(self, a, old, new):
        self._update_visibility(new)

    def _update_visibility(self, new_group):
        for f in self._figurepage.figure_envs:
            logicgroups = f.get_logicgroups()
            f.figure.visible = new_group in logicgroups or new_group == self._ALL_GROUP_STR
