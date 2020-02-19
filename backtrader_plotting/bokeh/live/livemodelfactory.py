from bokeh.models import ColumnDataSource, Model
from bokeh.models.widgets import Panel, Tabs
from bokeh.layouts import column, gridplot
from bokeh.models.widgets import Div
from bokeh.models.widgets import CheckboxGroup

from backtrader_plotting.html import metadata


class LiveModelFactory:
    def __init__(self, bokeh):
        self._bokeh = bokeh

    def _get_config_panel(self):
        def on_change_checkbox(val):
            i = 5
            i += 39
            pass

        names = []
        for d in self._bokeh._figurepages[0].strategy.datas:
            names.append(d._name)
        checkbox_group = CheckboxGroup(labels=names, active=[1 for _ in range(len(names))])
        checkbox_group.on_click(on_change_checkbox)

        childs = gridplot([[checkbox_group]], toolbar_options={'logo': None})
        return Panel(child=childs, title='Config')

    def generate_live_model(self, figurepage_idx = 0) -> Model:
        """Returns a model generated from internal blueprints"""
        #if figurepage_idx >= len(self._figurepages):
        #    raise RuntimeError(f'Cannot generate model for FigurePage with index {figurepage_idx} as there are only {len(self._figurepages)}.')

        figurepage = self._bokeh._figurepages[figurepage_idx]
        tabs = self._bokeh._generate_model_tabs(figurepage)

        # now append analyzer tab(s)
        analyzers = figurepage.analyzers
        panel_analyzer = self._bokeh._get_analyzer_panel(analyzers)
        if panel_analyzer is not None:
            tabs.append(panel_analyzer)

        # append meta tab
        assert figurepage.strategy is not None
        meta = Div(text=metadata.get_metadata_div(figurepage.strategy))
        metapanel = Panel(child=meta, title="Meta")
        tabs.append(metapanel)

        # append config panel
        tabs.append(self._get_config_panel())

        model = Tabs(tabs=tabs)

        return model