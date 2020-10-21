from backtrader_plotting.bokeh.bokeh import FigurePage


def assert_num_tabs(figs, *args):
    for idx, num in enumerate(args):
        assert len(figs[idx][0].model.tabs) == num


def assert_num_figures(figs, *args):
    for idx, num in enumerate(args):
        fp: FigurePage = figs[idx][0]
        assert len(fp.figures) == num
