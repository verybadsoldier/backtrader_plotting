def assert_num_tabs(figs, num_tabs):
    assert len(figs[0][0].model.tabs) == num_tabs


def assert_num_figures(figs, num_figs):
    assert len(figs[0][0].figures) == num_figs
