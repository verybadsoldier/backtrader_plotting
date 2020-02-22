def assert_num_tabs(figs, *args):
    for idx, num in enumerate(args):
        assert len(figs[idx][0].model.tabs) == num


def assert_num_figures(figs, *args):
    for idx, num in enumerate(args):
        assert len(figs[idx][0].figure_envs) == num
