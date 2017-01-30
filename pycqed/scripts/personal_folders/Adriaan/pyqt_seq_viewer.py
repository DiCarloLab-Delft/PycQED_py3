from qcodes.plots.pyqtgraph import QtPlot
from qcodes.plots.colors import color_cycle


def show_element_pyqt(element, QtPlot_win=None,
                      color_idx=None,
                      channels='all', ):
    if QtPlot_win is None:
        QtPlot_win = QtPlot(windowTitle='Seq_plot', figsize=(600, 400))

    t_vals, outputs_dict = element.waveforms()
    t_vals = t_vals*1e9
    xlabel = 'Time (ns)'
    ylabel = 'Analog output (V)'
    for i, key in enumerate(outputs_dict):
        if color_idx == None:
            color = color_cycle[i % len(color_cycle)]
        else:
            color = color_cycle[color_idx]
        if channels == 'all' or key in channels:
            QtPlot_win.add(
                x=t_vals, y=outputs_dict[key], name=key,
                color=color,
                symbol='o', symbolSize=5,
                xlabel=xlabel, ylabel=ylabel)
    return QtPlot_win
