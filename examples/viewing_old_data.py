import qcodes as qc
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools
import PyQt5
from qcodes.plots.pyqtgraph import QtPlot
station = qc.station


def show_analysis_values_pyqt(a_obj, QtPlot_win):

    if QtPlot_win is None:
        QtPlot_win = QtPlot(window_title='Analysis viewer',
                            figsize=(600, 400))
    for i, y in enumerate(a_obj.measured_values):
        if i+1 > len(QtPlot_win.subplots):
            QtPlot_win.win.nextRow()
        QtPlot_win.add(
            x=a_obj.sweep_points, y=y, name=a_obj.ylabels[i],
            subplot=i+1,
            symbol='o', symbolSize=5,
            xlabel=a_obj.xlabel, ylabel=a_obj.ylabels[i])
    p0 = QtPlot_win.subplots[0]
    for j, p in enumerate(QtPlot_win.subplots):
        if j > 0:
            p.setXLink(p0)
    return QtPlot_win


# Get yourself a range of timestamps to select the relevant data
timestamps = a_tools.get_timestamps_in_range(
    timestamp_start='20170101_000000',
    timestamp_end='20170101_000000',
    label='my_label')

try:
    vw.clear()
except:
    vw = None

for t_stamp in timestamps:
    a = ma.MeasurementAnalysis(auto=False, close_file=False, timestamp=t_stamp)
    a.get_naming_and_values()
    vw = show_analysis_values_pyqt(a, vw)
