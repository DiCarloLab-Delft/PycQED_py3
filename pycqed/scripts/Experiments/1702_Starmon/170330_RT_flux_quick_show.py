import pycqed.measurement.kernel_functions as kf
from pycqed.analysis import fitting_models as fit_mods
import numpy as np
reload(kf)
trace_folder = r'D:\Experiments\1702_Starmon\RT_kernel_traces'

# filename = r'\RefCurve_2017-03-30_8_163439.Wfm.csv'
filename = r'\RefCurve_2017-03-31_2_174708.Wfm.csv'

points_per_ns = 5
step_width_ns = 10

kf.kernel_dir = trace_folder

#
step_direct = kf.step_raw(filename, process_step=False)

try:
    vw.clear()
except Exception:
    from qcodes.plots.pyqtgraph import QtPlot
    vw = QtPlot(windowTitle='Seq_plot', figsize=(600, 400))

tvals = np.arange(len(step_direct))*1/5e9
vw.add(x=tvals, xlabel='Time', xunit='s',
       y=step_direct, ylabel='Raw amplitude', yunit='V',
       symbol='o', symbolSize=5, subplot=1)


output_dict = kf.get_all_sampled(filename, step_width_ns, points_per_ns,
                                 step_params=None)


#################################
# Initial plot


tvals = output_dict['t_step_raw']*1e-9  # Shifts the time vals
norm_amps = output_dict['step_raw']  # uses the normalized voltages
# vw.win.nextRow()
vw.add(x=tvals, xlabel='Time', xunit='s',
       y=norm_amps, ylabel='Normalized amplitude', yunit='',
       symbol='o', symbolSize=5, subplot=2)

tvals_k = output_dict['t_kernel']
amp_k = output_dict['kernel_step']


vw.add(x=tvals_k*1e-9, xlabel='Time', xunit='s',
       y=amp_k, ylabel='Kernel amplitude', yunit='V/s',
       symbol='o', symbolSize=5, subplot=3)