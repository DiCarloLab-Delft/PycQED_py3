import pycqed.measurement.kernel_functions as kf
from pycqed.analysis import fitting_models as fit_mods
import numpy as np
reload(kf)
trace_folder = r'D:\Experiments\1702_Starmon\RT_kernel_traces'
# filename = r'\RefCurve_2017-03-29_2_195314.Wfm.csv'
# filename = r'\RefCurve_2017-03-30_0_110407.Wfm.csv'
# filename = r'\RefCurve_2017-03-30_1_111929.Wfm.csv'
# filename = r'\RefCurve_2017-03-30_2_112928.Wfm.csv'

# filename = r'\RefCurve_2017-03-30_5_132742.Wfm.csv'
# filename = r'\RefCurve_2017-03-30_6_153026.Wfm.csv'
filename = r'\RefCurve_2017-03-30_7_155508.Wfm.csv'


def contains_nan(array):
    return np.isnan(array).any()

points_per_ns = 5
step_width_ns = 10

kf.kernel_dir = trace_folder


output_dict = kf.get_all_sampled(filename, step_width_ns, points_per_ns,
                                 step_params=None)

#################################
# Initial plot
try:
    vw.clear()
except Exception:
    from qcodes.plots.pyqtgraph import QtPlot
    vw = QtPlot(windowTitle='Seq_plot', figsize=(600, 400))


tvals = output_dict['t_step_raw']*1e-9  # Shifts the time vals
norm_amps = output_dict['step_raw']  # uses the normalized voltages
# vw.win.nextRow()
vw.add(x=tvals, xlabel='Time', xunit='s',
       y=norm_amps, ylabel='Normalized amplitude', yunit='',
       symbol='o', symbolSize=5, subplot=1)

tvals_k = output_dict['t_kernel']
amp_k = output_dict['kernel_step']


vw.add(x=tvals_k*1e-9, xlabel='Time', xunit='s',
       y=amp_k, ylabel='Kernel amplitude', yunit='V/s',
       symbol='o', symbolSize=5, subplot=2)
###################################################
# Extracting the fit results

import lmfit
fit_results = []
start_fit = .8e-6
start_idx = np.argmin(abs(tvals_k*1e-9-start_fit))

mod = lmfit.models.PolynomialModel(degree=2)
pars = mod.guess(amp_k[start_idx:], x=tvals_k[start_idx:]*1e-9)
fit_res = mod.fit(amp_k[start_idx:], pars, x=tvals_k[start_idx:]*1e-9)

fit_f = fit_res.eval(x=tvals_k[start_idx:]*1e-9,
                     c0=fit_res.params['c0'].value,
                     c1=fit_res.params['c1'].value,
                     c2=fit_res.params['c2'].value)
vw.add(x=tvals_k[start_idx:]*1e-9, xlabel='Time', xunit='s',
       y=fit_f, ylabel='Kernel amplitude', yunit='V/s',
       subplot=2)



##################################################

########## htilde

# Add the distortion to the kernel object
k0 = station.components['k0']
fit_kernel_step = fit_res.eval(x=np.arange(k0.corrections_length()*1e9)*1e-9,
                     c0=fit_res.params['c0'].value,
                     c1=fit_res.params['c1'].value,
                     c2=fit_res.params['c2'].value)
fit_kernel = kf.kernel_from_kernel_stepvec(fit_kernel_step)
ts = filename[10:-8]
save_file_name = 'corr0_'+ts
kf.save_kernel(fit_kernel, save_file=save_file_name)

try:
    k0.add_kernel_to_kernel_list(save_file_name+'.txt')
except ValueError as va:
    logging.warning(va)

