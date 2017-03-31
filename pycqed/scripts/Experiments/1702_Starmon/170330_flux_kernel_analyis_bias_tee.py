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

inversion_step_widths_ns = np.arange(1000, 49, -50)
# inversion_step_widths_ns = np.concatenate([np.arange(1000, 49, -50),
                                          # np.arange(50, 45, -5)])
inverted_waveforms = []
for step_width in inversion_step_widths_ns:
    output_dict = (kf.get_all_sampled(filename, step_width, points_per_ns,
                                      step_params=None, max_points=1000))
    inverted_waveforms.append((output_dict['t_kernel'],
                                output_dict['kernel_step']))



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

for (tvals_k, amp_k) in inverted_waveforms:
    vw.add(x=tvals_k*1e-9, xlabel='Time', xunit='s',
           y=amp_k, ylabel='Kernel amplitude', yunit='V/s',
           symbol='o', symbolSize=5, subplot=2)
###################################################
# Extracting the fit results

import lmfit
fit_results = []
for (tvals_k, amp_k) in inverted_waveforms:
    mod = lmfit.models.PolynomialModel(degree=2)
    pars = mod.guess(amp_k, x=tvals_k*1e-9)
    fit_res = mod.fit(amp_k, pars, x=tvals_k*1e-9)
    fit_results.append(fit_res.params)

c0_vec = [fr['c0'].value for fr in fit_results]
c1_vec = [fr['c1'].value for fr in fit_results]
c2_vec = [fr['c2'].value for fr in fit_results]

fit_f = fit_res.eval(x=tvals_k*1e-9,
                     c0=c0_vec[-1], c1=c1_vec[-1], c2=c2_vec[-1])
vw.add(x=tvals_k*1e-9, xlabel='Time', xunit='s',
       y=fit_f, ylabel='Kernel amplitude', yunit='V/s',
       subplot=2)

c0_vec

f, axs = plt.subplots(1,3)
axs[0].plot(inversion_step_widths_ns,c0_vec)
axs[1].plot(inversion_step_widths_ns,c1_vec)

axs[2].plot(inversion_step_widths_ns,c2_vec)


vw.add(x=inversion_step_widths_ns*1e-9, xlabel='Stepsize', xunit='s',
       y=c0_vec,
       symbol='o', symbolSize=5, subplot=3)

vw.add(x=inversion_step_widths_ns*1e-9, xlabel='Stepsize', xunit='s',
       y=c1_vec,
       symbol='o', symbolSize=5, subplot=4)

vw.add(x=inversion_step_widths_ns*1e-9, xlabel='Stepsize', xunit='s',
       y=c2_vec,
       symbol='o', symbolSize=5, subplot=5)

# Project fit results to 1ns stepsize

mod = lmfit.models.PolynomialModel(degree=2)
pars = mod.guess(c0_vec, x=inversion_step_widths_ns*1e-9)
fit_res = mod.fit(c0_vec, pars, x=inversion_step_widths_ns*1e-9)
c0_projected = fit_res.params['c0']


mod = lmfit.models.PolynomialModel(degree=2)
pars = mod.guess(c1_vec, x=inversion_step_widths_ns*1e-9)
fit_res = mod.fit(c1_vec, pars, x=inversion_step_widths_ns*1e-9)
c1_projected = fit_res.params['c0']

mod = lmfit.models.PolynomialModel(degree=2)
pars = mod.guess(c2_vec, x=inversion_step_widths_ns*1e-9)
fit_res = mod.fit(c2_vec, pars, x=inversion_step_widths_ns*1e-9)
c2_projected = fit_res.params['c0']

print('projected values')
print('c0: ', c0_projected.value)
print('c1: ', c1_projected.value)
print('c2: ', c2_projected.value)

##################################################


fit_f = fit_res.eval(x=tvals_k*1e-9,
                     c0=c0_projected.value, c1=c1_projected.value, c2=c2_projected.value)
vw.add(x=tvals_k*1e-9, xlabel='Time', xunit='s',
       y=fit_f, ylabel='Kernel amplitude', yunit='V/s',
       subplot=2)

# Add fit to plot

# vw.add(x=tvals_fit, xlabel='Time', xunit='s',
#        y=fit_amps, ylabel='Normalized amplitude', yunit='',
#        subplot=1)

########## htilde

# Add the distortion to the kernel object
k0 = station.components['k0']
fit_kernel_step = fit_res.eval(x=np.arange(k0.corrections_length()*1e9)*1e-9,
                     c0=c0_projected.value, c1=c1_projected.value, c2=c2_projected.value)
fit_kernel = kf.kernel_from_kernel_stepvec(fit_kernel_step)
ts = filename[10:-8]
save_file_name = 'corr0_'+ts
# print(fit_kernel)
# kf.save_kernel(fit_kernel, save_file=save_file_name)
# print(fit_kernel)

# try:
#     k0.add_kernel_to_kernel_list(save_file_name+'.txt')
# except ValueError as va:
#     logging.warning(va)

