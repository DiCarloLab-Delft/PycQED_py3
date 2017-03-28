import pycqed.measurement.kernel_functions as kf
from pycqed.analysis import fitting_models as fit_mods
import numpy as np
reload(kf)
trace_folder = r'D:\Experiments\1702_Starmon\RT_kernel_traces'
filename = r'\RefCurve_2017-03-27_2_152823.Wfm.csv'


points_per_ns = 5
step_width_ns = 10

kf.kernel_dir = trace_folder
output_dict = kf.get_all_sampled(filename, step_width_ns, points_per_ns,
                                 step_params=None)
try:
    vw.clear()
except Exception:
    from qcodes.plots.pyqtgraph import QtPlot
    vw = QtPlot(windowTitle='Seq_plot', figsize=(600, 400))

tvals = output_dict['t_step_raw']*1e-9  # Shifts the time vals
amps = output_dict['step_direct']  # uses the non normalized voltages

# vw.add(x=tvals, xlabel='Time', xunit='s',
#        y=amps, ylabel='Scope amplitude', yunit='V', symbol='o', symbolSize=5)


tvals_k = output_dict['t_kernel']
amp_k = output_dict['kernel_step']

# vw.add(x=tvals_k*1e-9, xlabel='Time', xunit='s',
#        y=amp_k, ylabel='Kernel amplitude', yunit='V/s', symbol='o', symbolSize=5,
#        subplot=2)


##########################
# Setting up the fitting model
triple_pole_mod = fit_mods.TripleExpDecayModel
triple_pole_mod.set_param_hint('amp1', max=0., value=-0.003, vary=True)
triple_pole_mod.set_param_hint('tau1', value=.5e-6, vary=True)
triple_pole_mod.set_param_hint('amp2', max=0, value=-.001, vary=True)
triple_pole_mod.set_param_hint('tau2', value=.2e-6, vary=True)
triple_pole_mod.set_param_hint('amp3', max=0., value=0., vary=False)
triple_pole_mod.set_param_hint('tau3', value=.2e-6, vary=False)
triple_pole_mod.set_param_hint('offset', min=0., value=1, vary=True)
triple_pole_mod.set_param_hint('n', value=1, vary=False)
my_tp_params = triple_pole_mod.make_params()

# the fit
start_time_fit = .5e-6
end_time_fit = 6e-6
fit_start = np.argmin(np.abs(tvals - start_time_fit))
fit_end = np.argmin(np.abs(tvals - end_time_fit))
tvals_fit = tvals[fit_start:fit_end]
norm_amps = output_dict['step_raw']
tp_fit_res = triple_pole_mod.fit(data=norm_amps[fit_start:fit_end],
                                 t=tvals_fit, params=my_tp_params)

###################################################
# Extracting the fit results
tau_1 = tp_fit_res.best_values['tau1']
amp_1 = tp_fit_res.best_values['amp1']
offset = tp_fit_res.best_values['offset']
tau_2 = tp_fit_res.best_values['tau2']
amp_2 = tp_fit_res.best_values['amp2']
offset = tp_fit_res.best_values['offset']
tau_3 = tp_fit_res.best_values['tau3']
amp_3 = tp_fit_res.best_values['amp3']
offset = tp_fit_res.best_values['offset']

# These are the analytical expressions for the kernel corrections
amp_kernel_1 = -amp_1/(1.+amp_1)
tau_kernel_1 = tau_1*(1+amp_1)
amp_kernel_2 = -amp_2/(1.+amp_2)
tau_kernel_2 = tau_2*(1+amp_2)
amp_kernel_3 = -amp_3/(1.+amp_3)
tau_kernel_3 = tau_3*(1+amp_3)
tpm_taus = [tau_kernel_1, tau_kernel_2, tau_kernel_3]
tpm_amps = [amp_kernel_1, amp_kernel_2, amp_kernel_3]

tau_idx = np.argmax(tpm_taus)
tau_m = tpm_taus[tau_idx]
amp_m = tpm_amps[tau_idx]
# We pick 8 us
kernel_length = 8000  # -> how many samples for the kernel (1GS/s -> ns)
t_kernel = np.arange(kernel_length)*1e-9
# fit_kernel_step = (1. + offset*amp_kernel_1*np.exp(-t_kernel/tau_kernel_1)
#                    + offset*amp_kernel_2*np.exp(-t_kernel/tau_kernel_2))/offset
#                    + offset*amp_kernel_3*np.exp(-t_kernel/tau_kernel_3))/offset

# it is good practice to correct only 1 order at a time
fit_kernel_step = (1. + offset*amp_m*np.exp(-t_kernel/tau_m))/offset
# FIXME -> we want to use the maximal tau here

# calculates the response of the delta function based on the kernel for the step
fit_kernel = kf.kernel_from_kernel_stepvec(fit_kernel_step)

##################################################

norm_amps = output_dict['step_raw']  # uses the normalized voltages
fit_amps = tp_fit_res.best_fit
# vw.win.nextRow()
vw.add(x=tvals, xlabel='Time', xunit='s',
       y=norm_amps, ylabel='Normalized amplitude', yunit='',
       symbol='o', symbolSize=5, subplot=1)

vw.add(x=tvals_fit, xlabel='Time', xunit='s',
       y=fit_amps, ylabel='Normalized amplitude', yunit='',
       subplot=1)

########## htilde
# tvals_h = output_dict['t_htilde_raw']*1e-9
# htilde_raw = output_dict['kernel_step']

# vw.add(x=tvals_h, xlabel='Time', xunit='s',
#        y=htilde_raw, ylabel='Kernel amplitude', yunit='V/s', symbol='o', symbolSize=5,
#        subplot=4)



# tvals_k = output_dict['t_kernel']
# amp_k = output_dict['kernel_step']

vw.add(x=tvals_k*1e-9, xlabel='Time', xunit='s',
       y=amp_k, ylabel='Kernel amplitude', yunit='V/s', symbol='o', symbolSize=5,
       subplot=2)

vw.add(x=t_kernel, xlabel='Time', xunit='s',
       y=fit_kernel_step, ylabel='Kernel amplitude', yunit='V/s',
       subplot=2)

# convention is to use the timestamp of the scope file as a base
ts = filename[10:-8]
save_file_name = 'corr0_'+ts

kf.save_kernel(fit_kernel, save_file=save_file_name)
print(save_file_name)



# Add the distortion to the kernel object
k0 = station.components['k0']
# k0.kernel_list([save_file_name+'.txt'])
k0.add_kernel_to_kernel_list(save_file_name+'.txt')


import pycqed.instrument_drivers.meta_instrument.kernel_object as k_obj
reload(k_obj)
# k0 = k_obj.Distortion(name='k0')
# station.add_component(k0)
