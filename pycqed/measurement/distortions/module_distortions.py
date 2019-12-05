"""
module for distortion correction
"""
import matplotlib.pyplot as plt
import sys
sys.path.append("D:/Repository/PyCQED_py3")
import pycqed.measurement.kernel_functions_vector as kf
from pycqed.analysis import fitting_models as fit_mods
import numpy as np
from imp import reload

reload(kf)


"""
prepare the kernel to calculate step response
"""

kernel_out = kf.kernel_from_kernel_stepvec(kernel_stepvec, width)
plt.plot(t, kernel_out, 'o-', label='Filter')
axes = plt.gca()
axes.set_xlim([0, 100])
# print(np.size(kernel_out),np.size(square_pulse))


# plt.plot(t_vec,np.real(convolution)[:len(square_pulse)],label='Filtered Signal')
# plt.legend(loc=2)

def convolve_kernel_input(input_signal, kernel):
    """
    takes as an input a signal and a kernel
    should be used twice in the whole module
    1. to get the distorted signal, then ideal_signal
    and kernel_out should be used
    2.to get the compensated signal, then distorted signal
    and correction_kernel must be used
    convolve the kernel_out with input signal to get distorted signal
    """

    convolution = np.convolve(input_signal, kernel)
    """
    plt.plot(convolution,label='Filtered')
    plt.plot(square_pulse,label='Input')
    ax = plt.gca()
    ax.set_ylabel('Signal amplitude (normalized)')
    ax.set_xlabel('Time (ns)')
    ax.legend()
    """
    return convolution

"""
Plot the kernel on QT graph and prepare for fittling
passed as a parameter to the output dictionary
to do: be more clear here
step_start : start of the signal 
step_end : end of the signal
baseline : baseline od the signal
"""
convolution = convolve_kernel_input(input_signal, kernel)
points_per_ns = 1
step_width_ns = 1
step_start = 10
step_end = 110
baseline = 0
output_dict = kf.get_all_sampled_vector(convolution,
                                        step_width_ns,
                                        points_per_ns,
                                        step_params={'baseline': baseline,
                                                     'step_start': step_start,
                                                     'step_end': step_end})

try:
    vw.clear()
except Exception:
    import PyQt5
    from qcodes.plots.pyqtgraph import QtPlot
    vw = QtPlot(windowTitle='Seq_plot', figsize=(600, 400))


def contains_nan(array):
    return np.isnan(array).any()


def load_exp_model():
    """
    Prepares a triple exponential model and its parameters
    Output:
            model
            parameters
    """
    triple_pole_mod = fit_mods.TripleExpDecayModel
    triple_pole_mod.set_param_hint('amp1', value=0.05, vary=True)
    triple_pole_mod.set_param_hint('tau1', value=.1e-6, vary=True)
    triple_pole_mod.set_param_hint(
        'amp2', value=0, vary=True)  # -.001, vary=True)
    triple_pole_mod.set_param_hint('tau2', value=.1e-6, vary=True)
    triple_pole_mod.set_param_hint('amp3', max=0., value=0., vary=False)
    triple_pole_mod.set_param_hint('tau3', value=.2e-6, vary=False)
    triple_pole_mod.set_param_hint('offset', min=0., value=1, vary=False)
    triple_pole_mod.set_param_hint('n', value=1, vary=False)
    my_tp_params = triple_pole_mod.make_params()
    return triple_pole_mod, my_tp_params


def fit_step(tvals, norm_amps, start_time_fit,
             end_time_fit, model='exp', verbose=0):
    """
    Produces a fit for a given step
    Inputs:
            sdf
    Outputs:
            sdfgsd
    """
    # loads the corresponding model
    if model == 'exp':
        fit_model, params_fit = load_exp_model()
    # slices the data for the fit
    fit_start = np.argmin(np.abs(tvals - start_time_fit))
    fit_end = np.argmin(np.abs(tvals - end_time_fit))
    tvals_fit = tvals[fit_start:fit_end]

    # makes sure the data is nicely behaved
    if contains_nan(norm_amps[fit_start:fit_end]):
        raise ValueError
    if contains_nan(tvals_fit):
        raise ValueError
    # does the fit
    tp_fit_res = fit_model.fit(data=norm_amps[fit_start:fit_end],
                               t=tvals_fit, params=params_fit)
    # if ok with verbosity, print
    if verbose > 0:
        print(tp_fit_res.fit_report())
    return tp_fit_res

# plotting the fit
#################################
# Initial plot
norm_amps = output_dict['step_raw']  # uses the normalized voltages
fit_amps = tp_fit_res.best_fit
# vw.win.nextRow()
vw.add(x=tvals, xlabel='Time', xunit='s',
       y=norm_amps, ylabel='Normalized amplitude', yunit='',
       symbol='o', symbolSize=5, subplot=1)

# extract fitting data and calculate analytically "correction kernel"
###################################################


def kernel_from_step_fit(tp_fit_res):
    """
    function to extract result(amplitude, offset, tau) from triple exponential model
    use them to calculate kernel correction
    takes as an input tp_fit_res
    """

    # TODO: give function the right argument, which is fit data
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
    # to do a better mathematical formula for amp_kernel_1 and tau_kernel_1
    # to do offset appears in both values
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
    kernel_length = 60000  # -> how many samples for the kernel (1GS/s -> ns)
    t_kernel = np.arange(kernel_length)*1e-9
    # fit_kernel_step = (1. + offset*amp_kernel_1*np.exp(-t_kernel/tau_kernel_1)
    #                    + offset*amp_kernel_2*np.exp(-t_kernel/tau_kernel_2))/offset
    #                    + offset*amp_kernel_3*np.exp(-t_kernel/tau_kernel_3))/offset

    # it is good practice to correct only 1 order at a time
    fit_kernel_step = (1. + offset*amp_m*np.exp(-t_kernel/tau_m))/offset
    # FIXME -> we want to use the maximal tau here

    # calculates the response of the delta function based on the kernel for
    # the step
    fit_kernel = kf.kernel_from_kernel_stepvec(fit_kernel_step)

    return fit_kernel, fit_kernel_step, t_kernel


##################################################

tvals = output_dict['t_step_raw']*1e-9  # Shifts the time vals
amps = output_dict['step_direct']  # uses the non normalized voltages

tvals_k = output_dict['t_kernel']
amp_k = output_dict['kernel_step']

# the fit
# start_time_fit = .1e-6
# end_time_fit = 9.5e-6
start_time_fit = .01*1e-6
end_time_fit = .09*1e-6
fit_start = np.argmin(np.abs(tvals - start_time_fit))
fit_end = np.argmin(np.abs(tvals - end_time_fit))
tvals_fit = tvals[fit_start:fit_end]
norm_amps = output_dict['step_raw']

#################################
# Initial plot
norm_amps = output_dict['step_raw']  # uses the normalized voltages
tp_fit_res = fit_step(tvals, norm_amps, start_time_fit, end_time_fit)
fit_amps = tp_fit_res.best_fit
# vw.win.nextRow()
vw.add(x=tvals, xlabel='Time', xunit='s',
       y=norm_amps, ylabel='Normalized amplitude', yunit='',
       symbol='o', symbolSize=5, subplot=1)
# calling functions
fit_kernel, fit_kernel_step, t_kernel = kernel_from_step_fit(tp_fit_res)
print(t_kernel)

# Add fit to plot

vw.add(x=tvals_fit, xlabel='Time', xunit='s',
       y=fit_amps, ylabel='Normalized amplitude', yunit='',
       subplot=1)

# htilde
# tvals_h = output_dict['t_htilde_raw']*1e-9
# htilde_raw = output_dict['kernel_step']

# vw.add(x=tvals_h, xlabel='Time', xunit='s',
#        y=htilde_raw, ylabel='Kernel amplitude', yunit='V/s', symbol='o', symbolSize=5,
#        subplot=4)


# tvals_k = output_dict['t_kernel']
# amp_k = output_dict['kernel_step']

vw.add(x=tvals_k*1e-9, xlabel='Time', xunit='s',
       y=amp_k, ylabel='Kernel amplitude', yunit='V/s',
       symbol='o', symbolSize=5,
       subplot=2)

vw.add(x=t_kernel, xlabel='Time', xunit='s',
       y=fit_kernel_step, ylabel='Kernel amplitude', yunit='V/s',
       subplot=2)

# convention is to use the timestamp of the scope file as a base
# ts = filename[10:-8]
# save_file_name = 'corr0_'+ts

# kf.save_kernel(fit_kernel, save_file=save_file_name)
# print(save_file_name)


# # Add the distortion to the kernel object
# k0 = station.components['k0']
# try:
#     k0.add_kernel_to_kernel_list(save_file_name+'.txt')
# except ValueError as va:
#     logging.warning(va)
