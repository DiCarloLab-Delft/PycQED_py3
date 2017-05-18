import matplotlib.pyplot as plt
import sys
sys.path.append("D:/Repository/PyCQED_py3")
import pycqed.measurement.kernel_functions_vector as kf
from pycqed.analysis import fitting_models as fit_mods
import numpy as np
from imp import reload

reload(kf)


def get_square_pulse(t, width, t0):
    square_pulse = kf.square(t, width, t0)
    return square_pulse


def get_sim_kernel(t, amp, width, tau):
    kernel_stepvec = 1. - amp*np.exp(-t/tau)
    return kernel_stepvec


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
    triple_pole_mod.set_param_hint('offset', min=0., value=1, vary=True)
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


def kernel_from_step_fit(tp_fit_res, kernel_length=60000):
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

