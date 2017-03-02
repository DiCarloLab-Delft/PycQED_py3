import numpy as np
import lmfit
from matplotlib import pyplot as plt


def PSD_Analysis(table, path):
    """
    Requires a table as input:
           Row  | Content
        --------+--------
            1   | dac
            2   | frequency
            3   | T1
            4   | T2 star
            5   | T2 echo
            6   | T1 exclusion mask (True where data is to be excluded)
            7   | T2 star exclusion mask (True where data is to be excluded)
            8   | T2 echo exclusion mask (True where data is to be excluded)
    """
    dac, freq, T1, Tramsey, Techo, T1_mask, T2s_mask, T2e_mask = table

    fit_result_arch = fit_frequencies(dac, freq)

    # convert dac in flux as unit of Phi_0
    flux = (dac-fit_result_arch.best_values['offset'])\
        / fit_result_arch.best_values['dac0']

    # calculate the derivative vs flux
    sensitivity_angular = partial_omega_over_flux(
        flux, fit_result_arch.best_values['Ec'],
        fit_result_arch.best_values['Ej'])
    sensitivity = sensitivity_angular/(2*np.pi)

    # Pure dephasing times
    # Calculate pure dephasings
    Gamma_1 = 1.0/T1[~T1_mask]

    Gamma_ramsey = 1.0/Tramsey[~T2s_mask]
    Gamma_echo = 1.0/Techo[~T2e_mask]

    Gamma_phi_ramsey = Gamma_ramsey - Gamma_1/2.0
    Gamma_phi_echo = Gamma_echo - Gamma_1/2.0

    plot_coherence_times(flux, freq, sensitivity,
                         T1, Tramsey, Techo)
    plot_ratios(flux, freq, sensitivity,
                Gamma_phi_ramsey, Gamma_phi_echo)

    fit_res_gammas = fit_gammas(sensitivity, Gamma_phi_ramsey, Gamma_phi_echo)

    intercept = fit_res_gammas.params['intercept'].value
    slope_ramsey = fit_res_gammas.params['slope_ramsey'].value
    slope_echo = fit_res_gammas.params['slope_echo'].value

    # after fitting gammas
    # Post processing
    Qc = 8100
    freq_resonator = 7.188e9
    chi_shift = 800e3

    # from flux noise
    A = slope_echo**2/np.log(2)
    print(r'Amplitude PSD (freq)= (%s u$\Phi_0$)^2' % (np.sqrt(A)/1e-6))
    print('Amplitude PSD (omega)= (%s u$\Phi_0$)^2' %
          (np.sqrt(A)/(2e-6*np.pi)))

    # from white noise
    # using Eq 5 in Nature com. 7,12964 (The flux qubit reviseited to enhanbce
    # coherence and reporducibility)
    k_r = 2*np.pi*freq_resonator/Qc
    eta = k_r**2/(k_r**2 + 4*chi_shift**2)
    n_avg = intercept*k_r/(4*chi_shift**2*eta)
    print('Estimated residual photon number: %s' % n_avg)


def prepare_input_table(dac, frequency, T1, T2_star, T2_echo,
                        T1_mask=None, T2_star_mask=None, T2_echo_mask=None):
    """
    Returns a table ready for PSD_Analysis input
    If sizes are different, it adds nans on the end.
    """
    assert(len(dac) == len(frequency))
    assert(len(dac) >= len(T1))
    assert(len(dac) >= len(T2_star))
    assert(len(dac) >= len(T2_echo))

    if T1_mask is None:
        T1_mask = np.zeros(len(T1), dtype=np.bool)
    if T2_star_mask is None:
        T2_star_mask = np.zeros(len(T2_star), dtype=np.bool)
    if T2_echo_mask is None:
        T2_echo_mask = np.zeros(len(T2_echo), dtype=np.bool)

    assert(len(T1) == len(T1_mask))
    assert(len(T2_star) == len(T2_star_mask))
    assert(len(T2_echo) == len(T2_echo_mask))

    table = np.ones((8, len(dac)))
    table = table * np.nan
    table[0, :] = dac
    table[1, :len(frequency)] = frequency
    table[2, :len(T1)] = T1
    table[3, :len(T2_star)] = T2_star
    table[4, :len(T2_echo)] = T2_echo
    table[5, :len(T1_mask)] = T1_mask
    table[6, :len(T2_star_mask)] = T2_star_mask
    table[7, :len(T2_echo_mask)] = T2_echo_mask

    return table


def arch(dac, Ec, Ej, offset, dac0):
    '''
    Function for frequency vs flux (in dac) for the transmon

    Input:
        - dac: voltage used in the DAC to generate the flux
        - Ec (Hz): Charging energy of the transmon in Hz
        - Ej (Hz): Josephson energy of the transmon in Hz
        - offset: voltage offset of the arch (same unit of the dac)
        - dac0: dac value to generate 1 Phi_0 (same unit of the dac)

    Note: the Phi_0 (periodicity) dac0
    '''
    model = np.sqrt(8*Ec*Ej*np.abs(np.cos((np.pi*(dac-offset))/dac0)))-Ec

    return model

# define the model (from the function) used to fit data
arch_model = lmfit.Model(arch)


# derivative of arch vs flux (in unit of Phi0)
# this is the sensitivity to flux noise
def partial_omega_over_flux(flux, Ec, Ej):
    '''
    Note: flux is in unit of Phi0
    Ej and Ec are in Hz

    Output: angular frequency over Phi_0
    '''
    model = -np.sign(np.cos(np.pi*flux)) * (np.pi**2)*np.sqrt(8*Ec*Ej) * \
        np.sin(np.pi*flux) / np.sqrt(np.abs(np.cos(np.pi*flux)))
    return model


def fit_frequencies(dac, freq):
    arch_model.set_param_hint('Ec', value=250e6, min=200e6, max=300e6)
    arch_model.set_param_hint('Ej', value=18e9, min=0)
    arch_model.set_param_hint('offset', value=0)
    arch_model.set_param_hint('dac0', value=2000, min=0)

    arch_model.make_params()

    fit_result_arch = arch_model.fit(freq, dac=dac)
    return fit_result_arch


def plot_coherence_times(flux, freq, sensitivity,
                         T1, Tramsey, Techo):
    # font = {'size': 16}
    # matplotlib.rc('font', **font)

    # f, ax = plt.subplots(1, 3, figsize=[18, 6], sharey=True)

    f, ax = plt.subplots(1, 3, sharey=True)

    ax[0].plot(flux/1e-3, T1/1e-6, '.', color='r', label='$T_1$')
    ax[0].plot(flux/1e-3, Tramsey/1e-6, '.', color='g', label='$T_2^*$')
    ax[0].plot(flux/1e-3, Techo/1e-6, '.', color='b', label='$T_2$')
    ax[0].set_title('$T_1$, $T_2^*$, $T_2$ vs flux')
    ax[0].set_ylabel('Coherence time ($\mu$s)')
    ax[0].set_xlabel('Flux (m$\Phi_0$)')
    ax[0].legend(loc=0)

    ax[1].plot(freq/1e9, T1/1e-6, '.', color='r', label='$T_1$')
    ax[1].plot(freq/1e9, Tramsey/1e-6, '.', color='g', label='$T_2^*$')
    ax[1].plot(freq/1e9, Techo/1e-6, '.', color='b', label='$T_2$')
    ax[1].set_title('$T_1$, $T_2^*$, $T_2$ vs frequency')
    ax[1].set_xlabel('Frequency (GHz)')
    ax[1].legend(loc=0)

    ax[2].plot(np.abs(sensitivity)/1e9, T1/1e-6, '.', color='r', label='$T_1$')
    ax[2].plot(np.abs(sensitivity)/1e9, Tramsey/1e-6,
               '.', color='g', label='$T_2^*$')
    ax[2].plot(
        np.abs(sensitivity)/1e9, Techo/1e-6, '.', color='b', label='$T_2$')
    ax[2].set_title('$T_1$, $T_2^*$, $T_2$ vs sensitivity')
    ax[2].set_xlabel(r'$|\partial\nu/\partial\Phi|$ (GHz/$\Phi_0$)')
    ax[2].legend(loc=0)

    f.tight_layout()

# ax[0].set_ylim([0,40])


def plot_ratios(flux, freq, sensitivity,
                Gamma_phi_ramsey, Gamma_phi_echo):
    # Pure dephaning times

    # # font size
    # font = {'size': 16}
    # matplotlib.rc('font', **font)
    # f, ax = plt.subplots(1, 3, figsize=[18, 6], sharey=True)

    f, ax = plt.subplots(1, 3, sharey=True)

    ratio_gamma = Gamma_phi_ramsey/Gamma_phi_echo

    ax[0].plot(flux/1e-3, ratio_gamma, '.', color='b')
    ax[0].set_title('$T_\phi^{echo}/T_\phi^{ramsey}$ vs flux')
    ax[0].set_ylabel('Ratio')
    ax[0].set_xlabel('Flux (m$\Phi_0$)')

    ax[1].plot(freq/1e9, ratio_gamma, '.', color='b')
    ax[1].set_title('$T_\phi^{echo}/T_\phi^{ramsey}$ vs frequency')
    ax[1].set_xlabel('Frequency (GHz)')

    ax[2].plot(np.abs(sensitivity)/1e9, ratio_gamma, '.', color='b')
    ax[2].set_title('$T_\phi^{echo}/T_\phi^{ramsey}$ vs sensitivity')
    ax[2].set_xlabel(r'$|\partial\nu/\partial\Phi|$ (GHz/$\Phi_0$)')

    f.tight_layout()


def residual_Gamma(pars_dict, sensitivity, Gamma_phi_ramsey, Gamma_phi_echo):
    slope_ramsey = pars_dict['slope_ramsey']
    slope_echo = pars_dict['slope_echo']
    intercept = pars_dict['intercept']

    gamma_values_ramsey = slope_ramsey*np.abs(sensitivity) + intercept
    residual_ramsey = Gamma_phi_ramsey - gamma_values_ramsey

    gamma_values_echo = slope_echo*np.abs(sensitivity) + intercept
    residual_echo = Gamma_phi_echo - gamma_values_echo

    return np.concatenate((residual_ramsey, residual_echo))


def super_residual(p):
    data = residual_Gamma(p)
    # print(type(data))
    return data.astype(float)


def fit_gammas(sensitivity, Gamma_phi_ramsey, Gamma_phi_echo, verbose=0):
    # create a parametrrer set for the initial guess
    p = lmfit.Parameters()
    p.add('slope_ramsey', value=100.0, vary=True)
    p.add('slope_echo', value=100.0, vary=True)
    p.add('intercept', value=100.0, vary=True)

    # mi = lmfit.minimize(super_residual, p)
    wrap_residual = lambda p: residual_Gamma(p,
                                             sensitivity=sensitivity,
                                             Gamma_phi_ramsey=Gamma_phi_ramsey,
                                             Gamma_phi_echo=Gamma_phi_echo)
    fit_result_gammas = lmfit.minimize(wrap_residual, p)
    verbose = 1
    if verbose > 0:
        lmfit.printfuncs.report_fit(fit_result_gammas.params)
    return fit_result_gammas


def plot_gamma_fit(sensitivity, Gamma_phi_ramsey, Gamma_phi_echo,
                   slope_ramsey, slope_echo, intercept):

    f, ax = plt.subplots(1, 1)

    ax.plot(np.abs(sensitivity)/1e9, Gamma_phi_ramsey,
             '.', color='g', label='$\Gamma_{Ramsey}$')
    ax.plot(np.abs(sensitivity)/1e9, slope_ramsey *
             np.abs(sensitivity)+intercept, color='g')

    ax.plot(np.abs(sensitivity)/1e9, Gamma_phi_echo,
             '.', color='b', label='$\Gamma_{Echo}$')
    ax.plot(np.abs(sensitivity)/1e9, slope_echo *
             np.abs(sensitivity)+intercept, color='b')

    ax.legend(loc=0)
    ax.set_title('Gamma vs |flux sensitivity|')
    ax.set_xlabel(r'$|\partial\nu/\partial\Phi|$ (GHz/$\Phi_0$)')
    ax.set_ylabel('$\Gamma$ (1/s)')


"""
Test code
# print the result of the fit and plot data
print('Offset = %s mV' % (fit_result_arch.best_values['offset']))
print('Dac/Phi_0 = %s mV' % (fit_result_arch.best_values['dac0']))
print('Ec = %s MHz' % (fit_result_arch.best_values['Ec']/1e6))
print('Ej = %s GHz' % (fit_result_arch.best_values['Ej']/1e9))

plt.plot(dac, freq/1e9, '.', label='data')
plt.plot(dac, fit_result_arch.best_fit/1e9, color='r', label='best fit')
plt.legend(loc=0)
plt.title('Full arch QR3')
plt.ylabel('Freq (GHz)')
plt.xlabel('Dac (mV)')
"""
