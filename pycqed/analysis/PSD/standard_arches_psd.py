import numpy as np
import lmfit
from matplotlib import pyplot as plt
import os


def PSD_Analysis(table, freq_resonator=None, Qc=None, chi_shift=None, path=None):
    """
    Requires a table as input:
           Row  | Content
        --------+--------
            1   | dac
            2   | frequency
            3   | T1
            4   | T2 star
            5   | T2 echo
            6   | Exclusion mask (True where data is to be excluded)

    Generates 7 plots:
        > T1, T2, Echo vs flux
        > T1, T2, Echo vs frequency
        > T1, T2, Echo vs flux sensitivity

        > ratio Ramsey/Echo vs flux
        > ratio Ramsey/Echo vs frequency
        > ratio Ramsey/Echo vs flux sensitivity

        > Dephasing rates Ramsey and Echo vs flux sensitivity

        If properties of resonator are provided (freq_resonator, Qc, chi_shift),
        it also calculates the number of noise photons.

    """
    dac, freq, T1, Tramsey, Techo, exclusion_mask = table
    exclusion_mask = np.array(exclusion_mask, dtype=bool)

    # Extract the dac arcs required for getting the sensitivities
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
    Gamma_1 = 1.0/T1[~exclusion_mask]
    Gamma_ramsey = 1.0/Tramsey[~exclusion_mask]
    Gamma_echo = 1.0/Techo[~exclusion_mask]

    Gamma_phi_ramsey = Gamma_ramsey - Gamma_1/2.0
    Gamma_phi_echo = Gamma_echo - Gamma_1/2.0

    plot_coherence_times(flux, freq, sensitivity,
                         T1, Tramsey, Techo, path)
    plot_ratios(flux, freq, sensitivity,
                Gamma_phi_ramsey, Gamma_phi_echo, path)

    fit_res_gammas = fit_gammas(sensitivity, Gamma_phi_ramsey, Gamma_phi_echo)

    intercept = fit_res_gammas.params['intercept'].value
    slope_ramsey = fit_res_gammas.params['slope_ramsey'].value
    slope_echo = fit_res_gammas.params['slope_echo'].value

    plot_gamma_fit(sensitivity, Gamma_phi_ramsey, Gamma_phi_echo,
                   slope_ramsey, slope_echo, intercept, path)

    # after fitting gammas
    # from flux noise
    # Martinis PRA 2003
    sqrtA_rams = slope_ramsey/(np.pi*np.sqrt(30))
    sqrtA_echo = slope_echo/(np.pi*np.sqrt(1.386))


    print('Amplitude echo PSD = (%s u\Phi_0)^2' % (sqrtA_echo/1e-6))
    print('Amplitude rams PSD = (%s u\Phi_0)^2' % (sqrtA_rams/1e-6))

    # from white noise
    # using Eq 5 in Nat. Comm. 7,12964 (The flux qubit revisited to enhance
    # coherence and reproducability)
    if not ((freq_resonator is None) and (Qc is None) and (chi_shift is None)):
        n_avg = calculate_n_avg(freq_resonator, Qc, chi_shift, intercept)
        print('Estimated residual photon number: %s' % n_avg)
    else:
        n_avg = np.nan

    return (sqrtA_echo/1e-6), n_avg

def calculate_n_avg(freq_resonator, Qc, chi_shift, intercept):
    """
    Returns the avg photon of white noise,assuming photon shot noise from the RO hanger.
    """
    k_r = 2*np.pi*freq_resonator/Qc
    eta = k_r**2/(k_r**2 + 4*chi_shift**2)
    n_avg = intercept*k_r/(4*chi_shift**2*eta)
    return n_avg


def prepare_input_table(dac, frequency, T1, T2_star, T2_echo,
                        T1_mask=None, T2_star_mask=None, T2_echo_mask=None):
    """
    Returns a table ready for PSD_Analysis input
    If sizes are different, it adds nans on the end.
    """
    assert(len(dac) == len(frequency))
    assert(len(dac) == len(T1))
    assert(len(dac) == len(T2_star))
    assert(len(dac) == len(T2_echo))


    if T1_mask is None:
        T1_mask = np.zeros(len(T1), dtype=bool)
    if T2_star_mask is None:
        T2_star_mask = np.zeros(len(T2_star), dtype=bool)
    if T2_echo_mask is None:
        T2_echo_mask = np.zeros(len(T2_echo), dtype=bool)

    assert(len(T1) == len(T1_mask))
    assert(len(T2_star) == len(T2_star_mask))
    assert(len(T2_echo) == len(T2_echo_mask))

    table = np.ones((6, len(dac)))
    table = table * np.nan
    table[0, :] = dac
    table[1, :len(frequency)] = frequency
    table[2, :len(T1)] = T1
    table[3, :len(T2_star)] = T2_star
    table[4, :len(T2_echo)] = T2_echo
    table[5, :len(T1_mask)] = np.logical_or(np.logical_or(T1_mask,
                                                          T2_star_mask),
                                            T2_echo_mask)

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

def plot_coherence_times_freq(flux, freq, sensitivity,
                         T1, Tramsey, Techo, path,
                         figname='Coherence_times.PNG'):
    f, ax = plt.subplots()

    ax.plot(freq/1e9, T1/1e-6, 'o', color='C3', label='$T_1$')
    ax.plot(freq/1e9, Tramsey/1e-6, 'o', color='C2', label='$T_2^*$')
    ax.plot(freq/1e9, Techo/1e-6, 'o', color='C0', label='$T_2$')
    ax.set_title('$T_1$, $T_2^*$, $T_2$ vs frequency')
    ax.set_xlabel('Frequency (GHz)')
    ax.legend(loc=0)


    f.tight_layout()
    if path is not None:
        savename = os.path.abspath(os.path.join(path, figname))
        f.savefig(savename, format='PNG', dpi=450)

def plot_coherence_times(flux, freq, sensitivity,
                         T1, Tramsey, Techo, path,
                         figname='Coherence_times.PNG'):
    # font = {'size': 16}
    # matplotlib.rc('font', **font)

    # f, ax = plt.subplots(1, 3, figsize=[18, 6], sharey=True)

    f, ax = plt.subplots(1, 3, figsize=(18, 5), sharey=True)

    ax[0].plot(flux/1e-3, T1/1e-6, 'o', color='C3', label='$T_1$')
    ax[0].plot(flux/1e-3, Tramsey/1e-6, 'o', color='C2', label='$T_2^*$')
    ax[0].plot(flux/1e-3, Techo/1e-6, 'o', color='C0', label='$T_2$')
    ax[0].set_title('$T_1$, $T_2^*$, $T_2$ vs flux', size=16)
    ax[0].set_ylabel('Coherence time ($\mu$s)', size=16)
    ax[0].set_xlabel('Flux (m$\Phi_0$)', size=16)

    ax[1].plot(freq/1e9, T1/1e-6, 'o', color='C3', label='$T_1$')
    ax[1].plot(freq/1e9, Tramsey/1e-6, 'o', color='C2', label='$T_2^*$')
    ax[1].plot(freq/1e9, Techo/1e-6, 'o', color='C0', label='$T_2$')
    ax[1].set_title('$T_1$, $T_2^*$, $T_2$ vs frequency', size=16)
    ax[1].set_xlabel('Frequency (GHz)', size=16)
    ax[1].legend(loc=0)

    ax[2].plot(np.abs(sensitivity)/1e9, T1/1e-6, 'o', color='C3', label='$T_1$')
    ax[2].plot(np.abs(sensitivity)/1e9, Tramsey/1e-6,
               'o', color='C2', label='$T_2^*$')
    ax[2].plot(
        np.abs(sensitivity)/1e9, Techo/1e-6, 'o', color='C0', label='$T_2$')
    ax[2].set_title('$T_1$, $T_2^*$, $T_2$ vs sensitivity', size=16)
    ax[2].set_xlabel(r'$|\partial\nu/\partial\Phi|$ (GHz/$\Phi_0$)', size=16)

    f.tight_layout()
    if path is not None:
        savename = os.path.abspath(os.path.join(path, figname))
        f.savefig(savename, format='PNG', dpi=450)



def plot_ratios(flux, freq, sensitivity,
                Gamma_phi_ramsey, Gamma_phi_echo, path,
                figname='Gamma_ratios.PNG'):
    # Pure dephaning times

    f, ax = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    ratio_gamma = Gamma_phi_ramsey/Gamma_phi_echo

    ax[0].plot(flux/1e-3, ratio_gamma, 'o', color='C0')
    ax[0].set_title('$T_\phi^{\mathrm{Echo}}/T_\phi^{\mathrm{Ramsey}}$ vs flux', size=16)
    ax[0].set_ylabel('Ratio', size=16)
    ax[0].set_xlabel('Flux (m$\Phi_0$)', size=16)

    ax[1].plot(freq/1e9, ratio_gamma, 'o', color='C0')
    ax[1].set_title('$T_\phi^{\mathrm{Echo}}/T_\phi^{\mathrm{Ramsey}}$ vs frequency', size=16)
    ax[1].set_xlabel('Frequency (GHz)', size=16)

    ax[2].plot(np.abs(sensitivity)/1e9, ratio_gamma, 'o', color='C0')
    ax[2].set_title('$T_\phi^{\mathrm{Echo}}/T_\phi^{\mathrm{Ramsey}}$ vs sensitivity', size=16)
    ax[2].set_xlabel(r'$|\partial\nu/\partial\Phi|$ (GHz/$\Phi_0$)', size=16)

    f.tight_layout()

    if path is not None:
        savename = os.path.abspath(os.path.join(path, figname))
        f.savefig(savename, format='PNG', dpi=450)


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
                   slope_ramsey, slope_echo, intercept, path,
                   f=None, ax=None,
                   figname='Gamma_Fit.PNG'):
    if f is None:
        f, ax = plt.subplots()

    ax.plot(np.abs(sensitivity)/1e9, Gamma_phi_ramsey,
            'o', color='C2', label='$\Gamma_{\phi,\mathrm{Ramsey}}$')
    ax.plot(np.abs(sensitivity)/1e9, slope_ramsey *
            np.abs(sensitivity)+intercept, color='C2')

    ax.plot(np.abs(sensitivity)/1e9, Gamma_phi_echo,
            'o', color='C0', label='$\Gamma_{\phi,\mathrm{Echo}}$')
    ax.plot(np.abs(sensitivity)/1e9, slope_echo *
            np.abs(sensitivity)+intercept, color='C0')

    ax.legend(loc=0)
    # ax.set_title('Pure dephasing vs flux sensitivity')
    ax.set_title('Previous cooldown')
    ax.set_xlabel(r'$|\partial f/\partial\Phi|$ (GHz/$\Phi_0$)')
    ax.set_ylabel('$\Gamma_{\phi}$ (1/s)')
    f.tight_layout()
    ax.set_ylim(0, np.max(Gamma_phi_ramsey)*1.05)
    if path is not None:
        savename = os.path.abspath(os.path.join(path, figname))
        f.savefig(savename, format='PNG', dpi=450)


"""
Test code
# print the result of the fit and plot data
print('Offset = %s mV' % (fit_result_arch.best_values['offset']))
print('Dac/Phi_0 = %s mV' % (fit_result_arch.best_values['dac0']))
print('Ec = %s MHz' % (fit_result_arch.best_values['Ec']/1e6))
print('Ej = %s GHz' % (fit_result_arch.best_values['Ej']/1e9))

plt.plot(dac, freq/1e9, 'o', label='data')
plt.plot(dac, fit_result_arch.best_fit/1e9, color='C3', label='best fit')
plt.legend(loc=0)
plt.title('Full arch QR3')
plt.ylabel('Freq (GHz)')
plt.xlabel('Dac (mV)')
"""
