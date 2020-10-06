import numpy as np
import matplotlib.pyplot as plt
from scipy.special import erfc
import lmfit
import logging
from pycqed.analysis import analysis_toolbox as a_tools
from pycqed.analysis.tools import data_manipulation as dm_tools
import string

#################################
#   Fitting Functions Library   #
#################################


def RandomizedBenchmarkingDecay(numCliff, Amplitude, p, offset):
    val = Amplitude * (p ** numCliff) + offset
    return val


def DoubleExpDampOscFunc(t, tau_1, tau_2,
                         freq_1, freq_2,
                         phase_1, phase_2,
                         amp_1, amp_2, osc_offset):
    if (tau_1>0):
        cos_1 = amp_1 * (np.cos(2 * np.pi * freq_1 * t + phase_1)) * np.exp(-(t / tau_1))
    else:
        cos_1 = np.zeros_like(t)
    if (tau_2>0):
        cos_2 = amp_2 * (np.cos(2 * np.pi * freq_2 * t + phase_2)) * np.exp(-(t / tau_2))
    else:
        cos_2 = np.zeros_like(t)
    return cos_1 + cos_2 + osc_offset


def double_RandomizedBenchmarkingDecay(numCliff, p, offset,
                                       invert=1):
    """
    A variety of the RB-curve that allows fitting both the inverting and
    non-inverting exponential.
    The amplitude of the decay curve is constrained to start at 0 or 1.
    The offset is the common point both curves converge to.

    pick invert to be 1 or 0
    """
    # Inverting clifford curve
    val_inv = (1 - offset) * (p ** numCliff) + offset
    # flipping clifford curve
    val_flip = -offset * (p ** numCliff) + offset
    # Using invert as a boolean but not using if statement to allow for
    # arrays to be input in the function
    val = (1 - invert) * val_flip + invert * val_inv
    return val


def LorentzFunc(f, amplitude, center, sigma):
    val = amplitude / np.pi * (sigma / ((f - center) ** 2 + sigma ** 2))
    return val


def Lorentzian(f, A, offset, f0, kappa):
    val = offset + A / np.pi * (kappa / ((f - f0) ** 2 + kappa ** 2))
    return val


def TwinLorentzFunc(f, A_gf_over_2, A, f0_gf_over_2, f0,
                    kappa_gf_over_2, kappa, background=0):
    """
    Twin lorentz with background.

    Args:
    f (float):          frequency sweep points in Hz
    A (float):          amplitude of the tallest/deepest Lorentzian structure
                        in the data
    A_gf_over_2 (float):    amplitude of the other Lorentzian structure in the
                            data; since this function is used for high power
                            qubit spectroscopy, this parameter refers to the
                            Lorentzian structure corresponding to the gf/2
                            transition
    f0 (float):         frequency of the tallest/deepest Lorentzian structure
                        in the data
    f0_gf_over_2 (float):   frequency of the other Lorentzian structure in the
                            data; since this function is used for high power
                            qubit spectroscopy, this parameter refers to the
                            Lorentzian structure corresponding to the gf/2
                            transition
    kappa (float):      kappa (FWHM) of the tallest/deepest Lorentzian structure
                        in the data
    kappa_gf_over_2 (float): kappa (FWHM) of the other Lorentzian structure in
                             the data; since this function is used for high
                             power qubit spectroscopy, this parameter refers to
                             the Lorentzian structure corresponding to the gf/2
                             transition
    background (float):     background offset
    """
    val = (A_gf_over_2 / np.pi * (kappa_gf_over_2 / ((f - f0_gf_over_2) ** 2 + kappa_gf_over_2 ** 2)) +
           A / np.pi * (kappa / ((f - f0) ** 2 + kappa ** 2)) + background)
    return val


def Qubit_dac_to_freq(dac_voltage, f_max, E_c,
                      dac_sweet_spot, V_per_phi0=None,
                      dac_flux_coefficient=None,
                      asymmetry=0, **kwargs):
    """
    The cosine Arc model for uncalibrated flux for asymmetric qubit.

    dac_voltage (V)
    f_max (Hz): sweet-spot frequency of the qubit
    E_c (Hz): charging energy of the qubit
    V_per_phi0 (V): volt per phi0 (convert voltage to flux)
    dac_sweet_spot (V): voltage at which the sweet-spot is found
    asym (dimensionless asymmetry param) = abs((EJ1-EJ2)/(EJ1+EJ2)),
    """
    if V_per_phi0 is None and dac_flux_coefficient is None:
        raise ValueError('Please specify "V_per_phi0".')

    if dac_flux_coefficient is not None:
        logging.warning('"dac_flux_coefficient" deprecated. Please use the '
                        'physically meaningful "V_per_phi0" instead.')
        V_per_phi0 = np.pi / dac_flux_coefficient

    qubit_freq = (f_max + E_c) * (
            asymmetry ** 2 + (1 - asymmetry ** 2) *
            np.cos(np.pi / V_per_phi0 *
                   (dac_voltage - dac_sweet_spot)) ** 2) ** 0.25 - E_c
    return qubit_freq


def Resonator_dac_to_freq(dac_voltage, f_max_qubit, f_0_res,
                          E_c, dac_sweet_spot,
                          coupling, V_per_phi0=None,
                          dac_flux_coefficient=None,
                          asymmetry=0, **kwargs):
    qubit_freq = Qubit_dac_to_freq(dac_voltage=dac_voltage, f_max=f_max_qubit, E_c=E_c,
                                   dac_sweet_spot=dac_sweet_spot, V_per_phi0=V_per_phi0,
                                   dac_flux_coefficient=dac_flux_coefficient,
                                   asymmetry=asymmetry)
    delta_qr = (qubit_freq - f_0_res)
    lamb_shift = (coupling ** 2 / delta_qr)
    resonator_freq = f_0_res - lamb_shift

    return resonator_freq


def Qubit_dac_to_detun(dac_voltage, f_max, E_c, dac_sweet_spot, V_per_phi0,
                       asymmetry=0):
    """
    The cosine Arc model for uncalibrated flux for asymmetric qubit.

    dac_voltage (V)
    f_max (Hz): sweet-spot frequency of the qubit
    E_c (Hz): charging energy of the qubit
    V_per_phi0 (V): volt per phi0 (convert voltage to flux)
    dac_sweet_spot (V): voltage at which the sweet-spot is found
    asymmetry (dimensionless asymmetry param) = abs((EJ1-EJ2)/(EJ1+EJ2))
    """
    return f_max - Qubit_dac_to_freq(dac_voltage,
                                     f_max=f_max, E_c=E_c,
                                     dac_sweet_spot=dac_sweet_spot,
                                     V_per_phi0=V_per_phi0,
                                     asymmetry=asymmetry)


def Qubit_freq_to_dac(frequency, f_max, E_c,
                      dac_sweet_spot, V_per_phi0=None,
                      dac_flux_coefficient=None, asymmetry=0,
                      branch='positive'):
    """
    The cosine Arc model for uncalibrated flux for asymmetric qubit.
    This function implements the inverse of "Qubit_dac_to_freq"

    frequency (Hz)
    f_max (Hz): sweet-spot frequency of the qubit
    E_c (Hz): charging energy of the qubit
    V_per_phi0 (V): volt per phi0 (convert voltage to flux)
    asym (dimensionless asymmetry param) = abs((EJ1-EJ2)/(EJ1+EJ2))
    dac_sweet_spot (V): voltage at which the sweet-spot is found
    branch (enum: 'positive' 'negative')
    """
    if V_per_phi0 is None and dac_flux_coefficient is None:
        raise ValueError('Please specify "V_per_phi0".')

    # asymm_term = (asymmetry**2 + (1-asymmetry**2))
    # dac_term = np.arccos(((frequency+E_c)/((f_max+E_c) * asymm_term))**2)

    dac_term = np.arccos(np.sqrt(
        (((frequency + E_c) / (f_max + E_c)) ** 4 - asymmetry ** 2) /
        (1 - asymmetry ** 2)))

    if dac_flux_coefficient is not None:
        logging.warning('"dac_flux_coefficient" deprecated. Please use the '
                        'physically meaningful "V_per_phi0" instead.')
        V_per_phi0 = np.pi / dac_flux_coefficient

    if branch == 'positive':
        dac_voltage = dac_term * V_per_phi0 / np.pi + dac_sweet_spot
    elif branch == 'negative':
        dac_voltage = -dac_term * V_per_phi0 / np.pi + dac_sweet_spot
    else:
        raise ValueError('branch {} not recognized'.format(branch))

    return dac_voltage


def Qubit_dac_sensitivity(dac_voltage, f_max: float, E_c: float,
                          dac_sweet_spot: float, V_per_phi0: float,
                          asymmetry: float = 0):
    """
    Derivative of the qubit detuning vs dac at dac_voltage.
    The returned quantity is "dfreq/dPhi (dac_voltage)"
    """
    cos_term = np.cos(np.pi / V_per_phi0 * (dac_voltage - dac_sweet_spot))
    sin_term = np.sin(np.pi / V_per_phi0 * (dac_voltage - dac_sweet_spot))
    return ((f_max + E_c) * (1 - asymmetry ** 2) * np.pi / (2 * V_per_phi0) *
            cos_term * sin_term * (asymmetry ** 2 + (1 - asymmetry ** 2) *
                                   cos_term ** 2) ** (-0.75))


def QubitFreqDac(dac_voltage, f_max, E_c,
                 dac_sweet_spot, dac_flux_coefficient, asymmetry=0):
    logging.warning('deprecated, replace QubitFreqDac with Qubit_dac_to_freq')
    return Qubit_dac_to_freq(dac_voltage, f_max, E_c,
                             dac_sweet_spot, dac_flux_coefficient, asymmetry)


def QubitFreqFlux(flux, f_max, E_c,
                  flux_zero, dac_offset=0):
    """The cosine Arc model for calibrated flux."""
    calculated_frequency = (f_max + E_c) * np.sqrt(np.abs(
        np.cos(np.pi * (flux - dac_offset) / flux_zero))) - E_c
    return calculated_frequency


def CosFunc(t, amplitude, frequency, phase, offset):
    """
    parameters:
        t, time in s
        amplitude a.u.
        frequency in Hz (f, not omega!)
        phase in rad
        offset a.u.
    """
    return amplitude * np.cos(2 * np.pi * frequency * t + phase) + offset


def CosFunc2(t, amplitude, frequency, phase, offset):
    """
    parameters:
        t, time in s
        amplitude a.u.
        frequency in Hz (f, not omega!)
        phase in rad
        offset a.u.
    """
    return amplitude * np.cos(2 * np.pi * frequency * (t + phase)) + offset


def ExpDecayFunc(t, tau, amplitude, offset, n):
    return amplitude * np.exp(-(t / tau) ** n) + offset


def idle_error_rate_exp_decay(N, N1, N2, A, offset):
    """
    exponential decay consisting of two components
    """
    return A * np.exp(-N / N1 - (N / N2) ** 2) + offset


def gain_corr_ExpDecayFunc(t, tau, amp, gc):
    """
    Specific form of an exponential decay used for flux corrections.
    Includes a "gain correction" parameter that is ignored when correcting
    the distortions.
    """

    y = gc * (1 + amp * np.exp(-t / tau))
    return y


def gain_corr_double_ExpDecayFunc(t, tau_A, tau_B, amp_A, amp_B, gc):
    """
    Specific form of an exponential decay used for flux corrections.
    Includes a "gain correction" parameter that is ignored when correcting
    the distortions.
    """
    y = gc * (1 + amp_A * np.exp(-t / tau_A) + amp_B * np.exp(-t / tau_B))
    return y


def ExpDampOscFunc(t, tau, n, frequency, phase, amplitude,
                   oscillation_offset, exponential_offset):
    return amplitude * np.exp(-(t / tau) ** n) * (np.cos(
        2 * np.pi * frequency * t + phase) + oscillation_offset) + exponential_offset

def ExpDampOscFuncComplex(t, tau, frequency, phase, amplitude, offset):
    return amplitude*np.exp(1j*(2 * np.pi * frequency * t + phase) - t/tau) + offset



def GaussExpDampOscFunc(t, tau, tau_2, frequency, phase, amplitude,
                        oscillation_offset, exponential_offset):
    return amplitude * np.exp(-(t / tau_2) ** 2 - (t / tau)) * (np.cos(
        2 * np.pi * frequency * t + phase) + oscillation_offset) + exponential_offset


def ExpDampDblOscFunc(t, tau, n, freq_1, freq_2, phase_1, phase_2,
                      amp_1, amp_2,
                      osc_offset_1, osc_offset_2, exponential_offset):
    """
    Exponential decay with double cosine modulation
    """
    exp_decay = np.exp(-(t / tau) ** n)
    cos_1 = (np.cos(
        2 * np.pi * freq_1 * t + phase_1) + osc_offset_1)
    cos_2 = (np.cos(
        2 * np.pi * freq_2 * t + phase_2) + osc_offset_2)

    return amp_1 * exp_decay * cos_1 + amp_2 * exp_decay * cos_2 + exponential_offset


def HangerFuncAmplitude(f, f0, Q, Qe, A, theta):
    """
    This is the function for a hanger  which does not take into account
    a possible slope.
    This function may be preferred over SlopedHangerFunc if the area around
    the hanger is small.
    In this case it may misjudge the slope
    Theta is the asymmetry parameter

    Note! units are inconsistent
    f is in Hz
    f0 is in GHz
    """
    return abs(A * (1. - Q / Qe * np.exp(1.j * theta) / (1. + 2.j * Q * (f / 1.e9 - f0) / f0)))



def hanger_func_complex_SI(f, f0, Q, Qe,
                           A, theta, phi_v, phi_0,
                           slope=1):
    """
    This is the complex function for a hanger (lamda/4 resonator).
    See equation 3.1 of the Asaad master thesis.

    It differens from the "old" HangerFuncComplex in that all inputs are SI
    (no longer f in SI and f0 not in SI...)


    Input:
        f   : frequency
        f0  : resonance frequency
        A   : background transmission amplitude
        Q  : loaded quality factor
        Qe  : extrinsic quality factor
        theta:  phase of Qe (in rad)
        phi_v:  phase to account for propagation delay to sample
        phi_0:  phase to account for propagation delay from sample
        slope:  slope of signal around the resonance


    The complex hanger function that has a list of parameters as input
    is now called hanger_func_complex_SI_pars


    """
    slope_corr = (1+slope*(f-f0)/f0)
    propagation_delay_corr = np.exp(1j * (phi_v * f + phi_0))
    hanger_contribution = (1 - Q / Qe * np.exp(1j * theta)/
                               (1 + 2.j * Q * (f  - f0) / f0))
    S21 = A * slope_corr * hanger_contribution * propagation_delay_corr

    return S21

def hanger_func_complex_SI_pars(f,pars):
    """

    This function is used in the minimization fitting which requires parameters.
    It calls the function hanger_func_complex_SI, see there for details.
    """

    f0 = pars['f0']
    Ql = pars['Ql']
    Qe = pars['Qe']
    A = pars['A']
    theta = pars['theta']
    phi_v = pars['phi_v']
    phi_0 = pars['phi_0']
    alpha = pars['alpha']
    return hanger_func_complex_SI(f, f0, Ql, Qe,
                           A, theta, phi_v, phi_0, alpha)



def PolyBgHangerFuncAmplitude(f, f0, Q, Qe, A, theta, poly_coeffs):
    # This is the function for a hanger (lambda/4 resonator) which takes into
    # account a possible polynomial background
    # NOT DEBUGGED
    return np.abs((1. + np.polyval(poly_coeffs, (f / 1.e9 - f0) / f0)) *
                  HangerFuncAmplitude(f, f0, Q, Qe, A, theta))



def SlopedHangerFuncAmplitude(f, f0, Q, Qe, A, theta, slope):
    # This is the function for a hanger (lambda/4 resonator) which takes into
    # account a possible slope df
    return np.abs((1. + slope * (f / 1.e9 - f0) / f0) *
                  HangerFuncAmplitude(f, f0, Q, Qe, A, theta))


def SlopedHangerFuncComplex(f, f0, Q, Qe, A, theta, phi_v, phi_0, slope):
    # This is the function for a hanger (lambda/4 resonator) which takes into
    # account a possible slope df
    return (1. + slope * (f / 1.e9 - f0) / f0) * np.exp(1.j * (phi_v * f + phi_0 - phi_v * f[0])) * \
           HangerFuncComplex(f, f0, Q, Qe, A, theta)


def linear_with_offset(x, a, b):
    """
    A linear signal with a fixed offset.
    """
    return a * x + b


def linear_with_background(x, a, b):
    """
    A linear signal with a fixed background.
    """
    return np.sqrt((a * x) ** 2 + b ** 2)


def linear_with_background_and_offset(x, a, b, c):
    """
    A linear signal with a fixed background.
    """
    return np.sqrt((a * x) ** 2 + b ** 2) + c


def gaussianCDF(x, amplitude, mu, sigma):
    """
    CDF of gaussian is P(X<=x) = .5 erfc((mu-x)/(sqrt(2)sig))
    """
    return 0.5 * amplitude * erfc((mu - x) / (np.sqrt(2) * sigma))


def double_gaussianCDF(x, A_amplitude, A_mu, A_sigma,
                       B_amplitude, B_mu, B_sigma):
    """
    CDF of two gaussians added on top of each other.

    uses "gaussianCDF"
    """
    CDF_A = gaussianCDF(x, amplitude=A_amplitude, mu=A_mu, sigma=A_sigma)
    CDF_B = gaussianCDF(x, amplitude=B_amplitude, mu=B_mu, sigma=B_sigma)
    return CDF_A + CDF_B

def ro_gauss(x, A_center, B_center, A_sigma, B_sigma, A_amplitude,
             B_amplitude, A_spurious, B_spurious):
    """
    Two double-gaussians with sigma and mu/center of the residuals equal to the
    according state.
    """
    gauss = lmfit.lineshapes.gaussian
    A_gauss = gauss(x=x[0], center=A_center, sigma=A_sigma, amplitude=A_amplitude)
    B_gauss = gauss(x=x[1], center=B_center, sigma=B_sigma, amplitude=B_amplitude)
    gauss0 = ((1-A_spurious)*A_gauss + A_spurious*B_gauss)
    gauss1 = ((1-B_spurious)*B_gauss + B_spurious*A_gauss)
    return [gauss0, gauss1]


def ro_CDF(x, A_center, B_center, A_sigma, B_sigma, A_amplitude,
           B_amplitude, A_spurious, B_spurious):
    cdf = gaussianCDF
    A_gauss = cdf(x=x[0], mu=A_center, sigma=A_sigma, amplitude=A_amplitude)
    B_gauss = cdf(x=x[1], mu=B_center, sigma=B_sigma, amplitude=B_amplitude)
    gauss0 = ((1-A_spurious)*A_gauss + A_spurious*B_gauss)
    gauss1 = ((1-B_spurious)*B_gauss + B_spurious*A_gauss)
    return [gauss0, gauss1]


def ro_CDF_discr(x, A_center, B_center, A_sigma, B_sigma, A_amplitude,
                 B_amplitude, A_spurious, B_spurious):
    #A_amplitude /= 1-A_spurious
    #B_amplitude /= 1-B_spurious
    return ro_CDF(x, A_center, B_center, A_sigma, B_sigma, A_amplitude,
                  B_amplitude, A_spurious=0, B_spurious=0)


def gaussian_2D(x, y, amplitude=1,
                center_x=0, center_y=0,
                sigma_x=1, sigma_y=1):
    """
    A 2D gaussian function. if you want to use this for fitting you need to
    flatten your data first.
    """
    gauss = lmfit.lineshapes.gaussian
    val = (gauss(x, amplitude, center_x, sigma_x) *
           gauss(y, amplitude, center_y, sigma_y))
    return val


def DoubleExpDecayFunc(t, tau1, tau2, amp1, amp2, offset, n):
    return (offset +
            amp1 * np.exp(-(t / tau1) ** n) +
            amp2 * np.exp(-(t / tau2) ** n))


def TripleExpDecayFunc(t, tau1, tau2, tau3, amp1, amp2, amp3, offset, n):
    return (offset +
            amp1 * np.exp(-(t / tau1) ** n) +
            amp2 * np.exp(-(t / tau2) ** n) +
            amp3 * np.exp(-(t / tau3) ** n))


def avoided_crossing_mediated_coupling(flux, f_bus, f_center1, f_center2,
                                       c1, c2, g, flux_state=0):
    """
    Calculates the frequencies of an avoided crossing for the following model.
        [f_b,  g,  g ]
        [g,   f_1, 0 ]
        [g,   0,  f_2]

    f1 = c1*flux + f_center1
    f2 = c2*flux + f_center2
    f_b = constant

    g:  the coupling strength, beware to relabel your variable if using this
        model to fit J1 or J2.
    flux_state:  this is a switch used for fitting. It determines which
        transition to return

    """
    if type(flux_state) == int:
        flux_state = [flux_state] * len(flux)

    frequencies = np.zeros([len(flux), 2])
    for kk, dac in enumerate(flux):
        f_1 = dac * c1 + f_center1
        f_2 = dac * c2 + f_center2
        matrix = [[f_bus, g, g],
                  [g, f_1, 0.],
                  [g, 0., f_2]]
        frequencies[kk, :] = np.linalg.eigvalsh(matrix)[:2]
    result = np.where(flux_state, frequencies[:, 0], frequencies[:, 1])
    return result


def avoided_crossing_direct_coupling(flux, f_center1, f_center2,
                                     c1, c2, g, flux_state=0):
    """
    Calculates the frequencies of an avoided crossing for the following model.
        [f_1, g ]
        [g,  f_2]

    f1 = c1*flux + f_center1
    f2 = c2*flux + f_center2

    g:  the coupling strength, beware to relabel your variable if using this
        model to fit J1 or J2.
    flux_state:  this is a switch used for fitting. It determines which
        transition to return
    """

    if type(flux_state) == int:
        flux_state = [flux_state] * len(flux)

    frequencies = np.zeros([len(flux), 2])
    for kk, dac in enumerate(flux):
        f_1 = dac * c1 + f_center1
        f_2 = dac * c2 + f_center2
        matrix = [[f_1, g],
                  [g, f_2]]
        frequencies[kk, :] = np.linalg.eigvalsh(matrix)[:2]
    result = np.where(flux_state, frequencies[:, 0], frequencies[:, 1])
    return result

def avoided_crossing_freq_shift(flux, a, b, g):
    """
    Calculates the frequency shift due to an avoided crossing for the following model:
        [delta_f,  g ]
        [g,        0 ]

    delta_f = a*flux + b

    Parameters
    ----------
    flux : array like
        flux bias values
    a, b : float
        parameters used to calculate frequency distance (delta) away from
        avoided crossing according to
            delta_f = a*flux+b

    g: float
        Coupling strength strength, beware to relabel your variable if using this
        model to fit J1 or J2.

    Returns
    -------
    frequency_shift : (float)


    Note: this model is useful for fitting the frequency shift due to an interaction
    in a chevron experiment (after fourier transforming the data).
    """

    frequencies = np.zeros([len(flux), 2])
    for kk, fl_i in enumerate(flux):
        f_1 = a * fl_i + b
        f_2 = 0
        matrix = [[f_1, g],
                  [g, f_2]]
        frequencies[kk, :] = np.linalg.eigvalsh(matrix)[:2]
    result = frequencies[:, 1] - frequencies[:, 0]
    return result


def resonator_flux(f_bare, g, A, f, t, sweetspot_cur):
    return f_bare - g/(A*np.sqrt(np.abs(np.cos(np.pi*f*(t-sweetspot_cur))))
                       - f_bare)


def ChevronFunc(amp, amp_center_1, amp_center_2, J2, detuning_swt_spt, t):
    """
    NB: [2020-03-11] This model was not thoroughly tested, implemented for
    chevron alignment were we care only about the center of the maxima

    Assuming we start in |0>, this function returns the population in |1>

    This model approximates the bare detuning with a quadratic dependence
    on amp.

    Args:
        amp (a.u.): flux square pulse amplitude
        amp_center_1 (a.u.): center of the chevron on one side of the flux arc
        amp_center_2 (a.u.): idem
        J2 (Hz): coupling of the interacting states
        detuning_swt_spt (Hz): detuning from the interaction point at sweet spot
            NB: not meaningful from a fit
    """
    # We approximate the bare detuning with a quadratic dependence on amp
    detuning = detuning_swt_spt * (amp - amp_center_1) * (amp - amp_center_2)
    d_sq = detuning ** 2
    J2_sq = J2 ** 2
    d4J_sq = d_sq + 4 * J2_sq
    return (4 * J2_sq * np.sin(np.pi * np.sqrt(d4J_sq) * t) ** 2) / d4J_sq


def ChevronInvertedFunc(amp, amp_center_1, amp_center_2, J2, detuning_swt_spt, t):
    """
    Assuming we start in |0>, this function returns the population in |0>

    See `ChevronFunc` for details
    """
    return 1 - ChevronFunc(amp, amp_center_1, amp_center_2, J2, detuning_swt_spt, t)

######################
# Residual functions #
######################


def residual_complex_fcn(pars, cmp_fcn, x, y):
    """
    Residual of a complex function with complex results 'y' and
    real input values 'x'
    For resonators 'x' is the the frequency, 'y' the complex transmission
    Input:
        pars = parameters dictionary (check the corresponding function 'cmp_fcn' for the parameters to pass)
        cmp_fcn = complex function
        x = input real values to 'cmp_fcn'
        y = output complex values from 'cmp_fcn'

    Author = Stefano Poletto
    """
    cmp_values = cmp_fcn(x, pars)

    res = cmp_values - y
    res = np.append(res.real, res.imag)

    return res


####################
# Guess functions  #
####################


def exp_dec_guess(model, data, t, vary_n=False, vary_off=True):
    """
    Assumes exponential decay in estimating the parameters
    """
    offs_guess = data[np.argmax(t)]
    amp_guess = data[np.argmin(t)] - offs_guess
    # guess tau by looking for value closest to 1/e
    tau_guess = t[np.argmin(abs((amp_guess * (1 / np.e) + offs_guess) - data))]

    model.set_param_hint('amplitude', value=amp_guess)
    model.set_param_hint('tau', value=tau_guess)
    if vary_n:
        model.set_param_hint('n', value=1.1, vary=vary_n, min=1)
    else:
        model.set_param_hint('n', value=1, vary=vary_n)
    if vary_off:
        model.set_param_hint('offset', value=offs_guess)
    else:
        model.set_param_hint('offset', value=0, vary=vary_off)

    params = model.make_params()
    return params


def SlopedHangerFuncAmplitudeGuess(data, f, fit_window=None):
    xvals = f
    peaks = a_tools.peak_finder(xvals, data)
      # Search for peak
    if peaks['dip'] is not None:    # look for dips first
        f0 = peaks['dip']
        amplitude_factor = -1.
    elif peaks['peak'] is not None:  # then look for peaks
        f0 = peaks['peak']
        amplitude_factor = 1.
    else:                                 # Otherwise take center of range
        f0 = np.median(xvals)
        amplitude_factor = -1.

    min_index = np.argmin(data)
    max_index = np.argmax(data)
    min_frequency = xvals[min_index]
    max_frequency = xvals[max_index]
    # print(min_frequency)
    # print(max_frequency)
    amplitude_guess = max(dm_tools.reject_outliers(data))

    # Creating parameters and estimations

    #Maybe this is not so good: sharp peaks will definitely be excluded
    S21min = (min(dm_tools.reject_outliers(data)) /
              max(dm_tools.reject_outliers(data)))
    # print(S21min)
    # S21min=  (min((data)) /
    #           max(dm_tools.reject_outliers(data)))
    # print(min((data)))
    # print(min(dm_tools.reject_outliers(data)))
    # print(max((data)))
    # print(max(dm_tools.reject_outliers(data)))

    # print(S21min)
    Q = f0 / abs(min_frequency - max_frequency)
    Qe = abs(Q / abs(1 - S21min))

    guess_dict = {'f0': {'value': f0*1e-9,
                         'min': min(xvals)*1e-9,
                         'max': max(xvals)*1e-9},
                  'A': {'value': amplitude_guess},
                  'Q': {'value': Q,
                        'min': 1,
                        'max': 50e6},
                  'Qe': {'value': Qe,
                         'min': 1,
                         'max': 50e6},
                  'Qi': {'expr': 'abs(1./(1./Q-1./Qe*cos(theta)))',
                         'vary': False},
                  'Qc': {'expr': 'Qe/cos(theta)',
                         'vary': False},
                  'theta': {'value': 0,
                            'min': -np.pi/2,
                            'max': np.pi/2},
                  'slope': {'value':0,
                            'vary':True}}
    return guess_dict


def hanger_func_complex_SI_Guess(data, f, fit_window=None):
    # This is complete garbage, just to get some return value
    xvals = f
    abs_data = np.abs(data)
    peaks = a_tools.peak_finder(xvals, abs_data)
    # Search for peak
    if peaks['dip'] is not None:    # look for dips first
        f0 = peaks['dip']
        amplitude_factor = -1.
    elif peaks['peak'] is not None:  # then look for peaks
        f0 = peaks['peak']
        amplitude_factor = 1.
    else:                                 # Otherwise take center of range
        f0 = np.median(xvals)
        amplitude_factor = -1.

    min_index = np.argmin(abs_data)
    max_index = np.argmax(abs_data)
    min_frequency = xvals[min_index]
    max_frequency = xvals[max_index]

    amplitude_guess = max(dm_tools.reject_outliers(abs_data))

    # Creating parameters and estimations
    S21min = (min(dm_tools.reject_outliers(abs_data)) /
              max(dm_tools.reject_outliers(abs_data)))

    Q = f0 / abs(min_frequency - max_frequency)

    Qe = abs(Q / abs(1 - S21min))
    guess_dict = {'f0': {'value': f0*1e-9,
                         'min': min(xvals)*1e-9,
                         'max': max(xvals)*1e-9},
                  'A': {'value': amplitude_guess},
                  'Qe': {'value': Qe,
                         'min': 1,
                         'max': 50e6},
                  'Ql': {'value': Q,
                         'min': 1,
                         'max': 50e6},
                  'theta': {'value': 0,
                            'min': -np.pi/2,
                            'max': np.pi/2},
                  'alpha': {'value':0,
                            'vary':True},
                  'phi_0': {'value':0,
                            'vary':True},
                  'phi_v': {'value':0,
                            'vary':True}}

    return guess_dict


def group_consecutives(vals, step=1):
    """Return list of consecutive lists of numbers from vals (number list)."""
    run = []
    result = [run]
    expect = None
    for v in vals:
        if (v == expect) or (expect is None):
            run.append(v)
        else:
            run = [v]
            result.append(run)
        expect = v + step
    return result


def arc_guess(freq, dac, dd=0.1):
    """
    Expects the dac values to be sorted!
    :param freq:
    :param dac:
    :param dd:
    :return:
    """
    p = round(max(dd * len(dac), 1))
    f_small = np.average(np.sort(freq)[:p]) + np.std(np.sort(freq)[:p])
    f_big = np.average(np.sort(freq)[-p:]) - np.std(np.sort(freq)[-p:])
    # print(f_small * 1e-9, f_big * 1e-9)

    fmax = np.max(freq)
    fmin = np.min(freq)

    dacs_ss = np.where(freq >= f_big)[0]
    dacs_as = np.where(freq <= f_small)[0]

    dacs_ss_groups = group_consecutives(vals=dacs_ss, step=1)
    dacs_as_groups = group_consecutives(vals=dacs_as, step=1)

    dacs_ss_single = []
    for g in dacs_ss_groups:
        ind = g[np.argmax(freq[g])]
        # ind = int(round(np.average(g)))
        dacs_ss_single.append(ind)

    dac_ss_group_index = np.argmin(np.abs(dac[dacs_ss_single]))
    dac_ss_index = dacs_ss_single[dac_ss_group_index]
    min_left = 0
    min_right = len(dac) - 1
    dacs_as_single = []
    for g in dacs_as_groups:
        if 0 in g:
            ind = 0
        elif len(dac) - 1 in g:
            ind = len(dac) - 1
        else:
            ind = int(round(np.average(g)))

        if ind < dac_ss_index:
            min_left = max(ind, min_left)
        elif ind > dac_ss_index:
            min_right = min(ind, min_right)
        dacs_as_single.append(ind)
    # print('maxs', dacs_ss_single)
    # print('mins', dacs_as_single)
    arc_len = (dac[min_right] - dac[min_left])

    # print('%d to %d = %.5f' % (min_left, min_right, arc_len))
    if min_left == 0 or min_right == len(dac) - 1:
        arc_len *= 2
    elif len(dacs_ss_groups) > 1:
        arc_len = np.average(dac[dacs_ss_single[1:]] - dac[dacs_ss_single[:-1]])

    return fmax, fmin, dac[dac_ss_index], arc_len


def Resonator_dac_arch_guess(model, data, dac_voltage,
                             E_c: float = None, values=None):
    if values is None:
        values = {}

    fmax, fmin, dac_ss, period = arc_guess(freq=data, dac=dac_voltage)

    # todo make better f_res guess
    f_res = np.mean(data)  # - (coup_guess ** 2 / (f_max_qubit - fmax))
    f_res = values.get('f_res', f_res)
    f_max_qubit = values.get('f_max_qubit', None)
    dac_ss = values.get('dac_sweet_spot', dac_ss)
    period = values.get('V_per_phi0', period)
    EC = values.get('E_c', 260e6)
    coup_guess = values.get('coup_guess', 15e6)
    asymmetry = values.get('asymmetry', 0)

    f_max_qubit_vary = f_max_qubit is None
    f_max_qubit = f_max_qubit or f_res - 500e6

    model.set_param_hint('f_0_res', value=f_res, min=f_res / 2, max=2 * f_res)
    model.set_param_hint('f_max_qubit', value=f_max_qubit, min=3e9, max=8.5e9, vary=f_max_qubit_vary)
    model.set_param_hint('dac_sweet_spot', value=dac_ss, min=(dac_ss - 0.005) / 2, max=2 * (dac_ss + 0.005))
    model.set_param_hint('V_per_phi0', value=period, min=(period - 0.005) / 3, max=5 * (period + 0.005))
    model.set_param_hint('asymmetry', value=asymmetry, max=1, min=-1)
    model.set_param_hint('coupling', value=coup_guess, min=1e6, max=80e6)
    model.set_param_hint('E_c', value=EC, min=50e6, max=400e6)

    params = model.make_params()
    return params


def Qubit_dac_arch_guess(model, data, dac_voltage, values=None):
    if values is None:
        values = {}
    fmax, fmin, dac_ss, period = arc_guess(freq=data, dac=dac_voltage)
    fmax = values.get('f_max', fmax)
    dac_ss = values.get('dac_sweet_spot', dac_ss)
    period = values.get('V_per_phi0', period)
    EC = values.get('E_c', 260e6)
    asymmetry = values.get('asymmetry', 0)

    model.set_param_hint('f_max', value=fmax, min=0.7 * fmax, max=1.3 * fmax)
    model.set_param_hint('dac_sweet_spot', value=dac_ss, min=(dac_ss - 0.005) / 2, max=2 * (dac_ss + 0.005))
    model.set_param_hint('V_per_phi0', value=period, min=(period - 0.005) / 3, max=5 * (period + 0.005))
    model.set_param_hint('asymmetry', value=asymmetry, max=1, min=-1)
    model.set_param_hint('E_c', value=EC, min=50e6, max=400e6)

    params = model.make_params()
    return params


def idle_err_rate_guess(model, data, N):
    """
    Assumes exponential decay in estimating the parameters
    """
    amp_guess = 0.5
    offset = np.mean(data)
    N1 = np.mean(N)
    N2 = np.mean(N)
    params = model.make_params(A=amp_guess,
                               N1=N1,
                               N2=N2,
                               offset=offset)
    return params


def fft_freq_phase_guess(data, t):
    """
    Guess for a cosine fit using FFT, only works for evenly spaced points
    """
    # Freq guess ! only valid with uniform sampling
    # Only first half of array is used, because the second half contains the
    # negative frequecy components, and we want a positive frequency.
    w = np.fft.fft(data)[:len(data) // 2]
    f = np.fft.fftfreq(len(data), t[1] - t[0])[:len(w)]
    w[0] = 0  # Removes DC component from fourier transform

    # Use absolute value of complex valued spectrum
    abs_w = np.abs(w)
    freq_guess = abs(f[abs_w == max(abs_w)][0])
    ph_guess = 2 * np.pi - (2 * np.pi * t[data == max(data)] * freq_guess)[0]
    # the condition data == max(data) can have several solutions
    #               (for example when discretization is visible)
    # to prevent errors we pick the first solution

    return freq_guess, ph_guess


def Cos_guess(model, data, t, **kwargs):
    """
    Tip: to use this assign this guess function as a method to a model use:
    model.guess = Cos_guess.__get__(
        model, model.__class__)
    """

    amp_guess = abs(max(data) - min(data)) / 2  # amp is positive by convention
    offs_guess = np.mean(data)

    freq_guess, ph_guess = fft_freq_phase_guess(data, t)

    model.set_param_hint('period', expr='1/frequency')
    params = model.make_params(amplitude=amp_guess,
                               frequency=freq_guess,
                               phase=ph_guess,
                               offset=offs_guess)
    params['amplitude'].min = 0  # Ensures positive amp
    params['frequency'].min = 0

    return params


def exp_damp_osc_guess(model, data, t):
    """
    Makes a guess for an exponentially damped oscillation.
    Uses the fft_freq_phase guess to guess the oscillation parameters.
    The guess for the exponential is simpler as it sets the exponent (n) at 1
    and the tau at 2/3 of the total range
    """
    amp_guess = abs(max(data) - min(data)) / 2  # amp is positive by convention
    freq_guess, ph_guess = fft_freq_phase_guess(data, t)
    osc_offs_guess = 0

    tau_guess = 2 / 3 * max(t)
    exp_offs_guess = np.mean(data)
    n_guess = 1

    params = model.make_params(amplitude=amp_guess,
                               frequency=freq_guess,
                               phase=ph_guess,
                               oscillation_offset=osc_offs_guess,
                               exponential_offset=exp_offs_guess,
                               n=n_guess,
                               tau=tau_guess)
    return params


def Cos_amp_phase_guess(model, data, f, t):
    """
    Guess for a cosine fit with fixed frequency f.
    """
    amp_guess = abs(max(data) - min(data)) / 2  # amp is positive by convention
    offs_guess = np.mean(data)

    ph_guess = (-2 * np.pi * t[data == max(data)] * f)[0]
    # the condition data == max(data) can have several solutions
    #               (for example when discretization is visible)
    # to prevent errors we pick the first solution

    # model.set_param_hint('period', expr='1')
    params = model.make_params(amplitude=amp_guess,
                               phase=ph_guess,
                               offset=offs_guess)
    params['amplitude'].min = 0  # Ensures positive amp

    return params


def gauss_2D_guess(model, data, x, y):
    """
    takes the mean of every row/column and then uses the regular gauss guess
    function to get a guess for the model parameters.

    Assumptions on input data
        * input is a flattened version of a 2D grid.
        * total surface under the gaussians sums up to 1.

    Potential improvements:
        Make the input also accept a 2D grid of data to prevent reshaping.
        Find a way to extract amplitude guess from data itself, note that
        taking the sum of the data (which should correspond to all data under
                                    the curve) does not do the trick.

    Note: possibly not compatible if the model uses prefixes.
    """
    dx = x[1:]-x[:-1]
    dy = y[1:]-y[:-1]
    sums = np.sum(((data[:-1,:-1]*dx).transpose()*dy))
    amp = np.sqrt(sums)
    data_grid = data.reshape(-1, len(np.unique(x)))
    x_proj_data = np.mean(data_grid, axis=0)
    y_proj_data = np.mean(data_grid, axis=1)
    x.sort()
    y.sort()
    xm = lmfit.models.GaussianModel()
    ym = lmfit.models.GaussianModel()
    x_guess = xm.guess(data=x_proj_data, x=np.unique(x))
    x_res = xm.fit(data=x_proj_data, x=np.unique(x), params=x_guess)
    y_guess = ym.guess(data=y_proj_data, x=np.unique(y))
    y_res = ym.fit(data=y_proj_data, x=np.unique(y), params=y_guess)

    x_guess = x_res.params
    y_guess = y_res.params
    model.set_param_hint('amplitude', value=amp, min=0.9*amp, max=1.1*amp,
                         vary=True)
    model.set_param_hint('sigma_x', value=x_guess['sigma'].value, min=0,
                         vary=True)
    model.set_param_hint('sigma_y', value=y_guess['sigma'].value, min=0,
                         vary=True)
    params = model.make_params(center_x=x_guess['center'].value,
                               center_y=y_guess['center'].value,)
    return params


def double_gauss_2D_guess(model, data, x, y):
    """
    takes the mean of every row/column and then uses the guess
    function of the double gauss.

    Assumptions on input data
        * input is a flattened version of a 2D grid.
    Note: possibly not compatible if the model uses prefixes.
    Note 2: see also gauss_2D_guess() for some notes on how to improve this
            function.
    """
    data_grid = data.reshape(-1, len(np.unique(x)))
    x_proj_data = np.mean(data_grid, axis=0)
    y_proj_data = np.mean(data_grid, axis=1)

    # The syntax here is slighly different than when calling a regular guess
    # function because I do not overwrite the class attribute properly.
    x_guess = double_gauss_guess(model=None, data=x_proj_data, x=np.unique(x))
    y_guess = double_gauss_guess(model=None, data=y_proj_data, x=np.unique(y))

    if model is not None:
        pars = model.make_params(A_sigma_x=x_guess['A_sigma'],
                                 A_sigma_y=y_guess['A_sigma'],
                                 A_center_x=x_guess['A_center'],
                                 A_center_y=y_guess['A_center'],
                                 A_amplitude=1,
                                 B_sigma_x=x_guess['B_sigma'],
                                 B_sigma_y=y_guess['B_sigma'],
                                 B_center_y=y_guess['B_center'],
                                 B_center_x=x_guess['B_center'],
                                 B_amplitude=1)
        return pars
    else:
        return x_guess, y_guess


def double_gauss_guess(model, data, x=None, **kwargs):
    """
    Finds a guess for the intial parametes of the double gauss model.
    Guess is based on taking the cumulative sum of the data and
    finding the points corresponding to 25% and 75%
    it finds sigma by using the property that ~33% of the data is contained
    in the range mu-sigma to mu+sigma.

    Tip: to use this assign this guess function as a method to a model use:
        model.guess = double_gauss_guess.__get__(
            model, model.__class__)
    """
    if x is None:
        x = np.arange(len(data))
    cdf = np.cumsum(data)
    norm_cdf = cdf / cdf[-1]
    par_dict = {'A_center': x[(np.abs(norm_cdf - 0.25)).argmin()],
                'B_center': x[(np.abs(norm_cdf - 0.75)).argmin()],
                'A_sigma': (x[(np.abs(norm_cdf - 0.25 - .33 / 2)).argmin()] -
                            x[(np.abs(norm_cdf - 0.25 + .33 / 2)).argmin()]),
                'B_sigma': (x[(np.abs(norm_cdf - 0.75 - .33 / 2)).argmin()] -
                            x[(np.abs(norm_cdf - 0.75 + .33 / 2)).argmin()])}

    amp = max(data) * (par_dict['A_sigma'] + par_dict['B_sigma']) / 2.
    if model is not None:
        # Specify explicitly because not all pars are set to those from the par
        # dict
        pars = model.make_params(A_center=par_dict['A_center'],
                                 B_center=par_dict['B_center'],
                                 A_sigma=par_dict['A_sigma'],
                                 B_sigma=par_dict['B_sigma'],
                                 A_amplitude=amp, B_amplitude=amp)
        return pars
    # The else clause is added explicitly to reuse this function for the
    # 2D double gauss model
    else:
        return par_dict

def ro_double_gauss_guess(model, data, x, fixed_p01 = False, fixed_p10 = False):
    # An initial guess is done on the binned data with single gaussians
    # to constrain the fit params and avoid fitting noise if
    # e.g., mmt. ind. rel. is very low
    gmod0 = lmfit.models.GaussianModel()
    guess0 = gmod0.guess(data=data[0], x=x[0])
    gmod1 = lmfit.models.GaussianModel()
    guess1 = gmod1.guess(data=data[1], x=x[1])


    model.set_param_hint(
        'A_center', vary=True, value=guess0['center'].value,
        min=guess0['center'] - 2 * guess0['sigma'],
        max=guess0['center'] + 2 * guess0['sigma'])
    model.set_param_hint(
        'B_center', vary=True, value=guess1['center'].value,
        min=guess1['center'] - 2 * guess1['sigma'],
        max=guess1['center'] + 2 * guess1['sigma'])
    model.set_param_hint('A_sigma', value=guess0['sigma'].value, vary=True)
    model.set_param_hint('B_sigma', value=guess1['sigma'].value, vary=True)

    # Amplitudes
    intarea0 = sum_int(x=x[0], y=data[0])[-1]
    intarea1 = sum_int(x=x[1], y=data[1])[-1]
    model.set_param_hint('A_amplitude', value=intarea0, vary=False)
    model.set_param_hint('B_amplitude', value=intarea1, vary=False)
    model.set_param_hint('SNR', expr='abs(A_center-B_center)*2/(A_sigma+B_sigma)',
                         vary=False)

    # Spurious excitement
    f = np.sqrt(2*np.pi)
    amp0 = 0.99 * np.max(data[0]) * guess0['sigma'] * f
    amp1 = 0.99 * np.max(data[1]) * guess1['sigma'] * f
    spurious0 = max(1-(amp0/intarea0), 1e-3)
    spurious1 = max(1-(amp1/intarea1), 1e-3)
    p01 = fixed_p01 if fixed_p01 is not False else spurious0
    p10 = fixed_p10 if fixed_p10 is not False else spurious1
    model.set_param_hint('A_spurious', value=p01, min=0, max=1,
                         vary=fixed_p01 is False)
    model.set_param_hint('B_spurious', value=p10, min=0, max=1,
                         vary=fixed_p10 is False)

    return model.make_params()


def ChevronGuess(model, data, amp, t, J2_hint=12.5e6, detuning_swt_spt_hint=1e9):

    model.set_param_hint("J2", value=J2_hint, min=1e6, max=100e6)

    model.set_param_hint(
        "detuning_swt_spt", value=detuning_swt_spt_hint, min=1e5, max=100e9
    )

    p_neg = np.array(data)
    p_neg[amp > 0] = 0.0
    amp_center_1 = amp[np.argmax(p_neg)]
    model.set_param_hint("amp_center_1", value=amp_center_1, min=-1000, max=1000)

    p_pos = np.array(data)
    p_pos[amp < 0] = 0.0
    amp_center_2 = amp[np.argmax(p_pos)]
    model.set_param_hint("amp_center_2", value=amp_center_2, min=-1000, max=1000)

    model.set_param_hint("t", value=t, min=0.0, max=100e-6, vary=False)

    return model.make_params()


def sum_int(x, y):
    return np.cumsum(y[:-1]*(x[1:]-x[:-1]))

#################################
#     User defined Models       #
#################################

# NOTE: it is actually better to instantiate the model within your analysis
# file, this prevents the model params having a memory.
# A valid reason to define it here would beexp_dec_guess if you want to add a guess function


CosModel = lmfit.Model(CosFunc)
CosModel.guess = Cos_guess
CosModel2 = lmfit.Model(CosFunc2)
ResonatorArch = lmfit.Model(resonator_flux)
ExpDecayModel = lmfit.Model(ExpDecayFunc)
TripleExpDecayModel = lmfit.Model(TripleExpDecayFunc)
ExpDecayModel.guess = exp_dec_guess  # todo: fix
ExpDampOscModel = lmfit.Model(ExpDampOscFunc)
GaussExpDampOscModel = lmfit.Model(GaussExpDampOscFunc)
ExpDampDblOscModel = lmfit.Model(ExpDampDblOscFunc)
DoubleExpDampOscModel = lmfit.Model(DoubleExpDampOscFunc)
HangerAmplitudeModel = lmfit.Model(HangerFuncAmplitude)
SlopedHangerAmplitudeModel = lmfit.Model(SlopedHangerFuncAmplitude)
PolyBgHangerAmplitudeModel = lmfit.Model(PolyBgHangerFuncAmplitude)
HangerComplexModel = lmfit.Model(hanger_func_complex_SI)
SlopedHangerComplexModel = lmfit.Model(SlopedHangerFuncComplex)
QubitFreqDacModel = lmfit.Model(QubitFreqDac)
QubitFreqFluxModel = lmfit.Model(QubitFreqFlux)
TwinLorentzModel = lmfit.Model(TwinLorentzFunc)
LorentzianModel = lmfit.Model(Lorentzian)
RBModel = lmfit.Model(RandomizedBenchmarkingDecay)
LinOModel = lmfit.Model(linear_with_offset)
LinBGModel = lmfit.Model(linear_with_background)
LinBGOModel = lmfit.Model(linear_with_background_and_offset)

# 2D models
Gaus2D_model = lmfit.Model(gaussian_2D, independent_vars=['x', 'y'])
Gaus2D_model.guess = gauss_2D_guess  # Note: not proper way to add guess func
DoubleGauss2D_model = (lmfit.Model(gaussian_2D, independent_vars=['x', 'y'],
                                   prefix='A_') +
                       lmfit.Model(gaussian_2D, independent_vars=['x', 'y'],
                                   prefix='B_'))
DoubleGauss2D_model.guess = double_gauss_2D_guess


def mkMultiGauss2DModel(n: int, prefixes: list = None):
    """
    Generates a bivariate multi Gaussian model

    Args:
        n (int): number of bivariate Gaussians in the model
        prefixes (list): if not specified upper case letter will be used
            to prefix the parameters of each bivariate Gaussian
    """
    assert n > 0
    if prefixes is None:
        prefixes = string.ascii_uppercase
    model = lmfit.Model(
        gaussian_2D,
        independent_vars=['x', 'y'],
        prefix=prefixes[0] + '_')
    for i in range(1, n):
        model += lmfit.Model(
            gaussian_2D,
            independent_vars=['x', 'y'],
            prefix=prefixes[i] + '_')
    return model

###################################
# Models based on lmfit functions #
###################################


LorentzModel = lmfit.Model(lmfit.models.lorentzian)
Lorentz_w_background_Model = lmfit.models.LorentzianModel() + \
    lmfit.models.LinearModel()
PolyBgHangerAmplitudeModel = (HangerAmplitudeModel *
    lmfit.models.PolynomialModel(degree=7))

DoubleGaussModel = (lmfit.models.GaussianModel(prefix='A_') +
                    lmfit.models.GaussianModel(prefix='B_'))
DoubleGaussModel.guess = double_gauss_guess  # defines a guess function


def plot_fitres2D_heatmap(fit_res, x, y, axs=None, cmap='viridis'):
    """
    Convenience function for plotting results of flattened 2D fits.

    It could be argued this does not belong in fitting models (it is not a
    model) but I put it here as it is closely related to all the stuff we do
    with lmfit. If anyone has a better location in mind, let me know (MAR).
    """
    # fixing the data rotation with [::-1]
    nr_cols = len(np.unique(x))
    data_2D = fit_res.data.reshape(-1, nr_cols, order='C')[::-1]
    fit_2D = fit_res.best_fit.reshape(-1, nr_cols, order='C')[::-1]
    guess_2D = fit_res.init_fit.reshape(-1, nr_cols, order='C')[::-1]
    if axs is None:
        f, axs = plt.subplots(1, 3, figsize=(14, 6))
    axs[0].imshow(data_2D, extent=[x[0], x[-1], y[0], y[-1]],
                  cmap=cmap, vmin=np.min(data_2D), vmax=np.max(data_2D))
    axs[1].imshow(fit_2D, extent=[x[0], x[-1], y[0], y[-1]],
                  cmap=cmap, vmin=np.min(data_2D), vmax=np.max(data_2D))
    axs[2].imshow(guess_2D, extent=[x[0], x[-1], y[0], y[-1]],
                  cmap=cmap, vmin=np.min(data_2D), vmax=np.max(data_2D))
    axs[0].set_title('data')
    axs[1].set_title('fit-result')
    axs[2].set_title('initial guess')
    return axs

# Before defining a new model, take a look at the built in models in lmfit.

# From http://lmfit.github.io/lmfit-py/builtin_models.html

# Built-in Fitting Models in the models module
# Peak-like models
# GaussianModel
# LorentzianModel
# VoigtModel
# PseudoVoigtModel
# Pearson7Model
# StudentsTModel
# BreitWignerModel
# LognormalModel
# DampedOcsillatorModel
# ExponentialGaussianModel
# SkewedGaussianModel
# DonaichModel
# Linear and Polynomial Models
# ConstantModel
# LinearModel
# QuadraticModel
# ParabolicModel
# PolynomialModel
# Step-like models
# StepModel
# RectangleModel
# Exponential and Power law models
# ExponentialModel
# PowerLawModel
