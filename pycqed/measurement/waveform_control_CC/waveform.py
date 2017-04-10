'''
    File:               waveform.py
    Author:             Wouter Vlothuizen and Adriaan Rol
    Purpose:            generate waveforms for all lookuptable based AWGs
    Based on:           pulse.py, pulse_library.py
    Prerequisites:
    Usage:
    Bugs:
'''

import numpy as np
from pycqed.analysis.fitting_models import Qubit_freq_to_dac


def gauss_pulse(amp, sigma_length, nr_sigma=4, sampling_rate=2e8,
                axis='x',
                motzoi=0, delay=0):
    '''
    All inputs are in s and Hz.
    '''
    sigma = sigma_length  # old legacy naming, to be replaced
    # nr_sigma_samples = int(sigma_length * sampling_rate)
    # nr_pulse_samples = int(nr_sigma*nr_sigma_samples)
    length = sigma*nr_sigma
    mu = length/2.

    t_step = 1/sampling_rate
    t = np.arange(0, nr_sigma*sigma + .1*t_step, t_step)

    gauss_env = amp*np.exp(-(0.5 * ((t-mu)**2) / sigma**2))
    deriv_gauss_env = motzoi * -1 * (t-mu)/(sigma**1) * gauss_env
    # substract offsets
    gauss_env -= (gauss_env[0]+gauss_env[-1])/2.
    deriv_gauss_env -= (deriv_gauss_env[0]+deriv_gauss_env[-1])/2.

    delay_samples = delay*sampling_rate

    # generate pulses
    if axis == 'x':
        pulse_I = gauss_env
        pulse_Q = deriv_gauss_env
    elif axis == 'y':
        pulse_I = -1*deriv_gauss_env
        pulse_Q = gauss_env
    Zeros = np.zeros(int(delay_samples))
    pulse_I = np.array(list(Zeros)+list(pulse_I))
    pulse_Q = np.array(list(Zeros)+list(pulse_Q))
    return pulse_I, pulse_Q


def block_pulse(amp, length, sampling_rate=2e8, delay=0, phase=0):
    '''
    Generates the envelope of a block pulse.
        length in s
        amp in V
        sampling_rate in Hz
        empty delay in s
        phase in degrees
    '''
    nr_samples = (length+delay)*sampling_rate
    delay_samples = delay*sampling_rate
    pulse_samples = nr_samples - delay_samples
    amp_I = amp*np.cos(phase*2*np.pi/360)
    amp_Q = amp*np.sin(phase*2*np.pi/360)
    block_I = amp_I * np.ones(int(pulse_samples))
    block_Q = amp_Q * np.ones(int(pulse_samples))
    Zeros = np.zeros(int(delay_samples))
    pulse_I = list(Zeros)+list(block_I)
    pulse_Q = list(Zeros)+list(block_Q)
    return pulse_I, pulse_Q

####################
# Pulse modulation #
####################


def mod_pulse(pulse_I, pulse_Q, f_modulation,
              Q_phase_delay=0, sampling_rate=2e8):
    '''
    inputs are in s and Hz.
    Q_phase_delay is in degree

    transformation:
    [I_mod] = [cos(wt)            sin(wt)] [I_env]
    [Q_mod]   [-sin(wt+phi)   cos(wt+phi)] [Q_env]

    phase delay is applied to Q_mod as a whole because it is to correct a
    mixer phase offset.
    To add phase to the pulse itself edit the envelope function.
    '''
    Q_phase_delay_rad = 2*np.pi * Q_phase_delay/360.
    nr_pulse_samples = len(pulse_I)
    f_mod_samples = f_modulation/sampling_rate
    pulse_samples = np.linspace(0, nr_pulse_samples, nr_pulse_samples,
                                endpoint=False)

    pulse_I_mod = pulse_I*np.cos(2*np.pi*f_mod_samples*pulse_samples) + \
        pulse_Q*np.sin(2*np.pi*f_mod_samples*pulse_samples)
    pulse_Q_mod = pulse_I*-np.sin(2*np.pi*f_mod_samples*pulse_samples +
                                  Q_phase_delay_rad) + \
        pulse_Q*np.cos(2*np.pi*f_mod_samples*pulse_samples + Q_phase_delay_rad)

    return pulse_I_mod, pulse_Q_mod


def simple_mod_pulse(pulse_I, pulse_Q, f_modulation,
                     Q_phase_delay=0, sampling_rate=2e8):
    '''
    inputs are in s and Hz.
    Q_phase_delay is in degree

    transformation:
    [I_mod] = [cos(wt)            0] [I_env]
    [Q_mod]   [0        sin(wt+phi)] [Q_env]

    phase delay is applied to Q_mod as a whole because it is to correct a
    mixer phase offset.
    To add phase to the pulse itself edit the envelope function.
    '''
    Q_phase_delay_rad = 2*np.pi * Q_phase_delay/360.
    nr_pulse_samples = len(pulse_I)
    f_mod_samples = f_modulation/sampling_rate
    pulse_samples = np.linspace(0, nr_pulse_samples, int(nr_pulse_samples),
                                endpoint=False)

    pulse_I_mod = pulse_I*np.cos(2*np.pi*f_mod_samples*pulse_samples)
    pulse_Q_mod = pulse_Q*np.sin(2*np.pi*f_mod_samples*pulse_samples +
                                 Q_phase_delay_rad)
    return pulse_I_mod, pulse_Q_mod


def martinis_flux_pulse(length, lambda_coeffs, theta_f,
                        f_01_max,
                        g2,
                        E_c,
                        dac_flux_coefficient,
                        f_interaction=None,
                        f_bus=None,
                        asymmetry=0,
                        sampling_rate=1e9,
                        return_unit='V'):
    """
    Returns the pulse specified by Martinis and Geller
    Phys. Rev. A 90 022307 (2014).

    \theta = \theta _0 + \sum_{n=1}^\infty  (\lambda_n*(1-\cos(n*2*pi*t/t_p))/2

    note that the lambda coefficients are rescaled to ensure that the center
    of the pulse has a value corresponding to theta_f.

    length          (float)
    lambda_coeffs   (list of floats)
    theta_f         (float) final angle of the interaction. This determines the
                    Voltage for the centerpoint of the waveform.

    f_01_max        (float) qubit sweet spot frequency (Hz).
    g2              (float) coupling between 11-02 (Hz),
                            approx sqrt(2) g1 (the 10-01 coupling).
    E_c             (float) Charging energy of the transmon (Hz).
        N.B. specify either f_interaction or f_bus
    f_interaction   (float) interaction frequency (Hz).
    f_bus           (float) frequency of the bus (Hz).
    dac_flux_coefficient  (float) conversion factor for AWG voltage to flux (1/V)
    asymmetry       (float) qubit asymmetry

    sampling_rate   (float)
    return_unit     (enum: ['V', 'eps', 'f01', 'theta']) whether to return the pulse
                    expressed in units of theta: the reference frame of the
                    interaction, units of epsilon: detuning to the bus
                    eps=f12-f_bus
    """
    lambda_coeffs = np.array(lambda_coeffs)
    nr_samples = int(np.round((length)*sampling_rate))  # rounds the nr samples
    length = nr_samples/sampling_rate  # gives back the rounded length
    t_step = 1/sampling_rate
    t = np.arange(0, length, t_step)
    if f_interaction is None:
        f_interaction = f_bus + E_c
    theta_0 = np.arctan(2*g2/(f_01_max-f_interaction))
    # you can not have weaker coupling than the initial coupling
    assert(theta_f > theta_0)
    odd_coeff_lambda_sum = np.sum(lambda_coeffs[::2])
    delta_theta = theta_f - theta_0
    # add a square pulse that reaches theta_f, for this, lambda0 is used
    lambda0 = 1-lambda_coeffs[0]  # only use lambda_coeffs[0] for scaling, this
    # enables fixing the square to 0 in optimizations by setting
    # lambda_coeffs[0]=1
    th_scale_factor = delta_theta/(lambda0+odd_coeff_lambda_sum)
    mart_pulse_theta = np.ones(nr_samples)*theta_0
    mart_pulse_theta += th_scale_factor*np.ones(nr_samples)*lambda0

    for i, lambda_coeff in enumerate(lambda_coeffs):
        n = i+1
        mart_pulse_theta += th_scale_factor * \
            lambda_coeff*(1-np.cos(n*2*np.pi*t/length))/2
    # adding square pulse scaling with lambda0 satisfying the condition
    # lamb0=1-lambda1
    if return_unit == 'theta':
        return mart_pulse_theta

    # Convert theta to detuning to the bus frequency
    mart_pulse_eps = (2*g2)/(np.tan(mart_pulse_theta))
    if return_unit == 'eps':
        return mart_pulse_eps

    # pulse parameterized in the f01 frequency
    mart_pulse_f01 = mart_pulse_eps + f_interaction
    if return_unit == 'f01':
        return mart_pulse_f01
    mart_pulse_V = Qubit_freq_to_dac(
        frequency=mart_pulse_f01,
        f_max=f_01_max, E_c=E_c,
        dac_sweet_spot=0,
        dac_flux_coefficient=dac_flux_coefficient,
        asymmetry=asymmetry, branch='positive')
    return mart_pulse_V


def mod_gauss(amp, sigma_length, f_modulation, axis='x',
              nr_sigma=4,
              motzoi=0, sampling_rate=2e8,
              Q_phase_delay=0, delay=0):
    '''
    Simple gauss pulse maker for CBOX. All inputs are in s and Hz.
    '''
    pulse_I, pulse_Q = gauss_pulse(amp, sigma_length, nr_sigma=nr_sigma,
                                   sampling_rate=sampling_rate, axis=axis,
                                   motzoi=motzoi, delay=delay)
    pulse_I_mod, pulse_Q_mod = mod_pulse(pulse_I, pulse_Q, f_modulation,
                                         sampling_rate=sampling_rate,
                                         Q_phase_delay=Q_phase_delay)
    return pulse_I_mod, pulse_Q_mod


def mixer_predistortion_matrix(alpha, phi):
    predistortion_matrix = np.array(
        [[1,  np.tan(phi*2*np.pi/360)],
         [0, 1/alpha * 1/np.cos(phi*2*np.pi/360)]])
    return predistortion_matrix
