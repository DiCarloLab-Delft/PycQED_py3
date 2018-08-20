'''
    File:               waveform.py
    Author:             Wouter Vlothuizen and Adriaan Rol
    Purpose:            generate waveforms for all lookuptable based AWGs
    Based on:           pulse.py, pulse_library.py
    Prerequisites:
    Usage:
    Bugs:
'''

import logging
import numpy as np
import scipy
from pycqed.analysis.fitting_models import Qubit_freq_to_dac


def gauss_pulse(amp: float, sigma_length: float, nr_sigma: int=4,
                sampling_rate: float=2e8, axis: str='x', phase: float=0,
                phase_unit: str='deg',
                motzoi: float=0, delay: float=0,
                subtract_offset: str='average'):
    '''
    All inputs are in s and Hz.
    phases are in degree.

    Args:
        amp (float):
            Amplitude of the Gaussian envelope.
        sigma_length (float):
            Sigma of the Gaussian envelope.
        nr_sigma (int):
            After how many sigma the Gaussian is cut off.
        sampling_rate (float):
            Rate at which the pulse is sampled.
        axis (str):
            Rotation axis of the pulse. If this is 'y', a 90-degree phase is
            added to the pulse, otherwise this argument is ignored.
        phase (float):
            Phase of the pulse.
        phase_unit (str):
            Unit of the phase (can be either "deg" or "rad")
        motzoi (float):
            DRAG-pulse parameter.
        delay (float):
            Delay of the pulse in s.
        subtract_offset (str):
            Instruction on how to subtract the offset in order to avoid jumps
            in the waveform due to the cut-off.
            'average': subtract the average of the first and last point.
            'first': subtract the value of the waveform at the first sample.
            'last': subtract the value of the waveform at the last sample.
            'none', None: don't subtract any offset.

    Returns:
        pulse_I, pulse_Q: Two quadratures of the waveform.
    '''
    sigma = sigma_length  # old legacy naming, to be replaced
    length = sigma*nr_sigma
    mu = length/2.

    t_step = 1/sampling_rate
    t = np.arange(0, nr_sigma*sigma + .1*t_step, t_step)

    gauss_env = amp*np.exp(-(0.5 * ((t-mu)**2) / sigma**2))
    deriv_gauss_env = motzoi * -1 * (t-mu)/(sigma**1) * gauss_env

    # Subtract offsets
    if subtract_offset.lower() == 'none' or subtract_offset is None:
        # Do not subtract offset
        pass
    elif subtract_offset.lower() == 'average':
        gauss_env -= (gauss_env[0]+gauss_env[-1])/2.
        deriv_gauss_env -= (deriv_gauss_env[0]+deriv_gauss_env[-1])/2.
    elif subtract_offset.lower() == 'first':
        gauss_env -= gauss_env[0]
        deriv_gauss_env -= deriv_gauss_env[0]
    elif subtract_offset.lower() == 'last':
        gauss_env -= gauss_env[-1]
        deriv_gauss_env -= deriv_gauss_env[-1]
    else:
        raise ValueError('Unknown value "{}" for keyword argument '
                         '"subtract_offset".'.format(subtract_offset))

    delay_samples = delay*sampling_rate

    # generate pulses
    Zeros = np.zeros(int(delay_samples))
    G = np.array(list(Zeros)+list(gauss_env))
    D = np.array(list(Zeros)+list(deriv_gauss_env))

    if axis == 'y':
        phase += 90

    pulse_I, pulse_Q = rotate_wave(G, D, phase=phase, unit=phase_unit)

    return pulse_I, pulse_Q


def single_channel_block(amp, length, sampling_rate=2e8, delay=0):
    '''
    Generates a block pulse.
        amp in V
        length in s
        sampling_rate in Hz
        empty delay in s
    '''
    nr_samples = int(np.round((length+delay)*sampling_rate))
    delay_samples = int(np.round(delay*sampling_rate))
    pulse_samples = nr_samples - delay_samples

    block = amp * np.ones(int(pulse_samples))
    Zeros = np.zeros(int(delay_samples))
    pulse = np.array(list(Zeros)+list(block))
    return pulse


def block_pulse(amp, length, sampling_rate=2e8, delay=0, phase=0):
    '''
    Generates the envelope of a block pulse for IQ modulation.
        length in s
        amp in V
        sampling_rate in Hz
        empty delay in s
        phase in degrees
    '''
    nr_samples = int(np.round((length+delay)*sampling_rate))
    delay_samples = int(np.round(delay*sampling_rate))
    pulse_samples = nr_samples - delay_samples
    amp_I = amp*np.cos(phase*2*np.pi/360)
    amp_Q = amp*np.sin(phase*2*np.pi/360)
    block_I = amp_I * np.ones(int(pulse_samples))
    block_Q = amp_Q * np.ones(int(pulse_samples))
    Zeros = np.zeros(int(delay_samples))
    pulse_I = list(Zeros)+list(block_I)
    pulse_Q = list(Zeros)+list(block_Q)
    return pulse_I, pulse_Q


def block_pulse_vsm(amp, length, sampling_rate=2e8, delay=0, phase=0):
    """
    4-channel compatible version of the vsm block pulse.
    """
    I, Q = block_pulse(amp=amp, length=length, sampling_rate=sampling_rate,
                       delay=delay, phase=phase)
    zeros = np.zeros(len(I))
    return I, zeros, zeros, Q


############################################################
# Pulse modulation  and correction functions               #
############################################################


def mod_pulse(pulse_I, pulse_Q, f_modulation: float,
              Q_phase_delay: float=0, sampling_rate: float=2e8):
    '''
    Single sideband modulation (SSB) of an input pulse_I, pulse_Q
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


def simple_mod_pulse(pulse_I, pulse_Q, f_modulation: float,
                     Q_phase_delay: float =0, sampling_rate: float=2e8):
    '''
    Double sideband modulation (DSB) of an input pulse_I, pulse_Q
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


def mixer_predistortion_matrix(alpha, phi):
    '''
    predistortion matrix correcting for a mixer with amplitude
    mismatch "mixer_alpha" and skewness "phi"

    M = [ 1            tan(phi) ]
        [ 0   1/mixer_alpha * sec(phi)]

    Notes on the procedure for acquiring this matrix can be found in
    PycQED/docs/notes/MixerSkewnessCalibration_LDC_150629.pdf
    '''
    predistortion_matrix = np.array(
        [[1,  np.tan(phi*2*np.pi/360)],
         [0, 1/alpha * 1/np.cos(phi*2*np.pi/360)]])
    return predistortion_matrix


def rotate_wave(wave_I, wave_Q, phase: float, unit: str = 'deg'):
    """
    Rotate a wave in the complex plane
        wave_I (array) : real component
        wave_Q (array) : imaginary component
        phase (float)  : desired rotation angle
        unit     (str) : either "deg" or "rad"
    returns:
        (rot_I, rot_Q) : arrays containing the rotated waves
    """
    if unit == 'deg':
        angle = np.deg2rad(phase)
    elif unit == 'rad':
        angle = angle
    else:
        raise ValueError('unit must be either "deg" or "rad"')
    rot_I = np.cos(angle)*wave_I - np.sin(angle)*wave_Q
    rot_Q = np.sin(angle)*wave_I + np.cos(angle)*wave_Q
    return rot_I, rot_Q


#####################################################
# Modulated standard waveforms
#####################################################

def mod_gauss(amp, sigma_length, f_modulation, axis='x', phase=0,
              nr_sigma=4,
              motzoi=0, sampling_rate=2e8,
              Q_phase_delay=0, delay=0):
    '''
    Simple modulated gauss pulse. All inputs are in s and Hz.
    '''
    pulse_I, pulse_Q = gauss_pulse(amp, sigma_length, nr_sigma=nr_sigma,
                                   sampling_rate=sampling_rate, axis=axis,
                                   phase=phase,
                                   motzoi=motzoi, delay=delay)
    pulse_I_mod, pulse_Q_mod = mod_pulse(pulse_I, pulse_Q, f_modulation,
                                         sampling_rate=sampling_rate,
                                         Q_phase_delay=Q_phase_delay)
    return pulse_I_mod, pulse_Q_mod


def mod_gauss_VSM(amp, sigma_length, f_modulation, axis='x', phase=0,
                  nr_sigma=4,
                  motzoi=0, sampling_rate=2e8,
                  Q_phase_delay=0, delay=0):
    '''
    4-channel VSM compatible DRAG pulse
    '''
    G, D = gauss_pulse(amp, sigma_length, nr_sigma=nr_sigma,
                       sampling_rate=sampling_rate, axis=axis,
                       phase=0,  # should always be 0
                       motzoi=motzoi, delay=delay)
    # The identity wave is used because the wave needs to be split up
    # to the VSM
    I = np.zeros(len(G))
    G_I, G_Q = rotate_wave(G, I, phase=phase, unit='deg')
    # D is in the Q quadrature because it should be 90 deg out of phase
    D_I, D_Q = rotate_wave(I, D, phase=phase, unit='deg')
    G_I_mod, G_Q_mod = mod_pulse(G_I, G_Q, f_modulation,
                                 sampling_rate=sampling_rate,
                                 Q_phase_delay=Q_phase_delay)
    D_I_mod, D_Q_mod = mod_pulse(D_I, D_Q, f_modulation,
                                 sampling_rate=sampling_rate,
                                 Q_phase_delay=Q_phase_delay)
    return G_I_mod, G_Q_mod, D_I_mod, D_Q_mod

def mod_square(amp, length, f_modulation,  phase=0,
              motzoi=0, sampling_rate=1e9):
    '''
    Simple modulated gauss pulse. All inputs are in s and Hz.
    '''
    pulse_I, pulse_Q = block_pulse(amp, length=length,
                                   sampling_rate=sampling_rate,
                                   phase=phase)
    pulse_I_mod, pulse_Q_mod = mod_pulse(pulse_I, pulse_Q, f_modulation,
                                         sampling_rate=sampling_rate)
    return pulse_I_mod, pulse_Q_mod


def mod_square_VSM(amp_G, amp_D, length, f_modulation,
                   phase_G: float = 0, phase_D: float=0,
                   sampling_rate: float=1e9):
    """
    4-channel square waveform modulated using SSB modulation.
    """
    G_I, G_Q = block_pulse(amp_G, length=length, sampling_rate=sampling_rate,
                           phase=phase_G)
    D_I, D_Q = block_pulse(amp_D, length=length, sampling_rate=sampling_rate,
                           phase=phase_D)
    G_I_mod, G_Q_mod = mod_pulse(G_I, G_Q, f_modulation,
                                 sampling_rate=sampling_rate)
    D_I_mod, D_Q_mod = mod_pulse(D_I, D_Q, f_modulation,
                                 sampling_rate=sampling_rate)
    return G_I_mod, G_Q_mod, D_I_mod, D_Q_mod


#####################################################
# Flux pulses
#####################################################


def martinis_flux_pulse(length: float, lambda_2: float, lambda_3: float,
                        theta_f: float,
                        f_01_max: float, J2: float,
                        V_offset: float=0, V_per_phi0: float=1,
                        E_c: float=250e6, f_bus: float =None,
                        f_interaction: float =None,
                        asymmetry: float =0, sampling_rate: float =1e9,
                        return_unit: str='V'):
    """
    Returns the pulse specified by Martinis and Geller
    Phys. Rev. A 90 022307 (2014).

    \theta = \theta _0 + \sum_{n=1}^\infty  (\lambda_n*(1-\cos(n*2*pi*t/t_p))/2

    note that the lambda coefficients are rescaled to ensure that the center
    of the pulse has a value corresponding to theta_f.

    length          (float) lenght of the waveform (s)
    lambda_2
    lambda_3

    theta_f         (float) final angle of the interaction in degrees.
                    Determines the Voltage for the center of the waveform.

    f_01_max        (float) qubit sweet spot frequency (Hz).
    J2              (float) coupling between 11-02 (Hz),
                    approx sqrt(2) J1 (the 10-01 coupling).
    E_c             (float) Charging energy of the transmon (Hz).
    f_bus           (float) frequency of the bus (Hz).
    f_interaction   (float) interaction frequency (Hz).
    asymmetry       (float) qubit asymmetry

    sampling_rate   (float) sampling rate of the AWG (Hz)
    return_unit     (enum: ['V', 'eps', 'f01', 'theta']) whether to return the
                    pulse expressed in units of theta: the reference frame of
                    the interaction, units of epsilon: detuning to the bus
                    eps=f12-f_bus
    """
    # Define number of samples and time points


    # Pulse is generated at a denser grid to allow for good interpolation
    # N.B. Not clear why interpolation is needed at all... -MAR July 2018
    fine_sampling_factor = 1  # 10
    nr_samples = int(np.round((length)*sampling_rate * fine_sampling_factor))
    rounded_length = nr_samples/(fine_sampling_factor * sampling_rate)
    tau_step = 1/(fine_sampling_factor * sampling_rate)  # denser points
    # tau is a virtual time/proper time
    taus = np.arange(0, rounded_length-tau_step/2, tau_step)
    # -tau_step/2 is to make sure final pt is excluded

    # Derived parameters
    if f_interaction is None:
        f_interaction = f_bus + E_c
    theta_i = np.arctan(2*J2 / (f_01_max - f_interaction))
    # Converting angle to radians as that is used under the hood
    theta_f = 2*np.pi*theta_f/360
    if theta_f < theta_i:
        raise ValueError(
            'theta_f ({:.2f} deg) < theta_i ({:.2f} deg):'.format(
                theta_f/(2*np.pi)*360, theta_i/(2*np.pi)*360)
            + 'final coupling weaker than initial coupling')

    # lambda_1 is scaled such that the final ("center") angle is theta_f
    lambda_1 = (theta_f - theta_i) / (2 + 2 * lambda_3)

    # Calculate the wave
    theta_wave = np.ones(nr_samples) * theta_i
    theta_wave += lambda_1 * (1 - np.cos(2 * np.pi * taus / rounded_length))
    theta_wave += (lambda_1 * lambda_2 *
                   (1 - np.cos(4 * np.pi * taus / rounded_length)))
    theta_wave += (lambda_1 * lambda_3 *
                   (1 - np.cos(6 * np.pi * taus / rounded_length)))

    # Clip wave to [theta_i, pi] to avoid poles in the wave expressed in freq
    theta_wave_clipped = np.clip(theta_wave, theta_i, np.pi-.01)
    if not np.array_equal(theta_wave, theta_wave_clipped):
        logging.warning(
            'Martinis flux wave form has been clipped to [{}, 180 deg]'
            .format(theta_i))

    # Transform from proper time to real time
    t = np.array([np.trapz(np.sin(theta_wave)[:i+1], dx=1/(10*sampling_rate))
                  for i in range(len(theta_wave))])

    # Interpolate pulse at physical sampling distance
    t_samples = np.arange(0, length, 1/sampling_rate)
    # Scaling factor for time-axis to get correct pulse length again
    scale = t[-1]/t_samples[-1]
    interp_wave = scipy.interpolate.interp1d(
        t/scale, theta_wave_clipped, bounds_error=False,
        fill_value='extrapolate')(t_samples)

    # Return in the specified units
    if return_unit == 'theta':
        # Theta is returned in radians here
        return np.nan_to_num(interp_wave)

    # Convert to detuning from f_interaction
    delta_f_wave = 2 * J2 / np.tan(interp_wave)
    if return_unit == 'eps':
        return np.nan_to_num(delta_f_wave)

    # Convert to parametrization of f_01
    f_01_wave = delta_f_wave + f_interaction
    if return_unit == 'f01':
        return np.nan_to_num(f_01_wave)

    # Convert to voltage
    voltage_wave = Qubit_freq_to_dac(
        frequency=f_01_wave,
        f_max=f_01_max,
        E_c=E_c,
        dac_sweet_spot=V_offset,
        V_per_phi0=V_per_phi0,
        asymmetry=asymmetry,
        branch='positive')

    # why sometimes the last sample is nan is not known,
    # but we will surely figure it out someday.
    # (Brian and Adriaan, 14.11.2017)
    # This may be caused by the fill_value of the interp_wave (~30 lines up)
    # that was set to 0 instead of extrapolate. This caused
    # the np.tan(interp_wave) to divide by zero. (MAR 10-05-2018)
    voltage_wave = np.nan_to_num(voltage_wave)

    return voltage_wave
############################################################################
#
############################################################################
