'''
    File:               waveform.py
    Author:             Wouter Vlothuizen and Adriaan Rol
    Purpose:            generate waveforms for all lookuptable based AWGs
    Based on:           pulse.py, pulse_library.py
    Prerequisites:
    Usage:
    Bugs:

Contains most basic waveforms, basic means having a few parameters and a
straightforward translation to AWG amplitude, i.e., no knowledge of qubit
parameters.

Examples of waveforms that are too advanced are flux pulses that require
knowledge of the flux sensitivity and interaction strengths and qubit
frequencies. See e.g., "waveform_control_CC/waveforms_flux.py".
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

    t_step = 1/sampling_rate
    mu = length/2. - 0.5*t_step  # center should be offset by half a sample
    t = np.arange(0, nr_sigma*sigma, t_step)

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
    nr_samples = int(np.round((length + delay) * sampling_rate))
    delay_samples = int(np.round(delay * sampling_rate))
    pulse_samples = nr_samples - delay_samples

    block = amp * np.ones(int(pulse_samples))
    Zeros = np.zeros(int(delay_samples))
    pulse = np.array(list(Zeros) + list(block))
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
    pulse_I = np.array(list(Zeros)+list(block_I))
    pulse_Q = np.array(list(Zeros)+list(block_Q))
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
