import logging
import os
import numpy as np
from copy import deepcopy
try:
    from math import gcd
except:  # Moved to math in python 3.5, this is to be 3.4 compatible
    from fractions import gcd
from ..waveform_control import pulsar
from ..waveform_control import element
from ..waveform_control.element import calculate_time_corr
from ..waveform_control import pulse
from ..waveform_control import sequence
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt

from importlib import reload
reload(pulse)
from ..waveform_control import pulse_library
reload(pulse_library)

station = None
reload(element)
kernel_dir_path = 'kernels/'
# You need to explicitly set this before running any functions from this module
# I guess there are cleaner solutions :)


def single_pulse_seq(pulse_pars=None,
                     verbose=False,
                     distortion_dict=None,
                     return_seq=False):
    '''

    '''
    if pulse_pars is None:
        pulse_pars = {'pulse_type': 'SquarePulse',
                      'pulse_delay': .1e-6,
                      'channel': 'ch3',
                      'amplitude': 0.5,
                      'length': .1e-6,
                      'dead_time_length': 10e-6}
    minus_pulse_pars = {'pulse_type': 'SquarePulse',
                  'pulse_delay': 3e-6 + pulse_pars['length'] + pulse_pars['pulse_delay'],
                  'channel': pulse_pars['channel'],
                  'amplitude': -pulse_pars['amplitude'],
                  'length': pulse_pars['length'],
                  'dead_time_length': 10e-6}

    dead_time_pulse = {'pulse_type': 'SquarePulse',
                       'pulse_delay': (minus_pulse_pars['length']),
                       'channel': pulse_pars['channel'],
                       'amplitude': 0,
                       'length': pulse_pars['dead_time_length']}
                       # 'length': 5e-6}
    seq_name = 'Square_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    for i, iter in enumerate([0, 1]):  # seq has to have at least 2 elts

        pulse_list = [pulse_pars, minus_pulse_pars,dead_time_pulse]
        # pulse_list = [pulse_pars, dead_time_pulse]

        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)

    if distortion_dict is not None:
        preloaded_kernels_vec = preload_kernels_func(distortion_dict)
    else:
        preloaded_kernels = []
    for i, el in enumerate(el_list):
        if distortion_dict is not None:
            el = distort_and_compensate(el, distortion_dict, preloaded_kernels_vec)
            el_list[i] = el
        seq.append_element(el, trigger_wait=True)
    station.components['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq


def chevron_seq_length(lengths, mw_pulse_pars, RO_pars, flux_pulse_pars=None,
                     verbose=False,
                     distortion_dict=None,
                     upload=True,
                     return_seq=False):
    '''

    '''
    preloaded_kernels_vec = preload_kernels_func(distortion_dict)
    if flux_pulse_pars is None:
        flux_pulse_pars = {'pulse_type': 'SquarePulse',
                      'pulse_delay': .1e-6,
                      'channel': 'ch3',
                      'amplitude': 0.5,
                      'length': .1e-6}
    # flux_pulse_pars['amplitude'] = 0.
    minus_flux_pulse_pars = {'pulse_type': 'SquarePulse',
                  'pulse_delay': 0., # will be overwritten
                  'channel': 'ch3',
                  'amplitude': -flux_pulse_pars['amplitude'],
                  'length': flux_pulse_pars['length']}
    original_delay = deepcopy(RO_pars)['pulse_delay']

    seq_name = 'Square_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    pulses = get_pulse_dict_from_pars(mw_pulse_pars)
    for i, lngt in enumerate(lengths):  # seq has to have at least 2 elts
        flux_pulse_pars['length'] = lngt
        minus_flux_pulse_pars['length'] = lngt
        # correcting timings
        pulse_buffer = 100e-9
        flux_pulse_pars['pulse_delay'] = pulse_buffer + (mw_pulse_pars['sigma'] *
                                                         mw_pulse_pars['nr_sigma'])
        msmt_buffer = 50e-9
        RO_pars['pulse_delay'] = msmt_buffer + lngt

        dead_time = 3e-6
        minus_flux_pulse_pars['pulse_delay'] = dead_time + RO_pars['length']
        pulse_list = [pulses['X180'], flux_pulse_pars, RO_pars, minus_flux_pulse_pars]
        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += 0.01e-6 + ((-int(lngt*1e9)) % 50)*1e-9

        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)

    for i, el in enumerate(el_list):
        if distortion_dict is not None:
            el = distort_and_compensate(el, distortion_dict, preloaded_kernels_vec)
            el_list[i] = el
        seq.append_element(el, trigger_wait=True)
    cal_points = 4
    RO_pars['pulse_delay'] = original_delay
    for i in range(int(cal_points/2)):
        pulse_list = [pulses['I'], RO_pars]
        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += 0.01e-6

        el = multi_pulse_elt(len(lengths)+i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    for i in range(int(cal_points/2)):
        pulse_list = [pulses['X180'], RO_pars]
        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += 0.01e-6

        el = multi_pulse_elt(len(lengths)+int(cal_points/2)+i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)


    if upload:
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq

def chevron_seq_amp(amps, mw_pulse_pars, RO_pars, flux_pulse_pars=None,
                     verbose=False,
                     distortion_dict=None,
                     upload=True,
                     return_seq=False):
    '''

    '''
    preloaded_kernels_vec = preload_kernels_func(distortion_dict)
    if flux_pulse_pars is None:
        flux_pulse_pars = {'pulse_type': 'SquarePulse',
                      'pulse_delay': .1e-6,
                      'channel': 'ch3',
                      'amplitude': 0.5,
                      'length': .1e-6}
    minus_flux_pulse_pars = {'pulse_type': 'SquarePulse',
                  'pulse_delay': 3e-6 + RO_pars['length'] + RO_pars['pulse_delay'],
                  'channel': 'ch3',
                  'amplitude': -flux_pulse_pars['amplitude'],
                  'length': flux_pulse_pars['length']}
    seq_name = 'Square_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    pulses = get_pulse_dict_from_pars(mw_pulse_pars)
    for i, am in enumerate(amps):  # seq has to have at least 2 elts
        flux_pulse_pars['amplitude'] = am
        minus_flux_pulse_pars['amplitude'] = am
        pulse_list = [pulses['X180'], flux_pulse_pars, RO_pars, minus_flux_pulse_pars]

        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)

    for i, el in enumerate(el_list):
        if distortion_dict is not None:
            el = distort_and_compensate(el, distortion_dict, preloaded_kernels_vec)
            el_list[i] = el
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq

def BusT1(times, mw_pulse_pars, RO_pars, flux_pulse_pars=None,
          verbose=False, distortion_dict=None,
          upload=True, return_seq=False):
    '''

    '''
    preloaded_kernels_vec = preload_kernels_func(distortion_dict)
    if flux_pulse_pars is None:
        raise ValueError('Need flux parameters for the gate.')
    minus_flux_pulse_pars = deepcopy(flux_pulse_pars)
    minus_flux_pulse_pars['amplitude']=-minus_flux_pulse_pars['amplitude']

    original_delay = deepcopy(RO_pars)['pulse_delay']

    seq_name = 'BusT1_seq'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    pulses = get_pulse_dict_from_pars(mw_pulse_pars)
    for i, tt in enumerate(times):
        # correcting timings
        pulse_buffer = 50e-9
        flux_pulse_pars['pulse_delay'] = pulse_buffer + (mw_pulse_pars['sigma'] *
                                                         mw_pulse_pars['nr_sigma'])

        flux_pulse_pars_2 = deepcopy(flux_pulse_pars)
        flux_pulse_pars_2['pulse_delay'] = tt + flux_pulse_pars['length']

        msmt_buffer = 50e-9
        RO_pars['pulse_delay'] = msmt_buffer + flux_pulse_pars['length']

        dead_time = 3e-6
        minus_flux_pulse_pars['pulse_delay'] = dead_time + RO_pars['length']

        minus_flux_pulse_pars_2 = deepcopy(flux_pulse_pars_2)
        minus_flux_pulse_pars_2['amplitude']=-minus_flux_pulse_pars_2['amplitude']

        pulse_list = [pulses['X180'], flux_pulse_pars, flux_pulse_pars_2,
                      RO_pars, minus_flux_pulse_pars, minus_flux_pulse_pars_2]

        #This ensures fixed point
        pulse_list[0]['pulse_delay'] += 0.01e-6 + ((-int(tt*1e9)) % 50)*1e-9

        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)

    for i, el in enumerate(el_list):
        if distortion_dict is not None:
            el = distort_and_compensate(el, distortion_dict, preloaded_kernels_vec)
            el_list[i] = el
        seq.append_element(el, trigger_wait=True)
    cal_points = 4
    RO_pars['pulse_delay'] = original_delay
    for i in range(int(cal_points/2)):
        pulse_list = [pulses['I'], RO_pars]
        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += 0.01e-6

        el = multi_pulse_elt(len(times)+i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    for i in range(int(cal_points/2)):
        pulse_list = [pulses['X180'], RO_pars]
        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += 0.01e-6

        el = multi_pulse_elt(len(times)+int(cal_points/2)+i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)


    if upload:
        station.components['AWG'].stop()
        station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq


def preload_kernels_func(distortion_dict):
    output_dict = {ch: [] for ch in distortion_dict['ch_list']}
    for ch in distortion_dict['ch_list']:
        for kernel in distortion_dict[ch]:
            if kernel is not '':
                print('Loading {}'.format(kernel_dir_path+kernel))
                # print(os.path.isfile('kernels/'+kernel))
                output_dict[ch].append(np.loadtxt(kernel_dir_path+kernel))
    return output_dict



def distort_and_compensate(element, distortion_dict, preloaded_kernels):
    t_vals, outputs_dict = element.waveforms()
    # print(len(t_vals),t_vals[-1])
    for ch in distortion_dict['ch_list']:
        element._channels[ch]['distorted'] = True
        length = len(outputs_dict[ch])
        for kernelvec in preloaded_kernels[ch]:
            outputs_dict[ch] = np.convolve(outputs_dict[ch], kernelvec)[:length]

        element.distorted_wfs[ch] = outputs_dict[ch][:len(t_vals)]
    return element

def get_pulse_dict_from_pars(pulse_pars):
    '''
    Returns a dictionary containing pulse_pars for all the primitive pulses
    based on a single set of pulse_pars.
    Using this function deepcopies the pulse parameters preventing accidently
    editing the input dictionary.

    input args:
        pulse_pars: dictionary containing pulse_parameters
    return:
        pulses: dictionary of pulse_pars dictionaries
    '''
    pi_amp = pulse_pars['amplitude']
    pi2_amp = pulse_pars['amplitude']*pulse_pars['amp90_scale']

    pulses = {'I': deepcopy(pulse_pars),
              'X180': deepcopy(pulse_pars),
              'mX180': deepcopy(pulse_pars),
              'X90': deepcopy(pulse_pars),
              'mX90': deepcopy(pulse_pars),
              'Y180': deepcopy(pulse_pars),
              'mY180': deepcopy(pulse_pars),
              'Y90': deepcopy(pulse_pars),
              'mY90': deepcopy(pulse_pars)}

    pulses['I']['amplitude'] = 0
    pulses['mX180']['amplitude'] = -pi_amp
    pulses['X90']['amplitude'] = pi2_amp
    pulses['mX90']['amplitude'] = -pi2_amp
    pulses['Y180']['phase'] = 90
    pulses['mY180']['phase'] = 90
    pulses['mY180']['amplitude'] = -pi_amp

    pulses['Y90']['amplitude'] = pi2_amp
    pulses['Y90']['phase'] = 90
    pulses['mY90']['amplitude'] = -pi2_amp
    pulses['mY90']['phase'] = 90

    return pulses
