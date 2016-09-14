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
from measurement.pulse_sequences.standard_elements import multi_pulse_elt

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
                      'dead_time_length': 3e-6}
    dead_time_pulse = {'pulse_type': 'SquarePulse',
                       'pulse_delay': (pulse_pars['length'] +
                                       pulse_pars['pulse_delay']),
                       'channel': 'ch3',
                       'amplitude': 0,
                       'length': pulse_pars['dead_time_length']}
    seq_name = 'Square_seq'
    seq = sequence.Sequence(seq_name)
    el_list = []
    for i, iter in enumerate([0, 1]):  # seq has to have at least 2 elts

        pulse_list = [pulse_pars, dead_time_pulse]

        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)

    for i, el in enumerate(el_list):
        if distortion_dict is not None:
            el = distort(el, distortion_dict)
            el_list[i] = el
        seq.append_element(el, trigger_wait=True)
    station.components['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq


def distort(element, distortion_dict):
    t_vals, outputs_dict = element.waveforms()
    for ch in distortion_dict['ch_list']:
        element._channels[ch]['distorted'] = True
        length = len(outputs_dict[ch])
        # kernels commute, since they are convolutions!
        for kernel in distortion_dict[ch]:
            print('Trying to open {}'.format(kernel_dir_path+kernel))
            # print(os.path.isfile('kernels/'+kernel))
            kernelvec = np.loadtxt(kernel_dir_path+kernel)
            outputs_dict[ch] = np.convolve(outputs_dict[ch], kernelvec)[:length]
        element.distorted_wfs[ch] = outputs_dict[ch]
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
