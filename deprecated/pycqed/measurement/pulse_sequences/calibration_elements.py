
from copy import deepcopy
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from ..waveform_control import sequence
import numpy as np

station = None


def cos_seq(amplitude, frequency, channels, phases,
            marker_channels=None, marker_lenght=20e-9,
            verbose=False, alphas=[1], phi_skews=[0], ):
    '''
    Cosine  sequence, plays a continuous cos on the specified channel


    Input pars:
        amplitude (float): amplitude in Vpp
        frequency (float): frequency in Hz
        channels (list[(str)]: channels on which to set a cos
        phases (list[(float)]: phases in degree

        marker_channels (list[(str)]: optionally specify markers to play

    '''
    seq_name = 'ModSquare'
    seq = sequence.Sequence(seq_name)
    el_list = []

    base_pars = {'pulse_type': 'ModSquare',
                'mod_frequency': frequency,
                'length': 2e-6,
                'amplitude': amplitude,
                'pulse_delay': 0}
    marker_pars = {'pulse_type': 'SquarePulse',
                   'length': marker_lenght,
                   'amplitude': 1,
                   'pulse_delay': 10e-9}

    pulse_list = []
    for i, phase in enumerate(phases):
        pulse = deepcopy(base_pars)
        pulse['I_channel'] = channels[i*2]
        pulse['Q_channel'] = channels[i*2+1]
        pulse['phase'] = phase
        pulse['alpha'] = alphas[i]
        pulse['phi_skew'] = phi_skews[i]
        pulse_list.append(pulse)
        # copy first element and set extra wait
    if marker_channels !=None:
        for i, marker_channel in enumerate(marker_channels):
            pulse = deepcopy(marker_pars)
            pulse['channel'] = marker_channel
            if i != 0:
                pulse['pulse_delay'] = 0
            pulse_list.append(pulse)

    el = multi_pulse_elt(0, station, pulse_list)
    el_list.append(el)
    seq.append_element(el, trigger_wait=False)
    # seq.append_element(el, trigger_wait=False)
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    return seq_name
