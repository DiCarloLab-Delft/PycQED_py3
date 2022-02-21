from copy import deepcopy
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from ..waveform_control import sequence
import numpy as np

station = None


def Pulsed_spec_seq(RO_pars,
                    spec_marker_channels,
                    spec_pulse_length=1e-6,
                    marker_interval=4e-6,
                    verbose=False,
                    trigger_wait=True):
    seq_name = 'Pulsed_spec_with_RF_mod'
    seq = sequence.Sequence(seq_name)
    RO_pars = deepcopy(RO_pars)
    spec_pulse_start = (
        marker_interval - (RO_pars['pulse_delay'] + RO_pars['length']))
    print(spec_pulse_start)
    RO_pars['pulse_delay'] += spec_pulse_length
    if spec_pulse_start < 0:
        err_str = ('marker_interval {} must'.format(marker_interval) +
                   'be larger than length of RO element')
        raise ValueError(err_str)

    marker_pars = {'pulse_type': 'SquarePulse',
                   'length': spec_pulse_length,
                   'amplitude': 1,
                   'channels': spec_marker_channels,
                   'pulse_delay': spec_pulse_start}
    print(spec_pulse_length)

    el_list = []
    for i in range(2):
        el = multi_pulse_elt(i, station, [marker_pars, RO_pars])
        el_list.append(el)
        seq.append_element(el, trigger_wait=trigger_wait)
        # Ensures a continuously running sequence
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
