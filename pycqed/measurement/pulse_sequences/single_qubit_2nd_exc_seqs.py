import logging
log = logging.getLogger(__name__)
from copy import deepcopy
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import get_pulse_dict_from_pars
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from ..waveform_control import sequence
import numpy as np

station = None

def echo_2nd_exc_seq(times, pulse_pars, pulse_pars_2nd, RO_pars,
                     cal_points=True, no_cal_points=6, artificial_detuning=None,
                     verbose=False, upload=True, return_seq=False,
                     last_ge_pulse=True):

    seq_name = 'Echo_2nd_exc_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    pulses_2nd = get_pulse_dict_from_pars(pulse_pars_2nd)

    center_X180ef = deepcopy(pulses_2nd['X180'])
    final_X90ef = deepcopy(pulses_2nd['X90'])
    center_X180ef['ref_point'] = 'start'
    final_X90ef['ref_point'] = 'start'

    for i, tau in enumerate(times):
        center_X180ef['pulse_delay'] = tau/2
        final_X90ef['pulse_delay'] = tau/2
        if artificial_detuning is not None:
            final_X90ef['phase'] = (tau-times[0]) * artificial_detuning * 360

        if cal_points and no_cal_points == 6 and \
                (i == (len(times)-6) or i == (len(times)-5)):
            el = multi_pulse_elt(i, station, [pulses['I'], pulses_2nd['I'],
                                              RO_pars])
        elif cal_points and no_cal_points == 6 and \
                (i == (len(times)-4) or i == (len(times)-3)):
            el = multi_pulse_elt(i, station, [pulses['X180'], pulses_2nd['I'],
                                              RO_pars])
        elif cal_points and no_cal_points == 6 and \
                (i == (len(times)-2) or i == (len(times)-1)):
            el = multi_pulse_elt(i, station, [pulses['X180'],
                                              pulses_2nd['X180'],
                                              RO_pars])
        elif cal_points and no_cal_points == 4 and \
                (i == (len(times)-4) or i == (len(times)-3)):
            el = multi_pulse_elt(i, station, [pulses['I'], pulses_2nd['I'],
                                              RO_pars])
        elif cal_points and no_cal_points == 4 and \
                (i == (len(times)-2) or i == (len(times)-1)):
            el = multi_pulse_elt(i, station, [pulses['X180'], pulses_2nd['I'],
                                              RO_pars])
        elif cal_points and no_cal_points == 2 and \
                (i == (len(times)-2) or i == (len(times)-1)):
            el = multi_pulse_elt(i, station, [pulses['I'], pulses_2nd['I'],
                                              RO_pars])
        else:
            pulse_list = [pulses['X180'], pulses_2nd['X90'],
                          center_X180ef, final_X90ef]
            if last_ge_pulse:
                pulse_list += [pulses['X180']]
            pulse_list += [RO_pars]
            el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
            return seq_name, el_list
    else:
        return seq


