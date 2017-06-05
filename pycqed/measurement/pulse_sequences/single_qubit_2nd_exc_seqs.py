from copy import deepcopy
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import get_pulse_dict_from_pars
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from ..waveform_control import sequence
import numpy as np

station = None


def Rabi_2nd_exc_seq(amps, pulse_pars, pulse_pars_2nd, RO_pars, n=1,
                     cal_points=True, no_cal_points=4, upload=True, return_seq=False,
                     post_msmt_delay=3e-6, verbose=False, last_ge_pulse=True):
    """
    Rabi sequence for the second excited state.
    Input pars:
        amps:            array of pulse amplitudes (V)
        pulse_pars:      dict containing the pulse parameters
        pulse_pars_2nd:  dict containing pulse_parameters for 2nd exc. state
        RO_pars:         dict containing the RO parameters
        n:               number of pulses (1 is conventional Rabi)
        post_msmt_delay: extra wait time for resetless compatibility
    """
    seq_name = 'Rabi_2nd_exc_sequence'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    pulses_2nd = get_pulse_dict_from_pars(pulse_pars_2nd)

    for i, amp in enumerate(amps):  # seq has to have at least 2 elts
        if cal_points and no_cal_points == 6 and  \
                (i == (len(amps)-6) or i == (len(amps)-5)):
                    el = multi_pulse_elt(i, station, [pulses['I'], pulses_2nd['I'], RO_pars])
        elif cal_points and no_cal_points == 6 and \
                (i == (len(amps)-4) or i == (len(amps)-3)):
                    el = multi_pulse_elt(i, station, [pulses['X180'], pulses_2nd['I'], RO_pars])
        elif cal_points and no_cal_points == 6 and \
                (i == (len(amps)-2) or i == (len(amps)-1)):
                    el = multi_pulse_elt(i, station, [pulses['X180'],
                                                      pulses_2nd['X180'],
                                                      RO_pars])
        elif cal_points and no_cal_points == 4 and \
                (i == (len(amps)-4) or i == (len(amps)-3)):
                    el = multi_pulse_elt(i, station, [pulses['I'], pulses_2nd['I'], RO_pars])
        elif cal_points and no_cal_points == 4 and \
                (i == (len(amps)-2) or i == (len(amps)-1)):
                    el = multi_pulse_elt(i, station, [pulses['X180'], pulses_2nd['I'], RO_pars])
        elif cal_points and no_cal_points == 2 and \
                (i == (len(amps)-2) or i == (len(amps)-1)):
                    el = multi_pulse_elt(i, station, [pulses['I'], pulses_2nd['I'], RO_pars])
        else:
            pulses_2nd['X180']['amplitude'] = amp

            pulse_list = [pulses['X180']]+n*[pulses_2nd['X180']]

            if last_ge_pulse:
                pulse_list += [pulses['X180']]

            pulse_list += [RO_pars]

            # copy first element and set extra wait
            pulse_list[0] = deepcopy(pulse_list[0])
            pulse_list[0]['pulse_delay'] += post_msmt_delay
            el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq_name, el_list
    else:
        return seq

def Ramsey_2nd_exc_seq(times, pulse_pars, pulse_pars_2nd, RO_pars, n=1,
                     cal_points=True, artificial_detuning =None,
                     post_msmt_delay=3e-6, verbose=False):
    '''
    Rabi sequence for the second excited state
    Input pars:
        amps:            array of pulse amplitudes (V)
        pulse_pars:      dict containing the pulse parameters
        pulse_pars_2nd:  dict containing pulse_parameters for 2nd exc. state
        RO_pars:         dict containing the RO parameters
        n:               number of pulses (1 is conventional Rabi)
        post_msmt_delay: extra wait time for resetless compatibility
    '''
    seq_name = 'Ramsey_2nd_exc_sequence'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    pulses_2nd = get_pulse_dict_from_pars(pulse_pars_2nd)
    for i, tau in enumerate(times):
        #if cal_points and (i == (len(times)-6) or
        #                   i == (len(times)-5)):
        #    el = multi_pulse_elt(i, station, [pulses['I'], RO_pars])
        if cal_points and (i == (len(times)-4) or
                             i == (len(times)-3)):
        #    el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
            el = multi_pulse_elt(i, station, [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or
                             i == (len(times)-1)):
        #    el = multi_pulse_elt(i, station, [pulses['X180'], pulses_2nd['X180'],
        #                                  RO_pars])
            el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        else:
            pulse_pars_x2 = deepcopy(pulses_2nd['X90'])
            pulse_pars_x2['pulse_delay'] = tau

            if artificial_detuning is not None:
                Dphase = ((tau-times[0]) * artificial_detuning * 360) % 360
            pulse_pars_x2['phase'] = Dphase

            pulse_list = ([pulses['X180']]+n*[pulses_2nd['X90'], pulse_pars_x2] +
                          [pulses['X180'], RO_pars])

            # copy first element and set extra wait
            pulse_list[0] = deepcopy(pulse_list[0])
            pulse_list[0]['pulse_delay'] += post_msmt_delay
            el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    return seq_name

def T1_2nd_exc_seq(times,
           pulse_pars, pulse_pars_2nd, RO_pars,
           cal_points=True,
           verbose=False, upload=True, return_seq=False):
    '''
    Rabi sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modulation used for RO
    Input pars:
        times:              array of times to wait after the 2nd excitation pi-pulse
        pulse_pars:         dict containing the pulse parameters
        pulse_pars_2nd:     dict containing the pulse parameters for ef excitation
        RO_pars:            dict containing the RO parameters
    '''
    seq_name = 'T1_2nd_exc_sequence'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    pulses_2nd = get_pulse_dict_from_pars(pulse_pars_2nd)
    pulses_x = deepcopy(pulses['X180'])

    for i, tau in enumerate(times):  # seq has to have at least 2 elts
        if cal_points and (i == (len(times)-6) or i == (len(times)-5)):
            el = multi_pulse_elt(i, station, [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-4) or i == (len(times)-3)):
            el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
            el = multi_pulse_elt(i, station, [pulses['X180'], pulses_2nd['X180'], RO_pars])
        else:
            pulses_x['pulse_delay'] = tau
            el = multi_pulse_elt(i, station, [pulses['X180'], pulses_2nd['X180'], pulses_x, RO_pars])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def SSRO_2nd_exc_state(pulse_pars, pulse_pars_2nd, RO_pars, verbose=False):

    seq_name = 'SSRO_2nd_exc'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    pulses_2nd = get_pulse_dict_from_pars(pulse_pars_2nd)
    # First element

    pulse_combinations = [[pulses['I']]+[RO_pars]]
    pulse_combinations += [[pulses['X180']] +[RO_pars]]
    pulse_combinations += [[pulses['X180']]+[pulses_2nd['X180']] +[RO_pars]]

    for i, pulse_list in enumerate(pulse_combinations):
        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    seq.append_element(el, trigger_wait=True)
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    return seq_name
