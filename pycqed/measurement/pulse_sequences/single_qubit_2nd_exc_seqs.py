import logging
log = logging.getLogger(__name__)
from copy import deepcopy
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import get_pulse_dict_from_pars
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from ..waveform_control import sequence
import numpy as np

station = None

def rabi_2nd_exc_seq(amps, pulse_pars, pulse_pars_2nd, RO_pars, n=1,
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
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    pulses_2nd = get_pulse_dict_from_pars(pulse_pars_2nd)
    for i, amp in enumerate(amps):  # seq has to have at least 2 elts
        if cal_points and no_cal_points == 6 and  \
                (i == (len(amps)-6) or i == (len(amps)-5)):
                    el = multi_pulse_elt(i, station, [pulses['I'],
                                                      pulses_2nd['I'], RO_pars])
        elif cal_points and no_cal_points == 6 and \
                (i == (len(amps)-4) or i == (len(amps)-3)):
                    el = multi_pulse_elt(i, station, [pulses['X180'],
                                                      pulses_2nd['I'], RO_pars])
        elif cal_points and no_cal_points == 6 and \
                (i == (len(amps)-2) or i == (len(amps)-1)):
                    el = multi_pulse_elt(i, station, [pulses['X180'],
                                                      pulses_2nd['X180'],
                                                      RO_pars])
        elif cal_points and no_cal_points == 4 and \
                (i == (len(amps)-4) or i == (len(amps)-3)):
                    el = multi_pulse_elt(i, station, [pulses['I'],
                                                      pulses_2nd['I'], RO_pars])
        elif cal_points and no_cal_points == 4 and \
                (i == (len(amps)-2) or i == (len(amps)-1)):
                    el = multi_pulse_elt(i, station, [pulses['X180'],
                                                      pulses_2nd['I'], RO_pars])
        elif cal_points and no_cal_points == 2 and \
                (i == (len(amps)-2) or i == (len(amps)-1)):
                    el = multi_pulse_elt(i, station, [pulses['I'],
                                                      pulses_2nd['I'], RO_pars])
        else:
            pulses_2nd_temp = deepcopy(pulses_2nd)
            pulses_2nd_temp['X180']['amplitude'] = amp
            pulse_list = [pulses['X180']]+n*[pulses_2nd_temp['X180']]

            # # ge rabi
            # pulses_temp = deepcopy(pulses)
            # pulses_temp['X180']['amplitude'] = amp
            # pulse_list = [pulses_temp['X180']]

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

def ramsey_2nd_exc_seq(times, pulse_pars, pulse_pars_2nd, RO_pars, n=1,
                     cal_points=True, no_cal_points=6, artificial_detuning=None,
                     post_msmt_delay=3e-6, verbose=False,
                     upload=True, return_seq=False, last_ge_pulse=True):
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
    if np.any(times>1e-3):
        logging.warning('The values in the times array might be too large.'
                        'The units should be seconds.')

    seq_name = 'Ramsey_2nd_exc_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    pulses_2nd = get_pulse_dict_from_pars(pulse_pars_2nd)
    for i, tau in enumerate(times):
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
            pulse_pars_x2 = deepcopy(pulses_2nd['X90'])
            pulse_pars_x2['pulse_delay'] = tau

            if artificial_detuning is not None:
                Dphase = ((tau-times[0]) * artificial_detuning * 360) % 360
            pulse_pars_x2['phase'] = Dphase

            pulse_list = ([pulses['X180']]+n*[pulses_2nd['X90'], pulse_pars_x2])
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

def ramsey_2nd_exc_seq_multiple_detunings(times, qb_name, operation_dict,
                                          cal_points, n=1,
                                          artificial_detunings=None,
                                          upload=True,
                                          preparation_type='wait',
                                          post_ro_wait=1e-6, reset_reps=1,
                                          final_reset_pulse=True, for_ef=False,
                                          last_ge_pulse=False):
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
    if np.any(times>1e-3):
        logging.warning('The values in the times array might be too large.'
                        'The units should be seconds.')
    if np.any(np.asarray(artificial_detunings)<1e3):
        logging.warning('The artificial detuning is too small. The units '
                        'should be Hz.')
    seq_name = 'Ramsey_2nd_exc_sequence_mult_det'
    prep_params = dict(preparation_type=preparation_type,
                       post_ro_wait=post_ro_wait,
                       repetitions=reset_reps,
                       final_reset_pulse=final_reset_pulse)
    station.pulsar.update_channel_settings()

    # Operations
    if for_ef:
        ramsey_ops = ["X180"] + ["X90_ef"] * 2 * n
        if last_ge_pulse:
            ramsey_ops += ["X180"]
    else:
        ramsey_ops = ["X90"] * 2 * n

    ramsey_ops += ["RO"]
    ramsey_ops = add_qb_name(ramsey_ops, qb_name)

    #pulses
    ramsey_pulses = [deepcopy(operation_dict[op]) for op in ramsey_ops]

    #name swept pulse:
    for i in range(n):
        idx = (2 if for_ef else 1 ) + i * 2
        ramsey_pulses[idx]["name"] = f"Ramsey_x2_{i}"

    #compute dephasing
    a_d = artificial_detunings
    dephasing = [((t-times[0]) * a_d[i % len(a_d)] * 360) % 360
                 for i, t in enumerate(times)]
    #sweep pulses
    params = {f'Ramsey_x2_{i}.tau': times for i in range(n)}
    params.update({f'Ramsey_x2_{i}.phase': dephasing for i in range(n)})

    swept_pulses = sweep_pulse_params(ramsey_pulses, params)
    seq = pulse_list_list_seq(swept_pulses_with_prep, seq_name, upload=False)

    # add calibration segments
    seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(len(seq.segments))

def qscale_2nd_exc_seq(qscales, pulse_pars,  pulse_pars_2nd, RO_pars,
           cal_points=True, verbose=False, upload=True, return_seq=False,
           last_ge_pulse=True, no_cal_points=6):
    '''
    Sequence used for calibrating the QScale factor used in the DRAG pulses.
    Applies X(pi/2)X(pi), X(pi/2)Y(pi), X(pi/2)Y(-pi) for each value of
    QScale factor.

    Beware that the elements alternate, in order to perform these 3
    measurements per QScale factor, the qscales sweep values must be
    repeated 3 times. This was chosen to be more easily compatible with
    standard detector functions and sweep pts.

    Input pars:
        qscales:             array of qscale factors
        pulse_pars:          dict containing the DRAG pulse parameters
        RO_pars:             dict containing the RO parameters
        cal_points:          if True, replaces the last 3*4 segments with
                             calibration points
    '''
    seq_name = 'QScale_2nd_exc_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulse_combinations=[['X90','X180'],['X90','Y180'],['X90','mY180']]
    pulses = get_pulse_dict_from_pars(pulse_pars)
    pulses_2nd = get_pulse_dict_from_pars(pulse_pars_2nd)
    for i, motzoi in enumerate(qscales):
        pulse_keys = pulse_combinations[i % 3]
        for p_name in ['X180', 'Y180', 'X90', 'mY180']:
            pulses_2nd[p_name]['motzoi'] = motzoi
        # if cal_points and (i == (len(times)-6) or
        #                   i == (len(times)-5)):
        #    el = multi_pulse_elt(i, station, [pulses['I'], RO_pars])
        # elif cal_points and (i == (len(qscales)-4) or
        #                            i == (len(qscales)-3)):
        #     el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        # elif cal_points and (i == (len(qscales)-2) or
        #                              i == (len(qscales)-1)):
        #     # pick motzoi for calpoint in the middle of the range
        #     pulses['X180']['motzoi'] = np.mean(qscales)
        #     el = multi_pulse_elt(i, station, [pulses_2nd['X180_ef'], RO_pars])
        if cal_points and no_cal_points == 6 and \
                (i == (len(qscales)-6) or i == (len(qscales)-5)):
            el = multi_pulse_elt(i, station, [pulses['I'], pulses_2nd['I'], RO_pars])
        elif cal_points and no_cal_points == 6 and \
                (i == (len(qscales)-4) or i == (len(qscales)-3)):
            el = multi_pulse_elt(i, station, [pulses['X180'], pulses_2nd['I'], RO_pars])
        elif cal_points and no_cal_points == 6 and \
                (i == (len(qscales)-2) or i == (len(qscales)-1)):
            pulses_2nd['X180']['motzoi'] = np.mean(qscales)
            el = multi_pulse_elt(i, station, [pulses['X180'],
                                              pulses_2nd['X180'],
                                              RO_pars])
        elif cal_points and no_cal_points == 4 and \
                (i == (len(qscales)-4) or i == (len(qscales)-3)):
            el = multi_pulse_elt(i, station, [pulses['I'], pulses_2nd['I'], RO_pars])
        elif cal_points and no_cal_points == 4 and \
                (i == (len(qscales)-2) or i == (len(qscales)-1)):
            el = multi_pulse_elt(i, station, [pulses['X180'], pulses_2nd['I'], RO_pars])
        elif cal_points and no_cal_points == 2 and \
                (i == (len(qscales)-2) or i == (len(qscales)-1)):
            el = multi_pulse_elt(i, station, [pulses['I'], pulses_2nd['I'], RO_pars])
        else:
            pulse_list = [pulses['X180']]
            pulse_list += [pulses_2nd[x] for x in pulse_keys]
            if last_ge_pulse:
                pulse_list += [pulses['X180']]
            pulse_list += [RO_pars]
            el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def T1_2nd_exc_seq(times,
           pulse_pars, pulse_pars_2nd, RO_pars,
           cal_points=True,
           no_cal_points=6,
           verbose=False, upload=True,
           return_seq=False,
           last_ge_pulse=True):
    '''
    Rabi sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modulation used for RO

    Args:
        times:
            array of times to wait after the 2nd excitation pi-pulse
        pulse_pars:
            dict containing the pulse parameters
        pulse_pars_2nd:
            dict containing the pulse parameters for ef excitation
        RO_pars:
            dict containing the RO parameters
    '''
    if np.any(times>1e-3):
        logging.warning('The values in the times array might be too large.'
                        'The units should be seconds.')
    seq_name = 'T1_2nd_exc_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    pulses_2nd = get_pulse_dict_from_pars(pulse_pars_2nd)
    pulses_x = deepcopy(pulses['X180'])
    RO_pulse_delay = RO_pars['pulse_delay']
    RO_pars = deepcopy(RO_pars)  # Prevents overwriting of the dict

    for i, tau in enumerate(times):  # seq has to have at least 2 elts
        if cal_points and no_cal_points == 6 and \
                (i == (len(times)-6) or i == (len(times)-5)):
                    el = multi_pulse_elt(i, station, [pulses['I'],
                                                      pulses_2nd['I'], RO_pars])
        elif cal_points and no_cal_points == 6 and \
                (i == (len(times)-4) or i == (len(times)-3)):
                    RO_pars['pulse_delay'] = RO_pulse_delay
                    el = multi_pulse_elt(i, station, [pulses['X180'],
                                                      pulses_2nd['I'], RO_pars])
        elif cal_points and no_cal_points == 6 and \
                (i == (len(times)-2) or i == (len(times)-1)):
                    RO_pars['pulse_delay'] = RO_pulse_delay
                    el = multi_pulse_elt(i, station, [pulses['X180'],
                                                      pulses_2nd['X180'],
                                                      RO_pars])
        elif cal_points and no_cal_points == 4 and \
                (i == (len(times)-4) or i == (len(times)-3)):
                    el = multi_pulse_elt(i, station, [pulses['I'],
                                                      pulses_2nd['I'], RO_pars])
        elif cal_points and no_cal_points == 4 and  \
                (i == (len(times)-2) or i == (len(times)-1)):
                    RO_pars['pulse_delay'] = RO_pulse_delay
                    el = multi_pulse_elt(i, station, [pulses['X180'],
                                                      pulses_2nd['I'], RO_pars])
        elif cal_points and no_cal_points == 2 and \
                (i == (len(times)-2) or i == (len(times)-1)):
                    el = multi_pulse_elt(i, station, [pulses['I'],
                                                      pulses_2nd['I'], RO_pars])
        else:
            pulse_list = [pulses['X180']]+[pulses_2nd['X180']]
            if last_ge_pulse:
                pulses_x['pulse_delay'] = tau
                pulse_list += [pulses_x]
            else:
                RO_pars['pulse_delay'] = RO_pulse_delay + tau
            pulse_list += [RO_pars]
            el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


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


