import logging
import numpy as np
from copy import deepcopy
try:
    from math import gcd
except:  # Moved to math in python 3.5, this is to be 3.4 compatible
    from fractions import gcd
from ..waveform_control import element
from ..waveform_control import pulse
from ..waveform_control import sequence
from pycqed.utilities.general import add_suffix_to_dict_keys
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt

from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import get_pulse_dict_from_pars
from importlib import reload
reload(pulse)
from ..waveform_control import pulse_library
reload(pulse_library)

station = None
reload(element)
kernel_dir_path = 'kernels/'
# You need to explicitly set this before running any functions from this module
# I guess there are cleaner solutions :)
cached_kernels = {}


def two_qubit_off_on(q0_pulse_pars, q1_pulse_pars, RO_pars,
                     return_seq=False, verbose=False):

    seq_name = '2_qubit_OffOn_sequence'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    # Create a dict with the parameters for all the pulses
    q0_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q0_pulse_pars), ' q0')
    q1_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q1_pulse_pars), ' q1')
    RO_dict = {'RO': RO_pars}

    pulse_dict = {}
    pulse_dict.update(q0_pulses)
    pulse_dict.update(q1_pulses)
    pulse_dict.update(RO_dict)

    # N.B. Identities not needed in all cases
    pulse_combinations = [['I q0', 'I q1', 'RO'],
                          ['X180 q0', 'I q1', 'RO'],
                          ['I q0', 'X180 q1', 'RO'],
                          ['X180 q0', 'X180 q1', 'RO']]

    for i, pulse_comb in enumerate(pulse_combinations):
        pulses = []
        for p in pulse_comb:
            pulses += [pulse_dict[p]]

        el = multi_pulse_elt(i, station, pulses)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.components['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def three_qubit_off_on(q0_pulse_pars, q1_pulse_pars, q2_pulse_pars, RO_pars,
                       return_seq=False, verbose=False):

    seq_name = '3_qubit_OffOn_sequence'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    # Create a dict with the parameters for all the pulses
    q0_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q0_pulse_pars), ' q0')
    q1_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q1_pulse_pars), ' q1')
    q2_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q2_pulse_pars), ' q2')
    RO_dict = {'RO': RO_pars}

    pulse_dict = {}
    pulse_dict.update(q0_pulses)
    pulse_dict.update(q1_pulses)
    pulse_dict.update(q2_pulses)
    pulse_dict.update(RO_dict)

    # N.B. Identities not needed in all cases
    pulse_combinations = [['I q0', 'I q1', 'I q2', 'RO'],
                          ['X180 q0', 'I q1', 'I q2', 'RO'],
                          ['I q0', 'X180 q1', 'I q2', 'RO'],
                          ['X180 q0', 'X180 q1', 'I q2', 'RO'],
                          ['I q0', 'I q1', 'X180 q2', 'RO'],
                          ['X180 q0', 'I q1', 'X180 q2', 'RO'],
                          ['I q0', 'X180 q1', 'X180 q2', 'RO'],
                          ['X180 q0', 'X180 q1', 'X180 q2', 'RO']]

    for i, pulse_comb in enumerate(pulse_combinations):
        pulses = []
        for p in pulse_comb:
            pulses += [pulse_dict[p]]

        el = multi_pulse_elt(i, station, pulses)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.components['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def four_qubit_off_on(q0_pulse_pars,
                      q1_pulse_pars,
                      q2_pulse_pars,
                      q3_pulse_pars,
                      RO_pars,
                      return_seq=False, verbose=False):

    seq_name = '4_qubit_OffOn_sequence'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    # Create a dict with the parameters for all the pulses
    q0_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q0_pulse_pars), ' q0')
    q1_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q1_pulse_pars), ' q1')
    q2_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q2_pulse_pars), ' q2')
    q3_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q3_pulse_pars), ' q3')
    RO_dict = {'RO': RO_pars}

    pulse_dict = {}
    pulse_dict.update(q0_pulses)
    pulse_dict.update(q1_pulses)
    pulse_dict.update(q2_pulses)
    pulse_dict.update(q3_pulses)
    pulse_dict.update(RO_dict)

    # N.B. Identities not needed in all cases
    pulse_combinations = [['I q0', 'I q1', 'I q2', 'I q3', 'RO'],
                          ['X180 q0', 'I q1', 'I q2', 'I q3', 'RO'],
                          ['I q0', 'X180 q1', 'I q2', 'I q3', 'RO'],
                          ['X180 q0', 'X180 q1', 'I q2', 'I q3', 'RO'],
                          ['I q0', 'I q1', 'X180 q2', 'I q3', 'RO'],
                          ['X180 q0', 'I q1', 'X180 q2', 'I q3', 'RO'],
                          ['I q0', 'X180 q1', 'X180 q2',  'I q3', 'RO'],
                          ['X180 q0', 'X180 q1', 'X180 q2',  'I q3', 'RO'],
                          ['I q0', 'I q1', 'I q2', 'X180 q3', 'RO'],
                          ['X180 q0', 'I q1', 'I q2', 'X180 q3', 'RO'],
                          ['I q0', 'X180 q1', 'I q2', 'X180 q3', 'RO'],
                          ['X180 q0', 'X180 q1', 'I q2', 'X180 q3', 'RO'],
                          ['I q0', 'I q1', 'X180 q2', 'X180 q3', 'RO'],
                          ['X180 q0', 'I q1', 'X180 q2', 'X180 q3', 'RO'],
                          ['I q0', 'X180 q1', 'X180 q2',  'X180 q3', 'RO'],
                          ['X180 q0', 'X180 q1', 'X180 q2',  'X180 q3', 'RO']]

    for i, pulse_comb in enumerate(pulse_combinations):
        pulses = []
        for p in pulse_comb:
            pulses += [pulse_dict[p]]

        el = multi_pulse_elt(i, station, pulses)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.components['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def five_qubit_off_on(q0_pulse_pars,
                      q1_pulse_pars,
                      q2_pulse_pars,
                      q3_pulse_pars,
                      q4_pulse_pars,
                      RO_pars,
                      return_seq=False, verbose=False):

    seq_name = '5_qubit_OffOn_sequence'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    # Create a dict with the parameters for all the pulses
    q0_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q0_pulse_pars), ' q0')
    q1_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q1_pulse_pars), ' q1')
    q2_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q2_pulse_pars), ' q2')
    q3_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q3_pulse_pars), ' q3')
    q4_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q4_pulse_pars), ' q4')
    RO_dict = {'RO': RO_pars}

    pulse_dict = {}
    pulse_dict.update(q0_pulses)
    pulse_dict.update(q1_pulses)
    pulse_dict.update(q2_pulses)
    pulse_dict.update(q3_pulses)
    pulse_dict.update(q4_pulses)
    pulse_dict.update(RO_dict)

    # N.B. Identities not needed in all cases
    pulse_combinations = [['I q0', 'I q1', 'I q2', 'I q3', 'I q4', 'RO'],
                          ['X180 q0', 'I q1', 'I q2', 'I q3', 'I q4', 'RO'],
                          ['I q0', 'X180 q1', 'I q2', 'I q3', 'I q4', 'RO'],
                          ['X180 q0', 'X180 q1', 'I q2', 'I q3', 'I q4', 'RO'],
                          ['I q0', 'I q1', 'X180 q2', 'I q3', 'I q4', 'RO'],
                          ['X180 q0', 'I q1', 'X180 q2', 'I q3', 'I q4', 'RO'],
                          ['I q0', 'X180 q1', 'X180 q2',
                              'I q3', 'I q4', 'RO'],
                          ['X180 q0', 'X180 q1', 'X180 q2',
                              'I q3', 'I q4', 'RO'],
                          ['I q0', 'I q1', 'I q2', 'X180 q3', 'I q4', 'RO'],
                          ['X180 q0', 'I q1', 'I q2', 'X180 q3', 'I q4', 'RO'],
                          ['I q0', 'X180 q1', 'I q2', 'X180 q3', 'I q4', 'RO'],
                          ['X180 q0', 'X180 q1', 'I q2',
                              'X180 q3', 'I q4', 'RO'],
                          ['I q0', 'I q1', 'X180 q2', 'X180 q3', 'I q4', 'RO'],
                          ['X180 q0', 'I q1', 'X180 q2',
                              'X180 q3', 'I q4', 'RO'],
                          ['I q0', 'X180 q1', 'X180 q2',
                              'X180 q3', 'I q4', 'RO'],
                          ['X180 q0', 'X180 q1', 'X180 q2',
                              'X180 q3', 'I q4', 'RO'],
                          ['I q0', 'I q1', 'I q2', 'I q3', 'X180 q4', 'RO'],
                          ['X180 q0', 'I q1', 'I q2', 'I q3', 'X180 q4', 'RO'],
                          ['I q0', 'X180 q1', 'I q2', 'I q3', 'X180 q4', 'RO'],
                          ['X180 q0', 'X180 q1', 'I q2',
                              'I q3', 'X180 q4', 'RO'],
                          ['I q0', 'I q1', 'X180 q2', 'I q3', 'X180 q4', 'RO'],
                          ['X180 q0', 'I q1', 'X180 q2',
                              'I q3', 'X180 q4', 'RO'],
                          ['I q0', 'X180 q1', 'X180 q2',
                              'I q3', 'X180 q4', 'RO'],
                          ['X180 q0', 'X180 q1', 'X180 q2',
                              'I q3', 'X180 q4', 'RO'],
                          ['I q0', 'I q1', 'I q2', 'X180 q3', 'X180 q4', 'RO'],
                          ['X180 q0', 'I q1', 'I q2',
                              'X180 q3', 'X180 q4', 'RO'],
                          ['I q0', 'X180 q1', 'I q2',
                              'X180 q3', 'X180 q4', 'RO'],
                          ['X180 q0', 'X180 q1', 'I q2',
                              'X180 q3', 'X180 q4', 'RO'],
                          ['I q0', 'I q1', 'X180 q2',
                              'X180 q3', 'X180 q4', 'RO'],
                          ['X180 q0', 'I q1', 'X180 q2',
                              'X180 q3', 'X180 q4', 'RO'],
                          ['I q0', 'X180 q1', 'X180 q2',
                              'X180 q3', 'X180 q4', 'RO'],
                          ['X180 q0', 'X180 q1', 'X180 q2',  'X180 q3', 'X180 q4', 'RO']]

    for i, pulse_comb in enumerate(pulse_combinations):
        pulses = []
        for p in pulse_comb:
            pulses += [pulse_dict[p]]

        el = multi_pulse_elt(i, station, pulses)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.components['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def two_qubit_AllXY(q0_pulse_pars, q1_pulse_pars, RO_pars,
                    double_points=True,
                    verbose=False, upload=True,
                    return_seq=False, X_mid=False, simultaneous=False):
    """
    Performs an AllXY on the first qubit will doing a pi-pulse before
    and after the AllXY on the second qubit
    AllXY q0 - RO
    X180 q1 - AllXY q0 - X180 q1 - RO
    """
    seq_name = '2_qubit_AllXY_sequence'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    # Create a dict with the parameters for all the pulses
    q0_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q0_pulse_pars), ' q0')
    q1_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q1_pulse_pars), ' q1')
    RO_dict = {'RO': RO_pars}

    pulse_dict = {}
    pulse_dict.update(q0_pulses)
    pulse_dict.update(q1_pulses)
    pulse_dict.update(RO_dict)


    AllXY_pulse_combinations = [
            ['I q0', 'I q0'], ['X180 q0', 'X180 q0'], ['Y180 q0', 'Y180 q0'],
            ['X180 q0', 'Y180 q0'], ['Y180 q0', 'X180 q0'],
            ['X90 q0', 'I q0'], ['Y90 q0', 'I q0'], ['X90 q0', 'Y90 q0'],
            ['Y90 q0', 'X90 q0'], ['X90 q0', 'Y180 q0'], ['Y90 q0', 'X180 q0'],
            ['X180 q0', 'Y90 q0'], ['Y180 q0', 'X90 q0'], ['X90 q0', 'X180 q0'],
            ['X180 q0', 'X90 q0'], ['Y90 q0', 'Y180 q0'], ['Y180 q0', 'Y90 q0'],
            ['X180 q0', 'I q0'], ['Y180 q0', 'I q0'], ['X90 q0', 'X90 q0'],
            ['Y90 q0', 'Y90 q0']]


    if double_points:
        AllXY_pulse_combinations = [val for val in AllXY_pulse_combinations
                                    for _ in (0, 1)]
    pulse_list = []
    for pulse_comb in (AllXY_pulse_combinations):
        pulse_list += [pulse_comb + ['RO']]
    for pulse_comb in (AllXY_pulse_combinations):

        if X_mid:
            if simultaneous:
                print('simultaneous!!! allxy and pi pulse gates')
                print("before", pulse_dict['X180 q1']['pulse_delay'])
                pulse_dict[pulse_comb[0]]['pulse_delay']=-20e-9
                # pulse_dict[pulse_comb[1]]['pulse_delay']=-20e-9
                pulse_dict['X180 q1']['pulse_delay']=-20
                print("after", pulse_dict['X180 q1']['pulse_delay'])
            pulse_list += [['X180 q1'] + [pulse_comb[0]] + ['X180 q1'] + [pulse_comb[1]]+ ['RO']]

        else:
            pulse_list += [['X180 q1'] + pulse_comb + ['X180 q1', 'RO']]



    for i, pulse_comb in enumerate(pulse_list):
        pulses = []
        for p in pulse_comb:
            pulses += [pulse_dict[p]]
        el = multi_pulse_elt(i, station, pulses)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    station.components['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def two_qubit_tomo_cardinal(cardinal,
                            q0_pulse_pars,
                            q1_pulse_pars,
                            RO_pars,
                            timings_dict,
                            verbose=False,
                            upload=True,
                            return_seq=False):

    seq_name = '2_qubit_Card_%d_seq' % cardinal
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    # Create a dict with the parameters for all the pulses
    q0_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q0_pulse_pars), ' q0')
    q1_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q1_pulse_pars), ' q1')
    RO_dict = {'RO': RO_pars}

    pulse_dict = {}
    pulse_dict.update(q0_pulses)
    pulse_dict.update(q1_pulses)
    pulse_dict.update(RO_dict)

    # Timings
    QQ_buffer = timings_dict['QQ_buffer']
    wait_time = timings_dict['wait_time']
    msmt_buffer = timings_dict['msmt_buffer']

    tomo_list_q0 = ['I q0', 'X180 q0', 'Y90 q0',
                    'mY90 q0', 'X90 q0', 'mX90 q0']
    tomo_list_q1 = ['I q1', 'X180 q1', 'Y90 q1',
                    'mY90 q1', 'X90 q1', 'mX90 q1']

    # inner loop on q0
    prep_idx_q0 = int(cardinal % 6)
    prep_idx_q1 = int(((cardinal - prep_idx_q0)/6) % 6)

    prep_pulse_q0 = pulse_dict[tomo_list_q0[prep_idx_q0]]
    prep_pulse_q1 = pulse_dict[tomo_list_q1[prep_idx_q1]]

    prep_pulse_q1['pulse_delay'] = QQ_buffer + (prep_pulse_q0['sigma'] *
                                                prep_pulse_q0['nr_sigma'])

    RO_pars['pulse_delay'] += msmt_buffer - (prep_pulse_q1['sigma'] *
                                             prep_pulse_q1['nr_sigma'])

    # Calibration points
    cal_points = [['I q0', 'I q1', 'RO'],
                  ['I q0', 'I q1', 'RO'],
                  ['I q0', 'I q1', 'RO'],
                  ['I q0', 'I q1', 'RO'],
                  ['I q0', 'I q1', 'RO'],
                  ['I q0', 'I q1', 'RO'],
                  ['I q0', 'I q1', 'RO'],
                  ['X180 q0', 'I q1', 'RO'],
                  ['X180 q0', 'I q1', 'RO'],
                  ['X180 q0', 'I q1', 'RO'],
                  ['X180 q0', 'I q1', 'RO'],
                  ['X180 q0', 'I q1', 'RO'],
                  ['X180 q0', 'I q1', 'RO'],
                  ['X180 q0', 'I q1', 'RO'],
                  ['I q0', 'X180 q1', 'RO'],
                  ['I q0', 'X180 q1', 'RO'],
                  ['I q0', 'X180 q1', 'RO'],
                  ['I q0', 'X180 q1', 'RO'],
                  ['I q0', 'X180 q1', 'RO'],
                  ['I q0', 'X180 q1', 'RO'],
                  ['I q0', 'X180 q1', 'RO'],
                  ['X180 q0', 'X180 q1', 'RO'],
                  ['X180 q0', 'X180 q1', 'RO'],
                  ['X180 q0', 'X180 q1', 'RO'],
                  ['X180 q0', 'X180 q1', 'RO'],
                  ['X180 q0', 'X180 q1', 'RO'],
                  ['X180 q0', 'X180 q1', 'RO'],
                  ['X180 q0', 'X180 q1', 'RO']]

    for i in range(36):
        tomo_idx_q0 = int(i % 6)
        tomo_idx_q1 = int(((i - tomo_idx_q0)/6) % 6)

        # print(i,tomo_idx_q0,tomo_idx_q1)

        tomo_pulse_q0 = pulse_dict[tomo_list_q0[tomo_idx_q0]]
        tomo_pulse_q1 = pulse_dict[tomo_list_q1[tomo_idx_q1]]

        tomo_pulse_q0['pulse_delay'] = wait_time + (prep_pulse_q1['sigma'] *
                                                    prep_pulse_q1['nr_sigma'])

        tomo_pulse_q1['pulse_delay'] = QQ_buffer + (tomo_pulse_q0['sigma'] *
                                                    tomo_pulse_q0['nr_sigma'])
        pulse_list = [prep_pulse_q0,
                      prep_pulse_q1,
                      tomo_pulse_q0,
                      tomo_pulse_q1,
                      RO_pars]
        pulse_list[0]['pulse_delay'] += 0.01e-6
        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    for i, pulse_comb in enumerate(cal_points):
        pulses = []
        for p in pulse_comb:
            pulses += [pulse_dict[p]]
        pulses[0]['pulse_delay'] += 0.01e-6

        el = multi_pulse_elt(35+i, station, pulses)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    station.components['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def two_qubit_tomo_bell(bell_state,
                        q0_pulse_pars,
                        q1_pulse_pars,
                        q0_flux_pars,
                        q1_flux_pars,
                        RO_pars,
                        distortion_dict,
                        timings_dict,
                        CPhase=True,
                        verbose=False,
                        upload=True,
                        return_seq=False):
    '''
        q0 is swap qubit
        q1 is cphase qubit
    '''

    seq_name = '2_qubit_Card_%d_seq' % bell_state
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    # Create a dict with the parameters for all the pulses
    q0_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q0_pulse_pars), ' q0')
    q1_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q1_pulse_pars), ' q1')
    # here I decide, q0 is the swap qubit, q1 is the CP qubit
    CPhase_qCP = {'CPhase q1': q1_flux_pars}# pulse incluse single qubit phase correction
    swap_qS = {'swap q0': q0_flux_pars}
    RO_dict = {'RO': RO_pars}

    pulse_dict = {}
    pulse_dict.update(q0_pulses)
    pulse_dict.update(q1_pulses)
    pulse_dict.update(CPhase_qCP)
    pulse_dict.update(swap_qS)
    pulse_dict.update(RO_dict)

    # Timings
    QQ_buffer = timings_dict['QQ_buffer']
    wait_time = timings_dict['wait_time']
    msmt_buffer = timings_dict['msmt_buffer']
    inter_swap_wait = timings_dict['inter_swap_wait']

    tomo_list_q0 = ['I q0', 'X180 q0', 'Y90 q0',
                    'mY90 q0', 'X90 q0', 'mX90 q0']
    tomo_list_q1 = ['I q1', 'X180 q1', 'Y90 q1',
                    'mY90 q1', 'X90 q1', 'mX90 q1']

    # defining pulses
    pulse_dict['mCPhase q1'] = deepcopy(pulse_dict['CPhase q1'])
    pulse_dict['mswap q0'] = deepcopy(pulse_dict['swap q0'])
    pulse_dict['mCPhase q1']['amplitude'] = -pulse_dict['CPhase q1']['amplitude']
    pulse_dict['mswap q0']['amplitude'] = -pulse_dict['swap q0']['amplitude']

    recovery_swap = deepcopy(pulse_dict['swap q0'])
    pulse_dict['phase corr q0'] = deepcopy(pulse_dict['swap q0'])
    pulse_dict['phase corr q0']['square_pulse_length'] = q0_flux_pars['phase_corr_pulse_length']
    pulse_dict['phase corr q0']['amplitude'] = q0_flux_pars['phase_corr_pulse_amp']
    pulse_dict['mphase corr q0'] = deepcopy(pulse_dict['swap q0'])
    pulse_dict['mphase corr q0']['square_pulse_length'] = q0_flux_pars['phase_corr_pulse_length']
    pulse_dict['mphase corr q0']['amplitude'] = -q0_flux_pars['phase_corr_pulse_amp']

    recovery_swap['pulse_delay'] = inter_swap_wait/2
    pulse_dict['recovery swap q0'] = recovery_swap
    pulse_dict['CPhase q1']['pulse_delay'] = inter_swap_wait/2

    # Pulse is used to set the starting refpoint for the compensation pulses
    pulse_dict.update({'dead_time_pulse':
                       {'pulse_type': 'SquarePulse',
                        'pulse_delay': q1_flux_pars['dead_time'],
                        'channel': q1_flux_pars['channel'],
                        'amplitude': 0,
                        'length': 0.}})

    # Calibration points
    cal_points = [['I q0', 'I q1', 'RO'],
                  ['I q0', 'I q1', 'RO'],
                  ['I q0', 'I q1', 'RO'],
                  ['I q0', 'I q1', 'RO'],
                  ['I q0', 'I q1', 'RO'],
                  ['I q0', 'I q1', 'RO'],
                  ['I q0', 'I q1', 'RO'],
                  ['X180 q0', 'I q1', 'RO'],
                  ['X180 q0', 'I q1', 'RO'],
                  ['X180 q0', 'I q1', 'RO'],
                  ['X180 q0', 'I q1', 'RO'],
                  ['X180 q0', 'I q1', 'RO'],
                  ['X180 q0', 'I q1', 'RO'],
                  ['X180 q0', 'I q1', 'RO'],
                  ['I q0', 'X180 q1', 'RO'],
                  ['I q0', 'X180 q1', 'RO'],
                  ['I q0', 'X180 q1', 'RO'],
                  ['I q0', 'X180 q1', 'RO'],
                  ['I q0', 'X180 q1', 'RO'],
                  ['I q0', 'X180 q1', 'RO'],
                  ['I q0', 'X180 q1', 'RO'],
                  ['X180 q0', 'X180 q1', 'RO'],
                  ['X180 q0', 'X180 q1', 'RO'],
                  ['X180 q0', 'X180 q1', 'RO'],
                  ['X180 q0', 'X180 q1', 'RO'],
                  ['X180 q0', 'X180 q1', 'RO'],
                  ['X180 q0', 'X180 q1', 'RO'],
                  ['X180 q0', 'X180 q1', 'RO']]

    if not CPhase:
        pulse_dict['CPhase q1']['amplitude'] = 0
        pulse_dict['mCPhase q1']['amplitude'] = 0
        pulse_dict['swap q0']['amplitude'] = 0
        pulse_dict['mswap q0']['amplitude'] = 0
        pulse_dict['recovery swap q0']['amplitude'] = 0
        pulse_dict['phase corr q0']['amplitude'] = 0
        pulse_dict['mCPhase q1']['phase_corr_pulse_amp'] = 0
        pulse_dict['CPhase q1']['phase_corr_pulse_amp'] = 0
        pulse_dict['mphase corr q0']['amplitude'] = 0
        after_pulse = pulse_dict['I q0']
        print('CPhase disabled')
    else:

        if bell_state == 0:
            gate1 = pulse_dict['Y90 q0']
            gate2 = pulse_dict['mY90 q1']
            after_pulse = pulse_dict['mY90 q0']
        elif bell_state == 1:
            gate1 = pulse_dict['Y90 q0']
            gate2 = pulse_dict['Y90 q1']
            after_pulse = pulse_dict['Y90 q0']
        elif bell_state == 2:
            gate1 = pulse_dict['mY90 q0']
            gate2 = pulse_dict['mY90 q1']
            after_pulse = pulse_dict['Y90 q0']
        elif bell_state == 3:
            gate1 = pulse_dict['Y90 q0']
            gate2 = pulse_dict['mY90 q1']
            after_pulse = pulse_dict['Y90 q0']


    for i in range(36):
        tomo_idx_q0 = int(i % 6)
        tomo_idx_q1 = int(((i - tomo_idx_q0)/6) % 6)

        # print(i,tomo_idx_q0,tomo_idx_q1)

        tomo_pulse_q0 = pulse_dict[tomo_list_q0[tomo_idx_q0]]
        tomo_pulse_q1 = pulse_dict[tomo_list_q1[tomo_idx_q1]]

        tomo_pulse_q0['pulse_delay'] = wait_time
        tomo_pulse_q1['pulse_delay'] = QQ_buffer + (tomo_pulse_q0['sigma'] *
                                                    tomo_pulse_q0['nr_sigma'])

        pulse_dict['swap q0']['pulse_delay'] = QQ_buffer + (gate2['sigma'] *
                                                    gate2['nr_sigma'])
        gate2['pulse_delay'] = QQ_buffer + (gate1['sigma'] * gate1['nr_sigma'])
        pulse_list = [gate2, gate1, pulse_dict['swap q0']] + \
                     [pulse_dict['CPhase q1']] + \
                     [pulse_dict['recovery swap q0'], pulse_dict['phase corr q0'], after_pulse] + \
                     [tomo_pulse_q0, tomo_pulse_q1, RO_pars] + \
                     [pulse_dict['dead_time_pulse']] + \
                     [pulse_dict['mswap q0']]*2+[pulse_dict['mCPhase q1']]+[pulse_dict['mphase corr q0']]
        # print(pulse_list)

        # pulse_list[0]['pulse_delay'] += 0.01e-6
        el = multi_pulse_elt(i, station, pulse_list)
        if distortion_dict is not None:
            el = distort_and_compensate(
                el, distortion_dict)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    for i, pulse_comb in enumerate(cal_points):
        pulses = []
        for p in pulse_comb:
            pulses += [pulse_dict[p]]
        pulses[0]['pulse_delay'] += 0.01e-6

        el = multi_pulse_elt(35+i, station, pulses)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    station.components['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name




def cphase_fringes(phases, q0_pulse_pars, q1_pulse_pars, RO_pars,
                   swap_pars_q0, cphase_pars_q1, timings_dict,
                   distortion_dict, verbose=False, upload=True, return_seq=False):
    '''
    '''
    preloaded_kernels_vec = preload_kernels_func(distortion_dict)
    original_delay = deepcopy(RO_pars)[0]['pulse_delay']
    seq_name = 'CPhase'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    # print(q0_pulse_pars)
    q0_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q0_pulse_pars[0]), ' q0')
    q1_pulses = add_suffix_to_dict_keys(
        get_pulse_dict_from_pars(q1_pulse_pars[0]), ' q1')

    pulse_dict = {}
    pulse_dict.update(q0_pulses)
    pulse_dict.update(q1_pulses)
    pulse_dict.update({'RO': RO_pars[0]})
    # print({'RO': RO_pars})

    # Timings
    buffer_mw_flux = timings_dict[0]['buffer_mw_flux']
    buffer_flux_mw = timings_dict[0]['buffer_flux_mw']
    msmt_buffer = timings_dict[0]['msmt_buffer']
    dead_time = timings_dict[0]['dead_time']
    # print(buffer_mw_flux,buffer_flux_mw,msmt_buffer,dead_time)

    # defining main pulses
    exc_pulse = deepcopy(pulse_dict['X180 q0'])
    exc_pulse['pulse_delay'] += 0.01e-6
    swap_pulse_1 = deepcopy(swap_pars_q0[0])
    # print(swap_pulse_1)
    swap_pulse_1['pulse_delay'] = buffer_mw_flux + \
        exc_pulse['sigma']*exc_pulse['nr_sigma']

    ramsey_1 = deepcopy(pulse_dict['Y90 q1'])
    ramsey_1['pulse_delay'] = buffer_flux_mw + swap_pulse_1['length']
    cphase_pulse = cphase_pars_q1[0]
    cphase_amp = cphase_pulse['amplitude']
    cphase_pulse['pulse_delay'] = buffer_mw_flux + \
        ramsey_1['sigma']*ramsey_1['nr_sigma']
    ramsey_2 = deepcopy(pulse_dict['X90 q1'])
    ramsey_2['pulse_delay'] = buffer_flux_mw + cphase_pulse['length']

    swap_pulse_2 = deepcopy(swap_pars_q0[0])
    swap_pulse_2['pulse_delay'] = buffer_mw_flux + \
        ramsey_2['sigma']*ramsey_2['nr_sigma']
    RO_pars[0]['pulse_delay'] = msmt_buffer + swap_pulse_2['length']

    # defining compensation pulses
    swap_comp_1 = deepcopy(swap_pulse_1)
    swap_pulse_1['pulse_delay'] = RO_pars[0]['length'] + dead_time
    cphase_comp = deepcopy(cphase_pulse)
    swap_comp_2 = deepcopy(swap_pulse_2)

    dead_time_pulse = {'pulse_type': 'SquarePulse',
                       'pulse_delay': RO_pars[0]['pulse_delay'],
                       'channel': swap_pars_q0[0]['channel'],
                       'amplitude': 0,
                       'length': dead_time}

    for i, ph2 in enumerate(phases[0]):
        # print(ph2)
        ramsey_2['phase'] = ph2

        cphase_pulse['amplitude'] = cphase_amp
        pulse_list = [exc_pulse,
                      swap_pulse_1,
                      ramsey_1,
                      cphase_pulse,
                      ramsey_2,
                      swap_pulse_2,
                      RO_pars[0],
                      swap_comp_1,
                      cphase_comp,
                      swap_comp_2,
                      dead_time_pulse]
        el = multi_pulse_elt(2*i, station, pulse_list)
        el_list.append(el)

        cphase_pulse['amplitude'] = 0.
        pulse_list = [exc_pulse,
                      swap_pulse_1,
                      ramsey_1,
                      cphase_pulse,
                      ramsey_2,
                      swap_pulse_2,
                      RO_pars[0],
                      swap_comp_1,
                      cphase_comp,
                      swap_comp_2,
                      dead_time_pulse]
        el = multi_pulse_elt(2*i+1, station, pulse_list)
        el_list.append(el)

    # Compensations
    for i, el in enumerate(el_list):
        if distortion_dict is not None:
            el = distort_and_compensate(
                el, distortion_dict, preloaded_kernels_vec)
            el_list[i] = el
        seq.append_element(el, trigger_wait=True)
    cal_points = 4
    RO_pars[0]['pulse_delay'] = original_delay

    # Calibration points
    cal_points = [['I q0', 'I q1', 'RO'],
                  ['I q0', 'I q1', 'RO'],
                  ['X180 q0', 'I q1', 'RO'],
                  ['X180 q0', 'I q1', 'RO'],
                  ['I q0', 'X180 q1', 'RO'],
                  ['I q0', 'X180 q1', 'RO'],
                  ['X180 q0', 'X180 q1', 'RO'],
                  ['X180 q0', 'X180 q1', 'RO']]

    for i, pulse_comb in enumerate(cal_points):
        pulses = []
        for p in pulse_comb:
            pulses += [pulse_dict[p]]
        pulses[0]['pulse_delay'] += 0.01e-6

        el = multi_pulse_elt(2*len(phases)+i, station, pulses)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    # upload
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
                if kernel in cached_kernels.keys():
                    print('Cached {}'.format(kernel_dir_path+kernel))
                    output_dict[ch].append(cached_kernels[kernel])
                else:
                    print('Loading {}'.format(kernel_dir_path+kernel))
                    # print(os.path.isfile('kernels/'+kernel))
                    kernel_vec = np.loadtxt(kernel_dir_path+kernel)
                    output_dict[ch].append(kernel_vec)
                    cached_kernels.update({kernel: kernel_vec})
    return output_dict

def distort_and_compensate(element, distortion_dict):
    """
    Distorts an element using the contenst of a distortion dictionary.
    The distortion dictionary should be formatted as follows.

    dist_dict{'ch_list': ['chx', 'chy'],
              'chx': np.array(.....),
              'chy': np.array(.....)}
    """

    t_vals, outputs_dict = element.waveforms()
    for ch in distortion_dict['ch_list']:
        element._channels[ch]['distorted'] = True
        length = len(outputs_dict[ch])
        kernelvec = distortion_dict[ch]
        outputs_dict[ch] = np.convolve(
            outputs_dict[ch], kernelvec)[:length]
        element.distorted_wfs[ch] = outputs_dict[ch][:len(t_vals)]
    return element

'''
def distort_and_compensate(element, distortion_dict, preloaded_kernels):
    t_vals, outputs_dict = element.waveforms()
    # print(len(t_vals),t_vals[-1])
    for ch in distortion_dict['ch_list']:
        element._channels[ch]['distorted'] = True
        length = len(outputs_dict[ch])
        for kernelvec in preloaded_kernels[ch]:
            outputs_dict[ch] = np.convolve(
                outputs_dict[ch], kernelvec)[:length]

        element.distorted_wfs[ch] = outputs_dict[ch][:len(t_vals)]
    return element


'''