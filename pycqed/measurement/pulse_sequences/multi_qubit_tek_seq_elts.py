import logging
import itertools
import numpy as np
from copy import deepcopy
from pycqed.measurement.waveform_control import element
from pycqed.measurement.waveform_control import pulse
from pycqed.measurement.waveform_control import sequence
from pycqed.utilities.general import add_suffix_to_dict_keys
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from pycqed.measurement.pulse_sequences.standard_elements import distort_and_compensate

from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import get_pulse_dict_from_pars
from importlib import reload
reload(pulse)
from ..waveform_control import pulse_library
reload(pulse_library)

station = None
reload(element)
kernel_dir = 'kernels/'
# You need to explicitly set this before running any functions from this module
# I guess there are cleaner solutions :)
cached_kernels = {}


def avoided_crossing_spec_seq(operation_dict, q0, q1, RO_target,
                              verbose=False,
                              upload=True):

    seq_name = 'avoidec_crossing_spec'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    sequencer_config = operation_dict['sequencer_config']

    # N.B. Identities not needed in all cases
    pulse_combinations = ['X180 '+q0, 'SpecPulse '+q1, 'RO '+RO_target]
    pulses = []
    for p in pulse_combinations:
        pulses += [operation_dict[p]]
    el = multi_pulse_elt(0, station, pulses, sequencer_config)
    el_list.append(el)
    seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    return seq, el_list


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
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
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
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
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
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
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
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def two_qubit_AllXY(operation_dict, q0='q0', q1='q1', RO_target='all',
                    sequence_type='simultaneous',
                    replace_q1_pulses_X180=False,
                    double_points=True,
                    verbose=False, upload=True,
                    return_seq=False):
    """
    Performs an AllXY sequence on two qubits.
    Has the option of replacing pulses on q1 with pi pulses

    Args:
        operation_dict (dict) : dictionary containing all pulse parameters
        q0, q1         (str) : target qubits for the sequence
        RO_target      (str) : target for the RO, can be a qubit name or 'all'
        sequence_type  (str) : sequential | interleaved | simultaneous | sandwiched
                              q0|q0|q1|q1   q0|q1|q0|q1   q01|q01      q1|q0|q0|q1
            describes the order of the AllXY pulses
        replace_q1_pulses_X180 (bool) : if True replaces all pulses on q1 with
            X180 pulses.

        double_points (bool) : if True measures each point in the AllXY twice
        verbose       (bool) : verbose sequence generation
        upload        (bool) :
    """
    seq_name = 'two_qubit_AllXY_{}_{}'.format(q0, q1)
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    sequencer_config = operation_dict['sequencer_config']

    AllXY_pulse_combinations = [['I ', 'I '], ['X180 ', 'X180 '], ['Y180 ', 'Y180 '],
                                ['X180 ', 'Y180 '], ['Y180 ', 'X180 '],
                                ['X90 ', 'I '], ['Y90 ', 'I '], [
                                    'X90 ', 'Y90 '],
                                ['Y90 ', 'X90 '], ['X90 ', 'Y180 '], [
                                    'Y90 ', 'X180 '],
                                ['X180 ', 'Y90 '], ['Y180 ', 'X90 '], [
                                    'X90 ', 'X180 '],
                                ['X180 ', 'X90 '], ['Y90 ', 'Y180 '], [
                                    'Y180 ', 'Y90 '],
                                ['X180 ', 'I '], ['Y180 ', 'I '], [
                                    'X90 ', 'X90 '],
                                ['Y90 ', 'Y90 ']]
    if double_points:
        AllXY_pulse_combinations = [val for val in AllXY_pulse_combinations
                                    for _ in (0, 1)]

    if sequence_type == 'simultaneous':
        operation_dict = deepcopy(operation_dict)  # prevents overwriting of dict
        for key in operation_dict.keys():
            if q1 in key:
                operation_dict[key]['refpoint'] = 'start'
                operation_dict[key]['pulse_delay'] = 0

    pulse_list = []
    if not replace_q1_pulses_X180:
        for pulse_comb in AllXY_pulse_combinations:
            if sequence_type == 'interleaved' or sequence_type == 'simultaneous':
                pulse_list += [[pulse_comb[0] + q0] + [pulse_comb[0] + q1] +
                               [pulse_comb[1] + q0] + [pulse_comb[1] + q1] +
                               ['RO ' + RO_target]]
            elif sequence_type == 'sequential':
                pulse_list += [[pulse_comb[0] + q0] + [pulse_comb[1] + q0] +
                               [pulse_comb[0] + q1] + [pulse_comb[1] + q1] +
                               ['RO ' + RO_target]]
            elif sequence_type == 'sandwiched':
                pulse_list += [[pulse_comb[0] + q1] + [pulse_comb[0] + q0] +
                               [pulse_comb[1] + q0] + [pulse_comb[1] + q1] +
                               ['RO ' + RO_target]]
            else:
                raise ValueError("sequence_type {} must be in".format(sequence_type) +
                                 " ['interleaved', simultaneous', 'sequential', 'sandwiched']")
    else:
        for pulse_comb in AllXY_pulse_combinations:
            if sequence_type == 'interleaved' or sequence_type == 'simultaneous':
                pulse_list += [[pulse_comb[0] + q0] + ['X180 ' + q1] +
                               [pulse_comb[1] + q0] + ['X180 ' + q1] +
                               ['RO ' + RO_target]]
            elif sequence_type == 'sequential':
                pulse_list += [[pulse_comb[0] + q0] + [pulse_comb[1] + q0] +
                               ['X180 ' + q1] + ['X180 ' + q1] +
                               ['RO ' + RO_target]]
            elif sequence_type == 'sandwiched':
                pulse_list += [['X180 ' + q1] + [pulse_comb[0] + q0] +
                               [pulse_comb[1] + q0] + ['X180 ' + q1] +
                               ['RO ' + RO_target]]
            else:
                raise ValueError("sequence_type {} must be in".format(sequence_type) +
                                 " ['interleaved', simultaneous', 'sequential', 'sandwiched']")

    for i, pulse_comb in enumerate(pulse_list):
        pulses = []
        for p in pulse_comb:
            pulses += [operation_dict[p]]

        el = multi_pulse_elt(i, station, pulses, sequencer_config)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
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
    # FIXME: This dictionary should not be required? -MAR
    # NOTE: required in the CPhase tomo as input but not used
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
        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    for i, pulse_comb in enumerate(cal_points):
        pulses = []
        for p in pulse_comb:
            pulses += [pulse_dict[p]]

        el = multi_pulse_elt(35+i, station, pulses)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def two_qubit_tomo_bell(bell_state,
                        operation_dict,
                        qS,
                        qCZ,
                        RO_target,
                        distortion_dict,
                        CZ_disabled=False,
                        cal_points_with_flux_pulses=True,
                        verbose=False,
                        upload=True):
    '''
        qS is swap qubit
        qCZ is cphase qubit
    '''
    sequencer_config = operation_dict['sequencer_config']

    seq_name = '2_qubit_Bell_Tomo_%d_seq' % bell_state
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    tomo_list_qS = []
    tomo_list_qCZ = []
    # Tomo pulses span a basis covering all the cardinal points
    tomo_pulses = ['I ', 'X180 ', 'Y90 ', 'mY90 ', 'X90 ', 'mX90 ']
    for tp in tomo_pulses:
        tomo_list_qS += [tp+qS]
        tomo_list_qCZ += [tp+qCZ]

    ###########################
    # Defining sub sequences #
    ###########################
    # This forms the base sequence, note that gate1, gate2 and after_pulse will
    # be replaced to prepare the desired state and tomo1 and tomo2 will be
    # replaced with tomography pulses
    base_sequence = (
        ['gate1 ' + qS, 'gate2 ' + qCZ,
         'SWAP '+qS, 'CZ ' + qCZ, 'rSWAP ' + qS,
         'SWAP_corr ' + qS, 'CZ_corr ' + qCZ,
         'after_pulse ' + qCZ, 'tomo1 '+qCZ, 'tomo2 '+qS, 'RO '+RO_target])

    # Calibration points
    # every calibration point is repeated 7 times to have 64 elts in totalb
    cal_points = [['I '+qCZ, 'I '+qS, 'RO '+RO_target]]*7 +\
                  [['I '+qCZ, 'X180 '+qS, 'RO '+RO_target]]*7 +\
                  [['X180 '+qCZ, 'I '+qS, 'RO '+RO_target]]*7 +\
                  [['X180 '+qCZ, 'X180 '+qS, 'RO '+RO_target]]*7

    if CZ_disabled:
        operation_dict['CZ '+qCZ]['amplitude'] = 0
        operation_dict['CZ '+qCZ]['phase_corr_pulse_amp'] = 0

    ################################################
    # Creating additional pulses for this sequence #
    ################################################
    # the recovery SWAP is identical to the regular SWAP operation, unless
    # an rSWAP is explicitly contained in the operation dict
    if ('rSWAP ' + qS) not in operation_dict.keys():
        operation_dict['rSWAP ' + qS] = deepcopy(operation_dict['SWAP ' + qS])
    operation_dict['CZ_corr ' + qCZ]['refpoint'] = 'simultaneous'

    ################
    # Bell states  #
    ################
    if bell_state == 0:  # |Phi_m>=|00>-|11>
        gate1 = 'Y90 ' + qS
        gate2 = 'Y90 ' + qCZ
        after_pulse = 'mY90 ' + qCZ
    elif bell_state == 1:  # |Phi_p>=|00>+|11>
        gate1 = 'mY90 ' + qS
        gate2 = 'Y90 ' + qCZ
        after_pulse = 'mY90 ' + qCZ
    elif bell_state == 2:  # |Psi_m>=|01> - |10>
        gate1 = 'Y90 ' + qS
        gate2 = 'mY90 ' + qCZ
        after_pulse = 'mY90 ' + qCZ
    elif bell_state == 3:  # |Psi_p>=|01> + |10>
        gate1 = 'mY90 ' + qS
        gate2 = 'mY90 ' + qCZ
        after_pulse = 'mY90 ' + qCZ

    # Below are states with the initial pulse on the CP-qubit disabled
    # these are not Bell states but are used for debugging
    elif bell_state == 0+10:  # |00>+|11>
        gate1 = 'Y90 ' + qS
        gate2 = 'I ' + qCZ
        after_pulse = 'mY90 ' + qCZ
    elif bell_state == 1+10:
        gate1 = 'mY90 ' + qS
        gate2 = 'I ' + qCZ
        after_pulse = 'mY90 ' + qCZ
    elif bell_state == 2+10:  # |01> - |10>
        gate1 = 'Y90 ' + qS
        gate2 = 'I ' + qCZ
        after_pulse = 'mY90 ' + qCZ
    elif bell_state == 3+10:
        gate1 = 'mY90 ' + qS
        gate2 = 'I ' + qCZ
        after_pulse = 'mY90 ' + qCZ

    # Below are states with the initial pulse on the SWAP-qubit disabled
    # these are not Bell states but are used for debugging
    elif bell_state == 0 + 20:  # |00>+|11>
        gate1 = 'I ' + qS
        gate2 = 'Y90 ' + qCZ
        after_pulse = 'mY90 ' + qCZ
    elif bell_state == 1 + 20:  # |01> - |10>
        gate1 = 'I ' + qS
        gate2 = 'Y90 ' + qCZ
        after_pulse = 'mY90 ' + qCZ
    elif bell_state == 2 + 20:
        gate1 = 'I ' + qS
        gate2 = 'mY90 ' + qCZ
        after_pulse = 'mY90 ' + qCZ
    elif bell_state == 3 + 20:
        gate1 = 'mY90 ' + qS
        gate2 = 'mY90 ' + qCZ
        after_pulse = 'mY90 ' + qCZ

    print('Compensation qCP {:.3f}'.format(
        operation_dict['CZ_corr ' + qCZ]['amplitude']))
    print('Compensation qS {:.3f}'.format(
        operation_dict['SWAP_corr ' + qS]['amplitude']))

    ########################################################
    #  Here the actual pulses of all elements get defined  #
    ########################################################
    # We start by replacing the state prepartion pulses
    base_sequence[0] = gate1
    base_sequence[1] = gate2
    base_sequence[7] = after_pulse

    seq_pulse_list = []

    for i in range(36):
        tomo_idx_qS = int(i % 6)
        tomo_idx_qCZ = int(((i - tomo_idx_qS)/6) % 6)
        base_sequence[8] = tomo_list_qCZ[tomo_idx_qCZ]
        base_sequence[9] = tomo_list_qS[tomo_idx_qS]
        seq_pulse_list += [deepcopy(base_sequence)]
    print(len(cal_points))
    for cal_pulses in cal_points:
        if cal_points_with_flux_pulses:
            base_sequence[0] = 'I ' + qS
            base_sequence[1] = 'I ' + qCZ
            base_sequence[7] = 'I ' + qCZ
            base_sequence[-3:] = cal_pulses
            seq_pulse_list += [deepcopy(base_sequence)]
        else:
            seq_pulse_list += [cal_pulses]

    for i, pulse_list in enumerate(seq_pulse_list):
        pulses = []
        for p in pulse_list:
            pulses += [operation_dict[p]]
        el = multi_pulse_elt(i, station, pulses, sequencer_config)
        if distortion_dict is not None:
            print('\rDistorting element {}/{} '.format(i+1,
                                                       len(seq_pulse_list)),
                  end='')
            el = distort_and_compensate(
                el, distortion_dict)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    return seq, el_list


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
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
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
                    print('Cached {}'.format(kernel_dir+kernel))
                    output_dict[ch].append(cached_kernels[kernel])
                else:
                    print('Loading {}'.format(kernel_dir+kernel))
                    # print(os.path.isfile('kernels/'+kernel))
                    kernel_vec = np.loadtxt(kernel_dir+kernel)
                    output_dict[ch].append(kernel_vec)
                    cached_kernels.update({kernel: kernel_vec})
    return output_dict

def two_qubit_tomo_cphase_cardinal(cardinal_state,
                        operation_dict,
                        qS,
                        qCZ,
                        RO_target,
                        distortion_dict,
                        CZ_disabled=False,
                        cal_points_with_flux_pulses=True,
                        verbose=False,
                        upload=True):
    '''
        qS is swap qubit
        qCZ is cphase qubit
    '''
    sequencer_config = operation_dict['sequencer_config']

    seq_name = '2_qubit_CPhase_Cardinal_Tomo_%d_seq' % cardinal_state
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    tomo_list_qS = []
    tomo_list_qCZ = []
    # Tomo pulses span a basis covering all the cardinal points
    tomo_pulses = ['I ', 'X180 ', 'Y90 ', 'mY90 ', 'X90 ', 'mX90 ']
    for tp in tomo_pulses:
        tomo_list_qS += [tp+qS]
        tomo_list_qCZ += [tp+qCZ]

    ###########################
    # Defining sub sequences #
    ###########################
    # This forms the base sequence, note that gate1, gate2 and after_pulse will
    # be replaced to prepare the desired state and tomo1 and tomo2 will be
    # replaced with tomography pulses
    base_sequence = (
        ['gate1 ' + qS, 'gate2 ' + qCZ,
         'SWAP '+qS, 'CZ ' + qCZ, 'rSWAP ' + qS,
         'SWAP_corr ' + qS, 'CZ_corr ' + qCZ,
         'after_pulse ' + qCZ, 'tomo1 '+qCZ, 'tomo2 '+qS, 'RO '+RO_target])

    # Calibration points
    # every calibration point is repeated 7 times to have 64 elts in total
    cal_points = [['I '+qCZ, 'I '+qS, 'RO '+RO_target]]*7 +\
                  [['I '+qCZ, 'X180 '+qS, 'RO '+RO_target]]*7 +\
                  [['X180 '+qCZ, 'I '+qS, 'RO '+RO_target]]*7 +\
                  [['X180 '+qCZ, 'X180 '+qS, 'RO '+RO_target]]*7

    if CZ_disabled:
        operation_dict['CZ '+qCZ]['amplitude'] = 0
        operation_dict['CZ '+qCZ]['phase_corr_pulse_amp'] = 0

    ################################################
    # Creating additional pulses for this sequence #
    ################################################
    # the recovery SWAP is identical to the regular SWAP operation, unless
    # an rSWAP is explicitly contained in the operation dict
    if ('rSWAP ' + qS) not in operation_dict.keys():
        operation_dict['rSWAP ' + qS] = deepcopy(operation_dict['SWAP ' + qS])
    operation_dict['CZ_corr ' + qCZ]['refpoint'] = 'simultaneous'

    ################
    # cardinal states  #
    ################
    # here select the qubit gates (depending on cardinal_state)
    prep_idx_qS = int(cardinal_state % 6)
    prep_idx_qCZ = int(((cardinal_state - prep_idx_qS)/6) % 6)

    print('Compensation qCP {:.3f}'.format(
        operation_dict['CZ_corr ' + qCZ]['amplitude']))
    print('Compensation qS {:.3f}'.format(
        operation_dict['SWAP_corr ' + qS]['amplitude']))

    ########################################################
    #  Here the actual pulses of all elements get defined  #
    ########################################################
    # We start by replacing the state prepartion pulses
    base_sequence[0] = tomo_list_qS[prep_idx_qS]
    base_sequence[1] = tomo_list_qCZ[prep_idx_qCZ]
    base_sequence[7] = 'I ' + qCZ

    seq_pulse_list = []

    for i in range(36):
        tomo_idx_qS = int(i % 6)
        tomo_idx_qCZ = int(((i - tomo_idx_qS)/6) % 6)
        base_sequence[8] = tomo_list_qCZ[tomo_idx_qCZ]
        base_sequence[9] = tomo_list_qS[tomo_idx_qS]
        seq_pulse_list += [deepcopy(base_sequence)]
    print(len(cal_points))
    for cal_pulses in cal_points:
        if cal_points_with_flux_pulses:
            base_sequence[0] = 'I ' + qS
            base_sequence[1] = 'I ' + qCZ
            base_sequence[7] = 'I ' + qCZ
            base_sequence[-3:] = cal_pulses
            seq_pulse_list += [deepcopy(base_sequence)]
        else:
            seq_pulse_list += [cal_pulses]

    for i, pulse_list in enumerate(seq_pulse_list):
        pulses = []
        for p in pulse_list:
            pulses += [operation_dict[p]]
        el = multi_pulse_elt(i, station, pulses, sequencer_config)
        if distortion_dict is not None:
            print('\rDistorting element {}/{} '.format(i+1,
                                                       len(seq_pulse_list)),
                  end='')
            el = distort_and_compensate(
                el, distortion_dict)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    return seq, el_list


def n_qubit_off_on(pulse_pars_list, RO_pars, return_seq=False, verbose=False,
                   parallel_pulses=False, preselection=False, upload=True,
                   RO_spacing=200e-9):
    n = len(pulse_pars_list)
    seq_name = '{}_qubit_OffOn_sequence'.format(n)
    seq = sequence.Sequence(seq_name)
    el_list = []

    # Create a dict with the parameters for all the pulses
    pulse_dict = {'RO': RO_pars}
    for i, pulse_pars in enumerate(pulse_pars_list):
        pars = pulse_pars.copy()
        if i != 0 and parallel_pulses:
            pars['refpoint'] = 'simultaneous'
        pulses = add_suffix_to_dict_keys(
            get_pulse_dict_from_pars(pars), ' {}'.format(i))
        pulse_dict.update(pulses)
    spacerpulse = {'pulse_type': 'SquarePulse',
                   'channel': RO_pars['acq_marker_channel'],
                   'amplitude': 0.0,
                   'length': RO_spacing,
                   'pulse_delay': 0}
    pulse_dict.update({'spacer': spacerpulse})

    # Create a list of required pulses
    pulse_combinations = []
    for pulse_list in itertools.product(*(n*[['I', 'X180']])):
        pulse_comb = (n+1)*['']
        for i, pulse in enumerate(pulse_list):
            pulse_comb[i] = pulse + ' {}'.format(i)
        pulse_comb[-1] = 'RO'
        if preselection:
            pulse_comb = ['RO', 'spacer'] + pulse_comb
        pulse_combinations.append(pulse_comb)

    for i, pulse_comb in enumerate(pulse_combinations):
        pulses = []
        for j, p in enumerate(pulse_comb):
            pulses += [pulse_dict[p]]
        el = multi_pulse_elt(i, station, pulses)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name

def n_qubit_reset(pulse_pars_list, RO_pars, feedback_delay, nr_resets=1,
                  return_seq=False, verbose=False, codeword_indices=None):
    n = len(pulse_pars_list)
    seq_name = '{}_qubit_{}_reset_sequence'.format(n, nr_resets)
    seq = sequence.Sequence(seq_name)
    el_list = []

    # Create a dict with the parameters for all the pulses
    pulse_dict = {'RO': RO_pars}
    for i, pulse_pars in enumerate(pulse_pars_list):
        pars = pulse_pars.copy()
        pulses = add_suffix_to_dict_keys(
            get_pulse_dict_from_pars(pars), ' {}'.format(i))
        pulse_dict.update(pulses)
    spacerpulse = {'pulse_type': 'SquarePulse',
                   'channel': RO_pars['acq_marker_channel'],
                   'amplitude': 0.0,
                   'length': feedback_delay,
                   'pulse_delay': 0}
    pulse_dict.update({'spacer': spacerpulse})

    # create the state-preparation elements and the state reset elements
    for state in range(2 ** n):
        pulses = []
        for qb in range(n):
            if state & (1 << qb):
                pulses += [pulse_dict['X180 {}'.format(qb)].copy()]
            else:
                pulses += [pulse_dict['I {}'.format(qb)].copy()]
            if qb != 0:
                pulses[-1]['refpoint'] = 'simultaneous'
        statename = str(state).zfill(len(str(2**n - 1)))
        el = multi_pulse_elt(state, station, pulses, name='prepare' + statename)
        el_list.append(el)
        # FIXME: these elements need to have the pulses at the very start.
        el = multi_pulse_elt(state, station, pulses, name='reset' + statename,
                             trigger=False)
        el_list.append(el)

    # create the reset element
    pulses = [pulse_dict['RO'], pulse_dict['spacer']]
    el = multi_pulse_elt(state, station, pulses, name='readout', trigger=False)
    el_list.append(el)

    # Create the sequence
    for state in range(2 ** n):
        statename = str(state).zfill(len(str(2 ** n - 1)))
        seq.insert('prepare'+statename, 'prepare'+statename, trigger_wait=True)
        for i in range(nr_resets):
            seq.insert('readout' + statename + '_' + str(i), 'readout',
                       trigger_wait=False, flags={'readout'})
            seq.insert('codeword' + statename + '_' + str(i), 'codeword',
                       trigger_wait=False)
        seq.insert('readout' + statename + '_final', 'readout',
                   trigger_wait=False, flags={'readout'})

    # Create the codeword table
    if codeword_indices is None:
        codeword_indices = np.arange(n)
    for state in range(2 ** n):
        statename = str(state).zfill(len(str(2 ** n - 1)))
        codeword = 0
        for qb in range(n):
            if state & (1 << qb):
                codeword |= (1 << codeword_indices[qb])
        seq.codewords[codeword] = 'reset' + statename

    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name
