import logging
log = logging.getLogger(__name__)
import itertools
import numpy as np
from copy import deepcopy
import pycqed.measurement.waveform_control.sequence as sequence
from pycqed.utilities.general import add_suffix_to_dict_keys
from pycqed.measurement.pulse_sequences.standard_elements import \
    multi_pulse_elt, distort_and_compensate
import pycqed.measurement.randomized_benchmarking.randomized_benchmarking as rb
import pycqed.measurement.randomized_benchmarking.two_qubit_clifford_group as tqc
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import \
    get_pulse_dict_from_pars, add_preparation_pulses, pulse_list_list_seq
from pycqed.measurement.gate_set_tomography.gate_set_tomography import \
    create_experiment_list_pyGSTi_qudev as get_exp_list
from pycqed.measurement.waveform_control import pulsar as ps
import pycqed.measurement.waveform_control.segment as segment

station = None
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


def n_qubit_off_on(pulse_pars_list, RO_pars_list, return_seq=False,
                   parallel_pulses=False, preselection=False, upload=True,
                   RO_spacing=2000e-9):
    n = len(pulse_pars_list)
    seq_name = '{}_qubit_OffOn_sequence'.format(n)
    seq = sequence.Sequence(seq_name)
    seg_list = []

    RO_pars_list_presel = deepcopy(RO_pars_list)
    
    for i, RO_pars in enumerate(RO_pars_list):
        RO_pars['pulse_name'] = 'RO_{}'.format(i)
        RO_pars['element_name'] = 'RO'
        if i != 0:
            RO_pars['ref_point'] = 'start'
    for i, RO_pars_presel in enumerate(RO_pars_list_presel):
        RO_pars_presel['ref_point'] = 'start'
        RO_pars_presel['element_name'] = 'RO_presel'
        RO_pars_presel['pulse_delay'] = -RO_spacing

    # Create a dict with the parameters for all the pulses
    pulse_dict = dict()
    for i, pulse_pars in enumerate(pulse_pars_list):
        pars = pulse_pars.copy()
        if i == 0 and parallel_pulses:
            pars['ref_pulse'] = 'segment_start'
        if i != 0 and parallel_pulses:
            pars['ref_point'] = 'start'
        pulses = add_suffix_to_dict_keys(
            get_pulse_dict_from_pars(pars), ' {}'.format(i))
        pulse_dict.update(pulses)

    # Create a list of required pulses
    pulse_combinations = []

    for pulse_list in itertools.product(*(n*[['I', 'X180']])):
        pulse_comb = (n)*['']
        for i, pulse in enumerate(pulse_list):
            pulse_comb[i] = pulse + ' {}'.format(i)
        pulse_combinations.append(pulse_comb)
    for i, pulse_comb in enumerate(pulse_combinations):
        pulses = []
        for j, p in enumerate(pulse_comb):
            pulses += [pulse_dict[p]]
        pulses += RO_pars_list
        if preselection:
            pulses = pulses + RO_pars_list_presel

        seg = segment.Segment('segment_{}'.format(i), pulses)
        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    if return_seq:
        return seq, seg_list
    else:
        return seq_name


def two_qubit_randomized_benchmarking_seq(qb1n, qb2n, operation_dict,
                                      nr_cliffords_value, #scalar
                                      nr_seeds,           #array
                                      max_clifford_idx=11520,
                                      CZ_pulse_name=None,
                                      net_clifford=0,
                                      clifford_decomposition_name='HZ',
                                      interleaved_gate=None,
                                      seq_name=None, upload=True,
                                      return_seq=False, verbose=False):

    """
    Args
        qb1n (str): name of qb1
        qb2n (str): name of qb2
        operation_dict (dict): dict with all operations from both qubits and
            with the multiplexed RO pulse pars
        nr_cliffords_value (int): number of random Cliffords to generate
        nr_seeds (array): array of the form np.arange(nr_seeds_value)
        CZ_pulse_name (str): pycqed name of the CZ pulse
        net_clifford (int): 0 or 1; whether the recovery Clifford returns
            qubits to ground statea (0) or puts them in the excited states (1)
        clifford_decomp_name (str): the decomposition of Clifford gates
            into primitives; can be "XY", "HZ", or "5Primitives"
        interleaved_gate (str): pycqed name for a gate
        seq_name (str): sequence name
        upload (bool): whether to upload sequence to AWGs
        return_seq (bool): whether to return seq and el_list or just seq
        verbose (bool): print detailed runtime information
    """

    if seq_name is None:
        seq_name = '2Qb_RB_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    # Set Clifford decomposition
    tqc.gate_decomposition = rb.get_clifford_decomposition(
        clifford_decomposition_name)

    for i in nr_seeds:
        cl_seq = rb.randomized_benchmarking_sequence_new(
            nr_cliffords_value,
            number_of_qubits=2,
            max_clifford_idx=max_clifford_idx,
            interleaving_cl=interleaved_gate,
            desired_net_cl=net_clifford)

        pulse_list = []
        for idx in cl_seq:
            pulse_tuples_list = tqc.TwoQubitClifford(idx).gate_decomposition
            pulsed_qubits = {qb1n, qb2n}
            for j, pulse_tuple in enumerate(pulse_tuples_list):
                if isinstance(pulse_tuple[1], list):
                    pulse_list += [operation_dict[CZ_pulse_name]]
                    pulsed_qubits = {qb1n, qb2n}
                else:
                    qb_name = qb1n if '0' in pulse_tuple[1] else qb2n
                    pulse_name = pulse_tuple[0]
                    if 'Z' not in pulse_name:
                        if qb_name not in pulsed_qubits:
                            pulse_name += 's'
                        else:
                            pulsed_qubits = set()
                        pulsed_qubits |= {qb_name}
                    pulse_list += [operation_dict[pulse_name + ' ' + qb_name]]
        pulse_list += [operation_dict['RO mux']]

        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq


def n_qubit_simultaneous_randomized_benchmarking_seq(qubit_names_list,
                                                     operation_dict,
                                                     nr_cliffords_value, #scalar
                                                     nr_seeds,           #array
                                                     net_clifford=0,
                                                     gate_decomposition='HZ',
                                                     interleaved_gate=None,
                                                     cal_points=False,
                                                     upload=True,
                                                     upload_all=True,
                                                     seq_name=None,
                                                     verbose=False,
                                                     return_seq=False):

    """
    Args:
        qubit_list (list): list of qubit names to perform RB on
        operation_dict (dict): operation dictionary for all qubits
        nr_cliffords_value (int): number of Cliffords in the sequence
        nr_seeds (numpy.ndarray): numpy.arange(nr_seeds_int) where nr_seeds_int
            is the number of times to repeat each Clifford sequence of
            length nr_cliffords_value
        net_clifford (int): 0 or 1; refers to the final state after the recovery
            Clifford has been applied. 0->gnd state, 1->exc state.
        gate_decomposition (str): 'HZ' or 'XY'
        interleaved_gate (str): used for regular single qubit Clifford IRB
            string referring to one of the gates in the single qubit
            Clifford group
        cal_points (bool): whether to use cal points
        upload (bool): upload sequence to AWG or not
        upload_all (bool): whether to upload to all AWGs
        seq_name (str): name of this sequences
        verbose (bool): print runtime info
        return_seq (bool): if True, returns seq, element list;
            if False, returns only seq_name
    """
    # get number of qubits
    n = len(qubit_names_list)

    for qb_nr, qb_name in enumerate(qubit_names_list):
        operation_dict['Z0 ' + qb_name] = \
            deepcopy(operation_dict['Z180 ' + qb_name])
        operation_dict['Z0 ' + qb_name]['basis_rotation'][qb_name] = 0

    if seq_name is None:
        seq_name = 'SRB_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    if upload_all:
        upload_AWGs = 'all'
    else:
        upload_AWGs = ['AWG1']
        for qbn in qubit_names_list:
            X90_pulse = deepcopy(operation_dict['X90 ' + qbn])
            upload_AWGs += [station.pulsar.get(X90_pulse['I_channel'] + '_AWG'),
                            station.pulsar.get(X90_pulse['Q_channel'] + '_AWG')]
        upload_AWGs = list(set(upload_AWGs))
    print(upload_AWGs)

    for elt_idx, i in enumerate(nr_seeds):

        if cal_points and (elt_idx == (len(nr_seeds)-2)):
            pulse_keys = n*['I']

            pulse_keys_w_suffix = []
            for k, pk in enumerate(pulse_keys):
                pk_name = pk if k == 0 else pk+'s'
                pulse_keys_w_suffix.append(pk_name+' '+qubit_names_list[k % n])

            pulse_list = []
            for pkws in pulse_keys_w_suffix:
                pulse_list.append(operation_dict[pkws])

        elif cal_points and (elt_idx == (len(nr_seeds)-1)):
            pulse_keys = n*['X180']

            pulse_keys_w_suffix = []
            for k, pk in enumerate(pulse_keys):
                pk_name = pk if k == 0 else pk+'s'
                pulse_keys_w_suffix.append(pk_name+' '+qubit_names_list[k % n])

            pulse_list = []
            for pkws in pulse_keys_w_suffix:
                pulse_list.append(operation_dict[pkws])

        else:
            # if clifford_sequence_list is None:
            clifford_sequence_list = []
            for index in range(n):
                clifford_sequence_list.append(
                    rb.randomized_benchmarking_sequence(
                    nr_cliffords_value, desired_net_cl=net_clifford,
                    interleaved_gate=interleaved_gate))

            pulse_keys = rb.decompose_clifford_seq_n_qubits(
                clifford_sequence_list,
                gate_decomp=gate_decomposition)

            # interleave pulses for each qubit to obtain [pulse0_qb0,
            # pulse0_qb1,..pulse0_qbN,..,pulseN_qb0, pulseN_qb1,..,pulseN_qbN]
            pulse_keys_w_suffix = []
            for k, lst in enumerate(pulse_keys):
                pulse_keys_w_suffix.append([x+' '+qubit_names_list[k % n]
                                            for x in lst])

            pulse_keys_by_qubit = []
            for k in range(n):
                pulse_keys_by_qubit.append([x for l in pulse_keys_w_suffix[k::n]
                                            for x in l])
            # # make all qb sequences the same length
            # max_len = 0
            # for pl in pulse_keys_by_qubit:
            #     if len(pl) > max_len:
            #         max_len = len(pl)
            # for ii, pl in enumerate(pulse_keys_by_qubit):
            #     if len(pl) < max_len:
            #         pl += (max_len-len(pl))*['I ' + qubit_names_list[ii]]

            pulse_list = []
            max_len = max([len(pl) for pl in pulse_keys_by_qubit])
            for j in range(max_len):
                for k in range(n):
                    if j < len(pulse_keys_by_qubit[k]):
                        pulse_list.append(deepcopy(operation_dict[
                                                   pulse_keys_by_qubit[k][j]]))

            # Make the correct pulses simultaneous
            for p in pulse_list:
                p['refpoint'] = 'end'
            a = [iii for iii in pulse_list if
                 iii['pulse_type'] == 'SSB_DRAG_pulse']
            a[0]['refpoint'] = 'end'
            refpoint = [a[0]['target_qubit']]

            for p in a[1:]:
                if p['target_qubit'] not in refpoint:
                    p['refpoint'] = 'start'
                    refpoint.append(p['target_qubit'])
                else:
                    p['refpoint'] = 'end'
                    refpoint = [p['target_qubit']]

        # add RO pulse pars at the end
        pulse_list += [operation_dict['RO mux']]
        el = multi_pulse_elt(elt_idx, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list,
                                    AWGs=upload_AWGs,
                                    channels='all',
                                    verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def two_qubit_tomo_bell_qudev_seq(bell_state,
                                  qb_c,
                                  qb_t,
                                  RO_pars,
                                  basis_pulses=None,
                                  cal_state_repeats=7,
                                  spacing=100e-9,
                                  CZ_disabled=False,
                                  num_flux_pulses=0,
                                  verbose=False,
                                  # separation=100e-9,
                                  upload=True):
    '''
                 |spacing|spacing|
    |qCZ> --gate1--------*--------after_pulse-----| tomo |
                         |
    |qS > --gate2--------*------------------------| tomo |

        qb_c (qCZ) is the control qubit (pulsed)
        qb_t (qS) is the target qubit

    Args:
        bell_state (int): which Bell state to prepare according to:
            0 -> phi_Minus
            1 -> phi_Plus
            2 -> psi_Minus
            3 -> psi_Plus
        qb_c (qubit object): control qubit
        qb_t (qubit object): target qubit
        RO_pars (dict): RO pulse pars for multiplexed RO on the qb_c, qb_t
        basis_pulses (tuple): tomo pulses to be applied on each qubit
            default: ('I', 'X180', 'Y90', 'mY90', 'X90', 'mX90')
        cal_state_repeats (int): number of times to repeat each cal state
        spacing (float): spacing before and after CZ pulse; see diagram above
        CZ_disabled (bool): True -> same bell state preparation but without
            the CZ pulse; False -> normal bell state preparation
        num_flux_pulses (int): number of times to apply a flux pulse with the
            same length as the one used in the CZ gate before the entire
            sequence (before bell state prep).
            CURRENTLY NOT USED!
        verbose (bool): print runtime info
        upload (bool): whether to upload sequence to AWG or not
    Returns:
        seq: Sequence object
        el_list: list of elements
    '''


    if station is None:
        logging.warning('No station specified.')
    if basis_pulses is None:
        basis_pulses = ('I', 'X180', 'Y90', 'mY90', 'X90', 'mX90')
        logging.warning('basis_pulses not specified. Using the'
                        'following basis:\n{}'.format(basis_pulses))

    sequencer_config = station.sequencer_config
    seq_name = '2_qubit_Bell_Tomo_%d_seq' % bell_state
    seq = sequence.Sequence(seq_name)
    el_list = []

    operation_dict = {'RO': RO_pars}
    operation_dict.update(qb_c.get_operation_dict())
    operation_dict.update(qb_t.get_operation_dict())

    qCZ = qb_c.name
    qS = qb_t.name

    # add spacer pulses and ICZ pulses
    ##### very specific to GHZ preparation! #####
    for qb_name in [qCZ, qS]:
        spacerpulse = {'pulse_type': 'SquarePulse',
                       'channel': RO_pars['acq_marker_channel'],
                       'amplitude': 0.0,
                       'length': spacing,
                       'pulse_delay': 0,
                       'target_qubit': qb_name}
        operation_dict.update({'spacer ' + qb_name: spacerpulse})

        if not CZ_disabled:
            # create ICZ as a a copy of the I pulse but with the same length
            # as the CZ pulse
            for i, CZ_op in enumerate(['CZ ' + qS + ' ' + qCZ,
                                       'CZ ' + qS + ' ' + qCZ]):
                operation_dict['ICZ ' + qb_name] = \
                    deepcopy(operation_dict['I ' + qb_name])
                operation_dict['ICZ ' + qb_name]['sigma'] = \
                    operation_dict[CZ_op]['pulse_length']
                operation_dict['ICZ ' + qb_name]['nr_sigma'] = 1

    # make all pulses for qubit1 and qubit2 be simultaneous with qubit0
    for pulse_names in operation_dict:
        if operation_dict[pulse_names]['target_qubit'] == qS:
            operation_dict[pulse_names]['refpoint'] = 'start'

    # This forms the base sequences, note that gate1, gate2 and after_pulse will
    # be replaced to prepare the desired state.
    if not CZ_disabled:
        base_sequence = ['gate2 ' + qCZ, 'gate1 ' + qS,
                         'spacer ' + qCZ, 'spacer ' + qS,
                         'CZ ' + qS + ' ' + qCZ, 'ICZ ' + qS,
                         'spacer ' + qCZ, 'spacer ' + qS,
                         'after_pulse', 'I ' + qS]
        # base_sequence = num_flux_pulses*['flux ' + qCZ]
        # base_sequence.extend(
        #     ['gate1 ' + qS, 'gate2 ' + qCZ, 'CZ_corr ' + qS, 'CZ_corr ' + qCZ,
        #      'CZ ' + qCZ, 'after_pulse',
        #      'tomo1 '+qCZ, 'tomo2 '+qS])#, 'RO '+RO_target])
        cal_base_sequence = \
            ['I '+qCZ, 'I '+qS,
              'spacer ' + qCZ, 'spacer ' + qS,
              'ICZ ' + qCZ, 'ICZ '+qS,
              'spacer ' + qCZ, 'spacer ' + qS,
              'I '+qCZ, 'I '+qS]
    else:
        base_sequence = ['gate2 ' + qCZ, 'gate1 ' + qS,
                         'after_pulse', 'I ' + qS]
        cal_base_sequence = ['I '+qCZ, 'I '+qS, 'I '+qCZ, 'I '+qS]

    ################
    # Bell states  #
    ################
    if bell_state == 0:  # |Phi_m>=|00>-|11>
        gate2 = 'Y90 ' + qCZ
        gate1 = 'Y90 ' + qS
        after_pulse = 'mY90 ' + qCZ
    elif bell_state == 1:  # |Phi_p>=|00>+|11>
        gate2 = 'Y90 ' + qCZ
        gate1 = 'mY90 ' + qS
        after_pulse = 'mY90 ' + qCZ
    elif bell_state == 2:  # |Psi_m>=|01> - |10>
        gate2 = 'mY90 ' + qCZ
        gate1 = 'Y90 ' + qS
        after_pulse = 'mY90 ' + qCZ
    elif bell_state == 3:  # |Psi_p>=|01> + |10>
        gate2 = 'Y90 ' + qCZ
        gate1 = 'Y90 ' + qS
        after_pulse = 'Y90 ' + qCZ

    ########################################################
    #  Here the actual pulses of all elements get defined  #
    ########################################################
    # We start by replacing the state prepartion pulses
    base_sequence[base_sequence.index('gate1 ' + qS)] = gate1
    base_sequence[base_sequence.index('gate2 ' + qCZ)] = gate2
    base_sequence[base_sequence.index('after_pulse')] = after_pulse

    # Tomo pulses
    tomo_pulses = get_tomography_pulses(*[qCZ, qS],
                                        basis_pulses=basis_pulses)

    seq_pulse_list = len(tomo_pulses)*['']
    for i, t in enumerate(tomo_pulses):
        seq_pulse_list[i] = base_sequence + t

    # Calibration points
    # every calibration point is repeated 7 times to have 64 elts in totalb
    cal_pulses = get_tomography_pulses(*[qCZ, qS],
                                       basis_pulses=('I', 'X180'))
    for i, cal_p in enumerate(cal_pulses):
        cal_pulses[i] = cal_base_sequence + cal_p
    cal_pulses = [list(i) for i in np.repeat(np.asarray(cal_pulses),
                                            cal_state_repeats, axis=0)]
    for cal_p in cal_pulses:
        seq_pulse_list += [cal_p]

    # # set correct timing before making the pulse list
    # flux_pulse = operation_dict['CZ ' + qS + ' ' + qCZ]
    # flux_pulse['refpoint'] = 'end'
    # print('\n',separation)
    # print(flux_pulse['pulse_delay'])
    #
    # operation_dict['mY90afs ' + qCZ] = operation_dict['mY90s ' + qCZ]
    # operation_dict['mY90afs ' + qCZ]['pulse_delay'] = \
    #     separation - flux_pulse['pulse_delay']
    # operation_dict['mY90afs ' + qCZ]['refpoint'] = 'start'
    # if bell_state==3:
    #     operation_dict['Y90afs ' + qCZ] = operation_dict['Y90s ' + qCZ]
    #     operation_dict['Y90afs ' + qCZ]['pulse_delay'] = \
    #         separation - flux_pulse['pulse_delay']
    #     operation_dict['Y90afs ' + qCZ]['refpoint'] = 'start'

    for i, pulse_list in enumerate(seq_pulse_list):
        # print('pulse_list ', pulse_list)
        pulses = []
        for p in pulse_list:
            pulses += [operation_dict[p]]
            # if 'CZ' in p:
            #     from pprint import pprint
            #     pprint(operation_dict[p])
        # print('pulses ', len(pulses))
        # from pprint import pprint
        # pprint(pulses)
        # print('\n')
        pulses += [RO_pars]
        el = multi_pulse_elt(i, station, pulses, sequencer_config)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    # return seq, seq_pulse_list
    return seq, el_list

def n_qubit_reset(qubit_names, operation_dict, cal_points,
                  prep_params=dict(),
                  upload=True):
    """

    Timing constraints:
        The reset_cycle_time and the readout fixed point should be commensurate
        with the UHFQC trigger grid and the granularity of the AWGs.

        When the -ro_acq_marker_delay of the readout pulse is larger than
        the drive pulse length, then it is important that its length is a
        multiple of the granularity of the AWG.
    """

    seq_name = '{}_reset_x{}_sequence'.format(','.join(qubit_names),
                                              prep_params.get('reset_reps',
                                                              '_default_n_reps'))
    seq = sequence.Sequence(seq_name)

    # add calibration segments
    seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    log.debug(seq)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())

def n_qubit_reset_old(qubit_names, operation_dict, reset_cycle_time, nr_resets=1,
                  return_seq=False, verbose=False, codeword_indices=None,
                  upload=True):
    """

    Timing constraints:
        The reset_cycle_time and the readout fixed point should be commensurate
        with the UHFQC trigger grid and the granularity of the AWGs.

        When the -ro_acq_marker_delay of the readout pulse is larger than
        the drive pulse length, then it is important that its length is a
        multiple of the granularity of the AWG.
    """

    n = len(qubit_names)
    seq_name = '{}_reset_x{}_sequence'.format(','.join(qubit_names), nr_resets)
    seq = sequence.Sequence(seq_name)
    el_list = []
    operation_dict = deepcopy(operation_dict)

    # set all qubit drive pulse delays to the maximum over the qubits
    max_delay = float('-inf')
    for qbn in qubit_names:
        max_delay = max(max_delay, operation_dict['X180 ' + qbn]['pulse_delay'])
    for qbn in qubit_names:
        for op in ['X180 ', 'X180s ', 'I ', 'Is ']:
            operation_dict[op + qbn]['pulse_delay'] = max_delay

    # sort qubits by pulse length
    pulse_lengths = []
    for qbn in qubit_names:
        pulse = operation_dict['X180 ' + qbn]
        pulse_lengths.append(pulse['sigma']*pulse['nr_sigma'])
    qubits_sorted = []
    for idx in np.argsort(pulse_lengths):
        qubits_sorted.append(qubit_names[idx])

    # increase the readout pulse delay such that the trigger would just
    # fit in the feedback element
    mw_pulse = operation_dict['X180 ' + qubits_sorted[-1]]
    ro_pulse = operation_dict['RO']
    ro_pulse['pulse_delay'] = max(ro_pulse['pulse_delay'],
                                  -ro_pulse['acq_marker_delay']
                                  - mw_pulse['pulse_delay']
                                  - mw_pulse['sigma']*mw_pulse['nr_sigma'])

    # create the wait pulse such that the length of the reset element would be
    # reset_cycle_time
    wait_time = mw_pulse['pulse_delay'] + mw_pulse['sigma']*mw_pulse['nr_sigma']
    wait_time += ro_pulse['pulse_delay'] + ro_pulse['length']
    wait_time = reset_cycle_time - wait_time
    wait_time -= station.pulsar.inter_element_spacing()
    wait_pulse = {'pulse_type': 'SquarePulse',
                  'channel': ro_pulse['acq_marker_channel'],
                  'amplitude': 0.0,
                  'length': wait_time,
                  'pulse_delay': 0}
    operation_dict.update({'I_fb': wait_pulse})

    # create the state-preparation elements and the state reset elements
    # the longest drive pulse is added last so that the readout does not overlap
    # with the drive pulses.
    qubit_order = np.arange(len(qubit_names))[np.argsort(pulse_lengths)]
    for state in range(2 ** n):
        pulses = []
        for i, qbn in enumerate(qubits_sorted):
            if state & (1 << qubit_names.index(qbn)):
                if i == 0:
                    pulses.append(operation_dict['X180 ' + qbn])
                else:
                    pulses.append(operation_dict['X180s ' + qbn])
            else:
                if i == 0:
                    pulses.append(operation_dict['I ' + qbn])
                else:
                    pulses.append(operation_dict['Is ' + qbn])
        pulses += [operation_dict['RO'], operation_dict['I_fb']]
        statename = str(state).zfill(len(str(2 ** n - 1)))
        el = multi_pulse_elt(state, station, pulses, name='prepare' + statename)
        el_list.append(el)
        el = multi_pulse_elt(state, station, pulses, name='reset' + statename,
                             trigger=False)
        el_list.append(el)

    # Create the sequence
    for state in range(2 ** n):
        statename = str(state).zfill(len(str(2 ** n - 1)))
        seq.insert('prepare'+statename, 'prepare'+statename, trigger_wait=True)
        for i in range(nr_resets):
            seq.insert('codeword' + statename + '_' + str(i), 'codeword',
                       trigger_wait=False)

    # Create the codeword table
    if codeword_indices is None:
        codeword_indices = np.arange(n)
    for state in range(2 ** n):
        statename = str(state).zfill(len(str(2 ** n - 1)))
        codeword = 0
        for qb_idx in range(n):
            if state & (1 << qb_idx):
                codeword |= (1 << codeword_indices[qb_idx])
        seq.codewords[codeword] = 'reset' + statename

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name

def parity_correction_seq(
        qb1n, qb2n, qb3n, operation_dict, CZ_pulses, feedback_delay=900e-9,
        prep_sequence=None, reset=True, nr_parity_measurements=1,
        tomography_basis=('I', 'X180', 'Y90', 'mY90', 'X90', 'mX90'),
        parity_op='ZZ', upload=True, verbose=False, return_seq=False, 
        preselection=False, ro_spacing=1e-6, dd_scheme=None, nr_dd_pulses=4,
        skip_n_initial_parity_checks=0, skip_elem='RO'):
    """

    |              elem 1               |  elem 2  |  (elem 3, 2)  | elem 4

    |q0> |======|---------*--------------------------(           )-|======|
         | prep |         |                         (   repeat    )| tomo |
    |q1> | q0,  |--mY90s--*--*--Y90--meas=Y180------(   parity    )| q0,  |
         | q2   |            |             ||       ( measurement )| q2   |
    |q2> |======|------------*------------Y180-------(           )-|======|
 
    required elements:
        elem_1:
            contains everything up to the first readout including preparation
            and first parity measurement
        elem_2 (x2):
            contains conditional Y180 on q1 and q2
        elem_3:
            additional parity measurements
        elem_4 (x6**2):
            tomography rotations for q0 and q2

    Args:
        parity_op: 'ZZ', 'XX', 'XX,ZZ' or 'ZZ,XX' specifies the type of parity 
                   measurement
    """
    if parity_op not in ['ZZ', 'XX', 'XX,ZZ', 'ZZ,XX']:
        raise ValueError("Invalid parity operator '{}'".format(parity_op))

    operation_dict['RO mux_presel'] = operation_dict['RO mux'].copy()
    operation_dict['RO mux_presel']['pulse_delay'] = \
        -ro_spacing - feedback_delay - operation_dict['RO mux']['length']
    operation_dict['RO mux_presel']['refpoint'] = 'end'
    operation_dict['RO mux_presel'].pop('basis_rotation', {})

    operation_dict['RO presel_dummy'] = {
        'pulse_type': 'SquarePulse',
        'channel': operation_dict['RO mux']['acq_marker_channel'],
        'amplitude': 0.0,
        'length': ro_spacing + feedback_delay,
        'pulse_delay': 0}
    operation_dict['I rr_decay'] = {
        'pulse_type': 'SquarePulse',
        'channel': operation_dict['RO mux']['acq_marker_channel'],
        'amplitude': 0.0,
        'length': 400e-9,
        'pulse_delay': 0}
    operation_dict['RO skip'] = {
        'pulse_type': 'SquarePulse',
        'channel': operation_dict['RO mux']['acq_marker_channel'],
        'amplitude': 0.0,
        'length': operation_dict['RO mux']['length'],
        'pulse_delay': 0
    }
    operation_dict['CZ1 skip'] = operation_dict[CZ_pulses[0]].copy()
    operation_dict['CZ1 skip']['amplitude'] = 0
    operation_dict['CZ2 skip'] = operation_dict[CZ_pulses[1]].copy()
    operation_dict['CZ2 skip']['amplitude'] = 0

    if dd_scheme is None:
        dd_pulses = [{
        'pulse_type': 'SquarePulse',
        'channel': operation_dict['RO mux']['acq_marker_channel'],
        'amplitude': 0.0,
        'length': feedback_delay,
        'pulse_delay': 0}]
    else:
        dd_pulses = get_dd_pulse_list(
            operation_dict,
            # [qb2n],
            [qb1n, qb3n],
            feedback_delay,
            nr_pulses=nr_dd_pulses,
            dd_scheme=dd_scheme,
            init_buffer=0)

    if prep_sequence is None:
        if parity_op[0] == 'X':
            prep_sequence = []
        else:
            prep_sequence = ['Y90 ' + qb1n, 'Y90s ' + qb3n]
    elif prep_sequence == 'mixed':
        prep_sequence = ['Y90 ' + qb1n, 'Y90s ' + qb3n, 'RO mux', 'I rr_decay']
    
    xx_sequence_first =  ['Y90 ' + qb1n, 'mY90s ' + qb2n, 'Y90s ' + qb3n,
                          CZ_pulses[0],
                          CZ_pulses[1],
                          # 'mY90 ' + qb1n,
                          'Y90 ' + qb2n,
                          'RO ' + qb2n]
    xx_sequence_after_z =  deepcopy(xx_sequence_first)
    xx_sequence_after_x =  ['mY90 ' + qb2n,
                            CZ_pulses[0],
                            CZ_pulses[1],
                            # 'mY90 ' + qb1n,
                            'Y90 ' + qb2n,
                            'RO ' + qb2n]
    zz_sequence_first =  ['mY90 ' + qb2n,
                          CZ_pulses[0],
                          CZ_pulses[1],
                          'Y90 ' + qb2n,
                          'RO ' + qb2n]
    zz_sequence_after_z =  deepcopy(zz_sequence_first)
    zz_sequence_after_x =  ['mY90 ' + qb2n, 'mY90s ' + qb3n, 'mY90s ' + qb1n,
                            CZ_pulses[0],
                            CZ_pulses[1],
                            'Y90 ' + qb2n,
                            'RO ' + qb2n]
    pretomo_after_z = []
    pretomo_after_x = ['mY90 ' + qb3n, 'mY90s ' + qb1n,]


    # create the elements
    el_list = []

    # first element
    if parity_op in ['XX', 'XX,ZZ']:        
        op_sequence = prep_sequence + xx_sequence_first
    else:
        op_sequence = prep_sequence + zz_sequence_first

    def skip_parity_check(op_sequence, skip_elem, CZ_pulses, qb2n):
        if skip_elem == 'RO':
            op_sequence = ['RO skip' if op == ('RO ' + qb2n) else op
                           for op in op_sequence]
        if skip_elem == 'CZ D1' or skip_elem == 'CZ both':
            op_sequence = ['CZ1 skip' if op == CZ_pulses[0] else op
                           for op in op_sequence]
        if skip_elem == 'CZ D2' or skip_elem == 'CZ both':
            op_sequence = ['CZ2 skip' if op == CZ_pulses[1] else op
                           for op in op_sequence]
        return op_sequence
    if skip_n_initial_parity_checks > 0:
        op_sequence = skip_parity_check(op_sequence, skip_elem, CZ_pulses, qb2n)
    pulse_list = [operation_dict[pulse] for pulse in op_sequence]
    pulse_list += dd_pulses
    if preselection:
        pulse_list.append(operation_dict['RO mux_presel'])
        # RO presel dummy is referenced to end of RO mux presel => it happens
        # before the preparation pulses!
        pulse_list.append(operation_dict['RO presel_dummy'])
    el_main = multi_pulse_elt(0, station, pulse_list, trigger=True, 
                              name='m')
    el_list.append(el_main)

    # feedback elements
    fb_sequence_0 = ['I ' + qb3n, 'Is ' + qb2n]
    fb_sequence_1 = ['X180 ' + qb3n] if reset else ['I ' + qb3n]
    fb_sequence_1 += ['X180s ' + qb2n]# if reset else ['Is ' + qb2n]
    pulse_list = [operation_dict[pulse] for pulse in fb_sequence_0]
    el_list.append(multi_pulse_elt(0, station, pulse_list, name='f0',
                                   trigger=False, previous_element=el_main))
    pulse_list = [operation_dict[pulse] for pulse in fb_sequence_1]
    el_fb = multi_pulse_elt(1, station, pulse_list, name='f1',
                            trigger=False, previous_element=el_main)
    el_list.append(el_fb)

    # repeated parity measurement element(s). Phase errors need to be corrected 
    # for by careful selection of qubit drive IF-s.
    if parity_op == 'ZZ':
        pulse_list = [operation_dict[pulse] for pulse in zz_sequence_after_z]
        pulse_list += dd_pulses
        el_repeat = multi_pulse_elt(0, station, pulse_list, trigger=False, 
                                    name='rz', previous_element=el_fb)
        el_list.append(el_repeat)
    elif parity_op == 'XX':
        pulse_list = [operation_dict[pulse] for pulse in xx_sequence_after_x]
        pulse_list += dd_pulses
        el_repeat = multi_pulse_elt(0, station, pulse_list, trigger=False, 
                                    name='rx', previous_element=el_fb)
        el_list.append(el_repeat)
    elif parity_op == 'ZZ,XX':
        pulse_list = [operation_dict[pulse] for pulse in xx_sequence_after_z]
        pulse_list += dd_pulses
        el_repeat_x = multi_pulse_elt(0, station, pulse_list, trigger=False, 
                                     name='rx', previous_element=el_fb)
        el_list.append(el_repeat_x)

        pulse_list = [operation_dict[pulse] for pulse in zz_sequence_after_x]
        pulse_list += dd_pulses
        el_repeat_z = multi_pulse_elt(1, station, pulse_list, trigger=False, 
                                     name='rz', previous_element=el_fb)
        el_list.append(el_repeat_z)
    elif parity_op == 'XX,ZZ':
        pulse_list = [operation_dict[pulse] for pulse in zz_sequence_after_x]
        pulse_list += dd_pulses
        el_repeat_z = multi_pulse_elt(0, station, pulse_list, trigger=False, 
                                      name='rz', previous_element=el_fb)
        el_list.append(el_repeat_z)

        pulse_list = [operation_dict[pulse] for pulse in xx_sequence_after_z]
        pulse_list += dd_pulses
        el_repeat_x = multi_pulse_elt(1, station, pulse_list, trigger=False, 
                                      name='rx', previous_element=el_fb)
        el_list.append(el_repeat_x)
    
    # repeated parity measurement element(s) with skipped parity check.
    if skip_n_initial_parity_checks > 1:
        if parity_op == 'ZZ':
            op_sequence = skip_parity_check(zz_sequence_after_z,
                                            skip_elem, CZ_pulses, qb2n)
            pulse_list = [operation_dict[pulse] for pulse in op_sequence]
            pulse_list += dd_pulses
            el_repeat_skip = multi_pulse_elt(0, station, pulse_list,
                                             trigger=False, name='rzs',
                                             previous_element=el_fb)
            el_list.append(el_repeat_skip)
        elif parity_op == 'XX':
            op_sequence = skip_parity_check(xx_sequence_after_x,
                                            skip_elem, CZ_pulses, qb2n)
            pulse_list = [operation_dict[pulse] for pulse in op_sequence]
            pulse_list += dd_pulses
            el_repeat_skip = multi_pulse_elt(0, station, pulse_list,
                                             trigger=False, name='rxs',
                                             previous_element=el_fb)
            el_list.append(el_repeat_skip)
        elif parity_op == 'ZZ,XX':
            op_sequence = skip_parity_check(xx_sequence_after_z,
                                            skip_elem, CZ_pulses, qb2n)
            pulse_list = [operation_dict[pulse] for pulse in op_sequence]
            pulse_list += dd_pulses
            el_repeat_x_skip = multi_pulse_elt(0, station, pulse_list,
                                               trigger=False, name='rxs',
                                               previous_element=el_fb)
            el_list.append(el_repeat_x_skip)

            op_sequence = skip_parity_check(zz_sequence_after_x,
                                            skip_elem, CZ_pulses, qb2n)
            pulse_list = [operation_dict[pulse] for pulse in op_sequence]
            pulse_list += dd_pulses
            el_repeat_z_skip = multi_pulse_elt(0, station, pulse_list,
                                               trigger=False, name='rzs',
                                               previous_element=el_fb)
            el_list.append(el_repeat_z_skip)
        elif parity_op == 'XX,ZZ':
            op_sequence = skip_parity_check(zz_sequence_after_x,
                                            skip_elem, CZ_pulses, qb2n)
            pulse_list = [operation_dict[pulse] for pulse in op_sequence]
            pulse_list += dd_pulses
            el_repeat_z_skip = multi_pulse_elt(0, station, pulse_list,
                                               trigger=False, name='rzs',
                                               previous_element=el_fb)
            el_list.append(el_repeat_z_skip)

            op_sequence = skip_parity_check(xx_sequence_after_z,
                                            skip_elem, CZ_pulses, qb2n)
            pulse_list = [operation_dict[pulse] for pulse in op_sequence]
            pulse_list += dd_pulses
            el_repeat_x_skip = multi_pulse_elt(0, station, pulse_list,
                                               trigger=False, name='rxs',
                                               previous_element=el_fb)
            el_list.append(el_repeat_x_skip)

    # check that the qubits do not acquire any phase over one round of parity 
    # correction
    for qbn in [qb1n, qb2n, qb3n]:
        ifreq = operation_dict['X180 ' + qbn]['mod_frequency']
        if parity_op in ['XX', 'ZZ']:
            elements_length = el_fb.ideal_length() + el_repeat.ideal_length()
            dynamic_phase = el_repeat.drive_phase_offsets.get(qbn, 0)
        else:
            elements_length = el_fb.ideal_length() + el_repeat_x.ideal_length()
            dynamic_phase = el_repeat_x.drive_phase_offsets.get(qbn, 0)
            print('Length difference of XX and ZZ cycles: {} s'.format(
                el_repeat_x.ideal_length() - el_repeat_z.ideal_length()
            ))
        dynamic_phase -= el_main.drive_phase_offsets.get(qbn, 0)
        phase_from_if = 360*ifreq*elements_length
        total_phase = phase_from_if + dynamic_phase
        total_mod_phase = (total_phase + 180) % 360 - 180
        print(qbn + ' aquires a phase of {} ≡ {} (mod 360)'.format(
            total_phase, total_mod_phase) + ' degrees each correction ' + 
            'cycle. You should reduce the intermediate frequency by {} Hz.'\
            .format(total_mod_phase/elements_length/360))

    # tomography elements
    if parity_op in ['XX', ['XX,ZZ', 'ZZ,XX'][nr_parity_measurements % 2]]:
        pretomo = pretomo_after_x
    else:
        pretomo = pretomo_after_z
    tomography_sequences = get_tomography_pulses(qb1n, qb3n,
                                                 basis_pulses=tomography_basis)
    for i, tomography_sequence in enumerate(tomography_sequences):
        pulse_list = [operation_dict[pulse] for pulse in 
                          pretomo + tomography_sequence + ['RO mux']]
        el_list.append(multi_pulse_elt(i, station, pulse_list, trigger=False,
                                       name='t{}'.format(i),
                                       previous_element=el_fb))

    # create the sequence
    seq_name = 'Two qubit entanglement by parity measurement'
    seq = sequence.Sequence(seq_name)
    seq.codewords[0] = 'f0'
    seq.codewords[1] = 'f1'
    for i in range(len(tomography_basis)**2):
        seq.append('m_{}'.format(i), 'm', trigger_wait=True)
        seq.append('f_{}_0'.format(i), 'codeword', trigger_wait=False)
        for j in range(1, nr_parity_measurements):
            if parity_op in ['XX', ['XX,ZZ', 'ZZ,XX'][j % 2]]:
                el_name = 'rx'
            else:
                el_name = 'rz'
            if j < skip_n_initial_parity_checks:
                el_name += 's'
            seq.append('r_{}_{}'.format(i, j), el_name,
                            trigger_wait=False)
            seq.append('f_{}_{}'.format(i, j), 'codeword',
                       trigger_wait=False)
        seq.append('t_{}'.format(i), 't{}'.format(i),
                   trigger_wait=False)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def parity_correction_no_reset_seq(
        q0n, q1n, q2n, operation_dict, CZ_pulses, feedback_delay=900e-9,
        prep_sequence=None, ro_spacing=1e-6, dd_scheme=None, nr_dd_pulses=0,
        tomography_basis=('I', 'X180', 'Y90', 'mY90', 'X90', 'mX90'),
        upload=True, verbose=False, return_seq=False, preselection=False):
    """

    |              elem 1               |  elem 2  | elem 3

    |q0> |======|---------*------------------------|======|
         | prep |         |                        | tomo |
    |q1> | q0,  |--mY90s--*--*--Y90--meas===-------| q0,  |
         | q2   |            |             ||      | q2   |
    |q2> |======|------------*------------X180-----|======|

    required elements:
        prep_sequence:
            contains everything up to the first readout
        Echo decoupling elements
            contains nr_echo_pulses X180 equally-spaced pulses on q0n, q2n
            FOR THIS TO WORK, ALL QUBITS MUST HAVE THE SAME PI-PULSE LENGTH
        feedback x 2 (for the two readout results):
            contains conditional Y80 on q1 and q2
        tomography x 6**2:
            measure all observables of the two qubits X/Y/Z
    """

    operation_dict['RO mux_presel'] = operation_dict['RO mux'].copy()
    operation_dict['RO mux_presel']['pulse_delay'] = \
        -ro_spacing - feedback_delay - operation_dict['RO mux']['length']
    operation_dict['RO mux_presel']['refpoint'] = 'end'

    operation_dict['RO presel_dummy'] = {
        'pulse_type': 'SquarePulse',
        'channel': operation_dict['RO mux']['acq_marker_channel'],
        'amplitude': 0.0,
        'length': ro_spacing+feedback_delay,
        'pulse_delay': 0}

    if dd_scheme is None:
        dd_pulses = [{
            'pulse_type': 'SquarePulse',
            'channel': operation_dict['RO mux']['acq_marker_channel'],
            'amplitude': 0.0,
            'length': feedback_delay,
            'pulse_delay': 0}]
    else:
        dd_pulses = get_dd_pulse_list(
            operation_dict,
            # [qb2n],
            [q0n, q2n],
            feedback_delay,
            nr_pulses=nr_dd_pulses,
            dd_scheme=dd_scheme,
            init_buffer=0)

    if prep_sequence is None:
        prep_sequence = ['Y90 ' + q0n, 'Y90s ' + q2n,
                         'mY90 ' + q1n,
                         CZ_pulses[0], CZ_pulses[1],
                         'Y90 ' + q1n,
                         'RO ' + q1n]

    pulse_list = [operation_dict[pulse] for pulse in prep_sequence] + dd_pulses

    if preselection:
        pulse_list.append(operation_dict['RO mux_presel'])
        # RO presel dummy is referenced to end of RO mux presel => it happens
        # before the preparation pulses!
        pulse_list.append(operation_dict['RO presel_dummy'])

    idle_length = operation_dict['X180 ' + q1n]['sigma']
    idle_length *= operation_dict['X180 ' + q1n]['nr_sigma']
    idle_length += 2*8/2.4e9
    idle_pulse = {
        'pulse_type': 'SquarePulse',
        'channel': operation_dict['RO mux']['acq_marker_channel'],
        'amplitude': 0.0,
        'length': idle_length,
        'pulse_delay': 0}
    pulse_list.append(idle_pulse)

    # tomography elements
    tomography_sequences = get_tomography_pulses(q0n, q2n,
                                                 basis_pulses=tomography_basis)
    print(len(tomography_sequences))
    # create the elements
    el_list = []
    for i, tomography_sequence in enumerate(tomography_sequences):
        tomography_sequence.append('RO mux')
        pulse_list_tomo = deepcopy(pulse_list) + \
                          [operation_dict[pulse] for pulse in
                           tomography_sequence]
        el_list.append(multi_pulse_elt(i, station, pulse_list_tomo,
                                       trigger=True,
                                       name='tomography_{}'.format(i)))

    # create the sequence
    seq_name = 'Two qubit entanglement by parity measurement'
    seq = sequence.Sequence(seq_name)
    # for i in range(len(tomography_basis)**2):
    for i, tomography_sequence in enumerate(tomography_sequences):
        seq.append('tomography_{}'.format(i), 'tomography_{}'.format(i),
                   trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def n_qubit_tomo_seq(
        qubit_names, operation_dict, prep_sequence=None,
        prep_name=None,
        rots_basis=('I', 'X180', 'Y90', 'mY90', 'X90', 'mX90'),
        upload=True, verbose=False, return_seq=False,
        preselection=False, ro_spacing=1e-6):
    """

    """

    # create the sequence
    if prep_name is None:
        seq_name = 'N-qubit tomography'
    else:
        seq_name = prep_name + ' tomography'
    seq = sequence.Sequence(seq_name)
    seg_list = []

    if prep_sequence is None:
        prep_sequence = ['Y90 ' + qubit_names[0]]

    # tomography elements
    tomography_sequences = get_tomography_pulses(*qubit_names,
                                                 basis_pulses=rots_basis)
    for i, tomography_sequence in enumerate(tomography_sequences):
        pulse_list = [operation_dict[pulse] for pulse in prep_sequence]
        # tomography_sequence.append('RO mux')
        # if preselection:
        #     tomography_sequence.append('RO mux_presel')
        #     tomography_sequence.append('RO presel_dummy')
        pulse_list.extend([operation_dict[pulse] for pulse in
                           tomography_sequence])
        ro_pulses = generate_mux_ro_pulse_list(qubit_names, operation_dict)
        pulse_list.extend(ro_pulses)

        if preselection:
            ro_pulses_presel = generate_mux_ro_pulse_list(qubit_names, 
                                                          operation_dict,
                                                          'RO_presel',
                                                          True, -ro_spacing)
            pulse_list.extend(ro_pulses_presel)
        seg = segment.Segment('tomography_{}'.format(i), pulse_list)
        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    if return_seq:
        return seq, seg_list
    else:
        return seq_name

def get_tomography_pulses(*qubit_names, basis_pulses=('I', 'X180', 'Y90',
                                                      'mY90', 'X90', 'mX90')):
    tomo_sequences = [[]]
    for i, qb in enumerate(qubit_names):
        if i == 0:
            qb = ' ' + qb
        else:
            qb = 's ' + qb
        tomo_sequences_new = []
        for sequence in tomo_sequences:
            for pulse in basis_pulses:
                tomo_sequences_new.append(sequence + [pulse+qb])
        tomo_sequences = tomo_sequences_new
    return tomo_sequences

def get_decoupling_pulses(*qubit_names, nr_pulses=4):
    if nr_pulses % 2 != 0:
        logging.warning('Odd number of dynamical decoupling pulses')
    echo_sequences = []
    for pulse in nr_pulses*['X180']:
        for i, qb in enumerate(qubit_names):
            if i == 0:
                qb = ' ' + qb
            else:
                qb = 's ' + qb
            echo_sequences.append(pulse+qb)
    return echo_sequences

def n_qubit_ref_seq(qubit_names, operation_dict, ref_desc, upload=True,
                    verbose=False, return_seq=False, preselection=False,
                    ro_spacing=1e-6):
    """
        Calibration points for arbitrary combinations

        Arguments:
            qubits: List of calibrated qubits for obtaining the pulse
                dictionaries.
            ref_desc: Description of the calibration sequence. Dictionary
                name of the state as key, and list of pulses names as values.
    """


    # create the elements
    seq_name = 'Calibration'
    seq = sequence.Sequence(seq_name)
    seg_list = []

    # calibration elements
    # calibration_sequences = []
    # for pulses in ref_desc:
    #     calibration_sequences.append(
    #         [pulse+' '+qb for qb, pulse in zip(qubit_names, pulses)])

    calibration_sequences = []
    for pulses in ref_desc:
        calibration_sequence_new = []
        for i, pulse in enumerate(pulses):
            if i == 0:
                qb = ' ' + qubit_names[i]
            else:
                qb = 's ' + qubit_names[i]
            calibration_sequence_new.append(pulse+qb)
        calibration_sequences.append(calibration_sequence_new)

    for i, calibration_sequence in enumerate(calibration_sequences):
        pulse_list = []
        pulse_list.extend(
            [operation_dict[pulse] for pulse in calibration_sequence])
        ro_pulses = generate_mux_ro_pulse_list(qubit_names, operation_dict)
        pulse_list.extend(ro_pulses)

        if preselection:
            ro_pulses_presel = generate_mux_ro_pulse_list(qubit_names, 
                                                          operation_dict,
                                                          'RO_presel',
                                                          True, -ro_spacing)
            pulse_list.extend(ro_pulses_presel)
        seg = segment.Segment('calibration_{}'.format(i), pulse_list)
        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    if return_seq:
        return seq, seg_list
    else:
        return seq_name


def n_qubit_ref_all_seq(qubit_names, operation_dict, upload=True, verbose=False,
                        return_seq=False,
                        preselection=False, ro_spacing=1e-6):
    """
        Calibration points for all combinations
    """

    return n_qubit_ref_seq(qubit_names, operation_dict,
                           ref_desc=itertools.product(["I", "X180"],
                                                      repeat=len(qubit_names)),
                           upload=upload, verbose=verbose,
                           return_seq=return_seq, preselection=preselection,
                           ro_spacing=ro_spacing)


def Ramsey_add_pulse_seq(times, measured_qubit_name,
                         pulsed_qubit_name, operation_dict,
                         artificial_detuning=None,
                         cal_points=True,
                         verbose=False,
                         upload=True, return_seq=False):

    if np.any(times > 1e-3):
        logging.warning('The values in the times array might be too large.'
                        'The units should be seconds.')

    seq_name = 'Ramsey_with_additional_pulse_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []


    pulse_pars_x1 = deepcopy(operation_dict['X90 ' + measured_qubit_name])
    pulse_pars_x1['refpoint'] = 'end'
    pulse_pars_x2 = deepcopy(pulse_pars_x1)
    pulse_pars_x2['refpoint'] = 'start'
    RO_pars = operation_dict['RO ' + measured_qubit_name]
    add_pulse_pars = deepcopy(operation_dict['X180 ' + pulsed_qubit_name])

    for i, tau in enumerate(times):
        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
            el = multi_pulse_elt(i, station, [
                operation_dict['I ' + measured_qubit_name], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
            el = multi_pulse_elt(i, station, [
                operation_dict['X180 ' + measured_qubit_name], RO_pars])
        else:
            pulse_pars_x2['pulse_delay'] = tau
            if artificial_detuning is not None:
                Dphase = ((tau-times[0]) * artificial_detuning * 360) % 360
                pulse_pars_x2['phase'] = Dphase

            if i % 2 == 0:
                el = multi_pulse_elt(
                    i, station, [operation_dict['X90 ' + measured_qubit_name],
                                 pulse_pars_x2, RO_pars])
            else:
                el = multi_pulse_elt(i, station,
                                     [add_pulse_pars, pulse_pars_x1,
                                     # [pulse_pars_x1, add_pulse_pars,
                                      pulse_pars_x2, RO_pars])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def Ramsey_add_pulse_sweep_phase_seq(
        phases, measured_qubit_name,
        pulsed_qubit_name, operation_dict,
        verbose=False,
        upload=True, return_seq=False,
        cal_points=True):

    seq_name = 'Ramsey_with_additional_pulse_sweep_phase_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    X90_2 = deepcopy(operation_dict['X90 ' + measured_qubit_name])
    for i, theta in enumerate(phases):
        X90_2['phase'] = theta*180/np.pi
        if cal_points and (theta == phases[-4] or theta == phases[-3]):
            el = multi_pulse_elt(i, station,
                                 [operation_dict['I ' + measured_qubit_name],
                                  operation_dict['RO ' + measured_qubit_name]])
        elif cal_points and (theta == phases[-2] or theta == phases[-1]):
            el = multi_pulse_elt(i, station,
                                 [operation_dict['X180 ' + measured_qubit_name],
                                  operation_dict['RO ' + measured_qubit_name]])
        else:
            if i % 2 == 0:
                el = multi_pulse_elt(
                    i, station, [operation_dict['X90 ' + measured_qubit_name],
                                 X90_2,
                                 operation_dict['RO ' + measured_qubit_name]])
            else:
                # X90_2['phase'] += 12
                # VZ = deepcopy(operation_dict['Z180 '+measured_qubit_name])
                # VZ['basis_rotation'][measured_qubit_name] = -12
                el = multi_pulse_elt(
                    i, station, [operation_dict['X90 ' + measured_qubit_name],
                                 operation_dict['X180s ' + pulsed_qubit_name],
                                 # VZ,
                                 X90_2,
                                 operation_dict['RO ' + measured_qubit_name]])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def fgge_gate_length_seq(lengths, qbt_name, qbm_name, fgge_pulse_name,
                         amplitude, mod_frequency, operation_dict,
                         upload=True, upload_all=True,
                         return_seq=False, cal_points=True,
                         verbose=False):

    seq_name = 'fgge_gate_seq'
    seq = sequence.Sequence(seq_name)
    el_list = []

    if amplitude > 1:
        raise ValueError('fgge pulse amplitude must be smaller than 1.')

    fgge_pulse = deepcopy(operation_dict[fgge_pulse_name])
    fgge_pulse['amplitude'] = amplitude
    fgge_pulse['mod_frequency'] = mod_frequency

    if upload_all:
        upload_AWGs = 'all'
    else:
        upload_AWGs = [station.pulsar.get(fgge_pulse['I_channel'] + '_AWG'),
                       station.pulsar.get(fgge_pulse['Q_channel'] + '_AWG')]

    for i, length in enumerate(lengths):
        if cal_points and (i == (len(lengths)-4) or i == (len(lengths)-3)):
            el = multi_pulse_elt(i, station,
                                 [operation_dict['I ' + qbm_name],
                                  operation_dict['RO ' + qbm_name]])
        elif cal_points and (i == (len(lengths)-2) or i == (len(lengths)-1)):
            el = multi_pulse_elt(i, station,
                                 [operation_dict['X180 ' + qbm_name],
                                  operation_dict['RO ' + qbm_name]])
        else:
            fgge_pulse['pulse_length'] = length - \
                                             (fgge_pulse['nr_sigma'] *
                                              fgge_pulse['gaussian_filter_sigma'])
            el = multi_pulse_elt(i, station,
                                 [operation_dict['X180 ' + qbt_name],
                                  fgge_pulse,
                                  operation_dict['RO ' + qbm_name]])

        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list,
                                    AWGs=upload_AWGs,
                                    channels='all',
                                    verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


#### Multi-qubit time-domain
def general_multi_qubit_seq(
        sweep_params,
        sweep_points,
        qb_names,
        operation_dict,   # of all qubits
        qb_names_DD=None,
        cal_points=True,
        no_cal_points=4,
        nr_echo_pulses=0,
        idx_DD_start=-1,
        UDD_scheme=True,
        upload=True,
        return_seq=False,
        verbose=False):

    """
    sweep_params = {
	        Pulse_type: (pulse_pars)
        }
    Ex:
        # Rabi
        sweep_params = (
	        ('X180', {'pulse_pars': {'amplitude': (lambda sp: sp),
	                 'repeat': 3}}), # for n-rabi; leave out if normal rabi
	    )

	    # Ramsey
        sweep_params = (
            ('X90', {}),
            ('X90', {'pulse_pars': {'refpoint': 'start',
                                    'pulse_delay': (lambda sp: sp),
                                    'phase': (lambda sp:
                        (            (sp-sweep_points[0]) *
                                    art_det * 360) % 360)}}),
        )

        # T1
	    sweep_params = (
	        ('X180', {}),
	        ('RO_mux', {'pulse_pars': {'pulse_delay': (lambda sp: sp)}})
        )

        # Echo
        sweep_params = (
            ('X90', {}),
            ('X180', {'pulse_pars': {'refpoint': 'start',
                                     'pulse_delay': (lambda sp: sp/2)}}),
            ('X90', {'pulse_pars': {
                'refpoint': 'start',
                'pulse_delay': (lambda sp: sp/2),
                'phase': (lambda sp:
                          ((sp-sweep_points[0]) * artificial_detuning *
                           360) % 360)}})

        )

        # QScale
        sweep_params = (
            ('X90', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                     'condition': (lambda i: i%3==0)}),
            ('X180', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                     'condition': (lambda i: i%3==0)}),
            ('X90', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                     'condition': (lambda i: i%3==1)}),
            ('Y180', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                     'condition': (lambda i: i%3==1)}),
            ('X90', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                     'condition': (lambda i: i%3==2)}),
            ('mY180', {'pulse_pars': {'motzoi': (lambda sp: sp)},
                     'condition': (lambda i: i%3==2)}),
            ('RO', {})
        )
    """

    seq_name = 'TD_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    len_sweep_pts = len(sweep_points)

    if (not isinstance(sweep_points, list) and
            not isinstance(sweep_points, np.ndarray)):
        if isinstance(sweep_points, dict):
            len_sweep_pts = len(sweep_points[list(sweep_points)[0]])
            assert (np.all([len_sweep_pts ==
                            len(sp) for sp in sweep_points.values()]))
        else:
            raise ValueError('Unrecognized type for "sweep_points".')

    for i in np.arange(len_sweep_pts):
        pulse_list = []
        if cal_points and no_cal_points == 4 and \
                (i == (len_sweep_pts-4) or i == (len_sweep_pts-3)):
            for qb_name in qb_names:
                qbn = ' ' + qb_name
                if qb_name != qb_names[0]:
                    qbn = 's ' + qb_name
                pulse_list += [operation_dict['I' + qbn]]
        elif cal_points and no_cal_points == 4 and \
                (i == (len_sweep_pts-2) or i == (len_sweep_pts-1)):
            for qb_name in qb_names:
                qbn = ' ' + qb_name
                if qb_name != qb_names[0]:
                    qbn = 's ' + qb_name
                pulse_list += [operation_dict['X180' + qbn]]
        elif cal_points and no_cal_points == 2 and \
                (i == (len_sweep_pts-2) or i == (len_sweep_pts-1)):
            for qb_name in qb_names:
                qbn = ' ' + qb_name
                if qb_name != qb_names[0]:
                    qbn = 's ' + qb_name
                pulse_list += [operation_dict['I' + qbn]]
        else:
            for sweep_tuple in sweep_params:
                pulse_key = [x for x in sweep_tuple if isinstance(x, str)][0]
                params_dict = [x for x in sweep_tuple if isinstance(x, dict)][0]

                proceed = True

                if 'condition' in params_dict:
                    if not params_dict['condition'](i):
                        proceed = False

                if proceed:
                    if 'mux' in pulse_key:
                        pulse_pars_dict = deepcopy(operation_dict[pulse_key])
                        if 'pulse_pars' in params_dict:
                            for pulse_par_name, pulse_par in \
                                    params_dict['pulse_pars'].items():
                                if hasattr(pulse_par, '__call__'):
                                    if isinstance(sweep_points, dict):
                                        if 'RO mux' in sweep_points.keys():
                                            pulse_pars_dict[pulse_par_name] = \
                                                pulse_par(sweep_points[
                                                              'RO mux'][i])
                                    else:
                                        pulse_pars_dict[pulse_par_name] = \
                                            pulse_par(sweep_points[i])
                                else:
                                    pulse_pars_dict[pulse_par_name] = \
                                        pulse_par
                        pulse_list += [pulse_pars_dict]
                    else:
                        for qb_name in qb_names:
                            pulse_pars_dict = deepcopy(
                                operation_dict[pulse_key + ' ' + qb_name])
                            if 'pulse_pars' in params_dict:
                                for pulse_par_name, pulse_par in \
                                        params_dict['pulse_pars'].items():
                                    if hasattr(pulse_par, '__call__'):
                                        if isinstance(sweep_points, dict):
                                            pulse_pars_dict[pulse_par_name] = \
                                                pulse_par(sweep_points[
                                                              qb_name][i])
                                        else:
                                            if pulse_par_name == \
                                                    'basis_rotation':
                                                pulse_pars_dict[
                                                    pulse_par_name] = {}
                                                pulse_pars_dict[pulse_par_name][
                                                    [qbn for qbn in qb_names if
                                                    qbn != qb_name][0]] = \
                                                    - pulse_par(sweep_points[i])

                                            else:
                                                pulse_pars_dict[
                                                    pulse_par_name] = \
                                                    pulse_par(sweep_points[i])
                                    else:
                                        pulse_pars_dict[pulse_par_name] = \
                                            pulse_par
                            if qb_name != qb_names[0]:
                                pulse_pars_dict['refpoint'] = 'simultaneous'

                            pulse_list += [pulse_pars_dict]

                if 'repeat' in params_dict:
                    n = params_dict['repeat']
                    pulse_list = n*pulse_list

            if nr_echo_pulses != 0:
                if qb_names_DD is None:
                    qb_names_DD = qb_names
                pulse_list = get_DD_pulse_list(operation_dict, qb_names,
                                               DD_time=sweep_points[i],
                                               qb_names_DD=qb_names_DD,
                                               pulse_dict_list=pulse_list,
                                               nr_echo_pulses=nr_echo_pulses,
                                               idx_DD_start=idx_DD_start,
                                               UDD_scheme=UDD_scheme)

        if not np.any([p['operation_type'] == 'RO' for p in pulse_list]):
            # print('in add mux')
            pulse_list += [operation_dict['RO mux']]
        el = multi_pulse_elt(i, station, pulse_list)

        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq


def get_dd_pulse_list(operation_dict, qb_names, dd_time, nr_pulses=4, 
                      dd_scheme='cpmg', init_buffer=0, refpoint='end'):
    drag_pulse_length = (operation_dict['X180 ' + qb_names[-1]]['nr_sigma'] * \
                         operation_dict['X180 ' + qb_names[-1]]['sigma'])
    pulse_list = []
    def cpmg_delay(i, nr_pulses=nr_pulses):
        if i == 0 or i == nr_pulses:
            return 1/nr_pulses/2
        else:
            return 1/nr_pulses

    def udd_delay(i, nr_pulses=nr_pulses):
        delay = np.sin(0.5*np.pi*(i + 1)/(nr_pulses + 1))**2
        delay -= np.sin(0.5*np.pi*i/(nr_pulses + 1))**2
        return delay

    if dd_scheme == 'cpmg':
        delay_func = cpmg_delay
    elif dd_scheme == 'udd':
        delay_func = udd_delay
    else:
        raise ValueError('Unknown decoupling scheme "{}"'.format(dd_scheme))

    for i in range(nr_pulses+1):
        delay_pulse = {
            'pulse_type': 'SquarePulse',
            'channel': operation_dict['X180 ' + qb_names[0]]['I_channel'],
            'amplitude': 0.0,
            'length': dd_time*delay_func(i),
            'pulse_delay': 0}
        if i == 0:
            delay_pulse['pulse_delay'] = init_buffer
            delay_pulse['refpoint'] = refpoint
        if i == 0 or i == nr_pulses:
            delay_pulse['length'] -= drag_pulse_length/2
        else:
            delay_pulse['length'] -= drag_pulse_length
        if delay_pulse['length'] > 0:
            pulse_list.append(delay_pulse)
        else:
            raise Exception("Dynamical decoupling pulses don't fit in the "
                            "specified dynamical decoupling duration "
                            "{:.2f} ns".format(dd_time*1e9))
        if i != nr_pulses:
            for j, qbn in enumerate(qb_names):
                pulse_name = 'X180 ' if j == 0 else 'X180s '
                pulse_pulse = deepcopy(operation_dict[pulse_name + qbn])
                pulse_list.append(pulse_pulse)
    return pulse_list

def get_DD_pulse_list(operation_dict, qb_names, DD_time,
                      qb_names_DD=None,
                      pulse_dict_list=None, idx_DD_start=-1,
                      nr_echo_pulses=4, UDD_scheme=True):

    n = len(qb_names)
    if qb_names_DD is None:
        qb_names_DD = qb_names
    if pulse_dict_list is None:
        pulse_dict_list = []
    elif len(pulse_dict_list) < 2*n:
        raise ValueError('The pulse_dict_list must have at least two entries.')

    idx_DD_start *= n
    pulse_dict_list_end = pulse_dict_list[idx_DD_start::]
    pulse_dict_list = pulse_dict_list[0:idx_DD_start]

    X180_pulse_dict = operation_dict['X180 ' + qb_names_DD[0]]
    DRAG_length = X180_pulse_dict['nr_sigma']*X180_pulse_dict['sigma']
    X90_separation = DD_time - DRAG_length

    if UDD_scheme:
        pulse_positions_func = \
            lambda idx, N: np.sin(np.pi*idx/(2*N+2))**2
        pulse_delays_func = (lambda idx, N: X90_separation*(
                pulse_positions_func(idx, N) -
                pulse_positions_func(idx-1, N)) -
                ((0.5 if idx == 1 else 1)*DRAG_length))

        if nr_echo_pulses*DRAG_length > X90_separation:
            pass
        else:
            for p_nr in range(nr_echo_pulses):
                for qb_name in qb_names_DD:
                    if qb_name == qb_names_DD[0]:
                        DD_pulse_dict = deepcopy(operation_dict[
                                                     'X180 ' + qb_name])
                        DD_pulse_dict['refpoint'] = 'end'
                        DD_pulse_dict['pulse_delay'] = pulse_delays_func(
                            p_nr+1, nr_echo_pulses)
                    else:
                        DD_pulse_dict = deepcopy(operation_dict[
                                                     'X180s ' + qb_name])
                    pulse_dict_list.append(DD_pulse_dict)

            for j in range(n):
                if j == 0:
                    pulse_dict_list_end[j]['refpoint'] = 'end'
                    pulse_dict_list_end[j]['pulse_delay'] = pulse_delays_func(
                        1, nr_echo_pulses)
                else:
                    pulse_dict_list_end[j]['pulse_delay'] = 0
    else:
        echo_pulse_delay = (X90_separation -
                            nr_echo_pulses*DRAG_length) / \
                           nr_echo_pulses
        if echo_pulse_delay < 0:
            pass
        else:
            start_end_delay = echo_pulse_delay/2
            for p_nr in range(nr_echo_pulses):
                for qb_name in qb_names_DD:
                    if qb_name == qb_names_DD[0]:
                        DD_pulse_dict = deepcopy(operation_dict[
                                                     'X180 ' + qb_name])
                        DD_pulse_dict['refpoint'] = 'end'
                        DD_pulse_dict['pulse_delay'] = \
                            (start_end_delay if p_nr == 0 else echo_pulse_delay)
                    else:
                        DD_pulse_dict = deepcopy(operation_dict[
                                                     'X180s ' + qb_name])
                    pulse_dict_list.append(DD_pulse_dict)
            for j in range(n):
                if j == 0:
                    pulse_dict_list_end[j]['refpoint'] = 'end'
                    pulse_dict_list_end[j]['pulse_delay'] = start_end_delay
                else:
                    pulse_dict_list_end[j]['pulse_delay'] = 0

    pulse_dict_list += pulse_dict_list_end

    return pulse_dict_list

def fgge_frequency_seq(mod_frequencies, length, amplitude,
                       qbt_name, qbm_name,
                       fgge_pulse_name, operation_dict,
                       upload_all=True,
                       verbose=False, cal_points=False,
                       upload=True, return_seq=False):



    seq_name = 'fgge_frequency_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    RO_pulse = operation_dict['RO ' + qbm_name]
    fgge_pulse = operation_dict[fgge_pulse_name]

    fgge_pulse['amplitude'] = amplitude
    fgge_pulse['pulse_length'] = length

    # if upload_all:
    upload_AWGs = 'all'
    upload_channels = 'all'
    # else:
    #     upload_AWGs = [station.pulsar.get(fgge_pulse['channel'] + '_AWG')]
    #     upload_channels = 'all'

    for i, frequency in enumerate(mod_frequencies):
        if cal_points and (i == (len(mod_frequencies)-4) or
                                   i == (len(mod_frequencies)-3)):
            el = multi_pulse_elt(i, station, [RO_pulse])
        elif cal_points and (i == (len(mod_frequencies)-2) or
                                     i == (len(mod_frequencies)-1)):
            el = multi_pulse_elt(i, station,
                                 [operation_dict['X180 ' + qbm_name],
                                  RO_pulse])
        else:
            fgge_pulse['mod_frequency'] = frequency
            el = multi_pulse_elt(i, station,
                                 # [operation_dict['X180 ' + qbt_name],
                                 [fgge_pulse,
                                  RO_pulse])

        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list,
                                    AWGs=upload_AWGs,
                                    channels=upload_channels,
                                    verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def pygsti_seq(qb_names, pygsti_listOfExperiments, operation_dict,
               preselection=True, ro_spacing=1e-6,
               seq_name=None, upload=True, upload_all=True,
               return_seq=False, verbose=False):

    if seq_name is None:
        seq_name = 'GST_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    tup_lst = [g.tup for g in pygsti_listOfExperiments]
    str_lst = []
    for t1 in tup_lst:
        s = ''
        for t in t1:
            s += str(t)
        str_lst += [s]
    experiment_lists = get_exp_list(filename='',
                                    pygstiGateList=str_lst,
                                    qb_names=qb_names)
    if preselection:
        RO_str = 'RO' if len(qb_names) == 1 else 'RO mux'
        operation_dict[RO_str+'_presel'] = \
            operation_dict[experiment_lists[0][-1]].copy()
        operation_dict[RO_str+'_presel']['pulse_delay'] = -ro_spacing
        operation_dict[RO_str+'_presel']['refpoint'] = 'start'
        operation_dict[RO_str+'presel_dummy'] = {
            'pulse_type': 'SquarePulse',
            'channel': operation_dict[
                experiment_lists[0][-1]]['acq_marker_channel'],
            'amplitude': 0.0,
            'length': ro_spacing,
            'pulse_delay': 0}

    if upload_all:
        upload_AWGs = 'all'
    else:
        upload_AWGs = ['AWG1']
        for qbn in qb_names:
            X90_pulse = deepcopy(operation_dict['X90 ' + qbn])
            upload_AWGs += [station.pulsar.get(X90_pulse['I_channel'] + '_AWG'),
                           station.pulsar.get(X90_pulse['Q_channel'] + '_AWG')]
        if len(qb_names) > 1:
            CZ_pulse_name = 'CZ {} {}'.format(qb_names[1], qb_names[0])
            if len([i for i in experiment_lists if CZ_pulse_name in i]) > 0:
                CZ_pulse = operation_dict[CZ_pulse_name]
                upload_AWGs += [station.pulsar.get(CZ_pulse['channel'] + '_AWG')]
                for ch in CZ_pulse['aux_channels_dict']:
                    upload_AWGs += [station.pulsar.get(ch + '_AWG')]
        upload_AWGs = list(set(upload_AWGs))
    print(upload_AWGs)
    for i, exp_lst in enumerate(experiment_lists):
        pulse_lst = [operation_dict[p] for p in exp_lst]
        from pprint import pprint
        if preselection:
            pulse_lst.append(operation_dict[RO_str+'_presel'])
            pulse_lst.append(operation_dict[RO_str+'presel_dummy'])
        el = multi_pulse_elt(i, station, pulse_lst)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list,
                                    AWGs=upload_AWGs,
                                    channels='all',
                                    verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def ro_dynamic_phase_seq(qbp_name, qbr_names,
                         phases, operation_dict,
                         pulse_separation, init_state,
                         verbose=False, cal_points=True,
                         upload=True, return_seq=False):

    """
    RO cross-dephasing measurement sequence. Measures the dynamic phase induced
    on qbr by a measurement tone on the pulsed qubit (qbp).
    Args:
        qbp_name: pulsed qubit name; RO pulse is applied on this qubit
        qbr_names: ramsey (measured) qubits
        phases: array of phases for the second piHalf pulse on qbr
        operation_dict: contains operation dicts from both qubits;
            !!! Must contain 'RO mux' which is the mux RO pulse only for
            the measured_qubits (qbr_names) !!!
        pulse_separation: separation between the two pi-half pulses, shouls be
            equal to integration length
        cal_points: use cal points
    """

    seq_name = 'ro_dynamic_phase_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    # put together n-qubit calibration point pulse lists
    # put together n-qubit ramsey pulse lists
    cal_I_pulses = []
    cal_X_pulses = []
    x90_1_pulses = []  # first ramsey pulse
    x90_2a_pulses = []  # with readout pulse
    x90_2b_pulses = []  # without readout pulse
    for j, qbr_name in enumerate(qbr_names):
        if j == 0:
            cal_I_pulses.append(deepcopy(operation_dict['I ' + qbr_name]))
            cal_X_pulses.append(deepcopy(operation_dict['X180 ' + qbr_name]))
            x90_1_pulses.append(deepcopy(operation_dict['X90 ' + qbr_name]))
            x90_2a_pulses.append(deepcopy(operation_dict['X90 ' + qbr_name]))
            x90_2b_pulses.append(deepcopy(operation_dict['X90 ' + qbr_name]))
            x90_2a_pulses[-1]['pulse_delay'] = pulse_separation
            x90_2a_pulses[-1]['refpoint'] = 'start'
            x90_2b_pulses[-1]['pulse_delay'] = pulse_separation
        else:
            cal_I_pulses.append(deepcopy(operation_dict['Is ' + qbr_name]))
            cal_X_pulses.append(deepcopy(operation_dict['X180s ' + qbr_name]))
            x90_1_pulses.append(deepcopy(operation_dict['X90s ' + qbr_name]))
            x90_2a_pulses.append(deepcopy(operation_dict['X90s ' + qbr_name]))
            x90_2b_pulses.append(deepcopy(operation_dict['X90s ' + qbr_name]))
    if init_state == 'e':
        x90_1_pulses.append(deepcopy(operation_dict['X180 ' + qbp_name]))
    cal_I_pulses.append(operation_dict['RO mux'])
    cal_X_pulses.append(operation_dict['RO mux'])

    for i, theta in enumerate(phases):
        if cal_points and (i == (len(phases)-4) or i == (len(phases)-3)):
            pulse_list = [operation_dict['I ' + qbp_name],
                          operation_dict['RO ' + qbp_name]]
            el = multi_pulse_elt(3*i, station, pulse_list)
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)
            for j in range(1, 3):
                el = multi_pulse_elt(3*i + j, station, cal_I_pulses)
                el_list.append(el)
                seq.append_element(el, trigger_wait=True)
        elif cal_points and (i == (len(phases)-2) or i == (len(phases)-1)):
            pulse_list = [operation_dict['X180 ' + qbp_name],
                          operation_dict['RO ' + qbp_name]]
            el = multi_pulse_elt(3*i, station, pulse_list)
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)
            for j in range(1, 3):
                el = multi_pulse_elt(3*i + j, station, cal_X_pulses)
                el_list.append(el)
                seq.append_element(el, trigger_wait=True)
        else:
            for qbr_X90_pulse in x90_2a_pulses + x90_2b_pulses:
                qbr_X90_pulse['phase'] = theta*180/np.pi

            pulse_list = x90_1_pulses + [operation_dict['RO ' + qbp_name]] + \
                         x90_2a_pulses + [operation_dict['RO mux']]
            el = multi_pulse_elt(3*i, station, pulse_list)
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)

            pulse_list = x90_1_pulses + x90_2b_pulses + \
                         [operation_dict['RO mux']]
            el = multi_pulse_elt(3*i + 1, station, pulse_list)
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name

def generate_mux_ro_pulse_list(qubit_names, operation_dict, element_name='RO',
                               ref_point='end', pulse_delay=0.0):
    ro_pulses = []
    for j, qb_name in enumerate(qubit_names):
        ro_pulse = deepcopy(operation_dict['RO ' + qb_name])
        ro_pulse['pulse_name'] = '{}_{}'.format(element_name, j)
        ro_pulse['element_name'] = element_name
        if j == 0:
            ro_pulse['pulse_delay'] = pulse_delay
            ro_pulse['ref_point'] = ref_point
        else:
            ro_pulse['ref_point'] = 'start'
        ro_pulses.append(ro_pulse)
    return ro_pulses

def interleaved_pulse_list_equatorial_seg(
        qubit_names, operation_dict, interleaved_pulse_list, phase, 
        pihalf_spacing=None, segment_name='segment'):
    pulse_list = []
    for notfirst, qbn in enumerate(qubit_names):
        pulse_list.append(deepcopy(operation_dict['X90 ' + qbn])) 
        pulse_list[-1]['ref_point'] = 'start'
        if not notfirst:
            pulse_list[-1]['name'] = 'refpulse'
    pulse_list += interleaved_pulse_list
    for notfirst, qbn in enumerate(qubit_names):
        pulse_list.append(deepcopy(operation_dict['X90 ' + qbn])) 
        pulse_list[-1]['phase'] = phase
        if notfirst:
            pulse_list[-1]['ref_point'] = 'start'
        elif pihalf_spacing is not None:
            pulse_list[-1]['ref_pulse'] = 'refpulse'
            pulse_list[-1]['ref_point'] = 'start'
            pulse_list[-1]['pulse_delay'] = pihalf_spacing
    pulse_list += generate_mux_ro_pulse_list(qubit_names, operation_dict)
    return segment.Segment(segment_name, pulse_list)

def interleaved_pulse_list_list_equatorial_seq(
        qubit_names, operation_dict, interleaved_pulse_list_list, phases, 
        pihalf_spacing=None, cal_points=True,
        sequence_name='equatorial_sequence', upload=True):
    seq = sequence.Sequence(sequence_name)
    for i, interleaved_pulse_list in enumerate(interleaved_pulse_list_list):
        for j, phase in enumerate(phases):
            seg = interleaved_pulse_list_equatorial_seg(
                qubit_names, operation_dict, interleaved_pulse_list, phase,
                pihalf_spacing=pihalf_spacing, segment_name=f'segment_{i}_{j}')
            seq.add(seg)
    if cal_points:
        # TODO: replace this part of code with more general cal point code
        for i, cal_pulse in enumerate(['I ', 'I ', 'X180 ', 'X180 ']):
            pulse_list = []
            for notfirst, qbn in enumerate(qubit_names):
                pulse_list.append(deepcopy(operation_dict[cal_pulse + qbn])) 
                pulse_list[-1]['ref_point'] = 'start'
            pulse_list += \
                generate_mux_ro_pulse_list(qubit_names, operation_dict)
            
            seg = segment.Segment(f'calibration_{i}', pulse_list)
            seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    return seq

def measurement_induced_dephasing_seq(
        measured_qubit_names, dephased_qubit_names, operation_dict, 
        ro_amp_scales, phases, pihalf_spacing=None, cal_points=True,
        sequence_name='measurement_induced_dephasing_seq', upload=True):
    interleaved_pulse_list_list = []
    for i, ro_amp_scale in enumerate(ro_amp_scales):
        interleaved_pulse_list = generate_mux_ro_pulse_list(
            measured_qubit_names, operation_dict, 
            element_name=f'interleaved_readout_{i}')
        for pulse in interleaved_pulse_list:
            pulse['amplitude'] *= ro_amp_scale
            pulse['operation_type'] = None
        interleaved_pulse_list_list.append(interleaved_pulse_list)
    return interleaved_pulse_list_list_equatorial_seq(
        dephased_qubit_names, operation_dict, interleaved_pulse_list_list, 
        phases, pihalf_spacing=pihalf_spacing, cal_points=cal_points,
        sequence_name=sequence_name, upload=upload)