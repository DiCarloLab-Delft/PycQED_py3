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
    get_pulse_dict_from_pars, add_preparation_pulses, pulse_list_list_seq, \
    prepend_pulses, add_suffix, sweep_pulse_params
from pycqed.measurement.gate_set_tomography.gate_set_tomography import \
    create_experiment_list_pyGSTi_qudev as get_exp_list
from pycqed.measurement.waveform_control import pulsar as ps
import pycqed.measurement.waveform_control.segment as segment

station = None
kernel_dir = 'kernels/'
# You need to explicitly set this before running any functions from this module
# I guess there are cleaner solutions :)
cached_kernels = {}


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

    repeat_dict = {}
    repeat_pattern = ((1.0 + int(preselection))*len(pulse_combinations),1)
    for i, RO_pars in enumerate(RO_pars_list):
        repeat_dict = seq.repeat(RO_pars, None, repeat_pattern)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    if return_seq:
        return seq, seg_list
    else:
        return seq_name


def two_qubit_randomized_benchmarking_seqs(
        qb1n, qb2n, operation_dict, cliffords, nr_seeds,
        max_clifford_idx=11520, cz_pulse_name=None, cal_points=None,
        net_clifford=0, clifford_decomposition_name='HZ',
        interleaved_gate=None, upload=True, prep_params=dict()):

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
        upload (bool): whether to upload sequence to AWGs
    """
    seq_name = '2Qb_RB_sequence'

    # Set Clifford decomposition
    tqc.gate_decomposition = rb.get_clifford_decomposition(
        clifford_decomposition_name)

    sequences = []
    for nCl in cliffords:
        pulse_list_list_all = []
        for _ in nr_seeds:
            cl_seq = rb.randomized_benchmarking_sequence_new(
                nCl,
                number_of_qubits=2,
                max_clifford_idx=max_clifford_idx,
                interleaving_cl=interleaved_gate,
                desired_net_cl=net_clifford)

            pulse_list = []
            pulsed_qubits = {qb1n, qb2n}
            for idx in cl_seq:
                pulse_tuples_list = tqc.TwoQubitClifford(idx).gate_decomposition
                for j, pulse_tuple in enumerate(pulse_tuples_list):
                    if isinstance(pulse_tuple[1], list):
                        pulse_list += [operation_dict[cz_pulse_name]]
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
                        pulse_list += [
                            operation_dict[pulse_name + ' ' + qb_name]]
            pulse_list += generate_mux_ro_pulse_list(
                [qb1n, qb2n], operation_dict)
            pulse_list_w_prep = add_preparation_pulses(
                pulse_list, operation_dict, [qb1n, qb2n], **prep_params)
            pulse_list_list_all.append(pulse_list_w_prep)
        seq = pulse_list_list_seq(pulse_list_list_all, seq_name+f'_{nCl}',
                                  upload=False)
        if cal_points is not None:
            seq.extend(cal_points.create_segments(operation_dict,
                                                  **prep_params))
        sequences.append(seq)

    # reuse sequencer memory by repeating readout pattern
    for s in sequences:
        s.repeat_ro(f"RO {qb1n}", operation_dict)
        s.repeat_ro(f"RO {qb2n}", operation_dict)

    if upload:
        ps.Pulsar.get_instance().program_awgs(sequences[0])

    return sequences, np.arange(sequences[0].n_acq_elements()), \
           np.arange(len(cliffords))


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


def n_qubit_reset(qb_names, operation_dict, prep_params=dict(), upload=True,
                  states=('g','e',)):
    """
    :param qb_names: list of qb_names to perform simultaneous reset upon
    :param states (tuple): ('g','e',) for active reset e, ('g','f',) for active
    reset f and ('g', 'e', 'f') for both.
    :param prep_params (dict): preparation parameters. Note: should be
        multi_qb_preparation_params, ie. threshold mapping should be of the
        form:  {qbi: thresh_map_qbi for qbi in qb_names}

    :return:
    """

    seq_name = '{}_reset_x{}_sequence'.format(qb_names,
                                              prep_params.get('reset_reps',
                                                              '_default_n_reps'))


    pulses = generate_mux_ro_pulse_list(qb_names, operation_dict)
    reset_and_last_ro_pulses = \
        add_preparation_pulses(pulses, operation_dict, qb_names, **prep_params)
    swept_pulses = []

    state_ops = dict(g=['I '], e=['X180 '], f=['X180 ', 'X180_ef '])
    for s in states:
        pulses = deepcopy(reset_and_last_ro_pulses)
        state_pulses = []
        segment_pulses = []
        # generate one sub list for each qubit, with qb pulse naming
        for qbn in qb_names:
            qb_state_pulses  = [deepcopy(operation_dict[op + qbn]) for op in
                            state_ops[s]]
            for op, p in zip(state_ops[s], qb_state_pulses):
                p['name'] = op + qbn
            state_pulses += [qb_state_pulses]
        # reference end of state pulse to start of first reset pulse,
        # to effectively prepend the state pulse

        for qb_state_pulses in state_pulses:
            segment_pulses += prepend_pulses(pulses, qb_state_pulses)[
                              :len(qb_state_pulses)]
        swept_pulses.append(segment_pulses + pulses)

    seq = pulse_list_list_seq(swept_pulses, seq_name, upload=False)

    # reuse sequencer memory by repeating readout pattern
    # 1. get all readout pulse names (if they are on different uhf,
    # they will be applied to different channels)
    ro_pulse_names = [p["pulse_name"] for p in
                      generate_mux_ro_pulse_list(qb_names, operation_dict)]
    # 2. repeat readout for each ro_pulse.
    [seq.repeat_ro(pn, operation_dict) for pn in ro_pulse_names]

    log.debug(seq)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())


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
            log.info('Length difference of XX and ZZ cycles: {} s'.format(
                el_repeat_x.ideal_length() - el_repeat_z.ideal_length()
            ))
        dynamic_phase -= el_main.drive_phase_offsets.get(qbn, 0)
        phase_from_if = 360*ifreq*elements_length
        total_phase = phase_from_if + dynamic_phase
        total_mod_phase = (total_phase + 180) % 360 - 180
        log.info(qbn + ' aquires a phase of {} ≡ {} (mod 360)'.format(
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


def parity_single_round_seq(ancilla_qubit_name, data_qubit_names, CZ_map,
                            preps, cal_points, prep_params, operation_dict,
                            upload=True):
    seq_name = 'Parity_1_round_sequence'
    qb_names = [ancilla_qubit_name] + data_qubit_names

    main_ops = ['Y90 ' + ancilla_qubit_name]
    for i, dqn in enumerate(data_qubit_names):
        op = 'CZ ' + ancilla_qubit_name + ' ' + dqn
        main_ops += CZ_map.get(op, [op])
        if i == len(data_qubit_names)/2 - 1:
            main_ops += ['Y180 ' + ancilla_qubit_name]
            # for dqnecho in enumerate(data_qubit_names):
            #             #     main_ops += ['Y180s ' + dqnecho]
    # if len(data_qubit_names)%2 == 0:
    main_ops += ['Y90 ' + ancilla_qubit_name]
    # else:
    # main_ops += ['mY90 ' + ancilla_qubit_name]

    all_opss = []
    for prep in preps:
        prep_ops = [{'g': 'I ', 'e': 'X180 ', '+': 'Y90 ', '-': 'mY90 '}[s] \
             + dqn for s, dqn in zip(prep, data_qubit_names)]
        end_ops = [{'g': 'I ', 'e': 'I ', '+': 'Y90 ', '-': 'Y90 '}[s] \
             + dqn for s, dqn in zip(prep, data_qubit_names)]
        all_opss.append(prep_ops + main_ops + end_ops)
    all_pulsess = []
    for all_ops, prep in zip(all_opss, preps):
        all_pulses = []
        for i, op in enumerate(all_ops):
            all_pulses.append(deepcopy(operation_dict[op]))
            # if i == 0:
            #     all_pulses[-1]['ref_pulse'] = 'segment_start'
            # elif 0 < i <= len(data_qubit_names):
            #     all_pulses[-1]['ref_point'] = 'start'
            if 'CZ' not in op:
                all_pulses[-1]['element_name'] = f'drive_{prep}'
            else:
                all_pulses[-1]['element_name'] = f'flux_{prep}'
        all_pulses += generate_mux_ro_pulse_list(qb_names, operation_dict)
        all_pulsess.append(all_pulses)

    # all_pulsess_with_prep = \
    #     [add_preparation_pulses(seg, operation_dict, qb_names, **prep_params)
    #      for seg in all_pulsess]
    all_pulsess_with_prep = all_pulsess

    seq = pulse_list_list_seq(all_pulsess_with_prep, seq_name, upload=False)

    # add calibration segments
    if cal_points is not None:
        seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    if upload:
       ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())


def parity_single_round__phases_seq(ancilla_qubit_name, data_qubit_names, CZ_map,
                                    phases, prep_anc,
                                    cal_points, prep_params,
                                    operation_dict,
                                    upload=True):
    seq_name = 'Parity_1_round_sequence'
    qb_names = [ancilla_qubit_name] + data_qubit_names

    if prep_anc=='e':
        main_ops = ['Y180 ' + ancilla_qubit_name]
    else:
        main_ops = ['I ' + ancilla_qubit_name]
    for i, dqn in enumerate(data_qubit_names):
        op = 'CZ ' + ancilla_qubit_name + ' ' + dqn
        main_ops += CZ_map.get(op, [op])
        if i == len(data_qubit_names)/2 - 1:
            main_ops += ['Y180 ' + ancilla_qubit_name]

    prep_ops = ['Y90' + (' ' if i==0 else 's ') + dqn for i,dqn in
                enumerate(data_qubit_names)]
    end_ops = ['mY90' + (' ' if i==0 else 's ') + dqn for i,dqn in
                enumerate(data_qubit_names)]

    all_pulsess = []
    for n, phase in enumerate(phases):
        all_pulses = []
        for i, op in enumerate(prep_ops+main_ops):
            all_pulses.append(deepcopy(operation_dict[op]))
            if 'CZ' not in op:
                all_pulses[-1]['element_name'] = f'drive_{n}'
            else:
                all_pulses[-1]['element_name'] = f'flux_{n}'
        for i, op in enumerate(end_ops):
            all_pulses.append(deepcopy(operation_dict[op]))
            all_pulses[-1]['element_name'] = f'drive_{n}'
            all_pulses[-1]['phase'] = phase/np.pi*180

        all_pulses += generate_mux_ro_pulse_list(qb_names, operation_dict)
        all_pulsess.append(all_pulses)

    all_pulsess_with_prep = \
        [add_preparation_pulses(seg, operation_dict, qb_names, **prep_params)
         for seg in all_pulsess]

    seq = pulse_list_list_seq(all_pulsess_with_prep, seq_name, upload=False)

    # add calibration segments
    if cal_points is not None:
        seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    if upload:
       ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())

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


def ramsey_add_pulse_seq_active_reset(
        times, measured_qubit_name, pulsed_qubit_name,
        operation_dict, cal_points, n=1, artificial_detunings = 0,
        upload=True, for_ef=False, last_ge_pulse=False, prep_params=dict()):
    '''
     Azz sequence:  Ramsey on measured_qubit
                    pi-pulse on pulsed_qubit
     Input pars:
         times:           array of delays (s)
         n:               number of pulses (1 is conventional Ramsey)
     '''
    seq_name = 'Ramsey_with_additional_pulse_sequence'

    # Operations
    if for_ef:
        ramsey_ops_measured = ["X180"] + ["X90_ef"] * 2 * n
        ramsey_ops_pulsed = ["X180"]
        if last_ge_pulse:
            ramsey_ops_measured += ["X180"]
    else:
        ramsey_ops_measured = ["X90"] * 2 * n
        ramsey_ops_pulsed = ["X180"]

    ramsey_ops_measured += ["RO"]
    ramsey_ops_measured = add_suffix(ramsey_ops_measured, " " + measured_qubit_name)
    ramsey_ops_pulsed = add_suffix(ramsey_ops_pulsed, " " + pulsed_qubit_name)
    ramsey_ops_init = ramsey_ops_pulsed + ramsey_ops_measured
    ramsey_ops_det = ramsey_ops_measured

    # pulses
    ramsey_pulses_init = [deepcopy(operation_dict[op]) for op in ramsey_ops_init]
    ramsey_pulses_det = [deepcopy(operation_dict[op]) for op in ramsey_ops_det]

    # name and reference swept pulse
    for i in range(n):
        idx = -2 #(2 if for_ef else 1) + i * 2 + 1
        ramsey_pulses_init[idx]["name"] = f"Ramsey_x2_{i}"
        ramsey_pulses_init[idx]['ref_point'] = 'start'
        ramsey_pulses_det[idx]["name"] = f"Ramsey_x2_{i}"
        ramsey_pulses_det[idx]['ref_point'] = 'start'

    # compute dphase
    a_d = artificial_detunings if np.ndim(artificial_detunings) == 1 \
        else [artificial_detunings]
    dphase = [((t - times[0]) * a_d[i % len(a_d)] * 360) % 360
                 for i, t in enumerate(times)]

    # sweep pulses
    params = {f'Ramsey_x2_{i}.pulse_delay': times for i in range(n)}
    params.update({f'Ramsey_x2_{i}.phase': dphase for i in range(n)})
    swept_pulses_init = sweep_pulse_params(ramsey_pulses_init, params)
    swept_pulses_det = sweep_pulse_params(ramsey_pulses_det, params)
    swept_pulses = np.ravel((swept_pulses_init,swept_pulses_det), order='F')

    # add preparation pulses
    swept_pulses_with_prep = \
        [add_preparation_pulses(p, operation_dict,
                                [pulsed_qubit_name, measured_qubit_name],
                                **prep_params)
         for p in swept_pulses]
    seq = pulse_list_list_seq(swept_pulses_with_prep, seq_name, upload=False)

    # add calibration segments
    seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    # reuse sequencer memory by repeating readout pattern
    seq.repeat_ro(f"RO {measured_qubit_name}", operation_dict)

    log.debug(seq)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())

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
    for i, exp_lst in enumerate(experiment_lists):
        pulse_lst = [operation_dict[p] for p in exp_lst]
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
        pihalf_spacing=None, prep_params=None, segment_name='equatorial_segment'):
    prep_params = {} if prep_params is None else prep_params
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
    pulse_list = add_preparation_pulses(pulse_list, operation_dict, qubit_names, **prep_params)
    return segment.Segment(segment_name, pulse_list)


def interleaved_pulse_list_list_equatorial_seq(
        qubit_names, operation_dict, interleaved_pulse_list_list, phases, 
        pihalf_spacing=None, prep_params=None, cal_points=None,
        sequence_name='equatorial_sequence', upload=True):
    prep_params = {} if prep_params is None else prep_params
    seq = sequence.Sequence(sequence_name)
    for i, interleaved_pulse_list in enumerate(interleaved_pulse_list_list):
        for j, phase in enumerate(phases):
            seg = interleaved_pulse_list_equatorial_seg(
                qubit_names, operation_dict, interleaved_pulse_list, phase,
                pihalf_spacing=pihalf_spacing, prep_params=prep_params, segment_name=f'segment_{i}_{j}')
            seq.add(seg)
    if cal_points is not None:
        seq.extend(cal_points.create_segments(operation_dict, **prep_params))
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    return seq, np.arange(seq.n_acq_elements())


def measurement_induced_dephasing_seq(
        measured_qubit_names, dephased_qubit_names, operation_dict, 
        ro_amp_scales, phases, pihalf_spacing=None, prep_params=None,
        cal_points=None, upload=True, sequence_name='measurement_induced_dephasing_seq'):
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
        phases, pihalf_spacing=pihalf_spacing, prep_params=prep_params,
        cal_points=cal_points, sequence_name=sequence_name, upload=upload)

def multi_parity_multi_round_seq(ancilla_qubit_names,
                                 data_qubit_names,
                                 parity_map,
                                 CZ_map,
                                 prep,
                                 operation_dict,
                                 mode='tomo',
                                 parity_seperation=800e-9,
                                 rots_basis=('I', 'Y90', 'X90'),
                                 parity_loops=1,
                                 cal_points=None,
                                 prep_params=None,
                                 upload=True):
    seq_name = 'Multi_Parity_{}_round_sequence'.format(parity_loops)
    qb_names = ancilla_qubit_names + data_qubit_names

    # dummy pulses
    dummy_ro_1 = {'pulse_type': 'GaussFilteredCosIQPulse',
                 'I_channel': 'UHF1_ch1',
                 'Q_channel': 'UHF1_ch2',
                 'amplitude': 0.00001,
                 'pulse_length': 50e-09,
                 'pulse_delay': 0,
                 'mod_frequency': 900.0e6,
                 'phase': 0,
                 'phi_skew': 0,
                 'alpha': 1,
                 'gaussian_filter_sigma': 1e-09,
                 'nr_sigma': 2,
                 'phase_lock': True,
                 'basis_rotation': {},
                 'operation_type': 'RO'}
    dummy_ro_2 = {'pulse_type': 'GaussFilteredCosIQPulse',
                 'I_channel': 'UHF2_ch1',
                 'Q_channel': 'UHF2_ch2',
                 'amplitude': 0.00001,
                 'pulse_length': 50e-09,
                 'pulse_delay': 0,
                 'mod_frequency': 900.0e6,
                 'phase': 0,
                 'phi_skew': 0,
                 'alpha': 1,
                 'gaussian_filter_sigma': 1e-09,
                 'nr_sigma': 2,
                 'phase_lock': True,
                 'basis_rotation': {},
                 'operation_type': 'RO'}

    echo_pulses = [('Y180' + 's ' + dqb)
                   for n, dqb in enumerate(data_qubit_names)]


    parity_ops_list = []
    echoed_round = {}
    for i in range(len(parity_map)):
        echoed_round[parity_map[i]['round']] = False

    for i in range(len(parity_map)):
        parity_ops = []
        anc_name = parity_map[i]['ancilla']

        basis_op = 'I'

        if parity_map[i]['type'] == 'Z':
            basis_op = 'I'
        elif parity_map[i]['type'] == 'X':
            basis_op = 'Y90'
        elif parity_map[i]['type'] == 'Y':
            basis_op = 'X90'

        for n, dqb in enumerate(parity_map[i]['data']):
            if dqb in data_qubit_names:
                op = basis_op + ('' if n==0 else 's') + ' ' + dqb
                parity_ops.append(op)
        parity_ops.append('Y90 ' + anc_name)

        for n, dqb in enumerate(parity_map[i]['data']):
            op = 'CZ ' + anc_name + ' ' + dqb
            op = CZ_map.get(op, [op])
            ops = parity_ops+op
            if n==1 and anc_name=='qb4':
                ops.append('Y180 ' + anc_name)
                ops.append('Z180 ' + parity_map[i]['data'][-1])
                ops.append('Z180 ' + parity_map[i]['data'][-2])
                print('ECHO')
            elif n==0 and (anc_name=='qb3' or anc_name=='qb5'):
                ops.append('Y180 ' + anc_name)
                ops.append('Z180 ' + parity_map[i]['data'][-1])
                print('ECHO')
            parity_ops = ops

        parity_ops.append('I ' + anc_name)
        parity_ops.append('Y90s ' + anc_name)
        print('ECHO')

        for n, dqb in enumerate(parity_map[i]['data']):
            if dqb in data_qubit_names:
                op = ('m' if basis_op is not 'I' else '') + basis_op + ('' if n==0
                                                                        else
                's') + ' ' + dqb
                # op =  basis_op + ('' if n==0 else 's') + ' ' + dqb
                parity_ops.append(op)
        parity_ops.append('I ' + parity_map[i]['data'][-1])
        parity_ops.append('RO ' + anc_name)
        parity_ops_list.append(parity_ops)



    prep_ops = [{'g': 'I ', 'e': 'X180 ', '+': 'Y90 ', '-': 'mY90 '}[s] \
             + dqn for s, dqn in zip(prep, data_qubit_names)]
    prep_mode = False

    end_sequence = []
    if mode=='tomo':
        end_sequences = get_tomography_pulses(*data_qubit_names,
                                              basis_pulses=rots_basis)
    elif mode=='onoff':
        end_sequences = [[rot + ('s ' if i==0 else ' ') + dqb
                          for i, dqb in enumerate(data_qubit_names)]
                         for rot in rots_basis]
    elif mode=='preps':
        prep_ops = [[{'g': 'I ', 'e': 'X180 ', '+': 'Y90 ', '-': 'mY90 '}[s] \
                    + dqn for s, dqn in zip(preps, data_qubit_names)]
                    for preps in rots_basis]
        end_sequences = [['I ' + data_qubit_names[0]] for preps in rots_basis]
        prep_mode = True
    else:
        end_sequences = ['I ' + data_qubit_names[0]]

    first_readout = dict()
    rounds = 0
    for k in range(len(parity_map)):
        first_readout[parity_map[k]['round']] = True
        if parity_map[k]['round'] > rounds:
            rounds = parity_map[k]['round']

    all_pulsess = []
    for t, end_sequence in enumerate(end_sequences):
        all_pulses = []

        if prep_mode:
            prep_pulses = [deepcopy(operation_dict[op]) for op in
                                    prep_ops[t]]
        else:
            prep_pulses = [deepcopy(operation_dict[op]) for op in
                                    prep_ops]

        for pulse in prep_pulses:
            # pulse['element_name'] = f'prep_tomo_{t}'
            pulse['element_name'] = f'drive_tomo_{t}'
            pulse['ref_pulse'] = 'segment_start'
            pulse['pulse_delay'] = -pulse['sigma']*pulse['nr_sigma']
        all_pulses += prep_pulses

        for m in range(parity_loops):
            for round in first_readout.keys():
                first_readout[round] = True

            for k in range(len(parity_map)):
                round = parity_map[k]['round']
                anc_name = parity_map[k]['ancilla']

                for i, op in enumerate(parity_ops_list[k]):
                    all_pulses.append(deepcopy(operation_dict[op]))
                    if op == 'I '+anc_name:
                        all_pulses[-1]['basis_rotation'] = parity_map[k][
                            'phases']
                    if i == 0:
                        all_pulses[-1]['ref_pulse'] = 'segment_start'
                        all_pulses[-1]['pulse_delay'] = np.sum(
                            parity_seperation[:round])+m*np.sum(parity_seperation)
                    if 'CZ' not in op and 'RO' not in op:
                        all_pulses[-1]['element_name'] = f'drive_tomo_{t}'
                        # all_pulses[-1]['element_name'] = f'drive_{round}_loop' + \
                        #                                  f'_{m}_tomo_{t}'
                    elif 'CZ' in op:
                        all_pulses[-1]['element_name'] = f'flux_tomo_{t}'
                    if 'RO' in op:
                        all_pulses[-1]['element_name'] = f'ro_{round}_loop' + \
                                                         f'_{m}_tomo_{t}'
                        if first_readout[round] is True:
                            all_pulses[-1]['name'] = f'first_ro_{round}_loop' + \
                                f'_{m}_tomo_{t}'
                        else:
                            all_pulses[-1]['ref_pulse'] = f'first_ro_{round}' + \
                                f'_loop_{m}_tomo_{t}'
                            all_pulses[-1]['ref_point'] = 'start'

                        first_readout[round] = False

                # more hacking for dummy
                all_pulses.append(deepcopy(dummy_ro_1))
                all_pulses[-1]['element_name'] = f'ro_{round}_loop' + \
                                                 f'_{m}_tomo_{t}'
                all_pulses[-1]['ref_pulse'] = f'first_ro_{round}' + \
                                              f'_loop_{m}_tomo_{t}'
                all_pulses[-1]['ref_point'] = 'start'

                all_pulses.append(deepcopy(dummy_ro_2))
                all_pulses[-1]['element_name'] = f'ro_{round}_loop' + \
                                                 f'_{m}_tomo_{t}'
                all_pulses[-1]['ref_pulse'] = f'first_ro_{round}' + \
                                              f'_loop_{m}_tomo_{t}'
                all_pulses[-1]['ref_point'] = 'start'
            # if  (m!=parity_loops-1)  or ( (parity_loops%2==0)
            #                               and  m==parity_loops-1):
            if m != parity_loops - 1:
                # print('Logical Echo')
                for i, op in enumerate(echo_pulses):
                    all_pulses.append(deepcopy(operation_dict[op]))
                    all_pulses[-1]['element_name'] = f'drive_tomo_{t}'
                    # all_pulses[-1]['pulse_delay'] = -50e-9

        end_pulses = [deepcopy(operation_dict[op]) for op in end_sequence]

        for pulse in end_pulses:
            pulse['element_name'] = f'drive_tomo_{t}'
            pulse['ref_pulse'] = f'first_ro_{rounds}' + \
                                 f'_loop_{parity_loops-1}_tomo_{t}'
            pulse['pulse_delay'] = 200e-9 # to account for UHF deadtime
            pulse['ref_point'] = 'end'
        all_pulses += end_pulses
        all_pulses += generate_mux_ro_pulse_list(qb_names, operation_dict)
        all_pulsess.append(all_pulses)


    if prep_params is not None:
        all_pulsess_with_prep = \
            [add_preparation_pulses(seg, operation_dict, qb_names, **prep_params)
             for seg in all_pulsess]
    else:
        all_pulsess_with_prep = all_pulsess

    seq = pulse_list_list_seq(all_pulsess_with_prep, seq_name, upload=False)

    # add calibration segments
    if cal_points is not None:
        seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    # This will currently only work with pre-selection
    ROs = (rounds+1)
    repeat_patern = (len(end_sequences), 1, (parity_loops, ROs), 1)
    for qbn in qb_names:
        pulse = 'RO ' + qbn
        repeat_dict = seq.repeat(pulse, operation_dict, repeat_patern)

    log.debug(repeat_dict)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    log.debug('sweep_points: ', seq.n_acq_elements())
    return seq, np.arange(seq.n_acq_elements())


def ro_dynamic_phase_seq(hard_sweep_dict, qbp_name, qbr_names,
                         operation_dict, pulse_separation, init_state,
                         cal_points=None, prep_params=dict(), upload=True):

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
    dummy_ro_1 = {'pulse_type': 'GaussFilteredCosIQPulse',
                  'I_channel': 'UHF1_ch1',
                  'Q_channel': 'UHF1_ch2',
                  'amplitude': 0.00001,
                  'pulse_length': 50e-09,
                  'pulse_delay': 0,
                  'mod_frequency': 900.0e6,
                  'phase': 0,
                  'phi_skew': 0,
                  'alpha': 1,
                  'gaussian_filter_sigma': 1e-09,
                  'nr_sigma': 2,
                  'phase_lock': True,
                  'basis_rotation': {},
                  'operation_type': 'RO'}
    dummy_ro_2 = {'pulse_type': 'GaussFilteredCosIQPulse',
                  'I_channel': 'UHF2_ch1',
                  'Q_channel': 'UHF2_ch2',
                  'amplitude': 0.00001,
                  'pulse_length': 50e-09,
                  'pulse_delay': 0,
                  'mod_frequency': 900.0e6,
                  'phase': 0,
                  'phi_skew': 0,
                  'alpha': 1,
                  'gaussian_filter_sigma': 1e-09,
                  'nr_sigma': 2,
                  'phase_lock': True,
                  'basis_rotation': {},
                  'operation_type': 'RO'}

    # put together n-qubit calibration point pulse lists
    # put together n-qubit ramsey pulse lists
    pulse_list = []
    for j, qbr_name in enumerate(qbr_names):
        pulse_list.append(deepcopy(operation_dict['X90 ' + qbr_name]))
        pulse_list[-1]['name'] = f'x1_{qbr_name}'
        pulse_list[-1]['ref_pulse'] = 'segment_start'
        pulse_list[-1]['refpoint'] = 'start'

    ro_probe = deepcopy(operation_dict['RO ' + qbp_name])
    pulse_list.append(ro_probe)
    pulse_list[-1]['name'] = f'ro_probe'
    pulse_list[-1]['element_name'] = f'ro_probe'
    pulse_list.append(dummy_ro_1)
    pulse_list[-1]['refpoint'] = 'start'
    pulse_list[-1]['element_name'] = f'ro_probe'
    pulse_list.append(dummy_ro_2)
    pulse_list[-1]['refpoint'] = 'start'
    pulse_list[-1]['element_name'] = f'ro_probe'

    for j, qbr_name in enumerate(qbr_names):
        pulse_list.append(deepcopy(operation_dict['X90 ' + qbr_name]))
        pulse_list[-1]['name'] = f'x2_{qbr_name}'
        pulse_list[-1]['ref_pulse'] = 'segment_start'
        pulse_list[-1]['refpoint'] = 'start'
        pulse_list[-1]['pulse_delay'] = pulse_separation

    ro_list = generate_mux_ro_pulse_list(qbr_names+[qbp_name], operation_dict)
    pulse_list += ro_list

    if init_state == 'e':
        pulse_list.append(deepcopy(operation_dict['X180 ' + qbp_name]))
        pulse_list[-1]['ref_pulse'] = 'segment_start'
        pulse_list[-1]['refpoint'] = 'start'
        pulse_list[-1]['pulse_delay'] = -100e-9

    params = {f'x2_{qbr_name}.{k}': v['values']
                   for k, v in hard_sweep_dict.items() for qbr_name in qbr_names}
    hsl = len(list(hard_sweep_dict.values())[0]['values'])
    params.update({f'ro_probe.amplitude': np.concatenate(
        [ro_probe['amplitude'] * np.ones(hsl // 2), np.zeros(hsl // 2)])})

    swept_pulses = sweep_pulse_params(pulse_list, params)

    swept_pulses_with_prep = \
        [add_preparation_pulses(p, operation_dict, [qbp_name], **prep_params)
         for p in swept_pulses]
    seq = pulse_list_list_seq(swept_pulses_with_prep, seq_name, upload=False)

    if cal_points is not None:
        # add calibration segments
        seq.extend(cal_points.create_segments(operation_dict, **prep_params))

    log.debug(seq)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq, np.arange(seq.n_acq_elements())
