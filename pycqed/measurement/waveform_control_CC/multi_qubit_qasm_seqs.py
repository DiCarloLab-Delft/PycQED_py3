from pycqed.utilities.general import mopen
from os.path import join, dirname

base_qasm_path = join(dirname(__file__), 'qasm_files')


def two_qubit_off_on(q0, q1, RO_target='all'):
    '''
    off_on sequence on two qubits.

    Args:
        q0, q1      (str) : target qubits for the sequence
        RO_target   (str) : target for the RO, can be a qubit name or 'all'
    '''
    filename = join(base_qasm_path, 'two_qubit_off_on.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    # off - off
    qasm_file.writelines('\ninit_all\n')
    qasm_file.writelines('I {}\n'.format(q0))
    qasm_file.writelines('I {}\n'.format(q1))
    qasm_file.writelines('RO {}  \n'.format(RO_target))

    # on - off
    qasm_file.writelines('\ninit_all\n')
    qasm_file.writelines('X180 {}\n'.format(q0))
    qasm_file.writelines('I {}\n'.format(q1))
    qasm_file.writelines('RO {}  \n'.format(RO_target))

    # off - on
    qasm_file.writelines('\ninit_all\n')
    qasm_file.writelines('I {}\n'.format(q0))
    qasm_file.writelines('X180 {}\n'.format(q1))
    qasm_file.writelines('RO {}  \n'.format(RO_target))

    # on - on
    qasm_file.writelines('\ninit_all\n')
    qasm_file.writelines('X180 {}\n'.format(q0))
    qasm_file.writelines('X180 {}\n'.format(q1))
    qasm_file.writelines('RO {}  \n'.format(RO_target))

    qasm_file.close()
    return qasm_file


def two_qubit_tomo_cardinal(cardinal,
                            q0,
                            q1,
                            RO_target='all'):
    '''
    Cardinal tomography for two qubits.

    Args:
        cardinal        (int) : index of prep gate
        q0, q1          (str) : target qubits for the sequence
        RO_target       (str) : target for the RO, can be a qubit name or 'all'
    '''
    tomo_pulses = ['I ', 'X180 ', 'Y90 ', 'mY90 ', 'X90 ', 'mX90 ']
    tomo_list_q0 = []
    tomo_list_q1 = []
    for tp in tomo_pulses:
        tomo_list_q0 += [tp + q0 + '\n']
        tomo_list_q1 += [tp + q1 + '\n']

    prep_index_q0 = int(cardinal % len(tomo_list_q0))
    prep_index_q1 = int(((cardinal - prep_index_q0) / len(tomo_list_q0) %
                         len(tomo_list_q1)))

    prep_pulse_q0 = tomo_list_q0[prep_index_q0]
    prep_pulse_q1 = tomo_list_q1[prep_index_q1]

    filename = join(base_qasm_path, 'two_qubit_tomo_cardinal.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    # Tomography pulses
    for p_q1 in tomo_list_q1:
        for p_q0 in tomo_list_q0:
            qasm_file.writelines('\ninit_all\n')
            qasm_file.writelines(prep_pulse_q0)
            qasm_file.writelines(prep_pulse_q1)
            qasm_file.writelines(p_q0)
            qasm_file.writelines(p_q1)
            qasm_file.writelines('RO ' + RO_target + '  \n')

    # Calibration pulses
    cal_points = [['I ', 'I '],
                  ['X180 ', 'I '],
                  ['I ', 'X180 '],
                  ['X180 ', 'X180 ']]
    cal_pulses = []
    # every calibration point is repeated 7 times. This is copied from the
    # script for Tektronix driven qubits. I do not know if this repetition
    # is important or even necessary here.
    for seq in cal_points:
        cal_pulses += [[seq[0] + q0 + '\n', seq[1] +
                        q1 + '\n', 'RO ' + RO_target + '\n']] * 7

    for seq in cal_pulses:
        qasm_file.writelines('\ninit_all\n')
        for p in seq:
            qasm_file.writelines(p)

    qasm_file.close()
    return qasm_file


def two_qubit_AllXY(q0, q1, RO_target='all',
                    sequence_type='sequential',
                    replace_q1_pulses_X180=False,
                    double_points=False):
    """
    AllXY sequence on two qubits.
    Has the option of replacing pulses on q1 with pi pulses

    Args:
        q0, q1         (str) : target qubits for the sequence
        RO_target      (str) : target for the RO, can be a qubit name or 'all'
        sequence_type  (str) : sequential | interleaved | simultaneous | sandwiched
                              q0|q0|q1|q1   q0|q1|q0|q1   q01|q01      q1|q0|q0|q1
                            N.B.!  simultaneous is currently not possible!
            describes the order of the AllXY pulses
        replace_q1_pulses_X180 (bool) : if True replaces all pulses on q1 with
            X180 pulses.

        double_points (bool) : if True measures each point in the AllXY twice
    """

    pulse_combinations = [['I', 'I'], ['X180', 'X180'], ['Y180', 'Y180'],
                          ['X180', 'Y180'], ['Y180', 'X180'],
                          ['X90', 'I'], ['Y90', 'I'], ['X90', 'Y90'],
                          ['Y90', 'X90'], ['X90', 'Y180'], ['Y90', 'X180'],
                          ['X180', 'Y90'], ['Y180', 'X90'], ['X90', 'X180'],
                          ['X180', 'X90'], ['Y90', 'Y180'], ['Y180', 'Y90'],
                          ['X180', 'I'], ['Y180', 'I'], ['X90', 'X90'],
                          ['Y90', 'Y90']]
    if double_points:
        pulse_combinations = [val for val in pulse_combinations
                              for _ in (0, 1)]
    filename = join(base_qasm_path, 'two_qubit_AllXY.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    if sequence_type == 'simultaneous':
        raise NotImplementedError('Simultaneous readout not implemented.')

    if replace_q1_pulses_X180:
        for pulse_comb in pulse_combinations:
            qasm_file.writelines('\ninit_all\n')
            if sequence_type == 'interleaved':
                qasm_file.writelines('{} {}\n'.format(pulse_comb[0], q0) +
                                     '{} {}\n'.format('X180', q1) +
                                     '{} {}\n'.format(pulse_comb[1], q0) +
                                     '{} {}\n'.format('X180', q1))
            elif sequence_type == 'sandwiched':
                qasm_file.writelines('{} {}\n'.format('X180', q1) +
                                     '{} {}\n'.format(pulse_comb[0], q0) +
                                     '{} {}\n'.format(pulse_comb[1], q0) +
                                     '{} {}\n'.format('X180', q1))
            elif sequence_type == 'sequential':
                qasm_file.writelines('{} {}\n'.format(pulse_comb[0], q0) +
                                     '{} {}\n'.format(pulse_comb[1], q0) +
                                     '{} {}\n'.format('X180', q1) +
                                     '{} {}\n'.format('X180', q1))
            elif sequence_type == 'simultaneous':
                # FIXME: Not implemented yet.
                pass
            else:
                raise ValueError("sequence_type {} ".format(sequence_type) +
                                 "['interleaved', 'simultaneous', " +
                                 "'sequential', 'sandwiched']")
            qasm_file.writelines('RO {}  \n'.format(RO_target))
    else:
        for pulse_comb in pulse_combinations:
            qasm_file.writelines('\ninit_all\n')
            if sequence_type == 'interleaved':
                qasm_file.writelines('{} {}\n'.format(pulse_comb[0], q0) +
                                     '{} {}\n'.format(pulse_comb[0], q1) +
                                     '{} {}\n'.format(pulse_comb[1], q0) +
                                     '{} {}\n'.format(pulse_comb[1], q1))
            elif sequence_type == 'sandwiched':
                qasm_file.writelines('{} {}\n'.format(pulse_comb[0], q1) +
                                     '{} {}\n'.format(pulse_comb[0], q0) +
                                     '{} {}\n'.format(pulse_comb[1], q0) +
                                     '{} {}\n'.format(pulse_comb[1], q1))
            elif sequence_type == 'sequential':
                qasm_file.writelines('{} {}\n'.format(pulse_comb[0], q0) +
                                     '{} {}\n'.format(pulse_comb[1], q0) +
                                     '{} {}\n'.format(pulse_comb[0], q1) +
                                     '{} {}\n'.format(pulse_comb[1], q1))
            elif sequence_type == 'simultaneous':
                # FIXME: Not implemented yet.
                pass
            else:
                raise ValueError("sequence_type {} ".format(sequence_type) +
                                 "['interleaved', 'simultaneous', " +
                                 "'sequential', 'sandwiched']")
            qasm_file.writelines('RO {}  \n'.format(RO_target))

    qasm_file.close()

    return qasm_file
