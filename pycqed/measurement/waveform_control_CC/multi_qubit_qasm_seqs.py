from pycqed.utilities.general import mopen
from os.path import join, dirname

base_qasm_path = join(dirname(__file__), 'qasm_files')


def AllXY(q0, q1, RO_target='all',
          sequence_type='sequential',
          replace_q1_pulses_X180=False,
          double_points=False):
    """
    AllXY sequence on two qubits.
    Has the option of replacing pulses on q1 with pi pulses

    Args:
        operation_dict (dict) : dictionary containing all pulse parameters
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
