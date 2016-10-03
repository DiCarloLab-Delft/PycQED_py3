'''
Definitions of all the standard single qubit sequences
code that generates the required QASM files

Take a look at http://www.media.mit.edu/quanta/qasm2circ/ for qasm
'''
import numpy as np
from os.path import join, dirname

from pycqed.utilities.general import mopen
from pycqed.measurement.randomized_benchmarking import randomized_benchmarking as rb
base_qasm_path = join(dirname(__file__), 'qasm_files')


def T1(qubit_name, times, clock_cycle=5e-9,
       cal_points=True):
    #
    clocks = np.round(times/clock_cycle)
    filename = join(base_qasm_path, 'T1.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    for i, cl in enumerate(clocks):
        qasm_file.writelines('init_all\n')
        if cal_points and (i == (len(clocks)-4) or
                           i == (len(clocks)-3)):
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
        elif cal_points and (i == (len(clocks)-2) or
                             i == (len(clocks)-1)):
            qasm_file.writelines('X180 {} \n'.format(qubit_name))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
        else:
            qasm_file.writelines('X180 {}     # exciting pi pulse\n'.format(
                                 qubit_name))
            qasm_file.writelines('I {} {:d} \n'.format(qubit_name, int(cl)))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file


def AllXY(qubit_name, double_points=False):
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

    filename = join(base_qasm_path, 'AllXY.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))

    for pulse_comb in pulse_combinations:
        qasm_file.writelines('init_all\n')
        if pulse_comb[0] != 'I':
            qasm_file.writelines('{} {}\n'.format(pulse_comb[0], qubit_name))
        if pulse_comb[1] != 'I':
            qasm_file.writelines('{} {}\n'.format(pulse_comb[1], qubit_name))
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file


def Rabi(qubit_name, amps, n=1):
    filename = join(base_qasm_path, 'Rabi_{}.qasm'.format(n))
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    for amp in amps:
        qasm_file.writelines('init_all\n')
        for i in range(n):
            qasm_file.writelines('Rx {} {} \n'.format(qubit_name, amp))
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file


def Ramsey(qubit_name, times):
    raise(NotImplementedError)


def echo(qubit_name, times):
    raise(NotImplementedError)


def off_on(qubit_name):
    filename = join(base_qasm_path, 'off_on.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))

    # Off
    qasm_file.writelines('init_all\n')
    qasm_file.writelines('RO {}  \n'.format(qubit_name))
    # On
    qasm_file.writelines('init_all\n')
    qasm_file.writelines('X180 {}     # On \n'.format(qubit_name))
    qasm_file.writelines('RO {}  \n'.format(qubit_name))

    qasm_file.close()
    return qasm_file


def butterfly(qubit):
    raise(NotImplementedError)


def randomized_benchmarking(qubit_name, nr_cliffords, nr_seeds,
                            net_clifford=0,
                            cal_points=True,
                            double_curves=True):
    '''
    Input pars:
        nr_cliffords:  list nr_cliffords for which to generate RB seqs
        nr_seeds:      int  nr_seeds for which to generate RB seqs
        net_clifford:  int index of net clifford the sequence should perform
                       0 corresponds to Identity and 3 corresponds to X180
        cal_points:    bool whether to replace the last two elements with
                       calibration points, set to False if you want
                       to measure a single element (for e.g. optimization)
        double_curves: Alternates between net clifford 0 and 3

    returns:
        qasm_file

    generates a qasm file for single qubit Clifford based randomized
    benchmarking.
    '''
    net_cliffords = [0, 3]  # Exists purely for the double curves mode
    filename = join(base_qasm_path, 'randomized_benchmarking.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    i = 0
    for seed in range(nr_seeds):
        for j, n_cl in enumerate(nr_cliffords):
            if cal_points and (j == (len(nr_cliffords)-4) or
                               j == (len(nr_cliffords)-3)):
                qasm_file.writelines('RO {}  \n'.format(qubit_name))
            elif cal_points and (j == (len(nr_cliffords)-2) or
                                 j == (len(nr_cliffords)-1)):
                qasm_file.writelines('X180 {} \n'.format(qubit_name))
                qasm_file.writelines('RO {}  \n'.format(qubit_name))
            else:
                if double_curves:
                    net_clifford = net_cliffords[i % 2]
                    i += 1
                cl_seq = rb.randomized_benchmarking_sequence(
                    n_cl, desired_net_cl=net_clifford)
                pulse_keys = rb.decompose_clifford_seq(cl_seq)
                for pulse in pulse_keys:
                    if pulse != 'I':
                        qasm_file.writelines('{} {}\n'.format(
                            pulse, qubit_name))
    qasm_file.close()
    return qasm_file


def MotzoiXY(qubit):
    raise(NotImplementedError)
