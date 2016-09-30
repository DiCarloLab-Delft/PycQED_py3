'''
Definitions of all the standard single qubit sequences
code that generates the required QASM files

Take a look at http://www.media.mit.edu/quanta/qasm2circ/ for qasm
'''
import numpy as np
from os.path import join, dirname
base_qasm_path = join(dirname(__file__), 'qasm_files')


def T1(qubit_name, times, clock_cycle=5e-9):
    #
    clocks = np.round(times/clock_cycle)

    qasm_file = open(join(base_qasm_path, 'T1.qasm'), mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    for cl in clocks:
        qasm_file.writelines('\ninit {}  \n'.format(qubit_name))
        qasm_file.writelines('X {}     # exciting pi pulse\n'.format(
                             qubit_name))
        qasm_file.writelines('I {} {:d} \n'.format(qubit_name, int(cl)))
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file






def AllXY(qubit, times):
    raise(NotImplementedError)


def Rabi(qubit, amps):
    raise(NotImplementedError)


def Ramsey(qubit, times):
    raise(NotImplementedError)


def echo(qubit, times):
    raise(NotImplementedError)


def off_on(qubit):
    raise(NotImplementedError)


def butterfly(qubit):
    raise(NotImplementedError)


def randomized_benchmarking(qubit):
    raise(NotImplementedError)


def MotzoiXY(qubit):
    raise(NotImplementedError)
