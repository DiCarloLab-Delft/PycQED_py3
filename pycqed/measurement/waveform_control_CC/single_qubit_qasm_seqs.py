'''
Definitions of all the standard single qubit sequences
code that generates the required QASM files

Take a look at http://www.media.mit.edu/quanta/qasm2circ/ for qasm
'''
import numpy as np
from os.path import join, dirname

from pycqed.utilities.general import mopen

base_qasm_path = join(dirname(__file__), 'qasm_files')


def T1(qubit_name, times, clock_cycle=5e-9):
    #
    clocks = np.round(times/clock_cycle)
    filename = join(base_qasm_path, 'T1.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    for cl in clocks:
        qasm_file.writelines('init_all\n')
        qasm_file.writelines('X {}     # exciting pi pulse\n'.format(
                             qubit_name))
        qasm_file.writelines('I {} {:d} \n'.format(qubit_name, int(cl)))
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file






def AllXY(qubit_name, times):
    raise(NotImplementedError)


def Rabi(qubit_name, amps):
    raise(NotImplementedError)


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
    qasm_file.writelines('X {}     # On \n'.format(qubit_name))
    qasm_file.writelines('RO {}  \n'.format(qubit_name))

    qasm_file.close()
    return qasm_file


def butterfly(qubit):
    raise(NotImplementedError)


def randomized_benchmarking(qubit):
    raise(NotImplementedError)


def MotzoiXY(qubit):
    raise(NotImplementedError)
