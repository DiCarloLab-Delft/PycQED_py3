'''
Definitions of all the standard single qubit sequences
code that generates the required QASM files

Take a look at http://www.media.mit.edu/quanta/qasm2circ/ for qasm
'''
import numpy as np


def T1(qubit, times, clock_cycle=5e-9):
    #
    clocks = np.round(times/clock_cycle)

    qasm_file = open('qasm_files/T1.qasm', mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit.name))
    for cl in clocks:
        qasm_file.writelines('\ninit {}  \n'.format(qubit.name))
        qasm_file.writelines('X {}     # exciting pi pulse\n'.format(
                             qubit.name))
        qasm_file.writelines('I {} {} \n'.format(qubit.name, cl))
        qasm_file.writelines('RO {}  \n'.format(qubit.name))
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
