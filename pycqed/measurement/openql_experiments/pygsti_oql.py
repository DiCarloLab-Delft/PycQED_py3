"""
This file reads in a pygsti dataset file and converts it to a valid
OpenQL sequence.
"""
import numpy as np
import pygsti
from os.path import join, dirname
import openql.openql as ql
from pycqed.utilities.general import suppress_stdout
import pycqed as pq
from openql.openql import Program, Kernel, Platform

base_qasm_path = join(dirname(__file__), 'qasm_files')
output_dir = join(dirname(__file__), 'output')
ql.set_output_dir(output_dir)

gst_exp_filepath = join(pq.__path__[0], 'measurement', 'gate_set_tomography')



# used to map pygsti gates to openQL gates
# for now (Jan 2018) only contains basic pygsti gates
gatemap = {'i': 'i',
          'x': 'rx90',
          'y': 'ry90',
          'cphase': 'fl_cw_02'}

def pygsti_expList_from_dataset(dataset_filename: str):
    """
    Takes a pygsti dataset file and extracts the experiment list from it.
    The expList is a list containing pygsti GateString objects.

    Args:
        dataset_filename (str) : filename of the dataset to read
    returns:
        expList (list): a list of pygsti GateString objects.
    """
    with open(dataset_filename, 'r') as f:
        rawLines = f.readlines()
    # lines[1:]

    lines = []
    for line in rawLines:
        lines.append(line.split(' ')[0])
    expList = [
        pygsti.objects.GateString(None, stringRepresentation=line)
        for line in lines[1:]]

    return expList


def openql_program_from_pygsti_expList(pygsti_gateset, program_name: str,
                                       qubits: list,
                                       platf_cfg: str):
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname=program_name,
                nqubits=platf.get_qubit_number(),
                p=platf)

    for i, gatestring in enumerate(pygsti_gateset):
        kernel_name = 'G {} {}'.format(i, gatestring)
        k = openql_kernel_from_gatestring(
            gatestring=gatestring, qubits=qubits,
            kernel_name=kernel_name, platf=platf)
        p.add_kernel(k)

    p.sweep_points = np.arange(len(pygsti_gateset), dtype=float)
    p.set_sweep_points(p.sweep_points, len(p.sweep_points))
    with suppress_stdout():
        p.compile()
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def openql_kernel_from_gatestring(gatestring, qubits: list,
                                  kernel_name:str, platf):
    """
    Generates an openQL kernel for a pygsti gatestring.
    """
    k = Kernel(kernel_name, p=platf)
    for q in qubits:
        k.prepz(q)

    for gate in gatestring:
        assert gate[0] == 'G' # for valid pyGSTi gatestrings
        if len(gate[1:]) == 1:
            # 1Q pygsti format e.g.,: Gi  or Gy
            k.gate(gatemap[gate[1]], qubits[0])

        elif len(gate[1:]) == 2:
            # 2Q pygsti format e.g.,: Gix  or Gyy
            k.gate(gatemap[gate[1]], qubits[0])
            k.gate(gatemap[gate[2]], qubits[1])
        elif gate == 'Gcphase':
            # only two qubit gate supported
            k.gate(gatemap[gate[1:]], qubits[0], qubits[1])
        else:
            raise NotImplementedError('Gate {} not supported'.format(gate))

    for q in qubits:
        k.measure(q)
    # ensures timing of readout is aligned
    k.gate('wait', qubits, 0)

    return k


def poor_mans_2q_gst(q0: int, q1: int, platf_cfg: str,):
    """
    Generates the QISA and QASM programs for poor_mans_GST, this is 2Q GST
    without the repetitions of any gate.
    """

    fp = join(gst_exp_filepath, 'PoorMans_2Q_GST.txt')
    expList = pygsti_expList_from_dataset(fp)
    p = openql_program_from_pygsti_expList(
        expList, 'PoorMans_GST', [2, 0], platf_cfg=platf_cfg)
    return p



