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
from pycqed.measurement.randomized_benchmarking import randomized_benchmarking as rb
from pycqed.measurement.randomized_benchmarking.two_qubit_clifford_group \
    import SingleQubitClifford, TwoQubitClifford
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




def randomized_benchmarking(qubits: list, platf_cfg: str,
                            nr_cliffords, nr_seeds: int,
                            net_clifford: int=0, restless: bool=False,
                            program_name: str='randomized_benchmarking',
                            cal_points: bool=True,
                            double_curves: bool=False):
    '''
    Input pars:
        qubits:         list of ints specifying qubit indices.
                        based on the length this function detects if it should
                        generate a single or two qubit RB sequence.
        platf_cfg:      filename of the platform config file
        nr_cliffords:   list nr_cliffords for which to generate RB seqs
        nr_seeds:       int  nr_seeds for which to generate RB seqs
        net_clifford:   int index of net clifford the sequence should perform
                            0 -> Idx
                            3 -> rx180
        restless:       bool, does not initialize if restless is True
        program_name:           some string that can be used as a label.
        cal_points:     bool whether to replace the last two elements with
                        calibration points, set to False if you want
                        to measure a single element (for e.g. optimization)

        double_curves: Alternates between net clifford 0 and 3

    Returns:
        p:              OpenQL Program object

    generates a program for single qubit Clifford based randomized
    benchmarking.
    '''
    net_cliffords = [0, 3]  # Exists purely for the double curves mode
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname=program_name, nqubits=platf.get_qubit_number(),
                p=platf)

    if len(qubits) ==1:
        qubit_map = {'q0': qubits[0]}
        number_of_qubits = 1
        Cl = SingleQubitClifford
    elif len(qubits) ==2:
        qubit_map = {'q0': qubits[0],
                     'q1': qubits[1]}
        number_of_qubits = 2
        Cl = TwoQubitClifford
    else:
        raise NotImplementedError()

    i = 0
    for seed in range(nr_seeds):
        for j, n_cl in enumerate(nr_cliffords):
            k = Kernel('RB_{}Cl_s{}'.format(n_cl, seed), p=platf)
            if not restless:
                for qubit_idx in qubit_map.values():
                    k.prepz(qubit_idx)
            if cal_points and (j == (len(nr_cliffords)-4) or
                               j == (len(nr_cliffords)-3)):
                k.measure(qubit_idx)

            elif cal_points and (j == (len(nr_cliffords)-2) or
                                 j == (len(nr_cliffords)-1)):
                k.x(qubit_idx)
                k.measure(qubit_idx)
            else:
                if double_curves:
                    net_clifford = net_cliffords[i % 2]
                    i += 1
                cl_seq = rb.randomized_benchmarking_sequence(
                    n_cl, number_of_qubits=number_of_qubits,
                    desired_net_cl=net_clifford)
                # pulse_keys = rb.decompose_clifford_seq(cl_seq)
                for cl in cl_seq:
                    gates = Cl(cl).gate_decomposition
                    for g, q in gates:
                        print(g, q)
                        if isinstance(q, str):
                            k.gate(g, qubit_map[q])
                        elif isinstance(q, list):
                            # proper codeword
                            k.gate(g, [qubit_map[q[0]], qubit_map[q[1]]])

                for qubit_idx in qubit_map.values():
                    k.measure(qubit_idx)
            p.add_kernel(k)
    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p