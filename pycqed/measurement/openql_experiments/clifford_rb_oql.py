"""
This file reads in a pygsti dataset file and converts it to a valid
OpenQL sequence.
"""

from os.path import join, dirname
import openql.openql as ql
from pycqed.utilities.general import suppress_stdout
from openql.openql import Program, Kernel, Platform
from pycqed.measurement.openql_experiments.single_qubit_oql import \
    add_single_qubit_cal_points
from pycqed.measurement.openql_experiments.multi_qubit_oql import \
    add_two_q_cal_points, add_multi_q_cal_points

from pycqed.measurement.randomized_benchmarking import randomized_benchmarking as rb
from pycqed.measurement.openql_experiments import openql_helpers as oqh
from pycqed.measurement.randomized_benchmarking.two_qubit_clifford_group \
    import SingleQubitClifford, TwoQubitClifford
base_qasm_path = join(dirname(__file__), 'qasm_files')
output_dir = join(dirname(__file__), 'output')
ql.set_output_dir(output_dir)


def randomized_benchmarking(qubits: list, platf_cfg: str,
                            nr_cliffords, nr_seeds: int,
                            net_cliffords: list=[0],
                            max_clifford_idx: int=11520,
                            initialize: bool=True,
                            interleaving_cliffords=[None],
                            program_name: str='randomized_benchmarking',
                            cal_points: bool=True,
                            f_state_cal_pts: bool=True,
                            recompile: bool=True):
    '''
    Input pars:
        qubits:         list of ints specifying qubit indices.
                        based on the length this function detects if it should
                        generate a single or two qubit RB sequence.
        platf_cfg:      filename of the platform config file
        nr_cliffords:   list nr_cliffords for which to generate RB seqs
        nr_seeds:       int  nr_seeds for which to generate RB seqs
        net_cliffords:  list of ints index of net clifford the sequence
                        should perform. See examples below on how to use this.
                            0 -> Idx
                            3 -> rx180
                            3*24+3 -> {rx180 q0 | rx180 q1}

        initialize:     if True initializes qubits to 0, disable for restless
            tuning
        program_name:           some string that can be used as a label.
        cal_points:     bool whether to replace the last two elements with
                        calibration points, set to False if you want
                        to measure a single element (for e.g. optimization)

        recompile:      True -> compiles the program,
                        'as needed' -> compares program to timestamp of config
                            and existence, if required recompile.
                        False -> compares program to timestamp of config.
                            if compilation is required raises a ValueError

                        If the program is more recent than the config
                        it returns an empty OpenQL program object with
                        the intended filename that can be used to upload the
                        previously compiled file.

    Returns:
        p:              OpenQL Program object

    ***************************************************************************
    Examples:
        1. Single qubit randomized benchmarking:

            p = cl_oql.randomized_benchmarking(
                qubits=[0],

                nr_cliffords=[2, 4, 8, 16, 32, 128, 512, 1024],
                nr_seeds=1,  # for CCL memory reasons
                platf_cfg=qubit.cfg_openql_platform_fn(),
                program_name='RB_{}'.format(i))

        2. Two qubit simultaneous randomized benchmarking:

            p = cl_oql.randomized_benchmarking(
                qubits=[0, 1],          # simultaneous RB on both qubits
                max_clifford_idx = 576, # to ensure only SQ Cliffords are drawn

                nr_cliffords=[2, 4, 8, 16, 32, 128, 512, 1024],
                nr_seeds=1,  # for CCL memory reasons
                platf_cfg=qubit.cfg_openql_platform_fn(),
                program_name='RB_{}'.format(i))

        3. Single qubit interleaved randomized benchmarking:
            p = cl_oql.randomized_benchmarking(
                qubits=[0],
                interleaving_cliffords=[None, 0, 16, 3],
                cal_points=False # relevant here because of data binning

                nr_cliffords=[2, 4, 8, 16, 32, 128, 512, 1024],
                nr_seeds=1,
                platf_cfg=qubit.cfg_openql_platform_fn(),
                program_name='Interleaved_RB_s{}_int{}_ncl{}_{}'.format(i))

    '''
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname=program_name, nqubits=platf.get_qubit_number(),
                p=platf)

    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')

    if not oqh.check_recompilation_needed(
            program_fn=p.filename, platf_cfg=platf_cfg, recompile=recompile):
        return p

    if len(qubits) == 1:
        qubit_map = {'q0': qubits[0]}
        number_of_qubits = 1
        Cl = SingleQubitClifford
    elif len(qubits) == 2:
        qubit_map = {'q0': qubits[0],
                     'q1': qubits[1]}
        number_of_qubits = 2
        Cl = TwoQubitClifford
    else:
        raise NotImplementedError()

    for seed in range(nr_seeds):
        for j, n_cl in enumerate(nr_cliffords):
            for interleaving_cl in interleaving_cliffords:
                for net_clifford in net_cliffords:
                    k = Kernel('RB_{}Cl_s{}_net{}_inter{}'.format(
                        n_cl, seed, net_clifford, interleaving_cl), p=platf)
                    if initialize:
                        for qubit_idx in qubit_map.values():
                            k.prepz(qubit_idx)

                    cl_seq = rb.randomized_benchmarking_sequence(
                        n_cl, number_of_qubits=number_of_qubits,
                        desired_net_cl=net_clifford,
                        max_clifford_idx=max_clifford_idx,
                        interleaving_cl=interleaving_cl)
                    for cl in cl_seq:
                        gates = Cl(cl).gate_decomposition
                        for g, q in gates:
                            if isinstance(q, str):
                                k.gate(g, qubit_map[q])
                            elif isinstance(q, list):
                                # proper codeword
                                k.gate(g, [qubit_map[q[0]], qubit_map[q[1]]])

                    # This hack is required to align multiplexed RO in openQL..
                    k.gate("wait",  list(qubit_map.values()), 0)
                    for qubit_idx in qubit_map.values():
                        k.measure(qubit_idx)
                    k.gate("wait",  list(qubit_map.values()), 0)
                    p.add_kernel(k)

        if cal_points:
            if number_of_qubits == 1:
                p = add_single_qubit_cal_points(
                    p, platf=platf, qubit_idx=qubits[0],
                    f_state_cal_pts=f_state_cal_pts)
            elif number_of_qubits == 2:

                if f_state_cal_pts:
                    combinations = ['00', '01', '10', '11', '02', '20', '22']
                else:
                    combinations = ['00', '01', '10', '11']
                p = add_multi_q_cal_points(p, platf=platf,
                                           qubits=qubits,
                                           combinations=combinations)


    with suppress_stdout():
        p.compile(verbose=False)

    return p
