import numpy as np
from os.path import join, dirname
import openql.openql as ql
from openql.openql import Program, Kernel, Platform

base_qasm_path = join(dirname(__file__), 'qasm_files')
output_dir = join(dirname(__file__), 'output')
ql.set_output_dir(output_dir)


def single_flux_pulse_seq(qubit_indices: tuple,
                          platf_cfg: str):
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="single_flux_pulse_seq",
                nqubits=platf.get_qubit_number(),
                p=platf)
    print(platf_cfg)
    k = Kernel("main", p=platf)
    for idx in qubit_indices:
        k.prepz(idx)  # to ensure enough separation in timing
    for i in range(7):
        k.gate('CW_00', i)
    k.gate('fl_cw_02', qubit_indices[0], qubit_indices[1])
    p.add_kernel(k)
    p.compile()
    # attribute is added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def flux_staircase_seq(platf_cfg: str):
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="flux_staircase_seq",
                nqubits=platf.get_qubit_number(),
                p=platf)
    print(platf_cfg)
    k = Kernel("main", p=platf)
    for i in range(1):
        k.prepz(i)  # to ensure enough separation in timing
    for i in range(1):
        k.gate('CW_00', i)
    k.gate('CW_00', 6)
    for cw in range(8):
        k.gate('fl_cw_{:02d}'.format(cw), 2, 0)
        k.gate('fl_cw_{:02d}'.format(cw), 3, 1)
        k.gate("wait", [0, 1, 2, 3], 200)  # because scheduling is wrong.

    p.add_kernel(k)
    p.compile()
    # attribute is added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def two_qubit_off_on(q0: int, q1: int, platf_cfg: str):
    '''
    off_on sequence on two qubits.

    Args:
        q0, q1      (str) : target qubits for the sequence
        platf_cfg: str
    '''

    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="two_qubit_off_on", nqubits=platf.get_qubit_number(),
                p=platf)
    p = add_two_q_cal_points(p, platf=platf, q0=0, q1=1)
    p.compile()
    # attribute is added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def two_qubit_tomo_cardinal(cardinal: int, q0: int, q1: int, platf_cfg: str):
    '''
    Cardinal tomography for two qubits.
    Args:
        cardinal        (int) : index of prep gate
        q0, q1          (int) : target qubits for the sequence
    '''
    tomo_pulses = ['i', 'rx180', 'ry90', 'rym90', 'rx90', 'rxm90']
    tomo_list_q0 = tomo_pulses
    tomo_list_q1 = tomo_pulses

    prep_index_q0 = int(cardinal % len(tomo_list_q0))
    prep_index_q1 = int(((cardinal - prep_index_q0) / len(tomo_list_q0) %
                         len(tomo_list_q1)))

    prep_pulse_q0 = tomo_list_q0[prep_index_q0]
    prep_pulse_q1 = tomo_list_q1[prep_index_q1]

    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="two_qubit_tomo_cardinal",
                nqubits=platf.get_qubit_number(), p=platf)

    # Tomography pulses
    i = 0
    for p_q1 in tomo_list_q1:
        for p_q0 in tomo_list_q0:
            i += 1
            kernel_name = '{}_{}_{}'.format(i, p_q0, p_q1)
            k = Kernel(kernel_name, p=platf)
            k.prepz(q0)
            k.prepz(q1)
            k.gate(prep_pulse_q0, q0)
            k.gate(prep_pulse_q1, q1)
            k.gate(p_q0, q0)
            k.gate(p_q1, q1)
            k.measure(q0)
            k.measure(q1)
            p.add_kernel(k)
    # every calibration point is repeated 7 times. This is copied from the
    # script for Tektronix driven qubits. I do not know if this repetition
    # is important or even necessary here.
    p = add_two_q_cal_points(p, platf=platf, q0=0, q1=1, reps_per_cal_pt=7)
    p.compile()
    # attribute is added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def two_qubit_AllXY(q0: int, q1: int, platf_cfg: str,
                    sequence_type='sequential',
                    replace_q1_pulses_X180: bool=False,
                    double_points: bool=False):
    """
    AllXY sequence on two qubits.
    Has the option of replacing pulses on q1 with pi pulses

    Args:
        q0, q1         (str) : target qubits for the sequence
        RO_target      (str) : target for the RO, can be a qubit name or 'all'
        sequence_type  (str) : Describes the timing/order of the pulses.
            options are: sequential | interleaved | simultaneous | sandwiched
                       q0|q0|q1|q1   q0|q1|q0|q1     q01|q01       q1|q0|q0|q1
            describes the order of the AllXY pulses
        replace_q1_pulses_X180 (bool) : if True replaces all pulses on q1 with
            X180 pulses.

        double_points (bool) : if True measures each point in the AllXY twice
    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="two_qubit_AllXY", nqubits=platf.get_qubit_number(),
                p=platf)

    pulse_combinations = [['i', 'i'], ['rx180', 'rx180'], ['ry180', 'ry180'],
                          ['rx180', 'ry180'], ['ry180', 'rx180'],
                          ['rx90', 'i'], ['ry90', 'i'], ['rx90', 'ry90'],
                          ['ry90', 'rx90'], ['rx90', 'ry180'],
                          ['ry90', 'rx180'],
                          ['rx180', 'ry90'], ['ry180', 'rx90'],
                          ['rx90', 'rx180'],
                          ['rx180', 'rx90'], ['ry90', 'ry180'],
                          ['ry180', 'ry90'],
                          ['rx180', 'i'], ['ry180', 'i'], ['rx90', 'rx90'],
                          ['ry90', 'ry90']]

    pulse_combinations_tiled = pulse_combinations + pulse_combinations
    if double_points:
        pulse_combinations = [val for val in pulse_combinations
                              for _ in (0, 1)]

    if replace_q1_pulses_X180:
        pulse_combinations_q1 = ['rx180' for val in pulse_combinations]

    pulse_combinations_q0 = pulse_combinations
    pulse_combinations_q1 = pulse_combinations_tiled

    i = 0
    for pulse_comb_q0, pulse_comb_q1 in zip(pulse_combinations_q0,
                                            pulse_combinations_q1):
        i += 1
        k = Kernel('AllXY_{}'.format(i), p=platf)
        k.prepz(q0)
        k.prepz(q1)
        # N.B. The identity gates are there to ensure proper timing
        if sequence_type == 'interleaved':
            k.gate(pulse_comb_q0[0], q0)
            k.gate('i', q1)

            k.gate('i', q0)
            k.gate(pulse_comb_q1[0], q1)

            k.gate(pulse_comb_q0[1], q0)
            k.gate('i', q1)

            k.gate('i', q0)
            k.gate(pulse_comb_q1[1], q1)

        elif sequence_type == 'sandwiched':
            k.gate('i', q0)
            k.gate(pulse_comb_q1[0], q1)

            k.gate(pulse_comb_q0[0], q0)
            k.gate('i', q1)
            k.gate(pulse_comb_q0[1], q0)
            k.gate('i', q1)

            k.gate('i', q0)
            k.gate(pulse_comb_q1[1], q1)

        elif sequence_type == 'sequential':
            k.gate(pulse_comb_q0[0], q0)
            k.gate('i', q1)
            k.gate(pulse_comb_q0[1], q0)
            k.gate('i', q1)
            k.gate('i', q0)
            k.gate(pulse_comb_q1[0], q1)
            k.gate('i', q0)
            k.gate(pulse_comb_q1[1], q1)

        elif sequence_type == 'simultaneous':
            k.gate(pulse_comb_q0[0], q0)
            k.gate(pulse_comb_q1[0], q1)
            k.gate(pulse_comb_q0[1], q0)
            k.gate(pulse_comb_q1[1], q1)
        else:
            raise ValueError("sequence_type {} ".format(sequence_type) +
                             "['interleaved', 'simultaneous', " +
                             "'sequential', 'sandwiched']")
        k.measure(q0)
        k.measure(q1)
        p.add_kernel(k)

    p.compile()
    # attribute is added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def chevron_seq(fluxing_qubit: str, spectator_qubit: str,
                excite_q1: bool=False, RO_target='all'):
    '''
    Single chevron sequence that does a swap on |01> <-> |10> or |11> <-> |20>.

    Args:
        fluxing_qubit (str): name of the qubit that is fluxed/
        spectator qubit (str): name of the qubit with which the fluxing
                            qubit interacts.
        RO_target   (str): can be q0, q1, or 'all'
        excite_q1   (bool): choose whether to excite q1, thus choosing
                            between the |01> <-> |10> and the |11> <-> |20>
                            swap.
    '''
    raise NotImplementedError()
    # filename = join(base_qasm_path, 'chevron_seq.qasm')
    # qasm_file = mopen(filename, mode='w')
    # qasm_file.writelines('qubit {} \nqubit {} \n'.format(fluxing_qubit,
    #                                                      spectator_qubit))

    # qasm_file.writelines('\ninit_all\n')
    # if excite_q1:
    #     qasm_file.writelines('X180 {} | X180 {}\n'.format(fluxing_qubit,
    #                                                       spectator_qubit))
    # else:
    #     qasm_file.writelines('X180 {}\n'.format(fluxing_qubit))
    # qasm_file.writelines('square {}\n'.format(fluxing_qubit))
    # if excite_q1:
    #     # fluxing_qubit is rotated to ground-state to have better contrast
    #     # (|0> and |2> instead of |1> and |2>)
    #     qasm_file.writelines('X180 {}\n'.format(fluxing_qubit))
    # if RO_target == 'all':
    #     qasm_file.writelines('RO {} | RO {}\n'.format(fluxing_qubit,
    #                                                   spectator_qubit))
    # else:
    #     qasm_file.writelines('RO {} \n'.format(RO_target))

    # qasm_file.close()
    # return qasm_file


def two_qubit_tomo_bell(bell_state, q0, q1,
                        wait_after_trigger=10e-9, wait_during_flux=260e-9,
                        clock_cycle=1e-9,
                        single_qubit_compiled_phase=False,
                        RO_target='all'):
    '''
    Two qubit bell state tomography.

    Args:
        bell_state      (int): index of prepared bell state
        q0, q1          (str): names of the target qubits
        wait_after_trigger (float): delay time in seconds after sending the
                                    trigger for the flux pulse
        clock_cycle     (float): period of the internal AWG clock
        wait_during_flux (int): wait time during the flux pulse
        single_qubit_compiled_phase (bool): wether to do single qubit phase
            correction in the recovery pulse
        RO_target   (str): can be q0, q1, or 'all'
    '''
    raise NotImplementedError()
    # tomo_pulses = ['I ', 'X180 ', 'Y90 ', 'mY90 ', 'X90 ', 'mX90 ']
    # tomo_list_q0 = []
    # tomo_list_q1 = []
    # for tp in tomo_pulses:
    #     tomo_list_q0 += [tp + q0 + '\n']
    #     tomo_list_q1 += [tp + q1 + '\n']

    # tomo_list_q0[0] = 'I 20\n'
    # tomo_list_q1[0] = 'I 20\n'

    # # Choose a bell state and set the corresponding preparation pulses
    # if bell_state % 10 == 0:  # |Phi_m>=|00>-|11>
    #     prep_pulse_q0 = 'Y90 {}\n'.format(q0)
    #     prep_pulse_q1 = 'Y90 {}\n'.format(q1)
    # elif bell_state % 10 == 1:  # |Phi_p>=|00>+|11>
    #     prep_pulse_q0 = 'mY90 {}\n'.format(q0)
    #     prep_pulse_q1 = 'Y90 {}\n'.format(q1)
    # elif bell_state % 10 == 2:  # |Psi_m>=|01>-|10>
    #     prep_pulse_q0 = 'Y90 {}\n'.format(q0)
    #     prep_pulse_q1 = 'mY90 {}\n'.format(q1)
    # elif bell_state % 10 == 3:  # |Psi_p>=|01>+|10>
    #     prep_pulse_q0 = 'mY90 {}\n'.format(q0)
    #     prep_pulse_q1 = 'mY90 {}\n'.format(q1)
    # else:
    #     raise ValueError('Bell state {} is not defined.'.format(bell_state))

    # # Recovery pulse is the same for all Bell states
    # if single_qubit_compiled_phase == False:
    #     after_pulse = 'mY90 {}\n'.format(q1)
    # else:
    #     after_pulse = 'recmY90 {}\n'.format(q1)

    # # Disable preparation pulse on one or the other qubit for debugging
    # if bell_state//10 == 1:
    #     prep_pulse_q1 = 'I 20'
    # elif bell_state//10 == 2:
    #     prep_pulse_q0 = 'I 20'

    # # Define compensation pulses
    # # FIXME: needs to be added
    # print('Warning: not using compensation pulses.')

    # # Write tomo sequence

    # filename = join(base_qasm_path, 'two_qubit_tomo_bell.qasm')
    # qasm_file = mopen(filename, mode='w')
    # qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    # for p_q1 in tomo_list_q1:
    #     for p_q0 in tomo_list_q0:
    #         qasm_file.writelines('\ninit_all\n')
    #         qasm_file.writelines('QWG trigger\n')
    #         qasm_file.writelines(
    #             'I {}\n'.format(int(wait_after_trigger//clock_cycle)))
    #         qasm_file.writelines(prep_pulse_q0)
    #         qasm_file.writelines(prep_pulse_q1)
    #         qasm_file.writelines(
    #             'I {}\n'.format(int(wait_during_flux//clock_cycle)))
    #         qasm_file.writelines(after_pulse)
    #         qasm_file.writelines(p_q1)
    #         qasm_file.writelines(p_q0)
    #         qasm_file.writelines('RO ' + RO_target + '  \n')

    # # Add calibration pulses
    # cal_pulses = []
    # # every calibration point is repeated 7 times. This is copied from the
    # # script for Tektronix driven qubits. I do not know if this repetition
    # # is important or even necessary here.
    # for seq in cal_points_2Q:
    #     cal_pulses += [[seq[0].format(q0), seq[1].format(q1),
    #                     'RO ' + RO_target + '\n']] * 7

    # for seq in cal_pulses:
    #     qasm_file.writelines('\ninit_all\n')
    #     for p in seq:
    #         qasm_file.writelines(p)

    # qasm_file.close()
    # return qasm_file


def CZ_calibration_seq(q0, q1, RO_target='all',
                       CZ_disabled=False,
                       cases=('no_excitation', 'excitation'),
                       wait_after_trigger=40e-9,
                       wait_during_flux=280e-9,
                       clock_cycle=1e-9,
                       mw_pulse_duration=40e-9):
    '''
    Sequence used to calibrate flux pulses for CZ gates.

    Timing of the sequence:
    q0:   --   X90  C-Phase  Rphi90   --      RO
    q1: (X180)  --     --       --   (X180)    RO

    Args:
        q0, q1      (str): names of the addressed qubits
        RO_target   (str): can be q0, q1, or 'all'
        CZ_disabled (bool): disable CZ gate
        excitations (bool/str): can be True, False, or 'both_cases'
        clock_cycle (float): period of the internal AWG clock
        wait_time   (int): wait time in seconds after triggering the flux
    '''
    raise NotImplementedError()

    # filename = join(base_qasm_path, 'CZ_calibration_seq.qasm')
    # qasm_file = mopen(filename, mode='w')
    # qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    # for case in cases:
    #     qasm_file.writelines('\ninit_all\n')
    #     qasm_file.writelines('QWG trigger\n')
    #     waitTime = wait_after_trigger
    #     if case == 'excitation':
    #         # Decrease wait time because there is an additional pulse
    #         waitTime -= mw_pulse_duration
    #     qasm_file.writelines(
    #         'I {}\n'.format(int(waitTime//clock_cycle)))
    #     if case == 'excitation':
    #         qasm_file.writelines('X180 {}\n'.format(q1))
    #     qasm_file.writelines('X90 {}\n'.format(q0))
    #     qasm_file.writelines(
    #         'I {}\n'.format(int(wait_during_flux//clock_cycle)))
    #     qasm_file.writelines('Rphi90 {}\n'.format(q0))
    #     if case == 'excitation':
    #         qasm_file.writelines('X180 {}\n'.format(q1))

    #     qasm_file.writelines('RO {}  \n'.format(RO_target))

    # qasm_file.close()
    # return qasm_file


def CZ_fast_calibration_seq(q0_name: str, q1_name: str, no_of_points: int,
                            cal_points: bool=True,
                            RO_target: str='all',
                            CZ_disabled: bool=False,
                            cases=('no_excitation', 'excitation'),
                            wait_after_trigger=40e-9,
                            wait_during_flux=280e-9,
                            clock_cycle=1e-9,
                            mw_pulse_duration=40e-9):
    '''
    Sequence used to (numerically) calibrate CZ gate, including single qubit
    phase corrections.
    Repeats the sequence below 'no_of_points' times, giving a new trigger
    instruction
        QWG trigger 'i'
    every time, where 'i' is the number of iteration (starting at 0).

    Timing of the sequence:
    q0:   --   mX90  C-Phase  X90   --      RO
    q1: (X180)  --     --       --   (X180)    RO

    Args:
        q0, q1      (str): names of the addressed qubits
        RO_target   (str): can be q0, q1, or 'all'
        CZ_disabled (bool): disable CZ gate
        excitations (bool/str): can be True, False, or 'both_cases'
        clock_cycle (float): period of the internal AWG clock
        wait_time   (int): wait time in seconds after triggering the flux
    '''
    raise NotImplementedError()
    # filename = join(base_qasm_path, 'CZ_fast_calibration_seq.qasm')
    # qasm_file = mopen(filename, mode='w')
    # qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0_name, q1_name))

    # for i in range(no_of_points):

    #     if cal_points and (i == no_of_points - 4 or i == no_of_points - 3):
    #         # Calibration point for |0>
    #         qasm_file.writelines('\ninit_all\n')
    #         qasm_file.writelines('RO {}  \n'.format(RO_target))
    #         pass
    #     elif cal_points and (i == no_of_points - 2 or i == no_of_points - 1):
    #         # Calibration point for |1>
    #         qasm_file.writelines('\ninit_all\n')
    #         qasm_file.writelines('X180 {} \n'.format(q0_name))
    #         qasm_file.writelines('X180 {} \n'.format(q1_name))
    #         qasm_file.writelines('RO {}  \n'.format(RO_target))
    #     else:
    #         for case in cases:
    #             qasm_file.writelines('\ninit_all\n')
    #             qasm_file.writelines('QWG_trigger_{}\n'.format(i))
    #             waitTime = wait_after_trigger
    #             if case == 'excitation':
    #                 # Decrease wait time because there is an additional pulse
    #                 waitTime -= mw_pulse_duration
    #             qasm_file.writelines(
    #                 'I {}\n'.format(int(waitTime//clock_cycle)))
    #             if case == 'excitation':
    #                 qasm_file.writelines('X180 {}\n'.format(q1_name))
    #             qasm_file.writelines('mX90 {}\n'.format(q0_name))
    #             qasm_file.writelines(
    #                 'I {}\n'.format(int(wait_during_flux//clock_cycle)))
    #             qasm_file.writelines('X90 {}\n'.format(q0_name))
    #             if case == 'excitation':
    #                 qasm_file.writelines('X180 {}\n'.format(q1_name))

    #             qasm_file.writelines('RO {}  \n'.format(RO_target))

    # qasm_file.close()
    # return qasm_file


def chevron_block_seq(q0_name, q1_name, no_of_points,
                      excite_q1=False, wait_after_trigger=40e-9,
                      wait_during_flux=400e-9, clock_cycle=1e-9,
                      RO_target='all', mw_pulse_duration=40e-9,
                      cal_points=True):
    '''
    Sequence for measuring a block of a chevron, i.e. using different codewords
    for different pulse lengths.

    Args:
        q0, q1        (str): names of the addressed qubits.
                             q0 is the pulse that experiences the flux pulse.
        RO_target     (str): can be q0, q1, or 'all'
        excite_q1    (bool): choose whether to excite q1, thus choosing
                             between the |01> <-> |10> and the |11> <-> |20>
                             swap.
        wait_after_trigger (float): delay time in seconds after sending the
                             trigger for the flux pulse
        clock_cycle (float): period of the internal AWG clock
        wait_time     (int): wait time between triggering QWG and RO
        cal_points   (bool): whether to use calibration points or not
    '''
    raise NotImplementedError()
    # filename = join(base_qasm_path, 'chevron_block_seq.qasm')
    # qasm_file = mopen(filename, mode='w')
    # qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0_name, q1_name))

    # for i in range(no_of_points):
    #     qasm_file.writelines('\ninit_all\n')

    #     qasm_file.writelines('QWG trigger {}\n'.format(i))
    #     if excite_q1:
    #         wait_after_trigger -= mw_pulse_duration
    #     qasm_file.writelines(
    #         'I {}\n'.format(int(wait_after_trigger//clock_cycle)))
    #     qasm_file.writelines('X180 {}\n'.format(q0_name))
    #     if excite_q1:
    #         qasm_file.writelines('X180 {}\n'.format(q1_name))
    #     qasm_file.writelines(
    #         'I {}\n'.format(int(wait_during_flux//clock_cycle)))
    #     if excite_q1:
    #         # q0 is rotated to ground-state to have better contrast
    #         # (|0> and |2> instead of |1> and |2>)
    #         qasm_file.writelines('X180 {}\n'.format(q0_name))
    #     qasm_file.writelines('RO {} \n'.format(RO_target))

    # if cal_points:
    #     # Add calibration pulses
    #     cal_pulses = []
    #     for seq in cal_points_2Q:
    #         cal_pulses += [[seq[0], seq[1], 'RO ' + RO_target + '\n']]

    # qasm_file.close()
    # return qasm_file


def CZ_state_cycling_light(q0: str, q1: str, N: int=1):
    """
    Implements a circuit that performs a permutation over all computational
    states. This light version performs this experiment for all 4 possible
    input states.

    Expected operation:
        U (|00>) -> |01>
        U (|01>) -> |11>
        U (|10>) -> |00>
        U (|11>) -> |10>

    Args:
        q0 (str): name of qubit q0
        q1 (str): name of qubit q1
        N  (int): number of times to apply U
    """
    raise NotImplementedError()
    # filename = join(base_qasm_path, 'CZ_state_cycling_light.qasm')
    # qasm_file = mopen(filename, mode='w')
    # qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    # U = ''
    # U += 'Y90 {} | mY90 {}\n'.format(q0, q1)
    # U += 'CZ {} {}\n'.format(q0, q1)
    # U += 'Y90 {} | Y90 {}\n'.format(q0, q1)
    # U += 'CZ {} {}\n'.format(q0, q1)
    # U += 'Y90 {} | Y90 {}\n'.format(q0, q1)

    # # Input |00>
    # qasm_file.writelines('init_all \n')
    # qasm_file.writelines('qwg_trigger_0 {}\n'.format(q0))
    # for n in range(N):
    #     qasm_file.writelines(U)
    # qasm_file.writelines('RO {}\n'.format(q0))

    # # Input |01>
    # qasm_file.writelines('init_all \n')
    # qasm_file.writelines('qwg_trigger_0 {}\n'.format(q0))
    # qasm_file.writelines('X180 {}\n'.format(q0))
    # for n in range(N):
    #     qasm_file.writelines(U)
    # qasm_file.writelines('RO {}\n'.format(q0))

    # # Input |10>
    # qasm_file.writelines('init_all \n')
    # qasm_file.writelines('qwg_trigger_0 {}\n'.format(q0))
    # qasm_file.writelines('X180 {}\n'.format(q1))
    # for n in range(N):
    #     qasm_file.writelines(U)
    # qasm_file.writelines('RO {}\n'.format(q0))

    # # Input |11>
    # qasm_file.writelines('init_all \n')
    # qasm_file.writelines('qwg_trigger_0 {}\n'.format(q0))
    # qasm_file.writelines('X180 {} | X180 {}\n'.format(q0, q1))
    # for n in range(N):
    #     qasm_file.writelines(U)
    # qasm_file.writelines('RO {}\n'.format(q0))

    # qasm_file.close()
    # return qasm_file


def CZ_restless_state_cycling(q0: str, q1: str, N: int=1):
    """
    Implements a circuit that performs a permutation over all computational
    states.

    Expected operation:
        U (|00>) -> |01>
        U (|01>) -> |11>
        U (|10>) -> |00>
        U (|11>) -> |10>

    Args:
        q0 (str): name of qubit q0
        q1 (str): name of qubit q1
        N  (int): number of times to apply U
    """
    raise NotImplementedError()
    # filename = join(base_qasm_path, 'CZ_state_cycling_light.qasm')
    # qasm_file = mopen(filename, mode='w')
    # qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    # U = ''
    # U += 'Y90 {} | mY90 {}\n'.format(q0, q1)
    # U += 'CZ {} {}\n'.format(q0, q1)
    # U += 'Y90 {} | Y90 {}\n'.format(q0, q1)
    # U += 'CZ {} {}\n'.format(q0, q1)
    # U += 'Y90 {} | Y90 {}\n'.format(q0, q1)

    # for n in range(N):
    #     qasm_file.writelines(U)
    # qasm_file.writelines('RO {}\n'.format(q0))


def add_two_q_cal_points(p, platf, q0: int, q1: int,
                         reps_per_cal_pt: int =1):
    """
    Returns a list of kernels containing calibration points for two qubits

    Args:
        p               : OpenQL  program to add calibration points to
        platf           : OpenQL platform used in the program
        q0, q1          : ints of two qubits
        reps_per_cal_pt : number of times to repeat each cal point
    Returns:
        kernel_list     : list containing kernels for the calibration points
    """
    kernel_list = []
    combinations = (["00"]*reps_per_cal_pt +
                    ["01"]*reps_per_cal_pt +
                    ["10"]*reps_per_cal_pt +
                    ["11"]*reps_per_cal_pt)
    for i, comb in enumerate(combinations):
        k = Kernel('cal{}_{}'.format(i, comb), p=platf)
        k.prepz(q0)
        k.prepz(q1)
        if comb[0] == '1':
            k.gate('rx180', q0)
        if comb[1] == '1':
            k.gate('rx180', q1)
        k.measure(q0)
        k.measure(q1)
        kernel_list.append(k)
        p.add_kernel(k)

    return p
