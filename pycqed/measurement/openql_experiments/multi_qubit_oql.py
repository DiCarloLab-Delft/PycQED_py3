from os.path import join, dirname
import numpy as np
import openql.openql as ql
from pycqed.utilities.general import suppress_stdout
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
        k.prepz(idx)  # to ensure enough separation in timing
        k.prepz(idx)  # to ensure enough separation in timing

    for i in range(7):
        k.gate('CW_00', i)
    k.gate('fl_cw_02', qubit_indices[0], qubit_indices[1])
    p.add_kernel(k)
    with suppress_stdout():
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
    with suppress_stdout():
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
    p = add_two_q_cal_points(p, platf=platf, q0=q0, q1=q1)
    with suppress_stdout():
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
    with suppress_stdout():
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

    with suppress_stdout():
        p.compile()
    # attribute is added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def Cryoscope(qubit_idx: int, buffer_time1=0, buffer_time2=0,
              platf_cfg: str=''):
    """
    Single qubit Ramsey sequence.
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        times:          the list of waiting times for each Ramsey element
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
    Returns:
        p:              OpenQL Program object containing

    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="Cryoscope", nqubits=platf.get_qubit_number(),
                p=platf)

    buffer_nanoseconds1 = int(round(buffer_time1/1e-9))
    buffer_nanoseconds2 = int(round(buffer_time2/1e-9))

    k = Kernel("RamZ_X", p=platf)
    k.prepz(qubit_idx)
    k.gate('rx90', qubit_idx)
    k.gate("wait", [qubit_idx], buffer_nanoseconds1)
    k.gate('fl_cw_02', 2, 0)
    k.gate("wait", [qubit_idx], buffer_nanoseconds2)
    k.gate('rx90', qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    k = Kernel("RamZ_Y", p=platf)
    k.prepz(qubit_idx)
    k.gate('rx90', qubit_idx)
    k.gate("wait", [qubit_idx], buffer_nanoseconds1)
    k.gate('fl_cw_02', 2, 0)
    k.gate("wait", [qubit_idx], buffer_nanoseconds2)
    k.gate('ry90', qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    # adding the calibration points
    # add_single_qubit_cal_points(p, platf=platf, qubit_idx=qubit_idx)

    with suppress_stdout():
        p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def CryoscopeGoogle(qubit_idx: int, buffer_time1, times, platf_cfg: str):
    """
    A Ramsey sequence with varying waiting times `times` around a flux pulse.
    Generates 2xlen(times) measurements (t1-x, t1-y, t2-x, t2-y. etc)
    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="CryoscopeGoogle", nqubits=platf.get_qubit_number(),
                p=platf)

    buffer_nanoseconds1 = int(round(buffer_time1/1e-9))

    for t in times:

        t_nanoseconds = int(round(t/1e-9))

        k = Kernel("RamZ_X", p=platf)
        k.prepz(qubit_idx)
        k.gate('rx90', qubit_idx)
        k.gate("wait", [qubit_idx], buffer_nanoseconds1)
        k.gate('fl_cw_02', 2, 0)
        k.gate("wait", [qubit_idx], t_nanoseconds)
        k.gate('rx90', qubit_idx)
        k.measure(qubit_idx)
        p.add_kernel(k)
        k = Kernel("RamZ_Y", p=platf)
        k.prepz(qubit_idx)
        k.gate('rx90', qubit_idx)
        k.gate("wait", [qubit_idx], buffer_nanoseconds1)
        k.gate('fl_cw_02', 2, 0)
        k.gate("wait", [qubit_idx], t_nanoseconds)
        k.gate('ry90', qubit_idx)
        k.measure(qubit_idx)
        p.add_kernel(k)

    with suppress_stdout():
        p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


# FIMXE: merge into the real chevron seq
def Chevron_hack(qubit_idx: int, qubit_idx_spec,
                 buffer_time, buffer_time2, platf_cfg: str):
    """
    Single qubit Ramsey sequence.
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        times:          the list of waiting times for each Ramsey element
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
    Returns:
        p:              OpenQL Program object containing

    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="Chevron", nqubits=platf.get_qubit_number(),
                p=platf)

    buffer_nanoseconds = int(round(buffer_time/1e-9))
    buffer_nanoseconds2 = int(round(buffer_time/1e-9))

    k = Kernel("Chevron", p=platf)
    k.prepz(qubit_idx)
    k.gate('rx90', qubit_idx_spec)
    k.gate('rx180', qubit_idx)
    k.gate("wait", [qubit_idx], buffer_nanoseconds)
    k.gate('fl_cw_02', 2, 0)
    k.gate('wait', [qubit_idx], buffer_nanoseconds2)
    k.gate('rx180', qubit_idx)
    k.measure(qubit_idx)
    # k.measure(qubit_idx_spec)
    p.add_kernel(k)

    # adding the calibration points
    # add_single_qubit_cal_points(p, platf=platf, qubit_idx=qubit_idx)

    with suppress_stdout():
        p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def Chevron(qubit_idx: int, qubit_idx_spec: int,
            buffer_time, buffer_time2, platf_cfg: str):
    """
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        qubit_idx:      int specifying the target qubit (starting at 0)
        qubit_idx_spec: int specifying the spectator qubit
        buffer_time   :
        buffer_time2  :

        platf_cfg:      filename of the platform config file
    Returns:
        p:              OpenQL Program object containing

    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="Chevron", nqubits=platf.get_qubit_number(),
                p=platf)

    buffer_nanoseconds = int(round(buffer_time/1e-9))
    buffer_nanoseconds2 = int(round(buffer_time2/1e-9))

    k = Kernel("Chevron", p=platf)
    k.prepz(qubit_idx)
    k.gate('rx90', qubit_idx_spec)
    k.gate('rx180', qubit_idx)
    k.gate("wait", [qubit_idx], buffer_nanoseconds)
    k.gate('fl_cw_02', 2, 0)
    k.gate('wait', [qubit_idx], buffer_nanoseconds2)
    k.gate('rx180', qubit_idx)
    k.measure(qubit_idx)
    k.measure(qubit_idx_spec)
    k.gate("wait", [qubit_idx, qubit_idx_spec], 0)
    p.add_kernel(k)

    with suppress_stdout():
        p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def two_qubit_tomo_bell(bell_state, q0, q1,
                        platf_cfg):
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
    tomo_gates = ['i', 'rx180', 'ry90', 'rym90', 'rx90', 'rxm90']

    # Choose a bell state and set the corresponding preparation pulses
    if bell_state == 0:  # |Phi_m>=|00>-|11>
        prep_pulse_q0, prep_pulse_q1 = 'ry90', 'ry90'
    elif bell_state % 10 == 1:  # |Phi_p>=|00>+|11>
        prep_pulse_q0, prep_pulse_q1 = 'rym90', 'ry90'
    elif bell_state % 10 == 2:  # |Psi_m>=|01>-|10>
        prep_pulse_q0, prep_pulse_q1 = 'ry90', 'rym90'
    elif bell_state % 10 == 3:  # |Psi_p>=|01>+|10>
        prep_pulse_q0, prep_pulse_q1 = 'rym90', 'rym90'
    else:
        raise ValueError('Bell state {} is not defined.'.format(bell_state))

    # Recovery pulse is the same for all Bell states
    after_pulse_q1 = 'rym90'

    # # Define compensation pulses
    # # FIXME: needs to be added
    # print('Warning: not using compensation pulses.')

    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="two_qubit_tomo_bell",
                nqubits=platf.get_qubit_number(),
                p=platf)
    for p_q1 in tomo_gates:
        for p_q0 in tomo_gates:
            k = Kernel("BellTomo_{}{}_{}{}".format(
                       q1, p_q1, q0, p_q0
                       ), p=platf)
            # next experiment
            k.prepz(q0)  # to ensure enough separation in timing
            k.prepz(q1)  # to ensure enough separation in timing
            # pre-rotations
            k.gate(prep_pulse_q0, q0)
            k.gate(prep_pulse_q1, q1)
            # FIXME hardcoded edge because of
            # brainless "directed edge recources" in compiler
            k.gate('fl_cw_01', 2, 0)
            # after-rotations
            k.gate(after_pulse_q1, q1)
            # tomo pulses
            k.gate(p_q1, q0)
            k.gate(p_q0, q1)
            # measure
            k.measure(q0)
            k.measure(q1)
            # sync barrier before tomo
            # k.gate("wait", [q0, q1], 0)
            k.gate("wait", [2, 0], 0)
            p.add_kernel(k)
    # 7 repetitions is because of assumptions in tomo analysis
    p = add_two_q_cal_points(p, platf=platf, q0=q0, q1=q1, reps_per_cal_pt=7)
    with suppress_stdout():
        p.compile()
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def two_qubit_DJ(q0, q1, platf_cfg):
    '''
    Two qubit Deutsch-Josza.

    Args:
        q0, q1          (str): names of the target qubits
    '''
    # Recovery pulse is the same for all Bell states
    after_pulse_q1 = 'rym90'

    # # Define compensation pulses
    # # FIXME: needs to be added
    # print('Warning: not using compensation pulses.')

    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="two_qubit_DJ",
                nqubits=platf.get_qubit_number(),
                p=platf)

    # experiments
    # 1
    k = Kernel("DJ1", p=platf)
    k.prepz(q0)  # to ensure enough separation in timing
    k.prepz(q1)  # to ensure enough separation in timing
    # prerotations
    k.gate('ry90', q0)
    k.gate('rym90', q1)
    # post rotations
    k.gate('ry90', q0)
    k.gate('ry90', q1)
    # measure
    k.measure(q0)
    k.measure(q1)
    p.add_kernel(k)

    # 2
    k = Kernel("DJ2", p=platf)
    k.prepz(q0)  # to ensure enough separation in timing
    k.prepz(q1)  # to ensure enough separation in timing
    # prerotations
    k.gate('ry90', q0)
    k.gate('rym90', q1)
    # rotations
    k.gate('rx180', q1)
    # post rotations
    k.gate('ry90', q0)
    k.gate('ry90', q1)
    # measure
    k.measure(q0)
    k.measure(q1)
    p.add_kernel(k)

    # 3
    k = Kernel("DJ3", p=platf)
    k.prepz(q0)  # to ensure enough separation in timing
    k.prepz(q1)  # to ensure enough separation in timing
    # prerotations
    k.gate('ry90', q0)
    k.gate('rym90', q1)
    # rotations
    k.gate('ry90', q1)
    k.gate('rx180', q0)
    k.gate('rx180', q1)

    # Hardcoded flux pulse, FIXME use actual CZ
    k.gate('wait', [2, 0], 100)
    k.gate('fl_cw_01', 2, 0)
    # FIXME hardcoded extra delays
    k.gate('wait', [2, 0], 200)

    k.gate('rx180', q0)
    k.gate('ry90', q1)

    # post rotations
    k.gate('ry90', q0)
    k.gate('ry90', q1)
    # measure
    k.measure(q0)
    k.measure(q1)
    p.add_kernel(k)

    # 4
    k = Kernel("DJ4", p=platf)
    k.prepz(q0)  # to ensure enough separation in timing
    k.prepz(q1)  # to ensure enough separation in timing
    # prerotations
    k.gate('ry90', q0)
    k.gate('rym90', q1)
    # rotations
    k.gate('rym90', q1)
    # Hardcoded flux pulse, FIXME use actual CZ
    k.gate('wait', [2, 0], 100)
    k.gate('fl_cw_01', 2, 0)
    # FIXME hardcoded extra delays
    k.gate('wait', [2, 0], 200)

    k.gate('rx180', q1)
    k.gate('rym90', q1)

    # post rotations
    k.gate('ry90', q0)
    k.gate('ry90', q1)
    # measure
    k.measure(q0)
    k.measure(q1)
    p.add_kernel(k)

    # 7 repetitions is because of assumptions in tomo analysis
    #p = add_two_q_cal_points(p, platf=platf, q0=q0, q1=q1, reps_per_cal_pt=7)
    with suppress_stdout():
        p.compile()
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def two_qubit_repeated_parity_check(qD: int, qA: int, platf_cfg: str,
                                    number_of_repetitions: int = 10,
                                    initialization_msmt: bool=False,
                                    initial_states=[0, 1]):
    """
    Implements a circuit for repeated parity checks.

    Circuit looks as follows:

    Data    (M)|------0------- | ^N- M
               |      |        |
    Ancilla (M)|--y90-0-y90-M- |   - M

    The initial "M" measurement is optional, the circuit is repated N times
    At the end both qubits are measured.

    Arguments:
        qD :        Data qubit, this is the qubit that the repeated parity
                    check will be performed on.
        qA :        Ancilla qubit, qubit that the parity will be mapped onto.
        platf_cfg:  filename of the platform config file
        number_of_repetitions: number of times to repeat the circuit
        initialization_msmt : whether to start with an initial measurement
                    to prepare the starting state.
    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="repeated_parity_check",
                nqubits=platf.get_qubit_number(), p=platf)


    for initial_state in initial_states:
        k = Kernel('repeated_parity_check_{}'.format(initial_state), p=platf)
        k.prepz(qD)
        k.prepz(qA)

        if initialization_msmt:
            k.measure(qA)
            k.measure(qD)
            k.gate('wait', [2, 0], 500)
        if initial_state== 1:
            k.gate('rx180', qD)
        for i in range(number_of_repetitions):
            # hardcoded barrier because of openQL #104
            k.gate('wait', [2, 0], 0)

            # k.gate('wait', [qA, qD], 0)
            k.gate('ry90', qA)
            if i == 0:
                # Flux cw_03 it he repeated cz gate
                k.gate('fl_cw_03', 2, 0)
                # k.gate('fl_cw_03', qA, qD)
            else:
                k.gate('fl_cw_00', 2, 0)
                # k.gate('fl_cw_00', qA, qD)
            k.gate('ry90', qA)
            k.measure(qA)

        k.measure(qD)
        # hardcoded barrier because of openQL #104
        k.gate('wait', [2, 0], 0)
        # k.gate('wait', [qA, qD], 0)
        p.add_kernel(k)

    with suppress_stdout():
        p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def conditional_oscillation_seq(q0: int, q1: int, platf_cfg: str,
                                CZ_disabled: bool=False,
                                angles=np.arange(0, 360, 20),
                                wait_time: int=0,
                                add_cal_points: bool=True,
                                cases: list=('no_excitation', 'excitation'),
                                flux_codeword:str='fl_cw_01' ):
    '''
    Sequence used to calibrate flux pulses for CZ gates.

    q0 is the oscilating qubit
    q1 is the spectator qubit

    Timing of the sequence:
    q0:   --   X90  C-Phase  Rphi90   --       RO
    q1: (X180)  --     --       --   (X180)    RO

    Args:
        q0, q1      (str): names of the addressed qubits
        RO_target   (str): can be q0, q1, or 'all'
        CZ_disabled (bool): disable CZ gate
        angles      (array): angles of the recovery pulse
        wait_time   (int): wait time in seconds after triggering the flux
    '''
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="conditional_oscillation_seq",
                nqubits=platf.get_qubit_number(),
                p=platf)
    # These angles correspond to special pi/2 pulses in the lutman
    for i, angle in enumerate(angles):
        for case in cases:
            # cw_idx corresponds to special hardcoded angles in the lutman
            cw_idx = angle//20 + 9

            k = Kernel("{}_{}".format(case, angle), p=platf)
            k.prepz(q0)
            k.prepz(q1)
            if case == 'excitation':
                k.gate('rx180', q1)
            k.gate('rx90', q0)
            if not CZ_disabled:
                k.gate(flux_codeword, 2, 0)
            k.gate('wait', [2, 0], wait_time)
            # hardcoded angles, must be uploaded to AWG
            if angle == 90:
                # special because the cw phase pulses go in mult of 20 deg
                k.gate('ry90', q0)
            else:
                k.gate('cw_{:02}'.format(cw_idx), q0)
            if case == 'excitation':
                k.gate('rx180', q1)

            k.measure(q0)
            k.measure(q1)
            # Implements a barrier to align timings
            # k.gate('wait', [q0, q1], 0)
            # hardcoded barrier because of openQL #104
            k.gate('wait', [2, 0], 0)

            p.add_kernel(k)
    if add_cal_points:
        p = add_two_q_cal_points(p, platf=platf, q0=q0, q1=q1)
    with suppress_stdout():
        p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')

    if add_cal_points:
        cal_pts_idx = [361, 362, 363, 364]
    else:
        cal_pts_idx = []

    p.sweep_points = np.concatenate([np.repeat(angles, len(cases)), cal_pts_idx])
    p.set_sweep_points(p.sweep_points, len(p.sweep_points))
    return p


def grovers_two_qubit_all_inputs(q0: int, q1: int, platf_cfg: str,
                                 precompiled_flux: bool=True,
                                 cal_points: bool=True):
    """
    Writes the QASM sequence for Grover's algorithm on two qubits.
    Sequence:
        q0: G0 -       - mY90 -    - mY90  - RO
                 CZ_ij          CZ
        q1: G1 -       - mY90 -    - mY90  - RO
    whit all combinations of (ij) = omega.
    G0 and G1 are Y90 or Y90, depending on the (ij).

    Args:
        q0_name, q1_name (string):
                Names of the qubits to which the sequence is applied.
        RO_target (string):
                Readout target. Can be a qubit name or 'all'.
        precompiled_flux (bool):
                Determies if the full waveform for the flux pulses is
                precompiled, thus only needing one trigger at the start,
                or if every flux pulse should be triggered individually.
        cal_points (bool):
                Whether to add calibration points.

    Returns:
        qasm_file: a reference to the new QASM file object.
    """

    if not precompiled_flux:
        raise NotImplementedError('Currently only precompiled flux pulses '
                                  'are supported.')

    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="Grovers_two_qubit_all_inputs",
                nqubits=platf.get_qubit_number(), p=platf)

    for G0 in ['ry90', 'rym90']:
        for G1 in ['ry90', 'rym90']:
            k = Kernel('Gr{}_{}'.format(G0, G1),  p=platf)
            k.prepz(q0)
            k.prepz(q1)
            k.gate(G0, q0)
            k.gate(G1, q1)
            k.gate('fl_cw_03', 2,0) # flux cw03 is the multi_cz pulse
            k.gate('ry90', q0)
            k.gate('ry90', q1)
            # k.gate('fl_cw_00', 2,0)
            k.gate('wait', [2, 0], 260)
            k.gate('ry90', q0)
            k.gate('ry90', q1)
            k.measure(q0)
            k.measure(q1)
            k.gate('wait', [2, 0], 0)
            p.add_kernel(k)

    if cal_points:
        p = add_two_q_cal_points(p, platf=platf, q0=q0, q1=q1)
    with suppress_stdout():
        p.compile()

    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p

def grovers_tomography(q0: int, q1: int, omega:int, platf_cfg: str,
                       precompiled_flux: bool=True,
                       cal_points: bool=True):

    if not precompiled_flux:
        raise NotImplementedError('Currently only precompiled flux pulses '
                                  'are supported.')

    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="Grovers_tomo_two_qubit_all_inputs",
                nqubits=platf.get_qubit_number(), p=platf)

    tomo_gates = ['i', 'rx180', 'ry90', 'rym90', 'rx90', 'rxm90']


    if omega == 0:
        G0 = 'ry90'
        G1 = 'ry90'
    elif omega == 1:
        G0 = 'ry90'
        G1 = 'rym90'
    elif omega == 2:
        G0 = 'rym90'
        G1 = 'ry90'
    elif omega == 3:
        G0 = 'rym90'
        G1 = 'rym90'
    else:
        raise ValueError('omega must be in [0, 3]')


    for p_q1 in tomo_gates:
        for p_q0 in tomo_gates:
            k = Kernel('Gr{}_{}_tomo_{}_{}'.format(G0, G1, p_q0, p_q1),
                       p=platf)
            k.prepz(q0)
            k.prepz(q1)
            k.gate(G0, q0)
            k.gate(G1, q1)
            k.gate('fl_cw_03', 2,0) # flux cw03 is the multi_cz pulse
            k.gate('ry90', q0)
            k.gate('ry90', q1)
            # k.gate('fl_cw_00', 2,0)
            k.gate('wait', [2, 0], 260)
            k.gate('ry90', q0)
            k.gate('ry90', q1)

            # tomo pulses
            k.gate(p_q1, q0)
            k.gate(p_q0, q1)

            k.measure(q0)
            k.measure(q1)
            k.gate('wait', [2, 0], 0)
            p.add_kernel(k)

    p = add_two_q_cal_points(p, platf=platf, q0=q0, q1=q1, reps_per_cal_pt=7)
    with suppress_stdout():
        p.compile()

    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def CZ_poisoned_purity_seq(q0, q1, platf_cfg: str, cal_points: bool=True):
    """
    Creates the |00> + |11> Bell state and does a partial tomography in
    order to determine the purity of both qubits.
    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="CZ_poisoned_purity_seq",
                nqubits=platf.get_qubit_number(), p=platf)
    tomo_list = ['rxm90', 'rym90', 'i']

    for p_pulse in tomo_list:
        k = Kernel("{}".format(p_pulse), p=platf)
        k.prepz(q0)
        k.prepz(q1)

        # Create a Bell state:  |00> + |11>
        k.gate('rym90', q0)
        k.gate('ry90', q1)
        k.gate('fl_cw_01', 2, 0)
        k.gate('rym90', q1)

        # Perform pulses to measure the purity of both qubits
        k.gate(p_pulse, q0)
        k.gate(p_pulse, q1)

        k.measure(q0)
        k.measure(q1)
        # Implements a barrier to align timings
        # k.gate('wait', [q0, q1], 0)
        # hardcoded because of openQL #104
        k.gate('wait', [2, 0], 0)

        p.add_kernel(k)
    if cal_points:
        k = Kernel("Cal 00", p=platf)
        k.prepz(q0)
        k.prepz(q1)
        k.measure(q0)
        k.measure(q1)
        k.gate('wait', [2, 0], 0)
        p.add_kernel(k)
        k = Kernel("Cal 11", p=platf)
        k.prepz(q0)
        k.prepz(q1)
        k.gate("rx180", q0)
        k.gate("rx180", q1)
        k.measure(q0)
        k.measure(q1)
        k.gate('wait', [2, 0], 0)
        p.add_kernel(k)


    with suppress_stdout():
        p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


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
        else:
            k.gate('i', q0)
        if comb[1] == '1':
            k.gate('rx180', q1)
        else:
            k.gate('i', q1)
        k.measure(q0)
        k.measure(q1)
        kernel_list.append(k)
        p.add_kernel(k)

    return p
