import numpy as np
import openql.openql as ql
import pycqed.measurement.openql_experiments.openql_helpers as oqh
from pycqed.utilities.general import int2base, suppress_stdout
from os.path import join


def single_flux_pulse_seq(qubit_indices: tuple,
                          platf_cfg: str):

    p = oqh.create_program("single_flux_pulse_seq", platf_cfg)

    k = oqh.create_kernel("main", p)
    for idx in qubit_indices:
        k.prepz(idx)  # to ensure enough separation in timing
        k.prepz(idx)  # to ensure enough separation in timing
        k.prepz(idx)  # to ensure enough separation in timing


    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0)
    k.gate('fl_cw_02', [qubit_indices[0], qubit_indices[1]])
    p.add_kernel(k)

    p = oqh.compile(p)
    return p


def flux_staircase_seq(platf_cfg: str):

    p = oqh.create_program("flux_staircase_seq", platf_cfg)

    k = oqh.create_kernel("main", p)
    for i in range(1):
        k.prepz(i)  # to ensure enough separation in timing
    for i in range(1):
        k.gate('CW_00', [i])
    k.gate('CW_00', [6])
    for cw in range(8):
        k.gate('fl_cw_{:02d}'.format(cw), [2, 0])
        k.gate('fl_cw_{:02d}'.format(cw), [3, 1])
        k.gate("wait", [0, 1, 2, 3], 200)  # because scheduling is wrong.
    p.add_kernel(k)

    p = oqh.compile(p)
    return p


def multi_qubit_off_on(qubits: list,  initialize: bool,
                       second_excited_state: bool, platf_cfg: str):
    """
    Performs an 'off_on' sequence on the qubits specified.
        off: (RO) - prepz -      -     - RO
        on:  (RO) - prepz - x180 -     - RO
        2nd  (RO) - prepz - X180 - X12 - RO  (if second_excited_state == True)

    Will cycle through all combinations of off and on. Last qubit in the list
    is considered the Least Significant Qubit (LSQ).

    Args:
        qubits (list) : list of integers denoting the qubits to use
        initialize (bool): if True does an extra initial measurement to
            allow post selecting data.
        second_excited_state (bool): if True includes the 2-state in the
            combinations.
        platf_cfg (str) : filepath of OpenQL platform config file
    """

    if second_excited_state:
        base = 3
    else:
        base = 2

    combinations = [int2base(i, base=base, fixed_length=len(qubits)) for
                    i in range(base**len(qubits))]

    p = oqh.create_program("multi_qubit_off_on", platf_cfg)

    for i, comb in enumerate(combinations):
        k = oqh.create_kernel('Prep_{}'.format(comb), p)

        # 1. Prepare qubits in 0
        for q in qubits:
            k.prepz(q)

        # 2. post-selection extra init readout
        if initialize:
            for q in qubits:
                k.measure(q)
            k.gate('wait', qubits, 0)

        # 3. prepare desired state
        for state, target_qubit in zip(comb, qubits):  # N.B. last is LSQ
            if state == '0':
                pass
            elif state == '1':
                k.gate('rx180', [target_qubit])
            elif state == '2':
                k.gate('rx180', [target_qubit])
                k.gate('rx12', [target_qubit])
        # 4. measurement of all qubits
        k.gate('wait', qubits, 0)
        # Used to ensure timing is aligned
        for q in qubits:
            k.measure(q)
        k.gate('wait', qubits, 0)
        p.add_kernel(k)

    p = oqh.compile(p)

    return p


def Ramsey_msmt_induced_dephasing(qubits: list, angles: list, platf_cfg: str,
                                  target_qubit_excited: bool=False, wait_time=0,
                                  extra_echo=False):
    """
    Ramsey sequence that varies azimuthal phase instead of time. Works for
    a single qubit or multiple qubits. The coherence of the LSQ is measured,
    while the whole list of qubits is measured.
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    note: executes the measurement between gates to measure the measurement
    induced dephasing

    Input pars:
        qubits:         list specifying the targeted qubit MSQ, and the qubit
                        of which the coherence is measured LSQ.
        angles:         the list of angles for each Ramsey element
        platf_cfg:      filename of the platform config file
    Returns:
        p:              OpenQL Program object containing

    """

    p = oqh.create_program("Ramsey_msmt_induced_dephasing", platf_cfg)

    for i, angle in enumerate(angles[:-4]):
        cw_idx = angle//20 + 9
        k = oqh.create_kernel("Ramsey_azi_"+str(angle), p)
        for qubit in qubits:
            k.prepz(qubit)
        if len(qubits)>1 and target_qubit_excited:
            for qubit in qubits[:-1]:
                k.gate('rx180', [qubit])
        k.gate('rx90', [qubits[-1]])
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        for qubit in qubits:
            k.measure(qubit)
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        if extra_echo:
            k.gate('rx180', [qubits[-1]])
            k.gate("wait", qubits, round(wait_time*1e9))
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        if len(qubits)>1 and target_qubit_excited:
            for qubit in qubits[:-1]:
                k.gate('rx180', [qubit])
        if angle == 90:
            # special because the cw phase pulses go in mult of 20 deg
            k.gate('ry90', [qubits[-1]])
        elif angle == 0:
            k.gate('rx90', [qubits[-1]])
        else:
            k.gate('cw_{:02}'.format(cw_idx), [qubits[-1]])
        p.add_kernel(k)

    # adding the calibration points
    oqh.add_single_qubit_cal_points(p, qubit_idx=qubits[-1], measured_qubits=qubits)

    p = oqh.compile(p)
    return p


def echo_msmt_induced_dephasing(qubits: list, angles: list, platf_cfg: str,
                                wait_time: float=0, target_qubit_excited: bool=False,
                                extra_echo: bool=False):
    """
    Ramsey sequence that varies azimuthal phase instead of time. Works for
    a single qubit or multiple qubits. The coherence of the LSQ is measured,
    while the whole list of qubits is measured.
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    note: executes the measurement between gates to measure the measurement
    induced dephasing

    Input pars:
        qubits:         list specifying the targeted qubit MSQ, and the qubit
                        of which the coherence is measured LSQ.
        angles:         the list of angles for each Ramsey element
        platf_cfg:      filename of the platform config file
        wait_time       wait time to acount for the measurement time in parts
                        of the echo sequence without measurement pulse

    Circuit looks as follows:

    qubits[:-1]    -----------------------(x180)[variable msmt](x180)

    qubits[-1]     - x90-wait-(x180)-wait- x180-wait-(x180)-wait-x90 - [strong mmt]



    Returns:
        p:              OpenQL Program object containing


    """
    p = oqh.create_program('echo_msmt_induced_dephasing', platf_cfg)

    for i, angle in enumerate(angles[:-4]):
        cw_idx = angle//20 + 9
        k = oqh.create_kernel('echo_azi_{}'.format(angle), p)
        for qubit in qubits:
            k.prepz(qubit)
        k.gate('rx90', [qubits[-1]])
        k.gate("wait", qubits, round(wait_time*1e9))
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        if extra_echo:
            k.gate('rx180', [qubits[-1]])
            k.gate("wait", qubits, round(wait_time*1e9))
        k.gate('rx180', [qubits[-1]])
        if len(qubits)>1 and target_qubit_excited:
            for qubit in qubits[:-1]:
                k.gate('rx180', [qubit])
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        for qubit in qubits:
            k.measure(qubit)
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        if extra_echo:
            k.gate('rx180', [qubits[-1]])
            k.gate("wait", qubits, round(wait_time*1e9))
        if len(qubits)>1 and target_qubit_excited:
            for qubit in qubits[:-1]:
                k.gate('rx180', [qubit])
        if angle == 90:
            # special because the cw phase pulses go in mult of 20 deg
            k.gate('ry90', [qubits[-1]])
        elif angle == 0:
            k.gate('rx90', [qubits[-1]])
        else:
            k.gate('cw_{:02}'.format(cw_idx), [qubits[-1]])
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        p.add_kernel(k)

    # adding the calibration points
    p = oqh.add_single_qubit_cal_points(p, qubit_idx=qubits[-1], measured_qubits=qubits)

    p = oqh.compile(p)

    return p


def two_qubit_off_on(q0: int, q1: int, platf_cfg: str):
    '''
    off_on sequence on two qubits.

    # FIXME: input arg should be "qubits" as a list

    Args:
        q0, q1      (int) : target qubits for the sequence
        platf_cfg: str
    '''
    p = oqh.create_program('two_qubit_off_on', platf_cfg)

    p = oqh.add_two_q_cal_points(p,  q0=q0, q1=q1)

    p = oqh.compile(p)
    return p


def two_qubit_tomo_cardinal(q0: int, q1: int, cardinal: int,  platf_cfg: str):
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

    p = oqh.create_program('two_qubit_tomo_cardinal', platf_cfg)

    # Tomography pulses
    i = 0
    for p_q1 in tomo_list_q1:
        for p_q0 in tomo_list_q0:
            i += 1
            kernel_name = '{}_{}_{}'.format(i, p_q0, p_q1)
            k = oqh.create_kernel(kernel_name, p)
            k.prepz(q0)
            k.prepz(q1)
            k.gate(prep_pulse_q0, [q0])
            k.gate(prep_pulse_q1, [q1])
            k.gate(p_q0, [q0])
            k.gate(p_q1, [q1])
            k.measure(q0)
            k.measure(q1)
            p.add_kernel(k)
    # every calibration point is repeated 7 times. This is copied from the
    # script for Tektronix driven qubits. I do not know if this repetition
    # is important or even necessary here.
    p = oqh.add_two_q_cal_points(p, q0=q1, q1=q0, reps_per_cal_pt=7)

    p = oqh.compile(p)
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
        sequence_type  (str) : Describes the timing/order of the pulses.
            options are: sequential | interleaved | simultaneous | sandwiched
                       q0|q0|q1|q1   q0|q1|q0|q1     q01|q01       q1|q0|q0|q1
            describes the order of the AllXY pulses
        replace_q1_pulses_X180 (bool) : if True replaces all pulses on q1 with
            X180 pulses.

        double_points (bool) : if True measures each point in the AllXY twice
    """
    p = oqh.create_program('two_qubit_AllXY', platf_cfg)

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

    pulse_combinations_q0 = pulse_combinations
    pulse_combinations_q1 = pulse_combinations_tiled

    if replace_q1_pulses_X180:
        pulse_combinations_q1 = [['rx180']*2 for val in pulse_combinations]

    i = 0
    for pulse_comb_q0, pulse_comb_q1 in zip(pulse_combinations_q0,
                                            pulse_combinations_q1):
        i += 1
        k = oqh.create_kernel('AllXY_{}'.format(i), p)
        k.prepz(q0)
        k.prepz(q1)
        # N.B. The identity gates are there to ensure proper timing
        if sequence_type == 'interleaved':
            k.gate(pulse_comb_q0[0], [q0])
            k.gate('i', [q1])

            k.gate('i', [q0])
            k.gate(pulse_comb_q1[0], [q1])

            k.gate(pulse_comb_q0[1], [q0])
            k.gate('i', [q1])

            k.gate('i', [q0])
            k.gate(pulse_comb_q1[1], [q1])

        elif sequence_type == 'sandwiched':
            k.gate('i', [q0])
            k.gate(pulse_comb_q1[0], [q1])

            k.gate(pulse_comb_q0[0], [q0])
            k.gate('i', [q1])
            k.gate(pulse_comb_q0[1], [q0])
            k.gate('i', [q1])

            k.gate('i', [q0])
            k.gate(pulse_comb_q1[1], [q1])

        elif sequence_type == 'sequential':
            k.gate(pulse_comb_q0[0], [q0])
            k.gate('i', [q1])
            k.gate(pulse_comb_q0[1], [q0])
            k.gate('i', [q1])
            k.gate('i', [q0])
            k.gate(pulse_comb_q1[0], [q1])
            k.gate('i', [q0])
            k.gate(pulse_comb_q1[1], [q1])

        elif sequence_type == 'simultaneous':
            k.gate(pulse_comb_q0[0], [q0])
            k.gate(pulse_comb_q1[0], [q1])
            k.gate(pulse_comb_q0[1], [q0])
            k.gate(pulse_comb_q1[1], [q1])
        else:
            raise ValueError("sequence_type {} ".format(sequence_type) +
                             "['interleaved', 'simultaneous', " +
                             "'sequential', 'sandwiched']")
        k.measure(q0)
        k.measure(q1)
        p.add_kernel(k)

    p = oqh.compile(p)
    return p


def residual_coupling_sequence(times, q0: int, q1: int, platf_cfg: str):
    """
    Sequence to measure the residual (ZZ) interaction between two qubits.
    Procedure is described in M18TR.

        (q0) --X90--(tau/2)-Y180-(tau/2)-Xm90--RO
        (q1) --X180-(tau/2)-X180-(tau/2)-------RO

    Input pars:
        times:          the list of waiting times in s for each Echo element
        q0              Phase measurement is performed on q0
        q1              Excitation is put in and removed on q1
        platf_cfg:      filename of the platform config file
    Returns:
        p:              OpenQL Program object containing

    """

    p = oqh.create_program("residual_coupling_sequence", platf_cfg)

    for i, time in enumerate(times[:-4]):

        k = oqh.create_kernel("residual_coupling_seq_{}".format(i), p)
        k.prepz(q0)
        k.prepz(q1)
        wait_nanoseconds = int(round(time/1e-9/2))
        k.gate('rx90', [q0])
        k.gate('rx180', [q1])
        k.gate("wait", [q0, q1], wait_nanoseconds)
        k.gate('ry180', [q0])
        k.gate('rx180', [q1])
        k.gate("wait", [q0, q1], wait_nanoseconds)
        k.gate('rxm90', [q0])
        k.measure(q0)
        k.measure(q1)
        k.gate("wait", [q0, q1], 0)
        p.add_kernel(k)

    # adding the calibration points
    p = oqh.add_two_q_cal_points(p, q0=q0, q1=q1)

    p = oqh.compile(p)
    return p


def Cryoscope(qubit_idx: int, buffer_time1=0, buffer_time2=0,
              flux_cw: str='fl_cw_02',
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
    p = oqh.create_program("Cryoscope", platf_cfg)
    buffer_nanoseconds1 = int(round(buffer_time1/1e-9))
    buffer_nanoseconds2 = int(round(buffer_time2/1e-9))

    k = oqh.create_kernel("RamZ_X", p)
    k.prepz(qubit_idx)
    k.gate('rx90', [qubit_idx])
    k.gate("wait", [qubit_idx], buffer_nanoseconds1)
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
    k.gate(flux_cw, [2, 0])
    #k.gate(flux_cw, [10, 8])
    
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
    k.gate("wait", [qubit_idx], buffer_nanoseconds2)
    k.gate('rx90', [qubit_idx])
    k.measure(qubit_idx)
    p.add_kernel(k)

    k = oqh.create_kernel("RamZ_Y", p)
    k.prepz(qubit_idx)
    k.gate('rx90', [qubit_idx])
    k.gate("wait", [qubit_idx], buffer_nanoseconds1)
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
    k.gate(flux_cw, [2, 0])
    #k.gate(flux_cw, [10, 8])
    
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
    k.gate("wait", [qubit_idx], buffer_nanoseconds2)
    k.gate('ry90', [qubit_idx])
    k.measure(qubit_idx)
    p.add_kernel(k)

    p = oqh.compile(p)
    return p


def CryoscopeGoogle(qubit_idx: int, buffer_time1, times, platf_cfg: str):
    """
    A Ramsey sequence with varying waiting times `times` around a flux pulse.
    Generates 2xlen(times) measurements (t1-x, t1-y, t2-x, t2-y. etc)
    """
    p = oqh.create_program("CryoscopeGoogle", platf_cfg)

    buffer_nanoseconds1 = int(round(buffer_time1/1e-9))

    for t in times:

        t_nanoseconds = int(round(t/1e-9))

        k = oqh.create_kernel("RamZ_X", p)
        k.prepz(qubit_idx)
        k.gate('rx90', [qubit_idx])
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        k.gate("wait", [qubit_idx], buffer_nanoseconds1)
        k.gate('fl_cw_02', [2, 0])
        k.gate("wait", [qubit_idx], t_nanoseconds)
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        k.gate('rx90', [qubit_idx])
        k.measure(qubit_idx)
        p.add_kernel(k)

        k = oqh.create_kernel("RamZ_Y", p)
        k.prepz(qubit_idx)
        k.gate('rx90', [qubit_idx])
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        k.gate("wait", [qubit_idx], buffer_nanoseconds1)
        k.gate('fl_cw_02', [2, 0])
        k.gate("wait", [qubit_idx], t_nanoseconds)
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        k.gate('ry90', [qubit_idx])
        k.measure(qubit_idx)
        p.add_kernel(k)

    p = oqh.compile(p)
    return p


def fluxed_ramsey(qubit_idx: int, wait_time: float,
                  flux_cw: str='fl_cw_02',
                  platf_cfg: str=''):
    """
    Single qubit Ramsey sequence.
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        maxtime:        longest plux pulse time
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
    Returns:
        p:              OpenQL Program object containing

    """
    p = oqh.create_program('OpenQL_Platform', platf_cfg)
    wait_time = wait_time/1e-9

    k = oqh.create_kernel("fluxed_ramsey_1", p)
    k.prepz(qubit_idx)
    k.gate('rx90', qubit_idx)
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
    k.gate(flux_cw, 2, 0)
    k.gate("wait", [qubit_idx], wait_time)
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
    k.gate('rx90', qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    k = oqh.create_kernel("fluxed_ramsey_2", p)
    k.prepz(qubit_idx)
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
    k.gate('rx90', qubit_idx)
    k.gate(flux_cw, 2, 0)
    k.gate("wait", [qubit_idx], wait_time)
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
    k.gate('ry90', qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    # adding the calibration points
    # add_single_qubit_cal_points(p, platf=platf, qubit_idx=qubit_idx)

    with suppress_stdout():
        p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_option('output_dir')
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
    p = oqh.create_program("Chevron_hack", platf_cfg)

    buffer_nanoseconds = int(round(buffer_time/1e-9))
    buffer_nanoseconds2 = int(round(buffer_time/1e-9))

    k = oqh.create_kernel("Chevron_hack", p)
    k.prepz(qubit_idx)
    k.gate('rx90', [qubit_idx_spec])
    k.gate('rx180', [qubit_idx])
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
    k.gate("wait", [qubit_idx], buffer_nanoseconds)
    k.gate('fl_cw_02', [2, 0])
    k.gate('wait', [qubit_idx], buffer_nanoseconds2)
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
    k.gate('rx180', [qubit_idx])
    k.measure(qubit_idx)
    k.measure(qubit_idx_spec)
    p.add_kernel(k)

    p = oqh.compile(p)
    return p


def Chevron(qubit_idx: int, qubit_idx_spec: int,
            buffer_time, buffer_time2, flux_cw: int, platf_cfg: str,
            target_qubit_sequence: str='ramsey'):
    """
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        qubit_idx:      int specifying the target qubit (starting at 0)
        qubit_idx_spec: int specifying the spectator qubit
        buffer_time   :
        buffer_time2  :

        platf_cfg:      filename of the platform config file
        target_qubit_sequence: selects whether to run a ramsey sequence on
            a target qubit ('ramsey'), keep it in gorund state ('ground')
            or excite it iat the beginning of the sequnce ('excited')
    Returns:
        p:              OpenQL Program object containing


    Circuit:
        q0    -x180-flux-x180-RO-
        qspec --x90-----------RO- (target_qubit_sequence='ramsey')

        q0    -x180-flux-x180-RO-
        qspec -x180-----------RO- (target_qubit_sequence='excited')

        q0    -x180-flux-x180-RO-
        qspec ----------------RO- (target_qubit_sequence='ground')

    """
    p = oqh.create_program("Chevron", platf_cfg)

    buffer_nanoseconds = int(round(buffer_time/1e-9))
    buffer_nanoseconds2 = int(round(buffer_time2/1e-9))
    if flux_cw is None:
        flux_cw = 2

    k = oqh.create_kernel("Chevron", p)
    k.prepz(qubit_idx)
    k.prepz(qubit_idx_spec)

    if target_qubit_sequence == 'ramsey':
        k.gate('rx90', [qubit_idx_spec])
    elif target_qubit_sequence == 'excited':
        k.gate('rx180', [qubit_idx_spec])
    elif target_qubit_sequence == 'ground':
        k.gate('i', [qubit_idx_spec])
    else:
        raise ValueError("target_qubit_sequence not recognized")
    k.gate('rx180', [qubit_idx])

    if buffer_nanoseconds > 0:
        k.gate("wait", [qubit_idx], buffer_nanoseconds)

    # For CCLight
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
    k.gate('fl_cw_{:02}'.format(flux_cw), [2, 0])
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround


    if buffer_nanoseconds2 > 0:
        k.gate('wait', [qubit_idx], buffer_nanoseconds2)
    k.gate('rx180', [qubit_idx])
    # k.gate("wait", [qubit_idx, qubit_idx_spec], 0)
    k.measure(qubit_idx)
    k.measure(qubit_idx_spec)

    p.add_kernel(k)

    p = oqh.compile(p)
    return p


def two_qubit_ramsey(times, qubit_idx: int, qubit_idx_spec: int,
                     platf_cfg: str, target_qubit_sequence: str='excited'):
    """
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        times:          the list of waiting times for each Ramsey element
        qubit_idx:      int specifying the target qubit (starting at 0)
        qubit_idx_spec: int specifying the spectator qubit

        platf_cfg:      filename of the platform config file
        target_qubit_sequence: selects whether to run a ramsey sequence on
            a target qubit ('ramsey'), keep it in gorund state ('ground')
            or excite it iat the beginning of the sequnce ('excited')
    Returns:
        p:              OpenQL Program object containing


    Circuit:
        q0    --x90-wait-x90-RO-
        qspec --x90----------RO- (target_qubit_sequence='ramsey')

        q0    --x90-wait-x90-RO-
        qspec -x180----------RO- (target_qubit_sequence='excited')

        q0    --x90-wait-x90-RO-
        qspec ---------------RO- (target_qubit_sequence='ground')

    """
    p = oqh.create_program("two_qubit_ramsey", platf_cfg)

    for i, time in enumerate(times):
        k = oqh.create_kernel("two_qubit_ramsey", p)
        k.prepz(qubit_idx)

        if target_qubit_sequence == 'ramsey':
            k.gate('rx90', [qubit_idx_spec])
        elif target_qubit_sequence == 'excited':
            k.gate('rx180', [qubit_idx_spec])
        elif target_qubit_sequence == 'ground':
            k.gate('i', [qubit_idx_spec])
        else:
            raise ValueError('target_qubit_sequence not recognized.')
        k.gate('rx90', [qubit_idx])

        wait_nanoseconds = int(round(time/1e-9))
        k.gate("wait", [qubit_idx, qubit_idx_spec], wait_nanoseconds)

        k.gate('i', [qubit_idx_spec])
        k.gate('rx90', [qubit_idx])

        k.measure(qubit_idx)
        k.measure(qubit_idx_spec)
        k.gate("wait", [qubit_idx, qubit_idx_spec], 0)
        p.add_kernel(k)

    # adding the calibration points
    oqh.add_two_q_cal_points(p, qubit_idx, qubit_idx_spec, reps_per_cal_pt=2)
    p = oqh.compile(p)
    return p


def two_qubit_tomo_bell(bell_state, q0, q1,
                        platf_cfg, wait_after_flux: float=None):
    '''
    Two qubit bell state tomography.

    Args:
        bell_state      (int): index of prepared bell state
        q0, q1          (str): names of the target qubits
        wait_after_flux (float): wait time after the flux pulse and
            after-rotation before tomographic rotations
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

    p = oqh.create_program("two_qubit_tomo_bell_{}_{}".format(q1, q0), platf_cfg)
    for p_q1 in tomo_gates:
        for p_q0 in tomo_gates:
            k = oqh.create_kernel(
                "BellTomo_{}{}_{}{}".format(q1, p_q1, q0, p_q0), p)
            # next experiment
            k.prepz(q0)  # to ensure enough separation in timing
            k.prepz(q1)  # to ensure enough separation in timing
            # pre-rotations
            k.gate(prep_pulse_q0, [q0])
            k.gate(prep_pulse_q1, [q1])
            # FIXME hardcoded edge because of
            # brainless "directed edge recources" in compiler
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6],  0) #alignment workaround
            k.gate('fl_cw_01', [2, 0])
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6],  0) #alignment workaround
            # after-rotations
            k.gate(after_pulse_q1, [q1])
            # possibly wait
            if wait_after_flux is not None:
                k.gate("wait", [q0, q1], round(wait_after_flux*1e9))
            # tomo pulses
            k.gate(p_q0, [q1])
            k.gate(p_q1, [q0])
            # measure
            k.measure(q0)
            k.measure(q1)
            # sync barrier before tomo
            # k.gate("wait", [q0, q1], 0)
            # k.gate("wait", [2, 0], 0)
            p.add_kernel(k)
    # 7 repetitions is because of assumptions in tomo analysis
    p = oqh.add_two_q_cal_points(p, q0=q0, q1=q1, reps_per_cal_pt=7)
    p = oqh.compile(p)
    return p


def two_qubit_tomo_bell_by_waiting(bell_state, q0, q1,
                                   platf_cfg, wait_time: int=20):
    '''
    Two qubit (bell) state tomography. There are no flux pulses applied,
    only waiting time. It is supposed to take advantage of residual ZZ to
    generate entanglement.

    Args:
        bell_state      (int): index of prepared bell state
        q0, q1          (str): names of the target qubits
        wait_time       (int): waiting time in which residual ZZ acts
                                    on qubits
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

    p = oqh.create_program("two_qubit_tomo_bell_by_waiting", platf_cfg)
    for p_q1 in tomo_gates:
        for p_q0 in tomo_gates:
            k = oqh.create_kernel("BellTomo_{}{}_{}{}".format(
                q1, p_q1, q0, p_q0), p)
            # next experiment
            k.prepz(q0)  # to ensure enough separation in timing
            k.prepz(q1)  # to ensure enough separation in timing
            # pre-rotations
            k.gate(prep_pulse_q0, [q0])
            k.gate(prep_pulse_q1, [q1])

            if wait_time > 0:
                k.wait([q0, q1], wait_time)

            k.gate(after_pulse_q1, [q1])
            # tomo pulses
            k.gate(p_q1, [q0])
            k.gate(p_q0, [q1])
            # measure
            k.measure(q0)
            k.measure(q1)
            # sync barrier before tomo
            # k.gate("wait", [q0, q1], 0)
            k.gate("wait", [2, 0], 0)
            p.add_kernel(k)
    # 7 repetitions is because of assumptions in tomo analysis
    p = oqh.add_two_q_cal_points(p, q0=q0, q1=q1, reps_per_cal_pt=7)
    p = oqh.compile(p)
    return p


def two_qubit_DJ(q0, q1, platf_cfg):
    '''
    Two qubit Deutsch-Josza.

    Args:
        q0, q1          (str): names of the target qubits
    '''

    p = oqh.create_program("two_qubit_DJ", platf_cfg)

    # experiments
    # 1
    k = oqh.create_kernel("DJ1", p)
    k.prepz(q0)  # to ensure enough separation in timing
    k.prepz(q1)  # to ensure enough separation in timing
    # prerotations
    k.gate('ry90', [q0])
    k.gate('rym90', [q1])
    # post rotations
    k.gate('ry90', [q0])
    k.gate('ry90', [q1])
    # measure
    k.measure(q0)
    k.measure(q1)
    p.add_kernel(k)

    # 2
    k = oqh.create_kernel("DJ2", p)
    k.prepz(q0)  # to ensure enough separation in timing
    k.prepz(q1)  # to ensure enough separation in timing
    # prerotations
    k.gate('ry90', [q0])
    k.gate('rym90', [q1])
    # rotations
    k.gate('rx180', [q1])
    # post rotations
    k.gate('ry90', [q0])
    k.gate('ry90', [q1])
    # measure
    k.measure(q0)
    k.measure(q1)
    p.add_kernel(k)

    # 3
    k = oqh.create_kernel("DJ3", p)
    k.prepz(q0)  # to ensure enough separation in timing
    k.prepz(q1)  # to ensure enough separation in timing
    # prerotations
    k.gate('ry90', [q0])
    k.gate('rym90', [q1])
    # rotations
    k.gate('ry90', [q1])
    k.gate('rx180', [q0])
    k.gate('rx180', [q1])

    # Hardcoded flux pulse, FIXME use actual CZ
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
    k.gate('wait', [2, 0], 100)
    k.gate('fl_cw_01', [2, 0])
    # FIXME hardcoded extra delays
    k.gate('wait', [2, 0], 200)
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround

    k.gate('rx180', [q0])
    k.gate('ry90', [q1])

    # post rotations
    k.gate('ry90', [q0])
    k.gate('ry90', [q1])
    # measure
    k.measure(q0)
    k.measure(q1)
    p.add_kernel(k)

    # 4
    k = oqh.create_kernel("DJ4", p)
    k.prepz(q0)  # to ensure enough separation in timing
    k.prepz(q1)  # to ensure enough separation in timing
    # prerotations
    k.gate('ry90', [q0])
    k.gate('rym90', [q1])
    # rotations
    k.gate('rym90', [q1])
    # Hardcoded flux pulse, FIXME use actual CZ
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
    k.gate('wait', [2, 0], 100)
    k.gate('fl_cw_01', [2, 0])
    # FIXME hardcoded extra delays
    k.gate('wait', [2, 0], 200)
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround

    k.gate('rx180', [q1])
    k.gate('rym90', [q1])

    # post rotations
    k.gate('ry90', [q0])
    k.gate('ry90', [q1])
    # measure
    k.measure(q0)
    k.measure(q1)
    p.add_kernel(k)

    # 7 repetitions is because of assumptions in tomo analysis
    p = oqh.add_two_q_cal_points(p, q0=q0, q1=q1, reps_per_cal_pt=7)
    p = oqh.compile(p)
    return p


def single_qubit_parity_check(qD: int, qA: int, platf_cfg: str,
                                    number_of_repetitions: int = 10,
                                    initialization_msmt: bool=False,
                                    initial_states=['0', '1'],
                                    flux_codeword: str = 'fl_cw_01',
                                    parity_axis='Z'):
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
    p = oqh.create_program("single_qubit_repeated_parity_check", platf_cfg)

    for initial_state in initial_states:
        k = oqh.create_kernel(
            'repeated_parity_check_{}'.format(initial_state), p)
        k.prepz(qD)
        k.prepz(qA)
        if initialization_msmt:
            k.measure(qA)
            k.measure(qD)
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 500) #wait on all
        if initial_state == '1':
            k.gate('ry180', [qD])
        elif initial_state == '+':
            k.gate('ry90', [qD])
        elif initial_state == '-':
            k.gate('rym90', [qD])
        elif initial_state == 'i':
            k.gate('rx90', [qD])
        elif initial_state == '-i':
            k.gate('rxm90', [qD])
        elif initial_state == '0':
            pass
        else:
            raise ValueError('initial_state= '+initial_state+' not recognized')
        for i in range(number_of_repetitions):
            k.gate('rym90', [qA])
            if parity_axis=='X':
                k.gate('rym90', [qD])
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            k.gate(flux_codeword, [2, 0])
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            k.gate('ry90', [qA])
            if parity_axis=='X':
                k.gate('ry90', [qD])
            k.measure(qA)

        k.measure(qD)
        # hardcoded barrier because of openQL #104
        k.gate('wait', [2, 0], 0)
        k.gate('wait', [qA, qD], 0)
        p.add_kernel(k)

    p = oqh.compile(p)
    return p

def two_qubit_parity_check(qD0: int, qD1: int, qA: int, platf_cfg: str,
                                    echo: bool=False,
                                    number_of_repetitions: int = 10,
                                    initialization_msmt: bool=False,
                                    initial_states=[['0','0'], ['0','1'], ['1','1',], ['1','0']],
                                    flux_codeword0: str = 'fl_cw_03',
                                    flux_codeword1: str = 'fl_cw_01',
                                    parity_axes=['ZZ'], tomo=False,
                                    tomo_after=False,
                                    ro_time=500e-9,
                                    echo_during_ancilla_mmt: bool=False,
                                    idling_time: float=40e-9,
                                    idling_time_echo: float=20e-9,
                                    idling_rounds: int=0):
    """
    Implements a circuit for repeated parity checks on two qubits.

    Circuit looks as follows:
                                                         ^N
    Data0   ----prep.|(my90)0--(y90)(wait) (echo) (wait)| (tomo) -MMMMMMMMMMMMMMMMMMMM
                     |      |                           |
    Ancilla (M)------|-my90-0-0-y90-MMMMMMMMMMMMMMMMMMMM|
                     |        |                         |
    Data1   ----prep.|(my90)--0(y90)(wait) (echo) (wait)| (tomo) -MMMMMMMMMMMMMMMMMMMM


    The initial "M" measurement is optional, the circuit is repated N times
    At the end both qubits are measured.

    Arguments:
        qD0 :       Data qubit, this is the qubit that the repeated parity
                    check will be performed on.
        qD1 :       Data qubit, this is the qubit that the repeated parity
                    check will be performed on.
        exho:       additional pi-pulse between the CZs
        qA :        Ancilla qubit, qubit that the parity will be mapped onto.
        platf_cfg:  filename of the platform config file
        number_of_repetitions: number of times to repeat the circuit
        initialization_msmt : whether to start with an initial measurement
                    to prepare the starting state.
    """
    print('new')
    p = oqh.create_program("two_qubit_parity_check", platf_cfg)
    data_qubits=[qD0,qD1]
    if tomo:
        tomo_gates = ['i', 'rx180', 'ry90', 'rym90', 'rx90', 'rxm90']
    else:
        tomo_gates = ['False']

    for p_q1 in tomo_gates:
        for p_q0 in tomo_gates:
            for initial_state in initial_states:
                k = oqh.create_kernel(
                    'repeated_parity_check_'+initial_state[0]+initial_state[1]+'_tomo0_'+p_q0+'_tomo1_'+p_q1,p)
                k.prepz(qD0)
                k.prepz(qD1)
                k.prepz(qA)
                #initialization
                if initialization_msmt:
                    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
                    # k.measure(qD0)
                    # k.measure(qD1)
                    k.measure(qA)
                    if echo_during_ancilla_mmt:
                        k.gate('wait', [qA, qD0, qD1], int(ro_time*1e9))
                    k.gate('wait', [qD0, qD1, qA], int(100)) #adding additional wait time to ensure good initialization
                    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
                #state preparation
                for i, initial_state_q in enumerate(initial_state):
                    if initial_state_q == '1':
                        k.gate('ry180', [data_qubits[i]])
                    elif initial_state_q == '+':
                        k.gate('ry90', [data_qubits[i]])
                    elif initial_state_q == '-':
                        k.gate('rym90', [data_qubits[i]])
                    elif initial_state_q == 'i':
                        k.gate('rx90', [data_qubits[i]])
                    elif initial_state_q == '-i':
                        k.gate('rxm90', [data_qubits[i]])
                    elif initial_state_q == '0':
                        pass
                    else:
                        raise ValueError('initial_state_q= '+initial_state_q+' not recognized')
                #parity measurement(s)
                for i in range(number_of_repetitions):
                    for parity_axis in parity_axes:
                        #k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
                        if parity_axis=='XX':
                            k.gate('rym90', [qD0])
                            k.gate('rym90', [qD1])
                        if parity_axis=='YY':
                            k.gate('rxm90', [qD0])
                            k.gate('rxm90', [qD1])
                        k.gate('rym90', [qA])
                        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
                        k.gate(flux_codeword0, [2, 0])
                        if echo:
                            k.gate('ry180', [qA])
                        k.gate(flux_codeword1, [2, 0])
                        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
                        k.gate('ry90', [qA])
                        if parity_axis=='XX':
                            k.gate('ry90', [qD0])
                            k.gate('ry90', [qD1])
                        elif parity_axis=='YY':
                            k.gate('rx90', [qD0])
                            k.gate('rx90', [qD1])
                        if (i is not number_of_repetitions-1) or (tomo_after): #last mmt can be multiplexed
                            k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0)
                            k.measure(qA)
                            if echo_during_ancilla_mmt:
                                k.gate('ry180', [qD0])
                                k.gate('ry180', [qD1])
                                k.gate('wait', [qA, qD0, qD1], int(ro_time*1e9))
                k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #separating parity from tomo
                if idling_rounds!=0:
                    for j in np.arange(idling_rounds):
                        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], int(idling_time_echo*1e9)) #alignment workaround
                        if echo_during_ancilla_mmt:
                            k.gate('ry180', [qD0])
                            k.gate('ry180', [qD1])
                        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], int((idling_time-idling_time_echo-20e-9)*1e9)) #alignment workaround
                #tomography
                if tomo:
                    k.gate("wait", [qD1, qD0], 0) #alignment workaround
                    k.gate(p_q0, [qD1])
                    k.gate(p_q1, [qD0])
                    k.gate("wait", [qD1, qD0], 0) #alignment workaround
                # measure
                if not tomo_after:
                    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
                    k.measure(qA)
                k.measure(qD0)
                k.measure(qD1)
                p.add_kernel(k)

    if tomo:
        #only add calbration points when doing tomography
        interleaved_delay=ro_time
        if echo_during_ancilla_mmt:
            interleaved_delay=ro_time
        if tomo_after:
            p = oqh.add_two_q_cal_points(p, q0=qD0, q1=qD1, reps_per_cal_pt=7, measured_qubits=[qD0, qD1],
                                     interleaved_measured_qubits=[qA],
                                     interleaved_delay=interleaved_delay,
                                     nr_of_interleaves=initialization_msmt+number_of_repetitions*len(parity_axes))

        else:
            p = oqh.add_two_q_cal_points(p, q0=qD0, q1=qD1, reps_per_cal_pt=7, measured_qubits=[qD0, qD1, qA],
                         interleaved_measured_qubits=[qA],
                         interleaved_delay=interleaved_delay, nr_of_interleaves=initialization_msmt+number_of_repetitions*len(parity_axes)-1)

    p = oqh.compile(p)
    return p


def conditional_oscillation_seq(q0: int, q1: int, platf_cfg: str,
                                CZ_disabled: bool=False,
                                angles=np.arange(0, 360, 20),
                                wait_time_between: int=0,
                                wait_time_after: int=0,
                                add_cal_points: bool=True,
                                CZ_duration: int=260,
                                nr_of_repeated_gates: int =1,
                                fixed_max_nr_of_repeated_gates: int=None,
                                cases: list=('no_excitation', 'excitation'),
                                flux_codeword: str='fl_cw_01',
                                q2: int=None,
                                flux_codeword2: str='fl_cw_03',
                                q2_excited: bool=False):
    '''
    Sequence used to calibrate flux pulses for CZ gates.

    q0 is the oscilating qubit
    q1 is the spectator qubit

    Timing of the sequence:
    q0:   --   X90  C-Phase  (second C-Phase) Rphi90    RO
    q1: (X180)  --  C-Phase     --            (X180)    RO
    q2: (X180)      C-Phase  (second C-Phase) (X180)

    Args:
        q0, q1      (str): names of the addressed qubits
        RO_target   (str): can be q0, q1, or 'all'
        CZ_disabled (bool): disable CZ gate
        angles      (array): angles of the recovery pulse
        wait_time_between (int) wait time in ns added after each flux pulse
        wait_time_after   (int): wait time in ns after triggering all flux
            pulses
    '''
    p = oqh.create_program("conditional_oscillation_seq", platf_cfg)
    # These angles correspond to special pi/2 pulses in the lutman
    for i, angle in enumerate(angles):
        for case in cases:
            # cw_idx corresponds to special hardcoded angles in the lutman
            cw_idx = angle//20 + 9

            k = oqh.create_kernel("{}_{}".format(case, angle), p)
            k.prepz(q0)
            k.prepz(q1)
            if q2 is not None:
                k.prepz(q2)
                if q2_excited:
                    k.gate('rx180', [q2])
            if case == 'excitation':
                k.gate('rx180', [q1])
            k.gate('rx90', [q0])
            if not CZ_disabled:
                for j in range(nr_of_repeated_gates):
                    if wait_time_between > 0:
                        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
                        k.gate('wait', [2, 0], wait_time_between)
                    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
                    k.gate(flux_codeword, [2, 0])
                    # k.gate(flux_codeword, [10, 8]) # Hack for QCC

                    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
                    if q2 is not None:
                        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
                        k.gate(flux_codeword2, [2, 0])
                        # k.gate(flux_codeword2, [10, 8]) # Hack for QCC

                        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            else:
                for j in range(nr_of_repeated_gates):
                    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
                    k.gate('wait', [2, 0], wait_time_between + CZ_duration)
                    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            if wait_time_after > 0:
                k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
                k.gate('wait', [2, 0], (wait_time_after))
                k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            # hardcoded angles, must be uploaded to AWG
            if angle == 90:
                # special because the cw phase pulses go in mult of 20 deg
                k.gate('ry90', [q0])
            elif angle == 0:
                k.gate('rx90', [q0])
            else:
                k.gate('cw_{:02}'.format(cw_idx), [q0])
            if case == 'excitation':
                k.gate('rx180', [q1])

            k.measure(q0)
            k.measure(q1)
            k.gate('wait', [q1, q0], 0)
            # Implements a barrier to align timings
            # k.gate('wait', [q0, q1], 0)
            # hardcoded barrier because of openQL #104
            # k.gate('wait', [2, 0], 0)

            p.add_kernel(k)
    if add_cal_points:
        p = oqh.add_two_q_cal_points(p, q0=q0, q1=q1,
                                     f_state_cal_pts=True,
                                     f_state_cal_pt_cw=31)
            # hardcoded requires ef pulses to be prepared
    p = oqh.compile(p)

    if add_cal_points:
        cal_pts_idx = [361, 362, 363, 364,
                       365, 366, 367]
    else:
        cal_pts_idx = []

    p.sweep_points = np.concatenate(
        [np.repeat(angles, len(cases)), cal_pts_idx])
    # FIXME: remove try-except, when we depend hardly on >=openql-0.6
    try:
        p.set_sweep_points(p.sweep_points)
    except TypeError:
        # openql-0.5 compatibility
        p.set_sweep_points(p.sweep_points, len(p.sweep_points))
    return p


def grovers_two_qubit_all_inputs(q0: int, q1: int, platf_cfg: str,
                                 precompiled_flux: bool=True,
                                 second_CZ_delay: int=0,
                                 CZ_duration: int=260,
                                 add_echo_pulses: bool=False,
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
        add_echo_pulses (bool): if True add's echo pulses before the
            second CZ gate.
        cal_points (bool):
                Whether to add calibration points.

    Returns:
        qasm_file: a reference to the new QASM file object.
    """

    if not precompiled_flux:
        raise NotImplementedError('Currently only precompiled flux pulses '
                                  'are supported.')

    p = oqh.create_program("grovers_two_qubit_all_inputs", platf_cfg)

    for G0 in ['ry90', 'rym90']:
        for G1 in ['ry90', 'rym90']:
            k = oqh.create_kernel('Gr{}_{}'.format(G0, G1),  p)
            k.prepz(q0)
            k.prepz(q1)
            k.gate(G0, [q0])
            k.gate(G1, [q1])
            k.gate('fl_cw_03', [2, 0])  # flux cw03 is the multi_cz pulse
            k.gate('ry90', [q0])
            k.gate('ry90', [q1])
            # k.gate('fl_cw_00', 2,0)
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            k.gate('wait', [2, 0], second_CZ_delay//2)
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            if add_echo_pulses:
                k.gate('rx180', [q0])
                k.gate('rx180', [q1])
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            k.gate('wait', [2, 0], second_CZ_delay//2)
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            if add_echo_pulses:
                k.gate('rx180', [q0])
                k.gate('rx180', [q1])

            k.gate('wait', [2, 0], CZ_duration)

            k.gate('ry90', [q0])
            k.gate('ry90', [q1])
            k.measure(q0)
            k.measure(q1)
            k.gate('wait', [2, 0], 0)
            p.add_kernel(k)

    if cal_points:
        p = oqh.add_two_q_cal_points(p, q0=q0, q1=q1)
    p = oqh.compile(p)
    return p




def grovers_two_qubits_repeated(qubits, platf_cfg: str,
                                nr_of_grover_iterations: int):
    """
    Writes the QASM sequence for Grover's algorithm on two qubits.
    Sequence:
        q0: G0 -       - mY90 -    - mY90  - RO
                 CZ             CZ
        q1: G1 -       - mY90 -    - mY90  - RO
    G0 and G1 are state preparation gates. Here G0 = 'ry90' and G1 = 'rym90'

    Parameters:
    -----------
    qubits: list of int
        List of the qubits (indices) to which the sequence is applied.
    """
    p = oqh.create_program("grovers_two_qubits_repeated", platf_cfg)
    q0 = qubits[-1]
    q1 = qubits[-2]

    G0 = {"phi": 90, "theta": 90}
    G1 = {"phi": 90, "theta": 90}
    for i in range(nr_of_grover_iterations):
        # k = p.new_kernel('Grover_iteration_{}'.format(i))
        k = oqh.create_kernel('Grover_iteration_{}'.format(i), p)
        k.prepz(q0)
        k.prepz(q1)
        # k.prepz()
        k.gate('ry90', [q0])
        k.gate('ry90', [q1])
        # k.rotate(q0, **G0)
        # k.rotate(q1, **G1)

        for j in range(i):
            # Oracle stage
            k.gate('cz', [2, 0]) #hardcoded fixme
            # k.cz(q0, q1)
            # Tagging stage
            if (j % 2 == 0):
                k.gate('rym90', [q0])
                k.gate('rym90', [q1])
                # k.ry(q0, -90)
                # k.ry(q1, -90)
            else:
                k.gate('ry90', [q0])
                k.gate('ry90', [q1])
                # k.ry(q0, 90)
                # k.ry(q1, 90)
            k.gate('cz', [2, 0]) #hardcoded fixme
            # k.cz(q0, q1)
            if (j % 2 == 0):
                k.gate('ry90', [q0])
                k.gate('ry90', [q1])
            else:
                k.gate('rym90', [q0])
                k.gate('rym90', [q1])
            # if (j % 2 == 0):
            #     k.ry(q0, 90)
            #     k.ry(q1, 90)
            # else:
            #     k.ry(q0, -90)
            #     k.ry(q1, -90)
        k.measure(q0)
        k.measure(q1)
        p.add_kernel(k)
    p = oqh.compile(p)
    # p.compile()
    return p






def grovers_tomography(q0: int, q1: int, omega: int, platf_cfg: str,
                       precompiled_flux: bool=True,
                       cal_points: bool=True, second_CZ_delay: int=260,
                       CZ_duration: int=260,
                       add_echo_pulses: bool=False):
    """
    Tomography sequence for Grover's algorithm.

        omega: int denoting state that the oracle prepares.
    """

    if not precompiled_flux:
        raise NotImplementedError('Currently only precompiled flux pulses '
                                  'are supported.')

    p = oqh.create_program("grovers_tomography",
                           platf_cfg)

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
            k = oqh.create_kernel('Gr{}_{}_tomo_{}_{}'.format(
                G0, G1, p_q0, p_q1), p)
            k.prepz(q0)
            k.prepz(q1)

            # Oracle
            k.gate(G0, [q0])
            k.gate(G1, [q1])
            k.gate('fl_cw_03', [2, 0])  # flux cw03 is the multi_cz pulse
            # Grover's search
            k.gate('ry90', [q0])
            k.gate('ry90', [q1])
            # k.gate('fl_cw_00', 2[,0])
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            k.gate('wait', [2, 0], second_CZ_delay//2)
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            if add_echo_pulses:
                k.gate('rx180', [q0])
                k.gate('rx180', [q1])
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            k.gate('wait', [2, 0], second_CZ_delay//2)
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            if add_echo_pulses:
                k.gate('rx180', [q0])
                k.gate('rx180', [q1])
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            k.gate('wait', [2, 0], CZ_duration)
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            k.gate('ry90', [q0])
            k.gate('ry90', [q1])

            # tomo pulses
            k.gate(p_q1, [q0])
            k.gate(p_q0, [q1])

            k.measure(q0)
            k.measure(q1)
            k.gate('wait', [2, 0], 0)
            p.add_kernel(k)

    p = oqh.add_two_q_cal_points(p, q0=q0, q1=q1, reps_per_cal_pt=7)
    p = oqh.compile(p)
    return p


def CZ_poisoned_purity_seq(q0, q1, platf_cfg: str,
                           nr_of_repeated_gates: int,
                           cal_points: bool=True):
    """
    Creates the |00> + |11> Bell state and does a partial tomography in
    order to determine the purity of both qubits.
    """
    p = oqh.create_program("CZ_poisoned_purity_seq",
                           platf_cfg)
    tomo_list = ['rxm90', 'rym90', 'i']

    for p_pulse in tomo_list:
        k = oqh.create_kernel("{}".format(p_pulse), p)
        k.prepz(q0)
        k.prepz(q1)

        # Create a Bell state:  |00> + |11>
        k.gate('rym90', [q0])
        k.gate('ry90', [q1])
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        for i in range(nr_of_repeated_gates):
            k.gate('fl_cw_01', [2, 0])
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        k.gate('rym90', [q1])

        # Perform pulses to measure the purity of both qubits
        k.gate(p_pulse, [q0])
        k.gate(p_pulse, [q1])

        k.measure(q0)
        k.measure(q1)
        # Implements a barrier to align timings
        # k.gate('wait', [q0, q1], 0)
        # hardcoded because of openQL #104
        k.gate('wait', [2, 0], 0)

        p.add_kernel(k)

    if cal_points:
        # FIXME: replace with standard add cal points function
        k = oqh.create_kernel("Cal 00", p)
        k.prepz(q0)
        k.prepz(q1)
        k.measure(q0)
        k.measure(q1)
        k.gate('wait', [2, 0], 0)
        p.add_kernel(k)
        k = oqh.create_kernel("Cal 11", p)
        k.prepz(q0)
        k.prepz(q1)
        k.gate("rx180", [q0])
        k.gate("rx180", [q1])
        k.measure(q0)
        k.measure(q1)
        k.gate('wait', [2, 0], 0)
        p.add_kernel(k)

    p = oqh.compile(p)
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


def Chevron_first_manifold(qubit_idx: int, qubit_idx_spec: int,
                           buffer_time, buffer_time2, flux_cw: int, platf_cfg: str):
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
    p = oqh.create_program("Chevron_first_manifold", platf_cfg)

    buffer_nanoseconds = int(round(buffer_time/1e-9))
    buffer_nanoseconds2 = int(round(buffer_time2/1e-9))
    if flux_cw is None:
        flux_cw = 2

    k = oqh.create_kernel("Chevron", p)
    k.prepz(qubit_idx)
    k.gate('rx180', [qubit_idx])
    k.gate("wait", [qubit_idx], buffer_nanoseconds)
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
    k.gate('fl_cw_{:02}'.format(flux_cw), [2, 0])
    k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
    k.gate('wait', [qubit_idx], buffer_nanoseconds2)
    k.measure(qubit_idx)
    k.measure(qubit_idx_spec)
    k.gate("wait", [qubit_idx, qubit_idx_spec], 0)
    p.add_kernel(k)

    p = oqh.compile(p)
    return p


def partial_tomography_cardinal(q0: int, q1: int, cardinal: int, platf_cfg: str,
                                precompiled_flux: bool=True,
                                cal_points: bool=True, second_CZ_delay: int=260,
                                CZ_duration: int=260,
                                add_echo_pulses: bool=False):
    """
    Tomography sequence for Grover's algorithm.

        cardinal: int denoting cardinal state prepared.
    """

    if not precompiled_flux:
        raise NotImplementedError('Currently only precompiled flux pulses '
                                  'are supported.')

    p = oqh.create_program("partial_tomography_cardinal",
                           platf_cfg)

    cardinal_gates = ['i', 'rx180', 'ry90', 'rym90', 'rx90', 'rxm90']

    if (cardinal > 35 or cardinal < 0):
        raise ValueError('cardinal must be in [0, 35]')

    idx_p0 = cardinal % 6
    idx_p1 = ((cardinal - idx_p0)//6) % 6
    # cardinal_gates[]
    #k.gate(string_of_the_gate, integer_from_qubit)
    tomo_gates = [('i', 'i'), ('i', 'rx180'), ('rx180', 'i'), ('rx180', 'rx180'),
                  ('ry90', 'ry90'), ('rym90', 'rym90'), ('rx90', 'rx90'), ('rxm90', 'rxm90')]

    for gates in tomo_gates:
        # strings denoting the gates
        SP0 = cardinal_gates[idx_p0]
        SP1 = cardinal_gates[idx_p1]
        t_q0 = gates[1]
        t_q1 = gates[0]
        k = oqh.create_kernel(
            'PT_{}_tomo_{}_{}'.format(cardinal, idx_p0, idx_p1), p)

        k.prepz(q0)
        k.prepz(q1)

        # Cardinal state preparation
        k.gate(SP0, [q0])
        k.gate(SP1, [q1])
        # tomo pulses
        # to be taken from list of tuples
        k.gate(t_q1, [q0])
        k.gate(t_q0, [q1])

        k.measure(q0)
        k.measure(q1)
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        k.gate('wait', [2, 0], 0)
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        p.add_kernel(k)

    p = oqh.add_two_q_cal_points(p, q0=q0, q1=q1, reps_per_cal_pt=2)
    p = oqh.compile(p)
    return p


def two_qubit_VQE(q0: int, q1: int, platf_cfg: str):
    """
    VQE tomography for two qubits.
    Args:
        cardinal        (int) : index of prep gate
        q0, q1          (int) : target qubits for the sequence
    """
    tomo_pulses = ['i', 'rx180', 'ry90', 'rym90', 'rx90', 'rxm90']
    tomo_list_q0 = tomo_pulses
    tomo_list_q1 = tomo_pulses

    p = oqh.create_program("two_qubit_VQE", platf_cfg)

    # Tomography pulses
    i = 0
    for p_q1 in tomo_list_q1:
        for p_q0 in tomo_list_q0:
            i += 1
            kernel_name = '{}_{}_{}'.format(i, p_q0, p_q1)
            k = oqh.create_kernel(kernel_name, p)
            k.prepz(q0)
            k.prepz(q1)
            k.gate('ry180', [q0])  # Y180 gate without compilation
            k.gate('i', [q0])  # Y180 gate without compilation
            k.gate("wait", [q1], 40)
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            k.gate('fl_cw_02', [2, 0])
            k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
            k.gate("wait", [q1], 40)
            k.gate(p_q0, [q0])  # compiled z gate+pre_rotation
            k.gate(p_q1, [q1])  # pre_rotation
            k.measure(q0)
            k.measure(q1)
            p.add_kernel(k)
    # every calibration point is repeated 7 times. This is copied from the
    # script for Tektronix driven qubits. I do not know if this repetition
    # is important or even necessary here.
    p = oqh.add_two_q_cal_points(p, q0=q1, q1=q0, reps_per_cal_pt=7)
    p = oqh.compile(p)
    return p


def sliding_flux_pulses_seq(
        qubits: list, platf_cfg: str,
        angles=np.arange(0, 360, 20), wait_time: int=0,
        flux_codeword_a: str='fl_cw_01', flux_codeword_b: str='fl_cw_01',
        ramsey_axis: str='x',
        add_cal_points: bool=True):
    """
    Experiment to measure effect flux pulses on each other.

    Timing of the sequence:
        q0:   -- flux_a -- wait -- X90 -- flux_b -- Rphi90 -- RO
        q1:   -- flux_a --      --     -- flux_b --        -- RO

    N.B. q1 only exists to satisfy flux tuples notion in CCL
    N.B.2 flux-tuples are now hardcoded to always be tuple [2,0] again
        because of OpenQL.

    Args:
        qubits      : list of qubits, LSQ (q0) is last entry in list
        platf_cfg   : openQL platform config
        angles      : angles along which to do recovery pulses
        wait_time   : time in ns after the first flux pulse and before the
            first microwave pulse.
        flux_codeword_a : flux codeword of the stimulus (1st) pulse
        flux_codeword_b : flux codeword of the spectator (2nd) pulse
        ramsey_axis : chooses between doing x90 or y90 rotation at the
            beginning of Ramsey sequence
        add_cal_points : if True adds calibration points at the end
    """

    p = oqh.create_program("sliding_flux_pulses_seq", platf_cfg)
    k = oqh.create_kernel("sliding_flux_pulses_seq", p)
    q0 = qubits[-1]
    q1 = qubits[-2]

    for i, angle in enumerate(angles):
        cw_idx = angle//20 + 9

        k.prepz(q0)
        k.gate(flux_codeword_a, [2, 0]) # edge hardcoded because of openql
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0)  # alignment workaround
        # hardcoded because of flux_tuples, [q1, q0])
        k.gate('wait', [q0, q1], wait_time)

        if ramsey_axis == 'x':
            k.gate('rx90', [q0])
        elif ramsey_axis == 'y':
            k.gate('ry90', [q0])
        else:
            raise ValueError('ramsey_axis must be "x" or "y"')
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0)  # alignment workaround
        k.gate(flux_codeword_b, [2, 0]) # edge hardcoded because of openql
        k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0)  # alignment workaround
        k.gate('wait', [q0, q1], 60)
        # hardcoded because of flux_tuples, [q1, q0])
        # hardcoded angles, must be uploaded to AWG
        if angle == 90:
            # special because the cw phase pulses go in mult of 20 deg
            k.gate('ry90', [q0])
        else:
            k.gate('cw_{:02}'.format(cw_idx), [q0])
        k.measure(q0)
        k.measure(q1)
        # Implements a barrier to align timings
        # k.gate('wait', [q0, q1], 0)
        # hardcoded barrier because of openQL #104
        k.gate('wait', [2, 0], 0)

    p.add_kernel(k)

    if add_cal_points:
        p = oqh.add_two_q_cal_points(p, q0=q0, q1=q1)
    p = oqh.compile(p)

    if add_cal_points:
        cal_pts_idx = [361, 362, 363, 364]
    else:
        cal_pts_idx = []

    p.sweep_points = np.concatenate([angles, cal_pts_idx])
    # FIXME: remove try-except, when we depend hardly on >=openql-0.6
    try:
        p.set_sweep_points(p.sweep_points)
    except TypeError:
        # openql-0.5 compatibility
        p.set_sweep_points(p.sweep_points, len(p.sweep_points))
    return p
