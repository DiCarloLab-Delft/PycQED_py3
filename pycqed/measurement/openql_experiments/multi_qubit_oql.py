import numpy as np
import openql.openql as ql
import pycqed.measurement.openql_experiments.openql_helpers as oqh
from pycqed.utilities.general import int2base, suppress_stdout
from os.path import join
from pycqed.instrument_drivers.meta_instrument.LutMans.flux_lutman import _def_lm as _def_lm_flux

def single_flux_pulse_seq(qubit_indices: tuple,
                          platf_cfg: str):

    p = oqh.create_program("single_flux_pulse_seq", platf_cfg)

    k = oqh.create_kernel("main", p)
    for idx in qubit_indices:
        k.prepz(idx)  # to ensure enough separation in timing
        k.prepz(idx)  # to ensure enough separation in timing
        k.prepz(idx)  # to ensure enough separation in timing


    k.gate("wait", [], 0)
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
        k = oqh.create_kernel('Prep_{}'.format(i), p)

        # 1. Prepare qubits in 0
        for q in qubits:
            k.prepz(q)

        # 2. post-selection extra init readout
        if initialize:
            for q in qubits:
                k.measure(q)
            k.gate('wait', [], 0)

        # 3. prepare desired state
        for state, target_qubit in zip(comb, qubits):  # N.B. last is LSQ
            if state == '0':
                pass
            elif state == '1':
                k.gate('rx180', [target_qubit])
            elif state == '2':
                k.gate('rx180', [target_qubit])
                k.gate('rx12', [target_qubit])
            # k.gate('wait', [], 0)
        # 4. measurement of all qubits
        k.gate('wait', [], 0)
        # Used to ensure timing is aligned
        for q in qubits:
            k.measure(q)
        k.gate('wait', [], 0)
        p.add_kernel(k)

    p = oqh.compile(p)

    return p


def targeted_off_on(qubits: list,
                    q_target: int,
                    pulse_comb:str,
                    platf_cfg: str):
    """
    Performs an 'off_on' sequence on the qubits specified.
        off: prepz -      - RO
        on:  prepz - x180 - RO

    Will cycle through all combinations of computational states of every
    qubit in <qubits> except the target qubit. The target qubit will be
    initialized according to <pulse_comb>. 'Off' initializes the qubit in
    the ground state and 'On' initializes the qubit in the excited state.

    Args:
        qubits (list) : list of integers denoting the qubits to use
        q_target (str) : targeted qubit.
        pulse_comb (str) : prepared state of target qubit.
        platf_cfg (str) : filepath of OpenQL platform config file
    """

    nr_qubits = len(qubits)
    idx = qubits.index(q_target)

    combinations = ['{:0{}b}'.format(i, nr_qubits-1) for i in range(2**(nr_qubits-1))]
    for i, comb in enumerate(combinations):
        comb = list(comb)#
        if 'on' in pulse_comb.lower():
            comb.insert(idx, '1')
        elif 'off' in pulse_comb.lower():
            comb.insert(idx, '0')
        else:
            raise ValueError()
        combinations[i] = ''.join(comb)

    p = oqh.create_program("Targeted_off_on", platf_cfg)

    for i, comb in enumerate(combinations):
        k = oqh.create_kernel('Prep_{}'.format(comb), p)

        # 1. Prepare qubits in 0
        for q in qubits:
            k.prepz(q)

        # 2. prepare desired state
        for state, target_qubit in zip(comb, qubits):  # N.B. last is LSQ
            if state == '0':
                pass
            elif state == '1':
                k.gate('rx180', [target_qubit])

        # 3. measurement of all qubits
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
        k.gate("wait", [], 0) #alignment workaround
        for qubit in qubits:
            k.measure(qubit)
        k.gate("wait", [], 0) #alignment workaround
        if extra_echo:
            k.gate('rx180', [qubits[-1]])
            k.gate("wait", qubits, round(wait_time*1e9))
        k.gate("wait", [], 0) #alignment workaround
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
        k.gate("wait", [], 0) #alignment workaround
        if extra_echo:
            k.gate('rx180', [qubits[-1]])
            k.gate("wait", qubits, round(wait_time*1e9))
        k.gate('rx180', [qubits[-1]])
        if len(qubits)>1 and target_qubit_excited:
            for qubit in qubits[:-1]:
                k.gate('rx180', [qubit])
        k.gate("wait", [], 0) #alignment workaround
        for qubit in qubits:
            k.measure(qubit)
        k.gate("wait", [], 0) #alignment workaround
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
        k.gate("wait", [], 0) #alignment workaround
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
                    replace_q1_pulses_with: str = None,
                    repetitions: int = 1):
    """
    AllXY sequence on two qubits.
    Has the option of replacing pulses on q1 with pi pulses

    Args:
        q0, q1         (str) : target qubits for the sequence
        sequence_type  (str) : Describes the timing/order of the pulses.
            options are: sequential | interleaved | simultaneous | sandwiched
                       q0|q0|q1|q1   q0|q1|q0|q1     q01|q01       q1|q0|q0|q1
            describes the order of the AllXY pulses
        replace_q1_pulses_with (bool) : if True replaces all pulses on q1 with
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

    pulse_combinations_q0 = np.repeat(pulse_combinations, repetitions, axis=0)

    if replace_q1_pulses_with is not None:
        # pulse_combinations_q1 = [[replace_q1_pulses_with]*2 for val in pulse_combinations]
        pulse_combinations_q1 = np.repeat(
            [[replace_q1_pulses_with]*2], len(pulse_combinations_q0), axis=0)
    else:
        pulse_combinations_q1 = np.tile(pulse_combinations, [repetitions, 1])
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

def multi_qubit_AllXY(Qubits_idx: list, platf_cfg: str, double_points: bool = True):
    """
    Single qubit AllXY sequence.
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
        double_points:  if true repeats every element twice
                        intended for evaluating the noise at larger time scales
    Returns:
        p:              OpenQL Program object containing


    """
    p = oqh.create_program("Multi_qubit_AllXY", platf_cfg)

    allXY = [['i', 'i'], ['rx180', 'rx180'], ['ry180', 'ry180'],
             ['rx180', 'ry180'], ['ry180', 'rx180'],
             ['rx90', 'i'], ['ry90', 'i'], ['rx90', 'ry90'],
             ['ry90', 'rx90'], ['rx90', 'ry180'], ['ry90', 'rx180'],
             ['rx180', 'ry90'], ['ry180', 'rx90'], ['rx90', 'rx180'],
             ['rx180', 'rx90'], ['ry90', 'ry180'], ['ry180', 'ry90'],
             ['rx180', 'i'], ['ry180', 'i'], ['rx90', 'rx90'],
             ['ry90', 'ry90']]

    # this should be implicit
    # FIXME: remove try-except, when we depend hard on >=openql-0.6
    try:
        p.set_sweep_points(np.arange(len(allXY), dtype=float))
    except TypeError:
        # openql-0.5 compatibility
        p.set_sweep_points(np.arange(len(allXY), dtype=float), len(allXY))

    for i, xy in enumerate(allXY):
        if double_points:
            js = 2
        else:
            js = 1
        for j in range(js):
            k = oqh.create_kernel("AllXY_{}_{}".format(i, j), p)
            for qubit in Qubits_idx:
              k.prepz(qubit)
              k.gate(xy[0], [qubit])
              k.gate(xy[1], [qubit])
              k.measure(qubit)
            p.add_kernel(k)

    p = oqh.compile(p)
    return p



def residual_coupling_sequence(times, q0: int, q_spectator_idx: list,
                               spectator_state: str, platf_cfg: str):
    """
    Sequence to measure the residual (ZZ) interaction between two qubits.
    Procedure is described in M18TR.

        (q0) --X90----(tau/2)---Y180-(tau/2)-Xm90--RO
        (qs) --[X180]-(tau/2)-[X180]-(tau/2)-------RO

    Input pars:
        times:           the list of waiting times in s for each Echo element
        q0               Phase measurement is performed on q0
        q_spectator_idx  Excitation is put in and removed on these qubits
                         as indicated
        spectator_state  Indicates on which qubit to put excitations.
        platf_cfg:       filename of the platform config file
    Returns:
        p:              OpenQL Program object containing

    """

    p = oqh.create_program("residual_coupling_sequence", platf_cfg)
    all_qubits = [q0]+q_spectator_idx
    n_qubits = len(all_qubits)

    gate_spec = [s.replace('0', 'i').replace('1', 'rx180') for s in spectator_state]

    for i, time in enumerate(times[:-2]):

        k = oqh.create_kernel("residual_coupling_seq_{}".format(i), p)
        k.prepz(q0)
        for q_s in q_spectator_idx:
            k.prepz(q_s)
        wait_nanoseconds = int(round(time/1e-9/2))
        k.gate('rx90', [q0])
        for i_s, q_s in enumerate(q_spectator_idx):
            k.gate(gate_spec[i_s], [q_s])
        k.gate("wait", all_qubits, wait_nanoseconds)
        k.gate('ry180', [q0])
        for i_s, q_s in enumerate(q_spectator_idx):
            k.gate(gate_spec[i_s], [q_s])
        k.gate("wait", all_qubits, wait_nanoseconds)
        # k.gate('rxm90', [q0])
        k.gate('ry90', [q0])
        k.measure(q0)
        for q_s in q_spectator_idx:
            k.measure(q_s)
        k.gate("wait", all_qubits, 0)
        p.add_kernel(k)

    # adding the calibration points
    p = oqh.add_multi_q_cal_points(p, qubits=all_qubits,
                                   combinations=['0'*n_qubits,'1'*n_qubits])

    p = oqh.compile(p)
    return p


def Cryoscope(
    qubit_idx: int,
    buffer_time1=0,
    buffer_time2=0,
    flux_cw: str = 'fl_cw_02',
    twoq_pair=[2, 0],
    platf_cfg: str = '',
    cc: str = 'CCL',
    double_projections: bool = True,
):
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
    buffer_nanoseconds1 = int(round(buffer_time1 / 1e-9))
    buffer_nanoseconds2 = int(round(buffer_time2 / 1e-9))

    if cc.upper() == 'CCL':
        flux_target = twoq_pair
    elif cc.upper() == 'QCC' or cc.upper() =='CC':
        flux_target = [qubit_idx]
        cw_idx = int(flux_cw[-2:])
        flux_cw = 'sf_{}'.format(_def_lm_flux[cw_idx]['name'].lower())
    else:
        raise ValueError('CC type not understood: {}'.format(cc))

    k = oqh.create_kernel("RamZ_X", p)
    k.prepz(qubit_idx)
    k.gate('rx90', [qubit_idx])
    k.gate("wait", [qubit_idx], buffer_nanoseconds1)
    k.gate("wait", [], 0)  # alignment workaround
    k.gate(flux_cw, flux_target)
    # k.gate(flux_cw, [10, 8])

    k.gate("wait", [], 0)  # alignment workaround
    k.gate("wait", [qubit_idx], buffer_nanoseconds2)
    k.gate('rx90', [qubit_idx])
    k.measure(qubit_idx)
    p.add_kernel(k)

    k = oqh.create_kernel("RamZ_Y", p)
    k.prepz(qubit_idx)
    k.gate('rx90', [qubit_idx])
    k.gate("wait", [qubit_idx], buffer_nanoseconds1)
    k.gate("wait", [], 0)  # alignment workaround
    k.gate(flux_cw, flux_target)
    # k.gate(flux_cw, [10, 8])

    k.gate("wait", [], 0)  # alignment workaround
    k.gate("wait", [qubit_idx], buffer_nanoseconds2)
    k.gate('ry90', [qubit_idx])
    k.measure(qubit_idx)
    p.add_kernel(k)

    if double_projections:
        k = oqh.create_kernel("RamZ_mX", p)
        k.prepz(qubit_idx)
        k.gate('rx90', [qubit_idx])
        k.gate("wait", [qubit_idx], buffer_nanoseconds1)
        k.gate("wait", [], 0)  # alignment workaround
        k.gate(flux_cw, flux_target)
        # k.gate(flux_cw, [10, 8])

        k.gate("wait", [], 0)  # alignment workaround
        k.gate("wait", [qubit_idx], buffer_nanoseconds2)
        k.gate('rxm90', [qubit_idx])
        k.measure(qubit_idx)
        p.add_kernel(k)

        k = oqh.create_kernel("RamZ_mY", p)
        k.prepz(qubit_idx)
        k.gate('rx90', [qubit_idx])
        k.gate("wait", [qubit_idx], buffer_nanoseconds1)
        k.gate("wait", [], 0)  # alignment workaround
        k.gate(flux_cw, flux_target)
        # k.gate(flux_cw, [10, 8])

        k.gate("wait", [], 0)  # alignment workaround
        k.gate("wait", [qubit_idx], buffer_nanoseconds2)
        k.gate('rym90', [qubit_idx])
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

    for i_t,t in enumerate(times):

        t_nanoseconds = int(round(t/1e-9))

        k = oqh.create_kernel("RamZ_X_{}".format(i_t), p)
        k.prepz(qubit_idx)
        k.gate('rx90', [qubit_idx])
        k.gate("wait", [], 0) #alignment workaround
        k.gate("wait", [qubit_idx], buffer_nanoseconds1)
        k.gate('fl_cw_02', [2, 0])
        k.gate("wait", [qubit_idx], t_nanoseconds)
        k.gate("wait", [], 0) #alignment workaround
        k.gate('rx90', [qubit_idx])
        k.measure(qubit_idx)
        p.add_kernel(k)

        k = oqh.create_kernel("RamZ_Y_{}".format(i_t), p)
        k.prepz(qubit_idx)
        k.gate('rx90', [qubit_idx])
        k.gate("wait", [], 0) #alignment workaround
        k.gate("wait", [qubit_idx], buffer_nanoseconds1)
        k.gate('fl_cw_02', [2, 0])
        k.gate("wait", [qubit_idx], t_nanoseconds)
        k.gate("wait", [], 0) #alignment workaround
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
    k.gate("wait", [], 0) #alignment workaround
    k.gate(flux_cw, 2, 0)
    k.gate("wait", [qubit_idx], wait_time)
    k.gate("wait", [], 0) #alignment workaround
    k.gate('rx90', qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    k = oqh.create_kernel("fluxed_ramsey_2", p)
    k.prepz(qubit_idx)
    k.gate("wait", [], 0) #alignment workaround
    k.gate('rx90', qubit_idx)
    k.gate(flux_cw, 2, 0)
    k.gate("wait", [qubit_idx], wait_time)
    k.gate("wait", [], 0) #alignment workaround
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
    k.gate("wait", [], 0) #alignment workaround
    k.gate("wait", [qubit_idx], buffer_nanoseconds)
    k.gate('fl_cw_02', [2, 0])
    k.gate('wait', [qubit_idx], buffer_nanoseconds2)
    k.gate("wait", [], 0) #alignment workaround
    k.gate('rx180', [qubit_idx])
    k.measure(qubit_idx)
    k.measure(qubit_idx_spec)
    p.add_kernel(k)

    p = oqh.compile(p)
    return p


def Chevron(qubit_idx: int, qubit_idx_spec: int, qubit_idx_park: int,
            buffer_time, buffer_time2, flux_cw: int, platf_cfg: str,
            measure_parked_qubit: bool = False,
            target_qubit_sequence: str = 'ramsey', cc: str = 'CCL',
            recover_q_spec: bool = False):
    """
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        qubit_idx:      int specifying the target qubit (starting at 0)
        qubit_idx_spec: int specifying the spectator qubit
        buffer_time   :
        buffer_time2  :
        measure_parked_qubit (bool): Whether we set a measurement on the parked qubit
        platf_cfg:      filename of the platform config file
        target_qubit_sequence: selects whether to run a ramsey sequence on
            a target qubit ('ramsey'), keep it in gorund state ('ground')
            or excite it iat the beginning of the sequnce ('excited')
        recover_q_spec (bool): applies the first gate of qspec at the end
            as well if `True`
    Returns:
        p:              OpenQL Program object containing


    Circuit:
        q0    -x180-flux-x180-RO-
        qspec --x90-----(x90)-RO- (target_qubit_sequence='ramsey')

        q0    -x180-flux-x180-RO-
        qspec -x180----(x180)-RO- (target_qubit_sequence='excited')

        q0    -x180-flux-x180-RO-
        qspec ----------------RO- (target_qubit_sequence='ground')

    """
    p = oqh.create_program("Chevron", platf_cfg)

    buffer_nanoseconds = int(round(buffer_time / 1e-9))
    buffer_nanoseconds2 = int(round(buffer_time2 / 1e-9))
    if flux_cw is None:
        flux_cw = 2
    flux_cw_name = _def_lm_flux[flux_cw]['name'].lower()

    k = oqh.create_kernel("Chevron", p)
    k.prepz(qubit_idx)
    k.prepz(qubit_idx_spec)
    if (qubit_idx_park is not None):
        k.prepz(qubit_idx_park)

    spec_gate_dict = {
        "ramsey": "rx90",
        "excited": "rx180",
        "ground": "i"
    }

    spec_gate = spec_gate_dict[target_qubit_sequence]

    k.gate(spec_gate, [qubit_idx_spec])
    k.gate('rx180', [qubit_idx])

    if buffer_nanoseconds > 0:
        k.gate("wait", [qubit_idx], buffer_nanoseconds)

    # For CCLight
    if cc.upper() == 'CCL':
        k.gate("wait", [], 0)  # alignment workaround
        k.gate('fl_cw_{:02}'.format(flux_cw), [2, 0])
        if qubit_idx_park is not None:
            k.gate('fl_cw_06', [qubit_idx_park])  # square pulse
        k.gate("wait", [], 0)  # alignment workaround
    elif cc.upper() == 'QCC' or cc.upper() == 'CC':
        k.gate("wait", [], 0)  # alignment workaround
        if qubit_idx_park is not None:
            k.gate('sf_square', [qubit_idx_park])
        k.gate('sf_{}'.format(flux_cw_name), [qubit_idx])
        k.gate("wait", [], 0)  # alignment workaround
    else:
        raise ValueError('CC type not understood: {}'.format(cc))

    if buffer_nanoseconds2 > 0:
        k.gate('wait', [qubit_idx], buffer_nanoseconds2)

    k.gate('rx180', [qubit_idx])

    if recover_q_spec:
        k.gate(spec_gate, [qubit_idx_spec])

    k.gate("wait", [], 0)  # alignment workaround
    k.measure(qubit_idx)
    k.measure(qubit_idx_spec)
    if (qubit_idx_park is not None) and measure_parked_qubit:
        k.measure(qubit_idx_park)

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
        k = oqh.create_kernel("two_qubit_ramsey_{}".format(i), p)
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
                        platf_cfg, wait_after_flux: float=None
                        , flux_codeword: str='cz'):
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
            k.gate("wait", [],  0)# Empty list generates barrier for all qubits in platf. only works with 0.8.0
            # k.gate('cz', [q0, q1])
            k.gate(flux_codeword, [q0, q1])
            k.gate("wait", [],  0)
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
    k.gate("wait", [], 0) #alignment workaround
    k.gate('wait', [2, 0], 100)
    k.gate('fl_cw_01', [2, 0])
    # FIXME hardcoded extra delays
    k.gate('wait', [2, 0], 200)
    k.gate("wait", [], 0) #alignment workaround

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
    k.gate("wait", [], 0) #alignment workaround
    k.gate('wait', [2, 0], 100)
    k.gate('fl_cw_01', [2, 0])
    # FIXME hardcoded extra delays
    k.gate('wait', [2, 0], 200)
    k.gate("wait", [], 0) #alignment workaround

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
                              flux_codeword: str = 'cz',
                              parity_axis='Z'):
    """
    Implements a circuit for repeated parity checks.

    Circuit looks as follows:

    Data    (M)|------0------- | ^N- M
 M
               |      |        |
    Ancilla (M)|-my90-0-y90-M- |   - M
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

    for k, initial_state in enumerate(initial_states):
        k = oqh.create_kernel(
            'repeated_parity_check_{}'.format(k), p)
        k.prepz(qD)
        k.prepz(qA)
        if initialization_msmt:
            k.measure(qA)
            k.measure(qD)
            k.gate("wait", []) #wait on all
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
            k.gate("wait", [], 0) #alignment workaround
            k.gate(flux_codeword, [qA, qD])
            k.gate("wait", [], 0) #alignment workaround
            k.gate('ry90', [qA])
            k.gate('wait', [qA, qD], 0)
            if parity_axis=='X':
                k.gate('ry90', [qD])
            k.measure(qA)


        k.measure(qD)
        # hardcoded barrier because of openQL #104
        # k.gate('wait', [2, 0], 0)
        k.gate('wait', [qA, qD], 0)
        p.add_kernel(k)

    p = oqh.compile(p)
    return p

def two_qubit_parity_check(qD0: int, qD1: int, qA: int, platf_cfg: str,
                           echo: bool=False,
                           number_of_repetitions: int = 10,
                           initialization_msmt: bool=False,
                           initial_states=[['0','0'], ['0','1'], ['1','1',], ['1','0']],
                           flux_codeword: str = 'cz',
                           # flux_codeword1: str = 'cz',
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
    Data0   ----prep.|------0-------(wait) (echo) (wait)| (tomo) -MMMMMMMMMMMMMMMMMMMM
                     |      |                           |
    Ancilla (M)------|-my90-0-0-y90-MMMMMMMMMMMMMMMMMMMM|
                     |        |                         |
    Data1   ----prep.|--------0-----(wait) (echo) (wait)| (tomo) -MMMMMMMMMMMMMMMMMMMM


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
        tomo_gates = ['i', 'rx180', 'rx90', 'rxm90', 'ry90', 'rym90']
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
                    k.gate("wait", [], 0) #alignment workaround
                    # k.measure(qD0)
                    # k.measure(qD1)
                    k.measure(qA)
                    if echo_during_ancilla_mmt:
                        k.gate('wait', [qA, qD0, qD1], int(ro_time*1e9))
                    k.gate('wait', [qD0, qD1, qA], int(100)) #adding additional wait time to ensure good initialization
                    k.gate("wait", [], 0) #alignment workaround
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
                        k.gate("wait", [], 0) #alignment workaround
                        if parity_axis=='XX':
                            k.gate('rym90', [qD0])
                            k.gate('rym90', [qD1])
                            k.gate("wait", [], 0) #alignment workaround
                        if parity_axis=='YY':
                            k.gate('rxm90', [qD0])
                            k.gate('rxm90', [qD1])
                            k.gate("wait", [], 0) #alignment workaround
                        k.gate('rym90', [qA])
                        # k.gate('ry90', [qD0])
                        # k.gate('ry90', [qD1])
                        k.gate("wait", [], 0) #alignment workaround
                        # k.gate(flux_codeword, [qA, qD1])
                        k.gate(flux_codeword, [qA, qD0])
                        k.gate("wait", [], 0)
                        # if echo:
                        #     k.gate('ry180', [qA])
                        k.gate(flux_codeword, [qA, qD1])
                        k.gate("wait", [], 0) #alignment workaround
                        k.gate('ry90', [qA])
                        # k.gate('rym90', [qD0])
                        # k.gate('rym90', [qD1])
                        k.gate("wait", [], 0)
                        if parity_axis=='XX':
                            k.gate('ry90', [qD0])
                            k.gate('ry90', [qD1])
                            k.gate("wait", [], 0) #alignment workaround
                        elif parity_axis=='YY':
                            k.gate('rx90', [qD0])
                            k.gate('rx90', [qD1])
                            k.gate("wait", [], 0) #alignment workaround
                        if (i is not number_of_repetitions-1) or (tomo_after): #last mmt can be multiplexed
                            k.gate("wait", [], 0)
                            k.measure(qA)
                            if echo_during_ancilla_mmt:
                                k.gate('ry180', [qD0])
                                k.gate('ry180', [qD1])
                                k.gate('wait', [qA, qD0, qD1], int(ro_time*1e9))
                k.gate("wait", [], 0) #separating parity from tomo
                if idling_rounds!=0:
                    for j in np.arange(idling_rounds):
                        k.gate("wait", [], int(idling_time_echo*1e9)) #alignment workaround
                        if echo_during_ancilla_mmt:
                            k.gate('ry180', [qD0])
                            k.gate('ry180', [qD1])
                        k.gate("wait", [], int((idling_time-idling_time_echo-20e-9)*1e9)) #alignment workaround
                #tomography
                if tomo:
                    k.gate("wait", [qD1, qD0], 0) #alignment workaround
                    k.gate(p_q1, [qD1])
                    k.gate(p_q0, [qD0])
                    k.gate("wait", [qD1, qD0], 0) #alignment workaround
                # measure
                if not tomo_after:
                    k.gate("wait", [], 0) #alignment workaround
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

def multi_qubit_parity_check(Data_qubits: list,
                             qA: int,
                             platf_cfg: str,
                             initialization_msmt: bool = False,
                             flux_codeword: str = 'cz'
                            ):

    print('new')
    p = oqh.create_program("Multi_qubit_parity_check", platf_cfg)

    n = len(Data_qubits)
    initial_states = [ '{:0{}b}'.format(i, n) for i in range(n**2) ]
    for initial_state in initial_states:
        k = oqh.create_kernel(
            'Parity_check_'+initial_state,p)
        for q in Data_qubits:
            k.prepz(q)
        k.prepz(qA)
        if initialization_msmt:
            for q in Data_qubits:
                k.measure(q)
            k.measure(qA)

        k.gate("wait", [], 0)

        #state preparation
        for i, s in enumerate(initial_state):
            if s == '1':
                k.gate('ry180', [Data_qubits[i]])
            elif s == '+':
                k.gate('ry90', [Data_qubits[i]])
            elif s == '-':
                k.gate('rym90', [Data_qubits[i]])
            elif s == 'i':
                ks.gate('rx90', [Data_qubits[i]])
            elif s == '-i':
                k.gate('rxm90', [Data_qubits[i]])
            elif s == '0':
                pass
            else:
                raise ValueError('initial_state_q= '+initial_state_q+' not recognized')
        #parity measurement(s)
        k.gate("wait", [], 0)
        k.gate('rym90', [qA])
        k.gate("wait", [], 0)

        # for q in Data_qubits[:2]:
        #     k.gate(flux_codeword, [q, qA])
        #     k.gate("wait", [], 0)

        # k.gate("wait", [], 0)
        # k.gate("rx180", [qA])
        # k.gate("wait", [], 0)

        # for q in Data_qubits[2:]:
        #     k.gate(flux_codeword, [q, qA])
        #     k.gate("wait", [], 0)

         # CZ gates
        k.gate('cz', [qA, Data_qubits[0]])
        # k.gate('ry180', [qH1])
        # k.gate('ry180', [qH1])
        # k.gate('i'    , [qH1])
        k.gate('wait', [], 0)
        k.gate('cz', [qA, Data_qubits[2]])
        k.gate('rx180', [Data_qubits[0]])
        k.gate('rx180', [Data_qubits[0]])
        k.gate('i', [Data_qubits[0]])
        k.gate('wait', [], 0)
        k.gate('rx180', [qA])
        k.gate('wait', [], 0)
        k.gate('cz', [qA, Data_qubits[1]])
        k.gate('wait', [], 0)
        k.gate('cz', [qA, Data_qubits[3]])
        k.gate('wait', [], 0)

        k.gate('ry90', [qA])
        k.gate("wait", [], 0)

        k.measure(qA)
        for q in Data_qubits:
            k.measure(q)

        p.add_kernel(k)

    p = oqh.compile(p)
    return p


def parity_assessment_seq(qA: int,
                          qA2: int,
                          qD1: int,
                          qD2: int,
                          qD3: int,
                          qD4: int,
                          platf_cfg: str):

    print('new')
    p = oqh.create_program("Parity_assessment_routine", platf_cfg)

    k = oqh.create_kernel('Single_parity_check', p)
    # state initialization
    k.prepz(qA)
    k.prepz(qD1)
    k.prepz(qD2)
    k.prepz(qD3)
    k.prepz(qD4)
    k.gate('wait', [], 0)

    ###############
    # X-Parity
    ###############
    k.gate('rym90', [qA])
    k.gate('rym90', [qD1])
    k.gate('rym90', [qD2])
    k.gate('rym90', [qD3])
    k.gate('rym90', [qD4])
    k.gate('wait', [], 0)
    k.gate('cz', [qA, qD2]) # CZ X-D2
    k.gate('wait', [], 0)
    k.gate('cz', [qA, qD1]) # CZ X-D1
    k.gate('rx180', [qD2])  # echo D2
    k.gate('rx180', [qD2])  # echo D2
    k.gate('i', [qD2])
    k.gate('wait', [], 0)
    k.gate('rx180', [qA])   # echo ancilla
    k.gate('wait', [], 0)
    k.gate('cz', [qA, qD4]) # CZ X-D4
    k.gate('wait', [], 0)
    k.gate('cz', [qA, qD3]) # CZ X-D3
    k.gate('wait', [], 0)
    k.gate('correction', [qA])
    k.gate('correction', [qD1])
    k.gate('correction', [qD2])
    k.gate('correction', [qD3])
    k.gate('correction', [qD4])
    k.gate('wait', [], 0)
    ##############
    # Echo
    ##############
    k.gate('wait', [], 0)
    k.gate('echo_corr', [qD1])
    k.gate("wait", [qD1], 160)
    k.gate('echo_corr', [qD2])
    k.gate("wait", [qD2], 60)
    k.gate('echo_corr', [qD3])
    k.gate("wait", [qD3], 280)
    k.gate('echo_corr', [qD4])
    k.gate("wait", [qD4], 300)
    k.measure(qA)
    k.measure(12)
    k.measure(10)
    k.gate('wait', [], 0)

    k.measure(qD1)  # measure data qubits
    k.measure(qD2)  #
    k.measure(qD3)  #
    k.measure(qD4)  #
    k.gate('wait', [], 0)
    p.add_kernel(k)


    k = oqh.create_kernel('Repeated_parity_check', p)
    # state initialization
    k.prepz(qA)
    k.prepz(qD1)
    k.prepz(qD2)
    k.prepz(qD3)
    k.prepz(qD4)
    k.gate('wait', [], 0)
    # XXXX-Parity check
    for i in range(2):
        ###############
        # X-Parity
        ###############
        k.gate('rym90', [qA])
        k.gate('rym90', [qD1])
        k.gate('rym90', [qD2])
        k.gate('rym90', [qD3])
        k.gate('rym90', [qD4])
        k.gate('wait', [], 0)
        k.gate('cz', [qA, qD2]) # CZ X-D2
        k.gate('wait', [], 0)
        k.gate('cz', [qA, qD1]) # CZ X-D1
        k.gate('rx180', [qD2])  # echo D2
        k.gate('rx180', [qD2])  # echo D2
        k.gate('i', [qD2])
        k.gate('wait', [], 0)
        k.gate('rx180', [qA])   # echo ancilla
        k.gate('wait', [], 0)
        k.gate('cz', [qA, qD4]) # CZ X-D4
        k.gate('wait', [], 0)
        k.gate('cz', [qA, qD3]) # CZ X-D3
        k.gate('wait', [], 0)
        k.gate('correction', [qA])
        k.gate('correction', [qD1])
        k.gate('correction', [qD2])
        k.gate('correction', [qD3])
        k.gate('correction', [qD4])
        k.gate('wait', [], 0)
        ##############
        # Echo
        ##############
        k.gate('wait', [], 0)
        k.gate('echo_corr', [qD1])
        k.gate("wait", [qD1], 160)
        k.gate('echo_corr', [qD2])
        k.gate("wait", [qD2], 60)
        k.gate('echo_corr', [qD3])
        k.gate("wait", [qD3], 280)
        k.gate('echo_corr', [qD4])
        k.gate("wait", [qD4], 300)
        k.measure(qA)
        k.measure(12)
        k.measure(10)
        k.gate('wait', [], 0)

    k.measure(qD1)  # measure data qubits
    k.measure(qD2)  #
    k.measure(qD3)  #
    k.measure(qD4)  #
    k.gate('wait', [], 0)
    p.add_kernel(k)

    k1 = oqh.create_kernel('zero_measurement', p)
    k1.prepz(qA)
    k1.prepz(qD1)
    k1.prepz(qD2)
    k1.prepz(qD3)
    k1.prepz(qD4)
    k1.gate('wait', [], 0)
    k1.measure(qD1)
    k1.measure(qD2)
    k1.measure(qD3)
    k1.measure(qD4)
    k1.measure(qA)
    k1.gate('wait', [], 0)
    p.add_kernel(k1)

    k = oqh.create_kernel('one_measurement', p)
    k.prepz(qA)
    k.prepz(qD1)
    k.prepz(qD2)
    k.prepz(qD3)
    k.prepz(qD4)
    k.gate('wait', [], 0)
    # k.gate('rx180', [qA])
    k.gate('rx180', [qD1])
    k.gate('rx180', [qD2])
    k.gate('rx180', [qD3])
    k.gate('rx180', [qD4])
    k.gate('wait', [], 0)
    k.measure(qD1)
    k.measure(qD2)
    k.measure(qD3)
    k.measure(qD4)
    k.measure(qA)
    k.gate('wait', [], 0)
    p.add_kernel(k)

    k = oqh.create_kernel('idle_measurement', p)
    for i in range(1):
        k.gate('wait', [], 200)
        k.measure(qD1)
        k.measure(qD2)
        k.measure(qD3)
        k.measure(qD4)
        k.measure(qA)
        k.gate('wait', [], 0)
    p.add_kernel(k)

    p = oqh.compile(p)
    return p


def surface_3(qD0: int, qD1: int, qA: int, qA2: int,
              platf_cfg: str,
              echo: bool=False,
              number_of_repetitions: int = 10,
              initialization_msmt: bool=False,
              initial_states=[['0','0'], ['0','1'], ['1','1',], ['1','0']],
              flux_codeword: str = 'cz',
              # flux_codeword1: str = 'cz',
              parity_axes=['ZZ'], tomo=False,
              tomo_after=False,
              ro_time_0=500e-9,
              ro_time_1=500e-9,
              echo_during_ancilla_mmt: bool=False,
              idling_time: float=40e-9,
              idling_time_echo: float=20e-9,
              idling_rounds: int=0,
              cal_points: int=7,
              idle_meas: int=0):

    print('new')
    p = oqh.create_program("Surface_3", platf_cfg)
    data_qubits=[qD0,qD1]

    if tomo:
        tomo_gates = ['i', 'rx180', 'rx90', 'rxm90', 'ry90', 'rym90']
    else:
        tomo_gates = ['False']

    for p_q1 in tomo_gates:
        for p_q0 in tomo_gates:
            for initial_state in initial_states:
                k = oqh.create_kernel(
                    'Surface_3_'+initial_state[0]+initial_state[1]+'_tomo0_'+p_q0+'_tomo1_'+p_q1,p)
                k.prepz(qD0)
                k.prepz(qD1)
                k.prepz(qA)
                # k.prepz(qA2)
                #initialization
                if initialization_msmt:
                    k.gate("wait", [], 0) #alignment workaround
                    # k.measure(qD0)
                    # k.measure(qD1)
                    k.measure(qA)
                    if echo_during_ancilla_mmt:
                        k.gate('wait', [qA, qD0, qD1], int(ro_time*1e9))
                    k.gate('wait', [qD0, qD1, qA], int(100)) #adding additional wait time to ensure good initialization
                    k.gate("wait", [], 0) #alignment workaround
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
                    for j, parity_axis in enumerate(parity_axes):
                        k.gate("wait", [], 0) #alignment workaround
                        if parity_axis=='XX':
                            k.gate('rym90', [qD0])
                            k.gate('rym90', [qD1])
                            # k.gate("wait", [], 0) #alignment workaround
                        if parity_axis=='YY':
                            k.gate('rxm90', [qD0])
                            k.gate('rxm90', [qD1])
                            # k.gate("wait", [], 0) #alignment workaround

                        k.gate('rym90', [qA])

                        k.gate("wait", [], 0) #alignment workaround
                        k.gate(flux_codeword, [qA, qD0])
                        k.gate("wait", [], 0)
                        k.gate('rx180', [qD0])
                        k.gate('rx180', [qD1])
                        k.gate("wait", [], 0)
                        k.gate(flux_codeword, [qA, qD1])
                        k.gate("wait", [], 0) #alignment workaround

                        k.gate('rym90', [qA])

                        if parity_axis=='XX':
                            k.gate('ry90', [qD0])
                            k.gate('ry90', [qD1])
                            k.gate("wait", [], 0) #alignment workaround
                        elif parity_axis=='YY':
                            k.gate('rx90', [qD0])
                            k.gate('rx90', [qD1])
                            k.gate("wait", [], 0) #alignment workaround
                        if (j != len(parity_axes)-1) or (tomo_after): #last mmt can be multiplexed
                            k.gate("wait", [], 0)
                            k.measure(qA)
                            if echo_during_ancilla_mmt:
                                k.gate('rx180', [qD0])
                                k.gate('wait', [qD0], int(140))
                                k.gate('rx180', [qD0])
                                k.gate('wait', [qD0], int(140))
                                k.gate('rx180', [qD0])
                                k.gate('wait', [qD0], int(140))

                                # k.gate('rx180', [qD0])
                                # k.gate('wait', [qD0], int(80))
                                # k.gate('rx180', [qD0])
                                # k.gate('wait', [qD0], int(80))
                                # k.gate('rx180', [qD0])
                                # k.gate('wait', [qD0], int(80))
                                # k.gate('rx180', [qD0])
                                # k.gate('wait', [qD0], int(80))
                                # k.gate('rx180', [qD0])
                                # k.gate('wait', [qD0], int(100))

                                k.gate('rx180', [qD1])
                                k.gate('wait', [qD1], int(ro_time_1*1e9))

                                # k.gate('rx180', [qD1])
                                # k.gate('wait', [qD1], int(220))
                                # k.gate('rx180', [qD1])
                                # k.gate('wait', [qD1], int(220))
                                # k.gate('rx180', [qD1])
                                # k.gate('rx180', [qD0])
                                # k.gate('rx180', [qD1])
                                # k.gate('wait', [qD0], int(ro_time_0*1e9))
                                # k.gate('wait', [qD1], int(ro_time_1*1e9))

                k.gate("wait", [], 0) #separating parity from tomo
                #tomography
                if tomo:
                    k.gate("wait", [qD1, qD0], 0) #alignment workaround
                    k.gate(p_q1, [qD1])
                    k.gate(p_q0, [qD0])
                    k.gate("wait", [qD1, qD0], 0) #alignment workaround
                # measure
                if not tomo_after:
                    k.gate("wait", [], 0) #alignment workaround
                    k.measure(qA)
                k.measure(qD0)
                k.measure(qD1)
                p.add_kernel(k)

    if tomo:
        #only add calbration points when doing tomography
        interleaved_delay=ro_time_0
        if echo_during_ancilla_mmt:
            interleaved_delay=ro_time_0
        if tomo_after:
            p = oqh.add_two_q_cal_points(p, q0=qD0, q1=qD1, reps_per_cal_pt=cal_points, measured_qubits=[qD0, qD1])#,
                                         # interleaved_measured_qubits=[qA],
                                         # interleaved_delay=interleaved_delay,
                                         # nr_of_interleaves=initialization_msmt+number_of_repetitions*len(parity_axes))
            if idle_meas != 0:
                k = oqh.create_kernel('Idle measurements', p)
                for i in range(idle_meas):
                    k.measure(qA)
                    k.gate("wait", [qA], 100)
                p.add_kernel(k)
        else:
            p = oqh.add_two_q_cal_points(p, q0=qD0, q1=qD1, reps_per_cal_pt=cal_points, measured_qubits=[qD0, qD1])

    p = oqh.compile(p)
    return p


def surface_4(qD0: int, qD1: int, qAX: int, qAZ: int, platf_cfg: str,
              echo: bool=False,
              number_of_repetitions: int = 10,
              initialization_msmt: bool=False,
              initial_states=[['+','0']],
              flux_codeword: str = 'cz',
              tomo=False,
              tomo_after=False,
              ro_time_0=500e-9,
              ro_time_1=500e-9,
              echo_during_ancilla_mmt: bool=False,
              echo_during_parity: bool=False,
              idling_time: float=40e-9,
              idling_time_echo: float=20e-9,
              idling_rounds: int=0,
              cal_points: int=7,
              idle_meas: int=0):

    print('new')
    p = oqh.create_program("Surface_4", platf_cfg)
    data_qubits=[qD0,qD1]

    if tomo:
        tomo_gates = ['i', 'rx180', 'rx90', 'rxm90', 'ry90', 'rym90']
    else:
        tomo_gates = ['False']

    for p_q1 in tomo_gates:
        for p_q0 in tomo_gates:
            for initial_state in initial_states:
                k = oqh.create_kernel(
                    'Surface_4_'+initial_state[0]+initial_state[1]+'_tomo0_'+p_q0+'_tomo1_'+p_q1,p)
                k.prepz(qD0)
                k.prepz(qD1)
                k.prepz(qAZ)
                k.prepz(qAX)
                #initialization
                if initialization_msmt:
                    k.gate("wait", [], 0) #alignment workaround
                    # k.measure(qD0)
                    # k.measure(qD1)
                    k.measure(qA)
                    if echo_during_ancilla_mmt:
                        k.gate('wait', [qA, qD0, qD1], int(ro_time*1e9))
                    k.gate('wait', [qD0, qD1, qA], int(100)) #adding additional wait time to ensure good initialization
                    k.gate("wait", [], 0) #alignment workaround
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
                    k.gate("wait", [], 0)

                    # X-parity operations
                    k.gate('rym90', [qD0])
                    k.gate('rym90', [qD1])
                    k.gate('rym90', [qAZ])
                    k.gate("wait", [], 0)
                    k.gate(flux_codeword, [qAZ, qD0])
                    k.gate("wait", [], 0)
                    k.gate('rx180', [qD0])
                    k.gate('rx180', [qD1])
                    k.gate("wait", [], 0)
                    k.gate(flux_codeword, [qAZ, qD1])
                    k.gate("wait", [], 0)
                    k.gate('rym90', [qAZ])
                    k.gate('ry90', [qD0])
                    k.gate('ry90', [qD1])
                    k.gate("wait", [], 0)

                    # Z-parity operations
                    k.gate('rym90', [qAX])
                    k.gate("wait", [], 0)
                    k.gate(flux_codeword, [qAX, qD0])
                    k.gate("wait", [], 0)
                    k.gate('rx180', [qD0])
                    k.gate('rx180', [qD1])
                    k.gate("wait", [], 0)
                    k.gate(flux_codeword, [qAX, qD1])
                    k.gate("wait", [], 0)
                    k.gate('rym90', [qAX])
                    k.gate("wait", [], 0)

                    if (i is not number_of_repetitions-1) or (tomo_after): #last mmt can be multiplexed
                        k.gate("wait", [], 0)
                        k.measure(qAZ)
                        k.measure(qAX)
                        if echo_during_ancilla_mmt:

                            k.gate('rx180', [qD0])
                            k.gate('wait', [qD0], int(140))
                            k.gate('rx180', [qD0])
                            k.gate('wait', [qD0], int(140))
                            k.gate('rx180', [qD0])
                            k.gate('wait', [qD0], int(140))

                            # k.gate('rx180', [qD1])
                            # k.gate('wait', [qD1], int(140))
                            # k.gate('rx180', [qD1])
                            # k.gate('wait', [qD1], int(140))
                            # k.gate('rx180', [qD1])
                            # k.gate('wait', [qD1], int(140))

                            # k.gate('rx180', [qD0])
                            # k.gate('wait', [qD0], int(ro_time_0*1e9))
                            k.gate('rx180', [qD1])
                            k.gate('wait', [qD1], int(ro_time_1*1e9))

                            # #CPMG D0
                            # n = 6
                            # Twait = 600/n - 20
                            # k.gate('ry90', [qD0])
                            # k.gate('wait', [qD0], int(Twait/2))
                            # k.gate('rx180', [qD0])
                            # for i in range(n-1):
                            #     k.gate('wait', [qD0], int(Twait))
                            #     k.gate('rx180', [qD0])
                            # k.gate('wait', [qD0], int(Twait/2))
                            # k.gate('rym90', [qD0])

                            # # CPMG D1
                            # n = 1
                            # Twait = 600/n - 20
                            # k.gate('ry90', [qD1])
                            # k.gate('wait', [qD1], int(Twait/2))
                            # k.gate('rx180', [qD1])
                            # for i in range(n-1):
                            #     k.gate('wait', [qD1], int(Twait))
                            #     k.gate('rx180', [qD1])
                            # k.gate('wait', [qD1], int(Twait/2))
                            # k.gate('rym90', [qD1])


                k.gate("wait", [], 0) #separating parity from tomo
                #tomography
                if tomo:
                    k.gate("wait", [qD1, qD0], 0) #alignment workaround
                    k.gate(p_q1, [qD1])
                    k.gate(p_q0, [qD0])
                    k.gate("wait", [qD1, qD0], 0) #alignment workaround
                # measure
                if not tomo_after:
                    k.gate("wait", [], 0) #alignment workaround
                    k.measure(qAZ)
                    k.measure(qAX)
                k.measure(qD0)
                k.measure(qD1)
                p.add_kernel(k)

    if tomo:
        #only add calbration points when doing tomography
        interleaved_delay=ro_time_0
        if echo_during_ancilla_mmt:
            interleaved_delay=ro_time_0
        if tomo_after:
            p = oqh.add_two_q_cal_points(p, q0=qD0, q1=qD1, reps_per_cal_pt=cal_points, measured_qubits=[qD0, qD1])#,
                                         # interleaved_measured_qubits=[qA],
                                         # interleaved_delay=interleaved_delay,
                                         # nr_of_interleaves=initialization_msmt+number_of_repetitions*len(parity_axes))
            if idle_meas != 0:
                k = oqh.create_kernel('Idle measurements', p)
                for i in range(idle_meas):
                    k.measure(qA)
                    k.gate("wait", [qA], 100)
                p.add_kernel(k)
        else:
            p = oqh.add_two_q_cal_points(p, q0=qD0, q1=qD1, reps_per_cal_pt=cal_points, measured_qubits=[qD0, qD1])#,

    p = oqh.compile(p)
    return p


def surface_4_pipelined(qD0: int, qD1: int,
                        qAX: int, qAZ: int,
                        qAX_d: int, qAZ_d: int,
                        ro_time: float,
                        platf_cfg: str,
                        number_of_repetitions: int = 1,
                        initial_states=[['+','0']],
                        flux_codeword: str = 'cz',
                        tomo=False,
                        echo_time_1:float=200,
                        echo_time_2:float=200,
                        cal_points: int=7,
                        idle_meas: int=0):

    print('new')
    p = oqh.create_program("Surface_4_pip", platf_cfg)
    data_qubits=[qD0,qD1]

    if tomo:
        tomo_gates = ['i', 'rx180', 'rx90', 'rxm90', 'ry90', 'rym90']
    else:
        tomo_gates = ['False']

    for p_q1 in tomo_gates:
        for p_q0 in tomo_gates:
            for initial_state in initial_states:
                k = oqh.create_kernel(
                    'Surface_4_'+initial_state[0]+initial_state[1]+'_tomo0_'+p_q0+'_tomo1_'+p_q1,p)
                k.prepz(qD0)
                k.prepz(qD1)
                k.prepz(qAZ)
                k.prepz(qAX)

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

                    k.gate("wait", [], 0)

                    # Trigger measurements on dummy qubits
                    k.measure(qAZ_d)
                    k.measure(qAX_d)

                    # X-parity operations
                    k.gate('rym90', [qD0])
                    k.gate('rym90', [qD1])
                    k.gate('rym90', [qAZ])
                    k.gate("wait", [qD0, qD1, qAX, qAZ], 0)
                    k.gate(flux_codeword, [qAZ, qD0])
                    k.gate("wait", [qD0, qD1, qAX, qAZ], 0)
                    k.gate("rx180", [qD0])
                    k.gate("rx180", [qD1])
                    k.gate("wait", [qD0, qD1, qAX, qAZ], 0)
                    k.gate(flux_codeword, [qAZ, qD1])
                    k.gate("wait", [qD0, qD1, qAX, qAZ], 0)
                    k.gate('rym90', [qAZ])
                    k.gate('ry90', [qD0])
                    k.gate('ry90', [qD1])
                    k.gate("wait", [qD0, qD1, qAX, qAZ], 0)

                    # # wait time between parity operations
                    # k.gate("wait", [qD0, qD1, qAX, qAZ], ro_time-220)
                    # Echo sequence 2
                    k.gate("wait", [qD0], echo_time_1)
                    k.gate("rx180", [qD0])
                    k.gate("wait", [qD0], 400-echo_time_1)
                    k.gate("wait", [qD1], echo_time_2)
                    k.gate("rx180", [qD1])
                    k.gate("wait", [qD1], 400-echo_time_2)

                    # Z-parity operations
                    k.gate("wait", [qD0, qD1, qAX, qAZ], 0)
                    k.gate('rym90', [qAX])
                    # k.gate('rym90', [qD0])
                    # k.gate('rym90', [qD1])
                    k.gate("wait", [qD0, qD1, qAX, qAZ], 0)
                    k.gate(flux_codeword, [qAX, qD0])
                    k.gate("wait", [qD0, qD1, qAX, qAZ], 0)
                    k.gate("rx180", [qD0])
                    k.gate("rx180", [qD1])
                    k.gate("wait", [qD0, qD1, qAX, qAZ], 0)
                    k.gate(flux_codeword, [qAX, qD1])
                    k.gate("wait", [qD0, qD1, qAX, qAZ], 0)
                    k.gate('rym90', [qAX])
                    # k.gate('ry90', [qD0])
                    # k.gate('ry90', [qD1])
                    k.gate("wait", [qD0, qD1, qAX, qAZ], 0)

                k.gate("wait", [], 0)

                #tomography
                if tomo:
                    k.gate("wait", [qD1, qD0], 0)
                    k.gate(p_q1, [qD1])
                    k.gate(p_q0, [qD0])
                    k.gate("wait", [qD1, qD0], 0)
                    k.measure(qD0)
                    k.measure(qD1)
                    k.measure(qAX_d) # last missing measurement for final Z-parity
                p.add_kernel(k)

    if tomo:
        p = oqh.add_two_q_cal_points(p, q0=qD0, q1=qD1, reps_per_cal_pt=cal_points, measured_qubits=[qD0, qD1])

    p = oqh.compile(p)
    return p

def Four_Qubit_tomo(qD1: int, qD2: int,
                    qD3: int, qD4: int,
                    qA1: int, qA2: int, qA3: int,
                    platf_cfg: str,
                    initial_states=[['0', '0', '0', '0']],
                    pipelined = False,
                    theta_gate=False,
                    idle_meas=0):

    print('new')
    p = oqh.create_program("Four_Qubit_tomo", platf_cfg)

    tomo_gates = ['i', 'rym90', 'rx90']

    data_qubits=[qD1, qD2, qD3, qD4]

    for p_q4 in tomo_gates:
        for p_q3 in tomo_gates:
            for p_q2 in tomo_gates:
                for p_q1 in tomo_gates:
                    for initial_state in initial_states:
                        k = oqh.create_kernel(
                            'Tomo__tomo1_'+p_q1+'_tomo2_'+p_q2+'_tomo3_'+p_q3+'_tomo4_'+p_q4, p)
                        k.prepz(qD1)
                        k.prepz(qD2)
                        k.prepz(qD3)
                        k.prepz(qD4)
                        k.prepz(qA1)
                        # k.prepz(qA2)
                        k.prepz(qA3)
                        k.gate("wait", [], 0)

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

                        k.gate("wait", [], 0)
                        if theta_gate:
                            k.gate('theta_gate', [qD1])
                            k.gate('theta_gate', [qD3])
                            k.gate("wait", [], 0)

                        if not pipelined:
                            ########################################################
                            # Parallel
                            ########################################################
                            ###############
                            # X-Parity
                            ###############
                            k.gate('rym90', [qA2])
                            k.gate('rym90', [qD1])
                            k.gate('rym90', [qD2])
                            k.gate('rym90', [qD3])
                            k.gate('rym90', [qD4])
                            k.gate('wait', [], 0)
                            k.gate('cz', [qA2, qD2]) # CZ X-D2
                            k.gate('wait', [], 0)
                            k.gate('cz', [qA2, qD1]) # CZ X-D1
                            k.gate('rx180', [qD2])  # echo D2
                            k.gate('rx180', [qD2])  # echo D2
                            k.gate('i', [qD2])
                            k.gate('wait', [], 0)
                            k.gate('rx180', [qA2])   # echo ancilla
                            k.gate('wait', [], 0)
                            k.gate('cz', [qA2, qD4]) # CZ X-D4
                            k.gate('wait', [], 0)
                            k.gate('cz', [qA2, qD3]) # CZ X-D3
                            k.gate('wait', [], 0)
                            k.gate('correction', [qA2])
                            k.gate('correction', [qD1])
                            k.gate('correction', [qD2])
                            k.gate('correction', [qD3])
                            k.gate('correction', [qD4])
                            k.gate('wait', [], 0)

                            ###############
                            # Z-Parity
                            ###############
                            k.gate('rym90', [qA1])
                            k.gate('rym90', [qA3])
                            k.gate("wait", [], 0)
                            k.gate('cz', [qA3, qD2]) # without parking
                            k.gate('cz', [qD1, qA1])
                            k.gate("wait", [], 0)
                            k.gate('cz', [qD4, qA3])
                            k.gate('cz', [qD3, qA1])
                            k.gate("wait", [], 0)
                            k.gate('ry90', [qA1])
                            k.gate('ry90', [qA3])
                            k.gate("wait", [], 0)

                            ##############
                            # Echo
                            ##############
                            k.gate('wait', [], 0)
                            k.gate('echo_corr', [qD1])
                            k.gate("wait", [qD1], 160)
                            k.gate('echo_corr', [qD2])
                            k.gate("wait", [qD2], 60)
                            k.gate('echo_corr', [qD3])
                            k.gate("wait", [qD3], 280)
                            k.gate('echo_corr', [qD4])
                            k.gate("wait", [qD4], 300)
                            k.measure(qA1)
                            k.measure(qA2)
                            k.measure(qA3)
                            # k.gate('rx180', [qD1])
                            # k.gate("wait", [qD1], 360)
                            # k.gate('rx180', [qD2])
                            # k.gate("wait", [qD2], 360)
                            # k.gate('rx180', [qD3])
                            # k.gate("wait", [qD3], 520)
                            # k.gate('rx180', [qD4])
                            # k.gate("wait", [qD4], 520)
                            k.gate('wait', [], 0)

                            #tomography
                            k.gate("wait", [qD1, qD2, qD3, qD4], 0)
                            k.gate(p_q1, [qD1])
                            k.gate(p_q2, [qD2])
                            k.gate(p_q3, [qD3])
                            k.gate(p_q4, [qD4])
                            k.gate("wait", [qD1, qD2, qD3, qD4], 0)

                            k.gate("wait", [], 0)
                            k.measure(qD1)
                            k.measure(qD2)
                            k.measure(qD3)
                            k.measure(qD4)
                            k.gate("wait", [], 0)

                        else:
                            # for i in range(2):
                            ########################################################
                            # Pipelined
                            ########################################################
                            ###############
                            # X-Parity
                            ###############
                            k.measure(4) # Qubit Z1_trig
                            k.measure(6) # Qubit Z2_trig
                            k.measure(1) # Qubit X_trig
                            k.gate('wait', [1, 4, 6], 280)
                            k.gate('rym90', [qA2])
                            k.gate('rym90', [qD1])
                            k.gate('rym90', [qD2])
                            k.gate('rym90', [qD3])
                            k.gate('rym90', [qD4])
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA2, qD2]) # CZ X-D2
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA2, qD1]) # CZ X-D1
                            k.gate('rx180', [qD2])  # echo D2
                            k.gate('rx180', [qD2])  # echo D2
                            k.gate('i', [qD2])
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('rx180', [qA2])   # echo ancilla
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA2, qD4]) # CZ X-D4
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA2, qD3]) # CZ X-D3
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            # k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            # k.gate('cz', [qA2, qD3]) # CZ X-D3
                            # k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            # k.gate('cz', [qA2, qD1]) # CZ X-D1
                            # k.gate('rx180', [qD2])  # echo D2
                            # k.gate('rx180', [qD2])  # echo D2
                            # k.gate('i', [qD2])
                            # k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            # k.gate('rx180', [qA2])   # echo ancilla
                            # k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            # k.gate('cz', [qA2, qD4]) # CZ X-D4
                            # k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            # k.gate('cz', [qA2, qD2]) # CZ X-D2
                            # k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('correction', [qA2])
                            k.gate('correction', [qD1])
                            k.gate('correction', [qD2])
                            k.gate('correction', [qD3])
                            k.gate('correction', [qD4])
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)

                            ##############
                            # Echo
                            ##############
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                            k.gate("wait", [qD1], 160)
                            k.gate('echo_corr', [qD1])
                            k.gate("wait", [qD1], 360-180)
                            k.gate("wait", [qD2], 160)
                            k.gate('echo_corr', [qD2])
                            k.gate("wait", [qD2], 360-180)
                            k.gate("wait", [qD3], 160)
                            k.gate('echo_corr', [qD3])
                            k.gate("wait", [qD3], 360-180)
                            k.gate("wait", [qD4], 160)
                            k.gate('echo_corr', [qD4])
                            k.gate("wait", [qD4], 360-180)
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                            k.gate('i', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])


                            ###############
                            # Z-Parity
                            ###############
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                            k.gate('rym90', [qA1])
                            k.gate('rym90', [qA3])
                            k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA3, qD2]) # without parking
                            k.gate('cz', [qD1, qA1])
                            k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qD4, qA3])
                            k.gate('cz', [qD3, qA1])
                            k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('ry90', [qA1])
                            k.gate('ry90', [qA3])
                            k.gate("wait", [], 0)

                            k.gate('rx180', [qD1]) # XL gate
                            k.gate('rx180', [qD3]) #
                            # k.gate('rx180', [qD1]) # ZL gate
                            # k.gate('rx180', [qD2]) #
                            # k.gate('ry180', [qD3]) #
                            # k.gate('ry180', [qD4]) #
                            k.gate("wait", [], 0)  #

                            ###############
                            # X-Parity
                            ###############
                            k.measure(4) # Qubit Z1_trig
                            k.measure(6) # Qubit Z2_trig
                            k.measure(1) # Qubit X_trig
                            k.gate('wait', [1, 4, 6], 280)
                            k.gate('rym90', [qA2])
                            k.gate('rym90', [qD1])
                            k.gate('rym90', [qD2])
                            k.gate('rym90', [qD3])
                            k.gate('rym90', [qD4])
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA2, qD2]) # CZ X-D2
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA2, qD1]) # CZ X-D1
                            k.gate('rx180', [qD2])  # echo D2
                            k.gate('rx180', [qD2])  # echo D2
                            k.gate('i', [qD2])
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('rx180', [qA2])   # echo ancilla
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA2, qD4]) # CZ X-D4
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA2, qD3]) # CZ X-D3
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            # k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            # k.gate('cz', [qA2, qD3]) # CZ X-D3
                            # k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            # k.gate('cz', [qA2, qD1]) # CZ X-D1
                            # k.gate('rx180', [qD2])  # echo D2
                            # k.gate('rx180', [qD2])  # echo D2
                            # k.gate('i', [qD2])
                            # k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            # k.gate('rx180', [qA2])   # echo ancilla
                            # k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            # k.gate('cz', [qA2, qD4]) # CZ X-D4
                            # k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            # k.gate('cz', [qA2, qD2]) # CZ X-D2
                            # k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('correction', [qA2])
                            k.gate('correction', [qD1])
                            k.gate('correction', [qD2])
                            k.gate('correction', [qD3])
                            k.gate('correction', [qD4])
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)

                            ##############
                            # Echo
                            ##############
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                            k.gate("wait", [qD1], 160)
                            k.gate('echo_corr', [qD1])
                            k.gate("wait", [qD1], 360-180)
                            k.gate("wait", [qD2], 160)
                            k.gate('echo_corr', [qD2])
                            k.gate("wait", [qD2], 360-180)
                            k.gate("wait", [qD3], 160)
                            k.gate('echo_corr', [qD3])
                            k.gate("wait", [qD3], 360-180)
                            k.gate("wait", [qD4], 160)
                            k.gate('echo_corr', [qD4])
                            k.gate("wait", [qD4], 360-180)
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                            k.gate('i', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])


                            ###############
                            # Z-Parity
                            ###############
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                            k.gate('rym90', [qA1])
                            k.gate('rym90', [qA3])
                            k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA3, qD2]) # without parking
                            k.gate('cz', [qD1, qA1])
                            k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qD4, qA3])
                            k.gate('cz', [qD3, qA1])
                            k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('ry90', [qA1])
                            k.gate('ry90', [qA3])
                            k.gate("wait", [], 0)

                            #tomography
                            k.gate("wait", [qD1, qD2, qD3, qD4], 0)
                            k.gate(p_q1, [qD1])
                            k.gate(p_q2, [qD2])
                            k.gate(p_q3, [qD3])
                            k.gate(p_q4, [qD4])
                            k.gate("wait", [qD1, qD2, qD3, qD4], 0)

                            k.gate("wait", [], 0)
                            k.measure(qD1)
                            k.measure(qD2)
                            k.measure(qD3)
                            k.measure(qD4)
                            k.measure(qA1)
                            k.measure(qA3)
                            k.gate("wait", [], 0)
                            #######################################################

                        p.add_kernel(k)

    p = oqh.add_multi_q_cal_points(p, qubits=[qD1, qD2, qD3, qD4], combinations=[ s1+s2+s3+s4 for s1 in ['0', '1']
                                                                                              for s2 in ['0', '1']
                                                                                              for s3 in ['0', '1']
                                                                                              for s4 in ['0', '1']],
                                   initialize=False)

    if idle_meas != 0 :
        k = oqh.create_kernel('idle_meas', p)
        k.prepz(qD1)
        k.prepz(qD2)
        k.prepz(qD3)
        k.prepz(qD4)
        for i in range(idle_meas):
            k.measure(qD1)
            k.measure(qD2)
            k.measure(qD3)
            k.measure(qD4)
        p.add_kernel(k)

    p = oqh.compile(p)
    return p

def S_gate_seq_tomo(qD1: int, qD2: int,
                    qD3: int, qD4: int,
                    qA1: int, qA2: int, qA3: int,
                    platf_cfg: str,
                    initial_states=[['0', '0', '0', '0']],
                    double_round=False,
                    idle_meas=0):

    print('new')
    p = oqh.create_program("S_gate_seq", platf_cfg)

    tomo_gates = ['i', 'rym90', 'rx90']

    data_qubits=[qD1, qD2, qD3, qD4]

    for p_q4 in tomo_gates:
        for p_q3 in tomo_gates:
            for p_q2 in tomo_gates:
                for p_q1 in tomo_gates:
                    for initial_state in initial_states:
                        k = oqh.create_kernel(
                            'Tomo__tomo1_'+p_q1+'_tomo2_'+p_q2+'_tomo3_'+p_q3+'_tomo4_'+p_q4, p)
                        k.prepz(qD1)
                        k.prepz(qD2)
                        k.prepz(qD3)
                        k.prepz(qD4)
                        # k.prepz(qA1)
                        # k.prepz(qA2)
                        # k.prepz(qA3)
                        k.gate("wait", [], 0)

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

                        k.gate("wait", [], 0)

                        ###############
                        # X-Parity
                        ###############
                        k.measure(4) # Qubit Z1_trig
                        k.measure(6) # Qubit Z2_trig
                        k.measure(1) # Qubit X_trig
                        k.gate('wait', [1, 4, 6], 280)
                        k.gate('rym90', [qA2])
                        k.gate('rym90', [qD1])
                        k.gate('rym90', [qD2])
                        k.gate('rym90', [qD3])
                        k.gate('rym90', [qD4])
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('cz', [qA2, qD2]) # CZ X-D2
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('cz', [qA2, qD1]) # CZ X-D1
                        k.gate('rx180', [qD2])  # echo D2
                        k.gate('rx180', [qD2])  # echo D2
                        k.gate('i', [qD2])
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('rx180', [qA2])   # echo ancilla
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('cz', [qA2, qD4]) # CZ X-D4
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('cz', [qA2, qD3]) # CZ X-D3
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('correction', [qA2])
                        k.gate('correction', [qD1])
                        k.gate('correction', [qD2])
                        k.gate('correction', [qD3])
                        k.gate('correction', [qD4])
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)

                        ##############
                        # Echo
                        ##############
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                        k.gate("wait", [qD1], 160)
                        k.gate('echo_corr', [qD1])
                        k.gate("wait", [qD1], 360-180)
                        k.gate("wait", [qD2], 160)
                        k.gate('echo_corr', [qD2])
                        k.gate("wait", [qD2], 360-180)
                        k.gate("wait", [qD3], 160)
                        k.gate('echo_corr', [qD3])
                        k.gate("wait", [qD3], 360-180)
                        k.gate("wait", [qD4], 160)
                        k.gate('echo_corr', [qD4])
                        k.gate("wait", [qD4], 360-180)
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                        k.gate('i', [11])
                        k.gate('sf_park', [11])
                        k.gate('sf_park', [11])
                        k.gate('sf_park', [11])
                        k.gate('sf_park', [11])
                        k.gate('sf_park', [11])
                        k.gate('sf_park', [11])

                        ###############
                        # Z-Parity
                        ###############
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                        k.gate('rym90', [qA1])
                        k.gate('rym90', [qA3])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('cz', [qA3, qD2]) # without parking
                        k.gate('cz', [qD1, qA1])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('cz', [qD4, qA3])
                        k.gate('cz', [qD3, qA1])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('ry90', [qA1])
                        k.gate('ry90', [qA3])
                        k.gate("wait", [], 0)

                        ###################
                        # Logical S
                        ###################
                        k.measure(4) # Qubit Z1_trig
                        k.measure(6) # Qubit Z2_trig
                        k.measure(1) # Qubit X_trig
                        k.gate('cw_26', [qA2])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('cz', [qD1, qA2])
                        k.gate('rx180', [qD2])
                        k.gate('rx180', [qD2])
                        k.gate('i', [qD2])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('rx180', [qA2])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('cz', [qA2, qD2])
                        k.gate('rx180', [qD1])
                        k.gate('rx180', [qD1])
                        k.gate('i', [qD1])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('correction_S', [qA2])
                        k.gate('correction_S', [qD1])
                        k.gate('correction_S', [qD2])
                        k.gate('correction_S', [qD3])
                        k.gate('correction_S', [qD4])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 360)
                        k.gate("wait", [], 0)

                        if double_round:
                            ###############
                            # X-Parity
                            ###############
                            k.measure(4) # Qubit Z1_trig
                            k.measure(6) # Qubit Z2_trig
                            k.measure(1) # Qubit X_trig
                            k.gate('wait', [1, 4, 6], 280)
                            k.gate('rym90', [qA2])
                            k.gate('rym90', [qD1])
                            k.gate('rym90', [qD2])
                            k.gate('rym90', [qD3])
                            k.gate('rym90', [qD4])
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA2, qD2]) # CZ X-D2
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA2, qD1]) # CZ X-D1
                            k.gate('rx180', [qD2])  # echo D2
                            k.gate('rx180', [qD2])  # echo D2
                            k.gate('i', [qD2])
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('rx180', [qA2])   # echo ancilla
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA2, qD4]) # CZ X-D4
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA2, qD3]) # CZ X-D3
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('correction', [qA2])
                            k.gate('correction', [qD1])
                            k.gate('correction', [qD2])
                            k.gate('correction', [qD3])
                            k.gate('correction', [qD4])
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)

                            ##############
                            # Echo
                            ##############
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                            k.gate("wait", [qD1], 160)
                            k.gate('echo_corr', [qD1])
                            k.gate("wait", [qD1], 360-180)
                            k.gate("wait", [qD2], 160)
                            k.gate('echo_corr', [qD2])
                            k.gate("wait", [qD2], 360-180)
                            k.gate("wait", [qD3], 160)
                            k.gate('echo_corr', [qD3])
                            k.gate("wait", [qD3], 360-180)
                            k.gate("wait", [qD4], 160)
                            k.gate('echo_corr', [qD4])
                            k.gate("wait", [qD4], 360-180)
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                            k.gate('i', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])

                            ###############
                            # Z-Parity
                            ###############
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                            k.gate('rym90', [qA1])
                            k.gate('rym90', [qA3])
                            k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA3, qD2]) # without parking
                            k.gate('cz', [qD1, qA1])
                            k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qD4, qA3])
                            k.gate('cz', [qD3, qA1])
                            k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('ry90', [qA1])
                            k.gate('ry90', [qA3])
                            k.gate("wait", [], 0)

                        # tomography
                        k.gate("wait", [qD1, qD2, qD3, qD4], 0)
                        k.gate(p_q1, [qD1])
                        k.gate(p_q2, [qD2])
                        k.gate(p_q3, [qD3])
                        k.gate(p_q4, [qD4])
                        k.gate("wait", [qD1, qD2, qD3, qD4], 0)
                        # measurement
                        k.gate("wait", [], 0)
                        k.measure(qD1)
                        k.measure(qD2)
                        k.measure(qD3)
                        k.measure(qD4)
                        k.measure(qA1)
                        k.measure(qA3)
                        k.gate("wait", [], 0)

                        p.add_kernel(k)

    p = oqh.add_multi_q_cal_points(p, qubits=[qD1, qD2, qD3, qD4], combinations=[ s1+s2+s3+s4 for s1 in ['0', '1']
                                                                                              for s2 in ['0', '1']
                                                                                              for s3 in ['0', '1']
                                                                                              for s4 in ['0', '1']])
    if idle_meas != 0 :
        k = oqh.create_kernel('idle_meas', p)
        k.prepz(qD1)
        k.prepz(qD2)
        k.prepz(qD3)
        k.prepz(qD4)
        for i in range(idle_meas):
            k.measure(qD1)
            k.measure(qD2)
            k.measure(qD3)
            k.measure(qD4)
        p.add_kernel(k)

    p = oqh.compile(p)
    return p


def X_gate_seq_tomo(qD1: int, qD2: int,
                    qD3: int, qD4: int,
                    qA1: int, qA2: int, qA3: int,
                    platf_cfg: str,
                    initial_states=[['0', '0', '0', '0']],
                    double_round=False,
                    idle_meas=0):

    print('new')
    p = oqh.create_program("S_gate_seq", platf_cfg)

    tomo_gates = ['i', 'rym90', 'rx90']

    data_qubits=[qD1, qD2, qD3, qD4]

    for p_q4 in tomo_gates:
        for p_q3 in tomo_gates:
            for p_q2 in tomo_gates:
                for p_q1 in tomo_gates:
                    for initial_state in initial_states:
                        k = oqh.create_kernel(
                            'Tomo__tomo1_'+p_q1+'_tomo2_'+p_q2+'_tomo3_'+p_q3+'_tomo4_'+p_q4, p)
                        k.prepz(qD1)
                        k.prepz(qD2)
                        k.prepz(qD3)
                        k.prepz(qD4)
                        # k.prepz(qA1)
                        # k.prepz(qA2)
                        # k.prepz(qA3)
                        k.gate("wait", [], 0)

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

                        k.gate("wait", [], 0)

                        ###############
                        # X-Parity
                        ###############
                        k.measure(4) # Qubit Z1_trig
                        k.measure(6) # Qubit Z2_trig
                        k.measure(1) # Qubit X_trig
                        k.gate('wait', [1, 4, 6], 280)
                        k.gate('rym90', [qA2])
                        k.gate('rym90', [qD1])
                        k.gate('rym90', [qD2])
                        k.gate('rym90', [qD3])
                        k.gate('rym90', [qD4])
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('cz', [qA2, qD2]) # CZ X-D2
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('cz', [qA2, qD1]) # CZ X-D1
                        k.gate('rx180', [qD2])  # echo D2
                        k.gate('rx180', [qD2])  # echo D2
                        k.gate('i', [qD2])
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('rx180', [qA2])   # echo ancilla
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('cz', [qA2, qD4]) # CZ X-D4
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('cz', [qA2, qD3]) # CZ X-D3
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('correction', [qA2])
                        k.gate('correction', [qD1])
                        k.gate('correction', [qD2])
                        k.gate('correction', [qD3])
                        k.gate('correction', [qD4])
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)

                        ##############
                        # Echo
                        ##############
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                        k.gate("wait", [qD1], 160)
                        k.gate('echo_corr', [qD1])
                        k.gate("wait", [qD1], 360-180)
                        k.gate("wait", [qD2], 160)
                        k.gate('echo_corr', [qD2])
                        k.gate("wait", [qD2], 360-180)
                        k.gate("wait", [qD3], 160)
                        k.gate('echo_corr', [qD3])
                        k.gate("wait", [qD3], 360-180)
                        k.gate("wait", [qD4], 160)
                        k.gate('echo_corr', [qD4])
                        k.gate("wait", [qD4], 360-180)
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                        k.gate('i', [11])
                        k.gate('sf_park', [11])
                        k.gate('sf_park', [11])
                        k.gate('sf_park', [11])
                        k.gate('sf_park', [11])
                        k.gate('sf_park', [11])
                        k.gate('sf_park', [11])

                        ###############
                        # Z-Parity
                        ###############
                        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                        k.gate('rym90', [qA1])
                        k.gate('rym90', [qA3])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('cz', [qA3, qD2]) # without parking
                        k.gate('cz', [qD1, qA1])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('cz', [qD4, qA3])
                        k.gate('cz', [qD3, qA1])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('ry90', [qA1])
                        k.gate('ry90', [qA3])
                        k.gate("wait", [], 0)

                        ###################
                        # Logical S
                        ###################
                        k.measure(4) # Qubit Z1_trig
                        k.measure(6) # Qubit Z2_trig
                        k.measure(1) # Qubit X_trig
                        k.gate('cw_26', [qA2])
                        k.gate('ry90', [qD1])
                        k.gate('ry90', [qD3])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('cz', [qD1, qA2])
                        k.gate('rx180', [qD3])
                        k.gate('rx180', [qD3])
                        k.gate('i', [qD3])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('rx180', [qA2])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('cz', [qA2, qD3])
                        k.gate('rx180', [qD1])
                        k.gate('rx180', [qD1])
                        k.gate('i', [qD1])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('correction_S', [qA2])
                        k.gate('cw_26', [qD1])
                        k.gate('cw_26', [qD3])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                        k.gate('correction_S', [qD1])
                        k.gate('correction_S', [qD2])
                        k.gate('correction_S', [qD3])
                        k.gate('correction_S', [qD4])
                        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 340)
                        k.gate("wait", [], 0)

                        if double_round:
                            ###############
                            # X-Parity
                            ###############
                            k.measure(4) # Qubit Z1_trig
                            k.measure(6) # Qubit Z2_trig
                            k.measure(1) # Qubit X_trig
                            k.gate('wait', [1, 4, 6], 280)
                            k.gate('rym90', [qA2])
                            k.gate('rym90', [qD1])
                            k.gate('rym90', [qD2])
                            k.gate('rym90', [qD3])
                            k.gate('rym90', [qD4])
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA2, qD2]) # CZ X-D2
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA2, qD1]) # CZ X-D1
                            k.gate('rx180', [qD2])  # echo D2
                            k.gate('rx180', [qD2])  # echo D2
                            k.gate('i', [qD2])
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('rx180', [qA2])   # echo ancilla
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA2, qD4]) # CZ X-D4
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA2, qD3]) # CZ X-D3
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('correction', [qA2])
                            k.gate('correction', [qD1])
                            k.gate('correction', [qD2])
                            k.gate('correction', [qD3])
                            k.gate('correction', [qD4])
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)

                            ##############
                            # Echo
                            ##############
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                            k.gate("wait", [qD1], 160)
                            k.gate('echo_corr', [qD1])
                            k.gate("wait", [qD1], 360-180)
                            k.gate("wait", [qD2], 160)
                            k.gate('echo_corr', [qD2])
                            k.gate("wait", [qD2], 360-180)
                            k.gate("wait", [qD3], 160)
                            k.gate('echo_corr', [qD3])
                            k.gate("wait", [qD3], 360-180)
                            k.gate("wait", [qD4], 160)
                            k.gate('echo_corr', [qD4])
                            k.gate("wait", [qD4], 360-180)
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                            k.gate('i', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])
                            k.gate('sf_park', [11])

                            ###############
                            # Z-Parity
                            ###############
                            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
                            k.gate('rym90', [qA1])
                            k.gate('rym90', [qA3])
                            k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qA3, qD2]) # without parking
                            k.gate('cz', [qD1, qA1])
                            k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('cz', [qD4, qA3])
                            k.gate('cz', [qD3, qA1])
                            k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
                            k.gate('ry90', [qA1])
                            k.gate('ry90', [qA3])
                            k.gate("wait", [], 0)

                        # tomography
                        k.gate("wait", [qD1, qD2, qD3, qD4], 0)
                        k.gate(p_q1, [qD1])
                        k.gate(p_q2, [qD2])
                        k.gate(p_q3, [qD3])
                        k.gate(p_q4, [qD4])
                        k.gate("wait", [qD1, qD2, qD3, qD4], 0)
                        # measurement
                        k.gate("wait", [], 0)
                        k.measure(qD1)
                        k.measure(qD2)
                        k.measure(qD3)
                        k.measure(qD4)
                        k.measure(qA1)
                        k.measure(qA3)
                        k.gate("wait", [], 0)

                        p.add_kernel(k)

    p = oqh.add_multi_q_cal_points(p, qubits=[qD1, qD2, qD3, qD4], combinations=[ s1+s2+s3+s4 for s1 in ['0', '1']
                                                                                              for s2 in ['0', '1']
                                                                                              for s3 in ['0', '1']
                                                                                              for s4 in ['0', '1']])
    if idle_meas != 0 :
        k = oqh.create_kernel('idle_meas', p)
        k.prepz(qD1)
        k.prepz(qD2)
        k.prepz(qD3)
        k.prepz(qD4)
        for i in range(idle_meas):
            k.measure(qD1)
            k.measure(qD2)
            k.measure(qD3)
            k.measure(qD4)
        p.add_kernel(k)

    p = oqh.compile(p)
    return p


def surface_7(qD1: int, qD2: int,
              qD3: int, qD4: int,
              qA1: int, qA2: int, qA3: int,
              rounds: int,
              platf_cfg: str,
              initial_state=['0', '0', '0', '0'],
              idle_meas=0):

    print('new')
    p = oqh.create_program("Surface_7", platf_cfg)

    tomo_gates = ['i', 'rym90', 'rx90']

    data_qubits=[qD1, qD2, qD3, qD4]

    k = oqh.create_kernel(
            'Tomo__tomo1_', p)
    k.prepz(qD1)
    k.prepz(qD2)
    k.prepz(qD3)
    k.prepz(qD4)
    k.prepz(qA1)
    k.prepz(qA2)
    k.prepz(qA3)
    k.gate("wait", [], 0)

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
    k.gate("wait", [], 0)

    for i in range(rounds):

        ###############
        # X-Parity
        ###############
        k.gate('rym90', [qA2])
        k.gate('rym90', [qD1])
        k.gate('rym90', [qD2])
        k.gate('rym90', [qD3])
        k.gate('rym90', [qD4])
        k.gate('wait', [], 0)
        k.gate('cz', [qA2, qD2]) # CZ X-D2
        k.gate('wait', [], 0)
        k.gate('cz', [qA2, qD1]) # CZ X-D1
        k.gate('rx180', [qD2])  # echo D2
        k.gate('rx180', [qD2])  # echo D2
        k.gate('i', [qD2])
        k.gate('wait', [], 0)
        k.gate('rx180', [qA2])   # echo ancilla
        k.gate('wait', [], 0)
        k.gate('cz', [qA2, qD4]) # CZ X-D4
        k.gate('wait', [], 0)
        k.gate('cz', [qA2, qD3]) # CZ X-D3
        k.gate('wait', [], 0)
        k.gate('correction', [qA2])
        k.gate('correction', [qD1])
        k.gate('correction', [qD2])
        k.gate('correction', [qD3])
        k.gate('correction', [qD4])
        k.gate('wait', [], 0)

        ###############
        # Z-Parity
        ###############
        k.gate('rym90', [qA1])
        k.gate('rym90', [qA3])
        k.gate("wait", [], 0)
        k.gate('cz', [qA3, qD2]) # without parking
        k.gate('cz', [qD1, qA1])
        k.gate("wait", [], 0)
        k.gate('cz', [qD4, qA3])
        k.gate('cz', [qD3, qA1])
        k.gate("wait", [], 0)
        k.gate('ry90', [qA1])
        k.gate('ry90', [qA3])
        k.gate("wait", [], 0)

        ##############
        # Echo
        ##############
        k.gate('wait', [], 0)
        k.gate('echo_corr', [qD1])
        k.gate("wait", [qD1], 160)
        k.gate('echo_corr', [qD2])
        k.gate("wait", [qD2], 60)
        k.gate('echo_corr', [qD3])
        k.gate("wait", [qD3], 280)
        k.gate('echo_corr', [qD4])
        k.gate("wait", [qD4], 300)
        k.measure(qA1)
        k.measure(qA2)
        k.measure(qA3)
        k.gate('wait', [], 0)

    k.gate("wait", [], 0)
    # k.gate('ry90', [qD1])
    # k.gate('ry90', [qD2])
    # k.gate('ry90', [qD3])
    # k.gate('ry90', [qD4])
    k.measure(qD1)
    k.measure(qD2)
    k.measure(qD3)
    k.measure(qD4)
    k.gate("wait", [], 0)

    p.add_kernel(k)

    # p = oqh.add_multi_q_cal_points(p, qubits=[qD1, qD2, qD3, qD4], combinations=[ s1+s2+s3+s4 for s1 in ['0', '1']
    #                                                                                           for s2 in ['0', '1']
    #                                                                                           for s3 in ['0', '1']
    #                                                                                           for s4 in ['0', '1']],
    #                                initialize=False)

    if idle_meas != 0 :
        k = oqh.create_kernel('idle_meas', p)
        k.prepz(qD1)
        k.prepz(qD2)
        k.prepz(qD3)
        k.prepz(qD4)
        for i in range(idle_meas):
            k.measure(qD1)
            k.measure(qD2)
            k.measure(qD3)
            k.measure(qD4)
        p.add_kernel(k)

    p = oqh.compile(p)
    return p


def surface_7_pipelined(qD1: int, qD2: int,
                        qD3: int, qD4: int,
                        qA1: int, qA2: int, qA3: int,
                        rounds: int,
                        platf_cfg: str,
                        initial_state=['0', '0', '0', '0'],
                        theta_gate = False,
                        hooked=False,
                        idle_meas=0):

    print('new')
    p = oqh.create_program("Surface_7_pipelined", platf_cfg)

    data_qubits=[qD1, qD2, qD3, qD4]

    k = oqh.create_kernel('Multiround_experiment', p)
    k.prepz(qD1)
    k.prepz(qD2)
    k.prepz(qD3)
    k.prepz(qD4)
    k.prepz(qA1)
    k.prepz(qA2)
    k.prepz(qA3)
    k.gate("wait", [], 0)

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
    k.gate("wait", [], 0)

    for i in range(rounds):

        ###############
        # X-Parity
        ###############
        k.measure(4) # Qubit Z1_trig
        k.measure(6) # Qubit Z2_trig
        k.measure(1) # Qubit X_trig
        k.gate('wait', [1, 4, 6], 280)
        k.gate('rym90', [qA2])
        k.gate('rym90', [qD1])
        k.gate('rym90', [qD2])
        k.gate('rym90', [qD3])
        k.gate('rym90', [qD4])
        if hooked:
            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
            k.gate('cz', [qA2, qD3]) # CZ X-D3
            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
            k.gate('cz', [qA2, qD1]) # CZ X-D1
            k.gate('rx180', [qD2])  # echo D2
            k.gate('rx180', [qD2])  # echo D2
            k.gate('i', [qD2])
            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
            k.gate('rx180', [qA2])   # echo ancilla
            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
            k.gate('cz', [qA2, qD4]) # CZ X-D4
            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
            k.gate('cz', [qA2, qD2]) # CZ X-D2
            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
        else:
            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
            k.gate('cz', [qA2, qD2]) # CZ X-D2
            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
            k.gate('cz', [qA2, qD1]) # CZ X-D1
            k.gate('rx180', [qD2])  # echo D2
            k.gate('rx180', [qD2])  # echo D2
            k.gate('i', [qD2])
            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
            k.gate('rx180', [qA2])   # echo ancilla
            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
            k.gate('cz', [qA2, qD4]) # CZ X-D4
            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
            k.gate('cz', [qA2, qD3]) # CZ X-D3
            k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
        k.gate('correction', [qA2])
        k.gate('correction', [qD1])
        k.gate('correction', [qD2])
        k.gate('correction', [qD3])
        k.gate('correction', [qD4])
        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)

        ##############
        # Echo
        ##############
        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
        k.gate("wait", [qD1], 160)
        k.gate('echo_corr', [qD1])
        k.gate("wait", [qD1], 360-180)
        k.gate("wait", [qD2], 160)
        k.gate('echo_corr', [qD2])
        k.gate("wait", [qD2], 360-180)
        k.gate("wait", [qD3], 160)
        k.gate('echo_corr', [qD3])
        k.gate("wait", [qD3], 360-180)
        k.gate("wait", [qD4], 160)
        k.gate('echo_corr', [qD4])
        k.gate("wait", [qD4], 360-180)
        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
        k.gate('i', [11])
        k.gate('sf_park', [11])
        k.gate('sf_park', [11])
        k.gate('sf_park', [11])
        k.gate('sf_park', [11])
        k.gate('sf_park', [11])
        k.gate('sf_park', [11])


        ###############
        # Z-Parity
        ###############
        k.gate('wait', [qD1, qD2, qD3, qD4, qA1, qA3], 0)
        k.gate('rym90', [qA1])
        k.gate('rym90', [qA3])
        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
        k.gate('cz', [qA3, qD2]) # without parking
        k.gate('cz', [qD1, qA1])
        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
        k.gate('cz', [qD4, qA3])
        k.gate('cz', [qD3, qA1])
        k.gate("wait", [qD1, qD2, qD3, qD4, qA1, qA2, qA3], 0)
        k.gate('ry90', [qA1])
        k.gate('ry90', [qA3])
        k.gate("wait", [], 0)
        if theta_gate:
            if i != rounds-1:
                k.gate('theta_gate', [qD2])
                k.gate("wait", [], 0)

    k.gate("wait", [], 0)
    # k.gate('ry90', [qD1])
    # k.gate('ry90', [qD2])
    # k.gate('ry90', [qD3])
    # k.gate('ry90', [qD4])
    # k.gate("wait", [], 0)
    k.measure(qD1)
    k.measure(qD2)
    k.measure(qD3)
    k.measure(qD4)
    k.measure(qA1)
    k.measure(qA3)
    k.gate("wait", [], 0)

    p.add_kernel(k)

    if idle_meas != 0 :
        k = oqh.create_kernel('idle_meas', p)
        k.gate('wait', [qD1, qD2, qD3, qD4], 200)
        for i in range(idle_meas):
            k.measure(qD1)
            k.measure(qD2)
            k.measure(qD3)
            k.measure(qD4)
        p.add_kernel(k)

    p = oqh.compile(p)
    return p


def conditional_oscillation_seq(q0: int, q1: int,
                                q2: int = None, q3: int = None,
                                platf_cfg: str = None,
                                disable_cz: bool = False,
                                disabled_cz_duration: int = 60,
                                cz_repetitions: int = 1,
                                angles=np.arange(0, 360, 20),
                                wait_time_before_flux: int = 0,
                                wait_time_after_flux: int = 0,
                                add_cal_points: bool = True,
                                cases: list = ('no_excitation', 'excitation'),
                                flux_codeword: str = 'cz',
                                flux_codeword_park: str = None,
                                parked_qubit_seq: str = 'ground',
                                disable_parallel_single_q_gates: bool = False,
                                offset_phase_calibration=0):
    '''
    Sequence used to calibrate flux pulses for CZ gates.

    q0 is the oscillating qubit
    q1 is the spectator qubit

    Timing of the sequence:
    q0:  X90   --  C-Phase  (repet. C-Phase) Rphi90 RO
    q1: X180/I --  C-Phase         --        X180   RO
    q2:  X90   -- PARK/C-Phase     --        Rphi90 RO
    q3: X180/I --  C-Phase         --        X180   RO

    Args:
        q0, q1      (str): names of the addressed qubits
        q2, q3      (str): names of optional extra qubit to either park or
            apply a CZ to.

        flux_codeword (str):
            the gate to be applied to the qubit pair q0, q1
        flux_codeword_park (str):
            optionally park qubits q2 (and q3) with either a 'park' pulse
            (single qubit operation on q2) or a 'cz' pulse on q2-q3.
        disable_cz (bool): disable CZ gate
        cz_repetitions (int): how many cz gates to apply consecutively
        angles      (array): angles of the recovery pulse
        wait_time_after_flux   (int): wait time in ns after triggering all flux
            pulses
    '''
    assert parked_qubit_seq in {"ground", "ramsey"}

    p = oqh.create_program("conditional_oscillation_seq", platf_cfg)

    # These angles correspond to special pi/2 pulses in the lutman
    for i, angle in enumerate(angles):
        for case in cases:

            k = oqh.create_kernel("{}_{}".format(case, angle), p)
            k.prepz(q0)
            k.prepz(q1)
            if q2 is not None:
                k.prepz(q2)
            if q3 is not None:
                k.prepz(q3)

            k.gate("wait", [], 0)  # alignment workaround

            # #################################################################
            # Single qubit ** parallel ** gates before flux pulses
            # #################################################################

            control_qubits = [q1]
            if q3 is not None:
                # In case of parallel cz
                control_qubits.append(q3)

            ramsey_qubits = [q0]
            if q2 is not None and parked_qubit_seq == "ramsey":
                # For parking and parallel cz
                ramsey_qubits.append(q2)

            if case == "excitation":
                # implicit identities otherwise
                for q in control_qubits:
                    k.gate("rx180", [q])
                    if disable_parallel_single_q_gates:
                        k.gate("wait", [], 0)

            for q in ramsey_qubits:
                k.gate("rym90", [q])
                if disable_parallel_single_q_gates:
                    k.gate("wait", [], 0)

            # # k.gate("wait", [], 0) # HARD CODED FOR
            # k.gate("rx180", [10]) # TUNE UP# SURFACE-4
            # #                       # TUNE UP

            k.gate("wait", [], 0)  # alignment workaround

            ##################################################################
            #Flux pulses
            ##################################################################

            k.gate('wait', [], wait_time_before_flux)

            for dummy_i in range(cz_repetitions):
                if not disable_cz:
                    # Parallel flux pulses below

                    k.gate("wait", [], 0)          # HARD CODED FOR
                    k.gate(flux_codeword, [11, 9])  # SURFACE-4
                    k.gate("wait", [], 0)          # TUNE UP

                    k.gate("wait", [], 0)          # HARD CODED FOR
                    k.gate(flux_codeword, [11, 8]) # SURFACE-4
                    k.gate("rx180", [9])           # TUNE UP
                    k.gate("rx180", [9])
                    k.gate("i",     [9])
                    k.gate("wait", [], 0)

                    k.gate("wait", [], 0) # TUNE UP
                    k.gate("rx180", [11])
                    k.gate("wait", [], 0) # HARD CODED FOR

                    k.gate("wait", [], 0)           # HARD CODE FOR
                    k.gate(flux_codeword, [11, 15]) # SURFACE-4
                    k.gate("wait", [], 0)           # TUNE UP

                    k.gate("wait", [], 0)           # HARD CODE FOR
                    k.gate(flux_codeword, [11, 14]) # SURFACE-4
                    k.gate("wait", [], 0)           # TUNE UP

                    # in case of parking and parallel cz
                    if flux_codeword_park == 'cz':
                        k.gate(flux_codeword_park, [q2, q3])
                    elif flux_codeword_park == 'park':
                        k.gate(flux_codeword_park, [q2])
                        if q3 is not None:
                            raise ValueError("Expected q3 to be None")
                    elif flux_codeword_park is None:
                        pass
                    else:
                        raise ValueError(
                            'flux_codeword_park "{}" not allowed'.format(
                                flux_codeword_park))
                else:
                    k.gate('wait', [q0, q1], disabled_cz_duration)

                k.gate("wait", [], 0)

            k.gate('wait', [], wait_time_after_flux)

            # #################################################################
            # Single qubit ** parallel ** gates post flux pulses
            # #################################################################

            if case == "excitation":
                for q in control_qubits:
                    k.gate("rx180", [q])
                    if disable_parallel_single_q_gates:
                        k.gate("wait", [], 0)

            # cw_idx corresponds to special hardcoded angles in the lutman
            # special because the cw phase pulses go in mult of 20 deg
            cw_idx = angle // 20 + 9
            phi_gate = None
            # if angle == 90+offset_phase_calibration:
            #     phi_gate = 'rx90'
            # elif angle == offset_phase_calibration:
            #     phi_gate = 'rym90'
            # else:
            phi_gate = 'cw_{:02}'.format(cw_idx)

            for q in ramsey_qubits:
                k.gate(phi_gate, [q])
                if disable_parallel_single_q_gates:
                    k.gate("wait", [], 0)

            k.gate('wait', [], 0)

            # #################################################################
            # Measurement
            # #################################################################

            k.measure(q0)
            k.measure(q1)
            if q2 is not None:
                k.measure(q2)
            if q3 is not None:
                k.measure(q3)
            k.gate('wait', [], 0)

            p.add_kernel(k)

    if add_cal_points:
        if q2 is None:
            states = ["00", "01", "10", "11"]
        else:
            states = ["000", "010", "101", "111"]

        qubits = [q0, q1] if q2 is None else [q0, q1, q2]
        oqh.add_multi_q_cal_points(
            p, qubits=qubits, f_state_cal_pt_cw=31,
            combinations=states, return_comb=False)

    p = oqh.compile(p)

    # [2020-06-24] parallel cz not supported (yet)

    if add_cal_points:
        cal_pts_idx = [361, 362, 363, 364]
    else:
        cal_pts_idx = []

    p.sweep_points = np.concatenate(
        [np.repeat(angles, len(cases)), cal_pts_idx])

    p.set_sweep_points(p.sweep_points)

    return p


def two_qubit_conditional_oscillation_seq(qH: int,
                                          qA: int,
                                          qL: int,
                                          platf_cfg: str = None,
                                          angles=np.arange(0, 360, 20),
                                          ramsey_qubit: int = None,
                                          echo: bool = False,
                                          cases: list = ['00', '01', '10', '11']):

    p = oqh.create_program("two_qubit_conditional_oscillation", platf_cfg)

    # These angles correspond to special pi/2 pulses in the lutman
    for i, angle in enumerate(angles):
        for case in cases:
            k = oqh.create_kernel("k{}_{}".format(case, angle), p)

            # Qubit initialization
            k.prepz(qH)
            k.prepz(qA)
            k.prepz(qL)
            k.gate('wait', [], 0)

            if ramsey_qubit is None:
                control_qubits = [qH, qL]
                ramsey_qubit = qA
            else:
                ramsey_qubit_idx = [qH, qA, qL].index(ramsey_qubit)
                control_qubits = [qH, qA, qL]
                control_qubits.pop(ramsey_qubit_idx)
            # prepare state of control qubits
            for i, state in enumerate(case):
                if state == '1':
                    k.gate('rx180', [control_qubits[i]])
                else:
                    pass
            k.gate("rx90", [ramsey_qubit])
            k.gate('wait', [], 0)

            # CZ gates
            k.gate('cz', [qH, qA])
            k.gate('rx180', [14])
            k.gate('rx180', [14])
            k.gate('i', [14])
            k.gate('wait', [], 0)
            if echo:
                k.gate('rx180', [qA])
                k.gate('wait', [], 0)
            k.gate('cz', [qA, qL])
            k.gate('rx180', [8])
            k.gate('rx180', [8])
            k.gate('i', [8])
            k.gate('wait', [], 0)

            # Return state of control qubits
            for i, state in enumerate(case):
                if state == '1':
                    k.gate('rx180', [control_qubits[i]])
                else:
                    pass

            # cw_idx corresponds to special hardcoded angles in the lutman
            # special because the cw phase pulses go in mult of 20 deg
            cw_idx = angle // 20 + 9
            phi_gate = None
            phi_gate = 'cw_{:02}'.format(cw_idx)
            k.gate(phi_gate, [ramsey_qubit])
            k.gate('wait', [], 0)

            # #################################################################
            # Measurement
            # #################################################################

            k.measure(qH)
            k.measure(qA)
            k.measure(qL)
            p.add_kernel(k)

    states = ["000", "001", "010", "011", "100", "101", "110", "111"]
    qubits = [qA, qH, qL]
    oqh.add_multi_q_cal_points(
        p, qubits=qubits,
        combinations=states, return_comb=False)

    p = oqh.compile(p)

    cal_pts_idx = [361, 362, 363, 364, 365, 366, 367, 368]

    p.sweep_points = np.concatenate(
        [np.repeat(angles, len(cases)), cal_pts_idx])
    p.set_sweep_points(p.sweep_points)

    return p


def four_qubit_conditional_oscillation_seq(qH1: int,
                                           qH2: int,
                                           qA: int,
                                           qL1: int,
                                           qL2: int,
                                           ramsey_qubit: int = None,
                                           platf_cfg: str = None,
                                           angles=np.arange(0, 360, 20),
                                           cases: list = ['0000', '0001', '0010', '0011',
                                                          '0100', '0101', '0110', '0111',
                                                          '1000', '1001', '1010', '1011',
                                                          '1100', '1101', '1110', '1111']):

    p = oqh.create_program("four_qubit_conditional_oscillation", platf_cfg)

    # These angles correspond to special pi/2 pulses in the lutman
    for i, angle in enumerate(angles):
        for case in cases:
            k = oqh.create_kernel("k{}_{}".format(case, angle), p)

            # Qubit initialization
            k.prepz(qH1)
            k.prepz(qH2)
            k.prepz(qA)
            k.prepz(qL1)
            k.prepz(qL2)
            k.gate('wait', [], 0)

            if ramsey_qubit is None:
                control_qubits = [qH1, qH2, qL1, qL2]
                ramsey_qubit = qA
            else:
                control_qubits = [qH1, qH2, qA, qL1, qL2]
                idx = control_qubits.index(ramsey_qubit)
                del control_qubits[idx]

            # prepare state of control qubits
            for i, state in enumerate(case):
                if state == '1':
                    k.gate('rx180', [control_qubits[i]])
                else:
                    pass
            k.gate("rym90", [ramsey_qubit])
            k.gate('wait', [], 0)

            # CZ gates
            k.gate('cz', [qA, qH2])
            k.gate('wait', [], 0)
            k.gate('cz', [qA, qH1])
            k.gate('rx180', [qH2])
            k.gate('rx180', [qH2])
            k.gate('i', [qH2])
            k.gate('wait', [], 0)
            k.gate('rx180', [qA])
            k.gate('wait', [], 0)
            k.gate('cz', [qA, qL2])
            k.gate('wait', [], 0)
            k.gate('cz', [qA, qL1])
            k.gate('wait', [], 0)

            # Return state of control qubits
            for i, state in enumerate(case):
                if state == '1':
                    k.gate('rx180', [control_qubits[i]])
                else:
                    pass

            # cw_idx = angle_2[i] // 20 + 9
            cw_idx = angle // 20 + 9
            phi_gate = 'cw_{:02}'.format(cw_idx)
            k.gate(phi_gate, [ramsey_qubit])
            k.gate('wait', [], 0)

            # #################################################################
            # Measurement
            # #################################################################

            k.measure(qH1)
            k.measure(qH2)
            k.measure(qA)
            k.measure(qL1)
            k.measure(qL2)
            p.add_kernel(k)

    # states = ["00000", "00000", "00010", "00011", "00100", "00101", "00110", "00111"]#,
    # #           "01000", "01001", "01010", "01011", "01100", "01101", "01110", "01111",
    # #           "10000", "10001", "10010", "10011", "10100", "10101", "10110", "10111",
    # #           "11000", "11001", "11010", "11011", "11100", "11101", "11110", "11111"]
    # qubits = [qA, qH1, qH2, qL1, qL2]
    # oqh.add_multi_q_cal_points(
    #     p, qubits=qubits,
    #     combinations=states, return_comb=False)

    p = oqh.compile(p)

    cal_pts_idx = []# [361+i for i in range(8)]

    p.sweep_points = np.concatenate(
        [np.repeat(angles, len(cases)), cal_pts_idx])
    p.set_sweep_points(p.sweep_points)

    return p

def phase_correction_calibration_seq(qH1: int,
                                     qH2: int,
                                     qA: int,
                                     qL1: int,
                                     qL2: int,
                                     platf_cfg: str,
                                     ramsey_qubit: int = None):

    p = oqh.create_program("four_qubit_conditional_oscillation", platf_cfg)

    k = oqh.create_kernel("Control_off", p)
    k.prepz(qH1)
    k.prepz(qH2)
    k.prepz(qA)
    k.prepz(qL1)
    k.prepz(qL2)
    k.gate('wait', [], 0)
    k.gate("rx90", [ramsey_qubit])
    k.gate('wait', [], 0)
    k.gate('cz', [qA, qH2])
    k.gate('wait', [], 0)
    k.gate('cz', [qA, qH1])
    k.gate('ry180', [qH2])
    k.gate('ry180', [qH2])
    k.gate('i', [qH2])
    k.gate('wait', [], 0)
    k.gate('rx180', [qA])
    k.gate('wait', [], 0)
    k.gate('cz', [qA, qL2])
    k.gate('wait', [], 0)
    k.gate('cz', [qA, qL1])
    k.gate('wait', [], 0)
    k.gate('correction', [ramsey_qubit])
    k.gate('wait', [], 0)
    k.measure(qA)
    k.measure(ramsey_qubit)
    p.add_kernel(k)

    k = oqh.create_kernel("Control_on", p)
    k.prepz(qH1)
    k.prepz(qH2)
    k.prepz(qA)
    k.prepz(qL1)
    k.prepz(qL2)
    k.gate('wait', [], 0)
    k.gate("rx90", [ramsey_qubit])
    k.gate('rx180', [qA])
    k.gate('wait', [], 0)
    k.gate('cz', [qA, qH2])
    k.gate('wait', [], 0)
    k.gate('cz', [qA, qH1])
    k.gate('ry180', [qH2])
    k.gate('ry180', [qH2])
    k.gate('i', [qH2])
    k.gate('wait', [], 0)
    k.gate('rx180', [qA])
    k.gate('wait', [], 0)
    k.gate('cz', [qA, qL2])
    k.gate('wait', [], 0)
    k.gate('cz', [qA, qL1])
    k.gate('wait', [], 0)
    k.gate('correction', [ramsey_qubit])
    k.gate('rx180', [qA])
    k.gate('wait', [], 0)
    k.measure(qA)
    k.measure(ramsey_qubit)
    p.add_kernel(k)

    p = oqh.compile(p)
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
            k.gate("wait", [], 0) #alignment workaround
            k.gate('wait', [2, 0], second_CZ_delay//2)
            k.gate("wait", [], 0) #alignment workaround
            if add_echo_pulses:
                k.gate('rx180', [q0])
                k.gate('rx180', [q1])
            k.gate("wait", [], 0) #alignment workaround
            k.gate('wait', [2, 0], second_CZ_delay//2)
            k.gate("wait", [], 0) #alignment workaround
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
            k.gate("wait", [], 0) #alignment workaround
            k.gate('wait', [2, 0], second_CZ_delay//2)
            k.gate("wait", [], 0) #alignment workaround
            if add_echo_pulses:
                k.gate('rx180', [q0])
                k.gate('rx180', [q1])
            k.gate("wait", [], 0) #alignment workaround
            k.gate('wait', [2, 0], second_CZ_delay//2)
            k.gate("wait", [], 0) #alignment workaround
            if add_echo_pulses:
                k.gate('rx180', [q0])
                k.gate('rx180', [q1])
            k.gate("wait", [], 0) #alignment workaround
            k.gate('wait', [2, 0], CZ_duration)
            k.gate("wait", [], 0) #alignment workaround
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
        k.gate("wait", [], 0) #alignment workaround
        for i in range(nr_of_repeated_gates):
            k.gate('fl_cw_01', [2, 0])
        k.gate("wait", [], 0) #alignment workaround
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
    k.gate("wait", [], 0) #alignment workaround
    k.gate('fl_cw_{:02}'.format(flux_cw), [2, 0])
    k.gate("wait", [], 0) #alignment workaround
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

    for i_g, gates in enumerate(tomo_gates):
        idx_g0 = i_g % 6
        idx_g1 = ((i_g - idx_g0)//6) % 6
        # strings denoting the gates
        SP0 = cardinal_gates[idx_p0]
        SP1 = cardinal_gates[idx_p1]
        t_q0 = gates[1]
        t_q1 = gates[0]
        k = oqh.create_kernel(
            'PT_{}_tomo_{}_{}'.format(cardinal, idx_g0, idx_g1), p)

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
        k.gate("wait", [], 0) #alignment workaround
        k.gate('wait', [2, 0], 0)
        k.gate("wait", [], 0) #alignment workaround
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
            k.gate("wait", [], 0) #alignment workaround
            k.gate('fl_cw_02', [2, 0])
            k.gate("wait", [], 0) #alignment workaround
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
        k.gate("wait", [], 0)  # alignment workaround
        # hardcoded because of flux_tuples, [q1, q0])
        k.gate('wait', [q0, q1], wait_time)

        if ramsey_axis == 'x':
            k.gate('rx90', [q0])
        elif ramsey_axis == 'y':
            k.gate('ry90', [q0])
        else:
            raise ValueError('ramsey_axis must be "x" or "y"')
        k.gate("wait", [], 0)  # alignment workaround
        k.gate(flux_codeword_b, [2, 0]) # edge hardcoded because of openql
        k.gate("wait", [], 0)  # alignment workaround
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


def two_qubit_state_tomography(qubit_idxs,
                               bell_state,
                               product_state,
                               platf_cfg,
                               wait_after_flux: float=None,
                               flux_codeword: str='cz'):

    p = oqh.create_program("state_tomography_2Q_{}_{}_{}".format(product_state,qubit_idxs[0], qubit_idxs[1]), platf_cfg)

    q0 = qubit_idxs[0]
    q1 = qubit_idxs[1]

    calibration_points = ['00', '01', '10', '11']
    measurement_pre_rotations = ['II', 'IF', 'FI', 'FF']
    bases = ['X', 'Y', 'Z']
    ## Explain this ?
    bases_comb = [basis_0+basis_1 for basis_0 in bases for basis_1 in bases]
    combinations = []
    combinations += [b+'-'+c for b in bases_comb for c in measurement_pre_rotations]
    combinations += calibration_points

    state_strings = ['0', '1', '+', '-', 'i', 'j']
    state_gate = ['i', 'rx180', 'ry90', 'rym90', 'rxm90', 'rx90']
    product_gate = ['0', '0', '0', '0']

    for basis in bases_comb:
        for pre_rot in measurement_pre_rotations: # tomographic pre-rotation
            k = oqh.create_kernel('TFD_{}-basis_{}'.format(basis, pre_rot), p)
            for q_idx in qubit_idxs:
                k.prepz(q_idx)

     # Choose a bell state and set the corresponding preparation pulses
            if bell_state is not None:
                        #
                # Q1 |0> --- P1 --o-- A1 -- R1 -- M
                #                 |
                # Q0 |0> --- P0 --o-- I  -- R0 -- M
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
                k.gate(prep_pulse_q0, [q0])
                k.gate(prep_pulse_q1, [q1])
                k.gate("wait", [],  0)# Empty list generates barrier for all qubits in platf. only works with 0.8.0
                # k.gate('cz', [q0, q1])
                k.gate(flux_codeword, [q0, q1])
                k.gate("wait", [],  0)
                # after-rotations
                k.gate(after_pulse_q1, [q1])
                # possibly wait
                if wait_after_flux is not None:
                    k.gate("wait", [q0, q1], round(wait_after_flux*1e9))
                k.gate("wait", [],  0)

            if product_state is not None:
                for i, string in enumerate(product_state):
                    product_gate[i] = state_gate[state_strings.index(string)]
                k.gate(product_gate[0], [q0])
                k.gate(product_gate[1], [q1])
                k.gate('wait', [], 0)

            if (product_state is not None) and (bell_state is not None):
                raise ValueError('Confusing requirements, both state {} and bell-state {}'.format(product_state,bell_state))

            # tomographic pre-rotations
            for rot_idx in range(2):
                q_idx = qubit_idxs[rot_idx]
                flip = pre_rot[rot_idx]
                qubit_basis = basis[rot_idx]
                # Basis rotations take the operator Z onto (Ri* Z Ri):
                #              Z     -Z       X       -X       -Y       Y
                # FLIPS        I      F       I        F        I       F
                # BASIS        Z      Z       X        X        Y       Y
                # tomo_gates = ['i', 'rx180', 'ry90', 'rym90', 'rx90', 'rxm90']
                prerot_Z = 'i'
                prerot_mZ = 'rx180'
                prerot_X = 'rym90'
                prerot_mX = 'ry90'
                prerot_Y = 'rx90'
                prerot_mY = 'rxm90'

                if flip == 'I' and qubit_basis == 'Z':
                    k.gate(prerot_Z, [q_idx])
                elif flip == 'F' and qubit_basis == 'Z':
                    k.gate(prerot_mZ, [q_idx])
                elif flip == 'I' and qubit_basis == 'X':
                    k.gate(prerot_X, [q_idx])
                elif flip == 'F' and qubit_basis == 'X':
                    k.gate(prerot_mX, [q_idx])
                elif flip == 'I' and qubit_basis == 'Y':
                    k.gate(prerot_Y, [q_idx])
                elif flip == 'F' and qubit_basis == 'Y':
                    k.gate(prerot_mY, [q_idx])
                else:
                    raise ValueError("flip {} and basis {} not understood".format(flip,basis))
                    k.gate('i', [q_idx])
            k.gate('wait', [], 0)
            for q_idx in qubit_idxs:
                k.measure(q_idx)
            k.gate('wait', [], 0)
            p.add_kernel(k)

    for cal_pt in calibration_points:
        k = oqh.create_kernel('Cal_{}'.format(cal_pt), p)
        for q_idx in qubit_idxs:
            k.prepz(q_idx)
        k.gate('wait', [], 0)
        for cal_idx, state in enumerate(cal_pt):
            q_idx = qubit_idxs[cal_idx]
            if state == '1':
                k.gate('rx180', [q_idx])
        k.gate('wait', [], 0) # barrier guarantees allignment
        for q_idx in qubit_idxs:
            k.measure(q_idx)
        k.gate('wait', [], 0)
        p.add_kernel(k)
    p = oqh.compile(p)
    p.combinations = combinations
    return p


def multi_qubit_Depletion(qubits: list, platf_cfg: str,
                          time: float):
    """

    Performs a measurement pulse and wait time followed by a simultaneous ALLXY on the
    specified qubits:

    |q0> - RO <--wait--> P0 - P1 - RO
    |q1> - RO <--time--> P0 - P1 - RO
     .
     .
     .

     args:
        qubits : List of qubits numbers.
        time   : wait time (s) after readout pulse.
    """

    p = oqh.create_program('multi_qubit_Depletion', platf_cfg)

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

    for i, pulse_comb in enumerate(pulse_combinations):
        for j in range(2): #double points
            k = oqh.create_kernel('Depletion_{}_{}'.format(j, i), p)
            for qubit in qubits:
                k.prepz(qubit)
                k.measure(qubit)

            wait_nanoseconds = int(round(time/1e-9))
            for qubit in qubits:
                k.gate("wait", [qubit], wait_nanoseconds)

            if sequence_type == 'simultaneous':
                for qubit in qubits:
                    k.gate(pulse_comb[0], [qubit])
                    k.gate(pulse_comb[1], [qubit])
                    k.measure(qubit)

            p.add_kernel(k)

    p = oqh.compile(p)
    return p


def two_qubit_Depletion(q0: int, q1: int, platf_cfg: str,
                        time: float,
                        sequence_type='sequential',
                        double_points: bool=False):
    """

    """
    p = oqh.create_program('two_qubit_Depletion', platf_cfg)

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

    i = 0
    for pulse_comb_q0, pulse_comb_q1 in zip(pulse_combinations_q0,
                                            pulse_combinations_q1):
        i += 1
        k = oqh.create_kernel('AllXY_{}'.format(i), p)
        k.prepz(q0)
        k.prepz(q1)
        k.measure(q0)
        k.measure(q1)

        wait_nanoseconds = int(round(time/1e-9))
        k.gate("wait", [q0], wait_nanoseconds)
        k.gate("wait", [q1], wait_nanoseconds)
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


def Two_qubit_RTE(QX: int , QZ: int, platf_cfg: str,
                  measurements: int, net='i', start_states: list = ['0'],
                  ramsey_time_1: int = 120, ramsey_time_2: int = 120,
                  echo: bool = False):
    """

    """
    p = oqh.create_program('RTE', platf_cfg)

    for state in start_states:
        k = oqh.create_kernel('RTE start state {}'.format(state), p)
        k.prepz(QX)
        k.prepz(QZ)
        if state == '1':
            k.gate('rx180', [QX])
            k.gate('rx180', [QZ])
        k.gate('wait', [QX, QZ], 0)
        ######################
        # Parity check
        ######################
        for m in range(measurements):
            # Superposition
            k.gate('rx90', [QX])
            k.gate('i', [QZ])
            # CZ emulation
            if echo:
                k.gate('wait', [QX, QZ], int((ramsey_time_1-20)/2) )
                k.gate('rx180', [QX])
                k.gate('i', [QZ])
                k.gate('wait', [QX, QZ], int((ramsey_time_1-20)/2) )
            else:
                k.gate('wait', [QX, QZ], ramsey_time_1)
            # intermidate sequential
            if net == 'pi' or echo:
                k.gate('rx90', [QX])
            else:
                k.gate('rxm90', [QX])
            k.gate('i', [QZ])
            k.gate('i', [QX])
            k.gate('rx90', [QZ])
            # CZ emulation
            if echo:
                k.gate('wait', [QX, QZ], int((ramsey_time_2-20)/2) )
                k.gate('rx180', [QZ])
                k.gate('i', [QX])
                k.gate('wait', [QX, QZ], int((ramsey_time_2-20)/2) )
            else:
                k.gate('wait', [QX, QZ], ramsey_time_2)
            # Recovery pulse
            k.gate('i', [QX])
            if net == 'pi' or echo:
                k.gate('rx90', [QZ])
            else:
                k.gate('rxm90', [QZ])
            k.gate('wait', [QX, QZ], 0)
            # Measurement
            k.measure(QX)
            k.measure(QZ)

        p.add_kernel(k)

    p = oqh.compile(p)
    return p

def Two_qubit_RTE_pipelined(QX:int, QZ:int, QZ_d:int, platf_cfg: str,
                            measurements:int, start_states:list = ['0'],
                            ramsey_time: int = 120, echo:bool = False):
    """

    """
    p = oqh.create_program('RTE_pipelined', platf_cfg)

    for state in start_states:
      k = oqh.create_kernel('RTE pip start state {}'.format(state), p)
      k.prepz(QX)
      k.prepz(QZ)
      if state == '1':
          k.gate('rx180', [QX])
          k.gate('rx180', [QZ])
      k.gate('wait', [QX, QZ, QZ_d], 0)
      # k.gate('wait', [QX], 0)
      ######################
      # Parity check
      #####################
      for m in range(measurements):

        k.measure(QZ_d)
        if echo is True:
            k.gate('wait', [QZ_d], ramsey_time+60)
        else:
            k.gate('wait', [QZ_d], ramsey_time+40)

        k.gate('rx90', [QZ])
        if echo is True:
            k.gate('wait', [QZ], ramsey_time/2)
            k.gate('rx180', [QZ])
            k.gate('wait', [QZ], ramsey_time/2)
            k.gate('rx90', [QZ])
        else:
            k.gate('wait', [QZ], ramsey_time)
            k.gate('rxm90', [QZ])
        k.gate('wait', [QZ], 500)

        k.measure(QX)
        k.gate('rx90', [QX])
        if echo is True:
            k.gate('wait', [QX], ramsey_time/2)
            k.gate('rx180', [QX])
            k.gate('wait', [QX], ramsey_time/2)
            k.gate('rx90', [QX])
        else:
            k.gate('wait', [QX], ramsey_time)
            k.gate('rxm90', [QX])

        k.gate('wait', [QX, QZ, QZ_d], 0)

      p.add_kernel(k)

    p = oqh.compile(p)
    return p



def Ramsey_cross(q_rams: int,
                 q_meas: int,
                 platf_cfg: str,
                 echo: bool = True,
                 wait_time: int= 0,
                 angles: list = np.arange(0,360, 20),
                 initial_state: str = '0'):
    """
    q_target is ramseyed
    q_spec is measured

    """
    p = oqh.create_program("Ramsey_msmt_induced_shift", platf_cfg)

    for i, angle in enumerate(angles[:]):

        k = oqh.create_kernel("Ramsey_azi_"+str(angle), p)

        k.prepz(q_rams)
        k.prepz(q_meas)
        k.gate("wait", [], 0)

        k.gate('rx90', [q_rams])
        if initial_state == '1':
            k.gate('rx180', [q_meas])

        k.gate("wait", [], 0)
        k.measure(q_meas)
        k.measure(12)

        if echo:
            k.gate('ry180', [q_rams])
            k.gate("wait", [q_rams], round(wait_time))
        k.gate("wait", [], 0)

        cw_idx = angle // 20 + 9
        k.gate('cw_{:02}'.format(cw_idx), [q_rams])
        k.measure(q_rams)
        k.measure(q_meas)

        p.add_kernel(k)

    # adding the calibration points
    oqh.add_single_qubit_cal_points(p, qubit_idx=q_rams,
                                    measured_qubits=[q_rams, q_meas])

    p.sweep_points = np.concatenate((np.repeat(angles,2), [360, 370, 380, 390]))
    p.set_sweep_points(p.sweep_points)

    p = oqh.compile(p)
    return p

def Ramsey_cross_2(q_rams: int,
                 q_meas: int,
                 platf_cfg: str,
                 echo: bool = True,
                 wait_time: int= 0,
                 angles: list = np.arange(0,360, 20),
                 nr_shots: int=2,
                 initial_state: str = '0'):
    """
    q_target is ramseyed
    q_spec is measured

    """
    p = oqh.create_program("Ramsey_msmt_induced_shift", platf_cfg)

    for i, angle in enumerate(angles[:]):

        k = oqh.create_kernel("Ramsey_azi_"+str(angle), p)

        k.prepz(q_rams)
        k.prepz(q_meas)
        k.gate("wait", [], 0)

        k.gate("rym90", [q_meas])
        if initial_state == '1':
            k.gate("rx180", [q_rams])
        k.gate('wait', [], 0)

        ##############
        # X parity
        ##############
        # k.gate('wait', [], 0)
        # k.gate('cz', [11, 9])
        # k.gate('wait', [], 0)
        # k.gate('cz', [11, 8])
        # k.gate('rx180', [9])
        # k.gate('rx180', [9])
        # k.gate('i', [9])
        # k.gate('wait', [], 0)
        # k.gate('rx180', [11])
        # k.gate('wait', [], 0)
        # k.gate('cz', [11, 15])
        # k.gate('wait', [], 0)
        # k.gate('cz', [11, 14])
        # k.gate('wait', [], 0)
        # k.gate('correction', [q_rams])
        # k.gate('correction', [q_meas])
        # k.gate("wait", [], 0)

        ##############
        # Z parity
        ##############
        k.gate("wait", [], 0)
        k.gate('cz', [12,  9]) # without parking
        k.gate('cz', [ 8, 10])
        k.gate("wait", [], 0)
        k.gate('cz', [15, 12])
        k.gate('cz', [14, 10])
        k.gate("wait", [], 0)
        k.gate('ry90', [q_meas])
        k.gate('ry90', [q_rams])
        k.gate("wait", [], 0)


        k.measure(q_meas)

        if echo:
            k.gate('echo_corr', [q_rams])
            k.gate("wait", [q_rams], round(wait_time))
        k.gate("wait", [], 0)

        cw_idx = angle // 20 + 9
        k.gate('cw_{:02}'.format(cw_idx), [q_rams])
        k.measure(q_rams)

        p.add_kernel(k)

    # adding the calibration points
    oqh.add_single_qubit_cal_points(p, qubit_idx=q_rams,
                                    measured_qubits=[q_rams, q_meas])

    p.sweep_points = np.concatenate((np.repeat(angles, nr_shots), [360, 370, 380, 390]))
    p.set_sweep_points(p.sweep_points)

    p = oqh.compile(p)
    return p



def Ramsey_cross_3(q_rams: int,
                 q_meas: int,
                 platf_cfg: str,
                 echo: bool = True,
                 wait_time: int= 0,
                 angles: list = np.arange(0,360, 20),
                 nr_shots: int=2,
                 initial_state: str = '0'):
    """
    q_target is ramseyed
    q_spec is measured
    calibrate phase of echo for pipelined surface-7

    """
    p = oqh.create_program("Ramsey_msmt_induced_shift", platf_cfg)

    for i, angle in enumerate(angles[:]):

        k = oqh.create_kernel("Ramsey_azi_"+str(angle), p)

        k.prepz(q_rams)
        k.prepz(q_meas)
        k.gate("wait", [], 0)

        if initial_state == '1':
            k.gate("rx180", [q_rams])
        k.gate('wait', [], 0)

        ########################################################
        # Pipelined
        ########################################################
        ###############
        # X-Parity
        ###############
        # k.measure(4) # Qubit Z1_trig
        # k.measure(6) # Qubit Z2_trig
        k.measure(1) # Qubit X_trig
        k.gate('wait', [1, 4, 6], 280)
        k.gate('rym90', [11])
        k.gate('wait', [8, 9, 14, 15, 10, 11, 12], 0)
        k.gate('cz', [11, 9]) # CZ X-D2
        k.gate('wait', [8, 9, 14, 15, 10, 11, 12], 0)
        k.gate('cz', [11, 8]) # CZ X-D1
        k.gate('rx180', [9])  # echo D2
        k.gate('rx180', [9])  # echo D2
        k.gate('i', [9])
        k.gate('wait', [8, 9, 14, 15, 10, 11, 12], 0)
        k.gate('rx180', [11])   # echo ancilla
        k.gate('wait', [8, 9, 14, 15, 10, 11, 12], 0)
        k.gate('cz', [11, 15]) # CZ X-D4
        k.gate('wait', [8, 9, 14, 15, 10, 11, 12], 0)
        k.gate('cz', [11, 14]) # CZ X-D3
        k.gate('wait', [8, 9, 14, 15, 10, 11, 12], 0)
        k.gate('correction', [11])
        k.gate('ry90', [q_rams])
        k.gate('wait', [8, 9, 14, 15, 10, 11, 12], 0)

        ##############
        # Echo
        ##############
        k.gate('wait', [8, 9, 14, 15, 10, 12], 0)
        k.gate("wait", [q_rams], wait_time)
        k.gate('echo_corr', [q_rams])
        k.gate("wait", [q_rams], 360-20-wait_time)
        k.gate('wait', [8, 9, 14, 15, 10, 12], 0)
        k.gate('i', [11])
        k.gate('sf_park', [11])
        k.gate('sf_park', [11])
        k.gate('sf_park', [11])
        k.gate('sf_park', [11])
        k.gate('sf_park', [11])
        k.gate('sf_park', [11])


        ###############
        # Z-Parity
        ###############
        k.gate('wait', [8, 9, 14, 15, 10, 12], 0)
        # k.gate('rym90', [10])
        # k.gate('rym90', [12])
        k.gate('i', [12])
        k.gate("wait", [8, 9, 14, 15, 10, 11, 12], 0)
        k.gate('cz', [12, 9]) # without parking
        k.gate('cz', [8, 10])
        k.gate("wait", [8, 9, 14, 15, 10, 11, 12], 0)
        k.gate('cz', [15, 12])
        k.gate('cz', [14, 10])
        k.gate("wait", [8, 9, 14, 15, 10, 11, 12], 0)
        # k.gate('ry90', [10])
        # k.gate('ry90', [12])
        k.gate('i', [12])

        cw_idx = angle // 20 + 9
        k.gate('cw_{:02}'.format(cw_idx), [q_rams])
        k.gate("wait", [], 0)

        k.measure(q_rams)

        p.add_kernel(k)

    # adding the calibration points
    oqh.add_single_qubit_cal_points(p, qubit_idx=q_rams,
                                    measured_qubits=[q_rams, q_meas])

    p.sweep_points = np.concatenate((np.repeat(angles, nr_shots), [360, 370, 380, 390]))
    p.set_sweep_points(p.sweep_points)

    p = oqh.compile(p)
    return p



def TEST_RTE(QX:int , QZ:int, platf_cfg: str,
             measurements:int):
    """

    """
    p = oqh.create_program('Multi_RTE', platf_cfg)

    k = oqh.create_kernel('Multi_RTE', p)
    k.prepz(QX)
    k.prepz(QZ)
    ######################
    # Parity check
    ######################
    for m in range(measurements):
        # Superposition
        k.gate('ry90', [QX])
        k.gate('i', [QZ])
        # CZ emulation
        k.gate('i', [QZ, QX])
        k.gate('i', [QZ, QX])
        k.gate('i', [QZ, QX])
        # CZ emulation
        k.gate('i', [QZ, QX])
        k.gate('i', [QZ, QX])
        k.gate('i', [QZ, QX])
        # intermidate sequential
        k.gate('rym90', [QX])
        k.gate('i', [QZ])
        k.gate('i', [QX])
        k.gate('ry90', [QZ])
        # CZ emulation
        k.gate('i', [QZ, QX])
        k.gate('i', [QZ, QX])
        k.gate('i', [QZ, QX])
        # CZ emulation
        k.gate('i', [QZ, QX])
        k.gate('i', [QZ, QX])
        k.gate('i', [QZ, QX])
        # Recovery pulse
        k.gate('i', [QX])
        k.gate('rym90', [QZ])
        # Measurement
        k.measure(QX)
        k.measure(QZ)

    p.add_kernel(k)

    p = oqh.compile(p)
    return p




def off_on_fluxed(qubit_idx: int,
                  qubit_dummy_idx: int,
                  pulse_comb: str,
                  platf_cfg: str):
    """
    Performs an 'off_on' sequence on the qubit specified.
        off: (RO) - prepz -      - RO
        on:  (RO) - prepz - x180 - RO
    Args:
        qubit_idx (int) :
        pulse_comb (list): What pulses to play valid options are
            "off", "on", "off_on"
        initialize (bool): if True does an extra initial measurement to
            post select data.
        platf_cfg (str) : filepath of OpenQL platform config file

    Pulses can be optionally enabled by putting 'off', respectively 'on' in
    the pulse_comb string.
    """
    p = oqh.create_program('off_on_fluxed', platf_cfg)

    # # Off
    if 'off' in pulse_comb.lower():
        k = oqh.create_kernel("off", p)
        k.prepz(qubit_idx)
        k.gate('wait', [], 0)
        k.measure(qubit_dummy_idx)
        k.gate('sf_park', [qubit_idx])
        k.gate('wait', [qubit_idx], 120)
        p.add_kernel(k)

    if 'on' in pulse_comb.lower():
        k = oqh.create_kernel("on", p)
        k.prepz(qubit_idx)
        k.gate('rx180', [qubit_idx])
        k.gate('wait', [], 0)
        k.measure(qubit_dummy_idx)
        k.gate('sf_park', [qubit_idx])
        k.gate('wait', [qubit_idx], 120)
        p.add_kernel(k)

    if ('on' not in pulse_comb.lower()) and ('off' not in pulse_comb.lower()):
        raise ValueError()

    p = oqh.compile(p)
    return p