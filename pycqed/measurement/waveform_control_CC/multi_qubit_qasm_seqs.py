from pycqed.utilities.general import mopen
from os.path import join, dirname
base_qasm_path = join(dirname(__file__), 'qasm_files')

#  Calibration pulses for all two qubit sequences globally defined
cal_points_2Q = [['', ''],
                 ['X180 {}\n', ''],
                 ['', 'X180 {}\n'],
                 ['X180 {}\n', 'X180 {}\n']]


def two_qubit_off_on(q0, q1, RO_target='all'):
    '''
    off_on sequence on two qubits.

    Args:
        q0, q1      (str) : target qubits for the sequence
        RO_target   (str) : target for the RO, can be a qubit name or 'all'
    '''
    filename = join(base_qasm_path, 'two_qubit_off_on.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    # off - off
    qasm_file.writelines('\ninit_all\n')
    if RO_target == 'all':
        qasm_file.writelines('RO {} | RO {} \n'.format(q0, q1))
    else:
        qasm_file.writelines('RO {}  \n'.format(RO_target))

    # on - off
    qasm_file.writelines('\ninit_all\n')
    qasm_file.writelines('X180 {}\n'.format(q0))
    if RO_target == 'all':
        qasm_file.writelines('RO {} | RO {} \n'.format(q0, q1))
    else:
        qasm_file.writelines('RO {}  \n'.format(RO_target))

    # off - on
    qasm_file.writelines('\ninit_all\n')
    qasm_file.writelines('X180 {}\n'.format(q1))
    if RO_target == 'all':
        qasm_file.writelines('RO {} | RO {} \n'.format(q0, q1))
    else:
        qasm_file.writelines('RO {}  \n'.format(RO_target))

    # on - on
    qasm_file.writelines('\ninit_all\n')
    qasm_file.writelines('X180 {}\n'.format(q0))
    qasm_file.writelines('X180 {}\n'.format(q1))
    if RO_target == 'all':
        qasm_file.writelines('RO {} | RO {} \n'.format(q0, q1))
    else:
        qasm_file.writelines('RO {}  \n'.format(RO_target))

    qasm_file.close()
    return qasm_file


def two_qubit_tomo_cardinal(cardinal,
                            q0,
                            q1,
                            RO_target='all'):
    '''
    Cardinal tomography for two qubits.

    Args:
        cardinal        (int) : index of prep gate
        q0, q1          (str) : target qubits for the sequence
        RO_target       (str) : target for the RO, can be a qubit name or 'all'
    '''
    tomo_pulses = ['I ', 'X180 ', 'Y90 ', 'mY90 ', 'X90 ', 'mX90 ']
    tomo_list_q0 = []
    tomo_list_q1 = []
    for tp in tomo_pulses:
        tomo_list_q0 += [tp + q0 + '\n']
        tomo_list_q1 += [tp + q1 + '\n']

    prep_index_q0 = int(cardinal % len(tomo_list_q0))
    prep_index_q1 = int(((cardinal - prep_index_q0) / len(tomo_list_q0) %
                         len(tomo_list_q1)))

    prep_pulse_q0 = tomo_list_q0[prep_index_q0]
    prep_pulse_q1 = tomo_list_q1[prep_index_q1]

    filename = join(base_qasm_path, 'two_qubit_tomo_cardinal.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    # Tomography pulses
    for p_q1 in tomo_list_q1:
        for p_q0 in tomo_list_q0:
            qasm_file.writelines('\ninit_all\n')
            qasm_file.writelines(prep_pulse_q0)
            qasm_file.writelines(prep_pulse_q1)
            qasm_file.writelines(p_q0)
            qasm_file.writelines(p_q1)
            qasm_file.writelines('RO ' + RO_target + '  \n')
    cal_pulses = []
    # every calibration point is repeated 7 times. This is copied from the
    # script for Tektronix driven qubits. I do not know if this repetition
    # is important or even necessary here.
    for seq in cal_points_2Q:
        cal_pulses += [[seq[0].format(q0) +
                        seq[1].format(q1) +
                        'RO {} \n'.format(RO_target)]]

    for seq in cal_pulses:
        qasm_file.writelines('\ninit_all\n')
        for p in seq:
            qasm_file.writelines(p)

    qasm_file.close()
    return qasm_file


def two_qubit_AllXY(q0, q1, RO_target='all',
                    sequence_type='sequential',
                    replace_q1_pulses_X180=False,
                    double_points=False):
    """
    AllXY sequence on two qubits.
    Has the option of replacing pulses on q1 with pi pulses

    Args:
        q0, q1         (str) : target qubits for the sequence
        RO_target      (str) : target for the RO, can be a qubit name or 'all'
        sequence_type  (str) : sequential | interleaved | simultaneous | sandwiched
                              q0|q0|q1|q1   q0|q1|q0|q1   q01|q01      q1|q0|q0|q1
                            N.B.!  simultaneous is currently not possible!
            describes the order of the AllXY pulses
        replace_q1_pulses_X180 (bool) : if True replaces all pulses on q1 with
            X180 pulses.

        double_points (bool) : if True measures each point in the AllXY twice
    """

    pulse_combinations = [['I', 'I'], ['X180', 'X180'], ['Y180', 'Y180'],
                          ['X180', 'Y180'], ['Y180', 'X180'],
                          ['X90', 'I'], ['Y90', 'I'], ['X90', 'Y90'],
                          ['Y90', 'X90'], ['X90', 'Y180'], ['Y90', 'X180'],
                          ['X180', 'Y90'], ['Y180', 'X90'], ['X90', 'X180'],
                          ['X180', 'X90'], ['Y90', 'Y180'], ['Y180', 'Y90'],
                          ['X180', 'I'], ['Y180', 'I'], ['X90', 'X90'],
                          ['Y90', 'Y90']]

    pulse_combinations_tiled = pulse_combinations + pulse_combinations
    if double_points:
        pulse_combinations = [val for val in pulse_combinations
                              for _ in (0, 1)]

    if replace_q1_pulses_X180:
        pulse_combinations_q1 = ['X180' for val in pulse_combinations]

    pulse_combinations_q0 = pulse_combinations
    pulse_combinations_q1 = pulse_combinations_tiled


    filename = join(base_qasm_path, 'two_qubit_AllXY.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    for pulse_comb_q0, pulse_comb_q1 in zip(pulse_combinations_q0,
                                            pulse_combinations_q1):
        qasm_file.writelines('\ninit_all\n')
        if sequence_type == 'interleaved':
            qasm_file.writelines('{} {}\n'.format(pulse_comb_q0[0], q0) +
                                 '{} {}\n'.format(pulse_comb_q1[0], q1) +
                                 '{} {}\n'.format(pulse_comb_q0[1], q0) +
                                 '{} {}\n'.format(pulse_comb_q1[1], q1))
        elif sequence_type == 'sandwiched':
            qasm_file.writelines('{} {}\n'.format(pulse_comb_q1[0], q1) +
                                 '{} {}\n'.format(pulse_comb_q0[0], q0) +
                                 '{} {}\n'.format(pulse_comb_q0[1], q0) +
                                 '{} {}\n'.format(pulse_comb_q1[1], q1))
        elif sequence_type == 'sequential':
            qasm_file.writelines('{} {}\n'.format(pulse_comb_q0[0], q0) +
                                 '{} {}\n'.format(pulse_comb_q0[1], q0) +
                                 '{} {}\n'.format(pulse_comb_q1[0], q1) +
                                 '{} {}\n'.format(pulse_comb_q1[1], q1))
        elif sequence_type == 'simultaneous':
            qasm_file.writelines('{} {} |'.format(pulse_comb_q0[0], q0) +
                                 '{} {}\n'.format(pulse_comb_q1[0], q1) +
                                 '{} {} |'.format(pulse_comb_q0[1], q0) +
                                 '{} {}\n'.format(pulse_comb_q1[1], q1))
        else:
            raise ValueError("sequence_type {} ".format(sequence_type) +
                             "['interleaved', 'simultaneous', " +
                             "'sequential', 'sandwiched']")
        qasm_file.writelines('RO {}  \n'.format(RO_target))

    qasm_file.close()
    return qasm_file


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
    filename = join(base_qasm_path, 'chevron_seq.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(fluxing_qubit,
                                                         spectator_qubit))

    qasm_file.writelines('\ninit_all\n')
    if excite_q1:
        qasm_file.writelines('X180 {} | X180 {}\n'.format(fluxing_qubit,
                                                          spectator_qubit))
    else:
        qasm_file.writelines('X180 {}\n'.format(fluxing_qubit))
    qasm_file.writelines('square {}\n'.format(fluxing_qubit))
    if excite_q1:
        # fluxing_qubit is rotated to ground-state to have better contrast
        # (|0> and |2> instead of |1> and |2>)
        qasm_file.writelines('X180 {}\n'.format(fluxing_qubit))
    if RO_target == 'all':
        qasm_file.writelines('RO {} | RO {}\n'.format(fluxing_qubit,
                                                      spectator_qubit))
    else:
        qasm_file.writelines('RO {} \n'.format(RO_target))

    qasm_file.close()
    return qasm_file


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
    tomo_pulses = ['I ', 'X180 ', 'Y90 ', 'mY90 ', 'X90 ', 'mX90 ']
    tomo_list_q0 = []
    tomo_list_q1 = []
    for tp in tomo_pulses:
        tomo_list_q0 += [tp + q0 + '\n']
        tomo_list_q1 += [tp + q1 + '\n']

    tomo_list_q0[0] = 'I 20\n'.format(q0)
    tomo_list_q1[0] = 'I 20\n'.format(q1)

    # Choose a bell state and set the corresponding preparation pulses
    if bell_state % 10 == 0:  # |Phi_m>=|00>-|11>
        prep_pulse_q0 = 'Y90 {}\n'.format(q0)
        prep_pulse_q1 = 'Y90 {}\n'.format(q1)
    elif bell_state % 10 == 1:  # |Phi_p>=|00>+|11>
        prep_pulse_q0 = 'mY90 {}\n'.format(q0)
        prep_pulse_q1 = 'Y90 {}\n'.format(q1)
    elif bell_state % 10 == 2:  # |Psi_m>=|01>-|10>
        prep_pulse_q0 = 'Y90 {}\n'.format(q0)
        prep_pulse_q1 = 'mY90 {}\n'.format(q1)
    elif bell_state % 10 == 3:  # |Psi_p>=|01>+|10>
        prep_pulse_q0 = 'mY90 {}\n'.format(q0)
        prep_pulse_q1 = 'mY90 {}\n'.format(q1)
    else:
        raise ValueError('Bell state {} is not defined.'.format(bell_state))

    # Recovery pulse is the same for all Bell states
    if single_qubit_compiled_phase == False:
        after_pulse = 'mY90 {}\n'.format(q1)
    else:
        after_pulse = 'recmY90 {}\n'.format(q1)

    # Disable preparation pulse on one or the other qubit for debugging
    if bell_state//10 == 1:
        prep_pulse_q1 = 'I 20'
    elif bell_state//10 == 2:
        prep_pulse_q0 = 'I 20'

    # Define compensation pulses
    # FIXME: needs to be added
    print('Warning: not using compensation pulses.')

    # Write tomo sequence

    filename = join(base_qasm_path, 'two_qubit_tomo_bell.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    for p_q1 in tomo_list_q1:
        for p_q0 in tomo_list_q0:
            qasm_file.writelines('\ninit_all\n')
            qasm_file.writelines('QWG trigger\n')
            qasm_file.writelines(
                'I {}\n'.format(int(wait_after_trigger//clock_cycle)))
            qasm_file.writelines(prep_pulse_q0)
            qasm_file.writelines(prep_pulse_q1)
            qasm_file.writelines(
                'I {}\n'.format(int(wait_during_flux//clock_cycle)))
            qasm_file.writelines(after_pulse)
            qasm_file.writelines(p_q1)
            qasm_file.writelines(p_q0)
            qasm_file.writelines('RO ' + RO_target + '  \n')

    # Add calibration pulses
    cal_pulses = []
    # every calibration point is repeated 7 times. This is copied from the
    # script for Tektronix driven qubits. I do not know if this repetition
    # is important or even necessary here.
    for seq in cal_points_2Q:
        cal_pulses += [[seq[0].format(q0), seq[1].format(q1), 'RO ' + RO_target + '\n']] * 7

    for seq in cal_pulses:
        qasm_file.writelines('\ninit_all\n')
        for p in seq:
            qasm_file.writelines(p)

    qasm_file.close()
    return qasm_file


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

    filename = join(base_qasm_path, 'CZ_calibration_seq.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    for case in cases:
        qasm_file.writelines('\ninit_all\n')
        qasm_file.writelines('QWG trigger\n')
        waitTime = wait_after_trigger
        if case == 'excitation':
            # Decrease wait time because there is an additional pulse
            waitTime -= mw_pulse_duration
        qasm_file.writelines(
            'I {}\n'.format(int(waitTime//clock_cycle)))
        if case == 'excitation':
            qasm_file.writelines('X180 {}\n'.format(q1))
        qasm_file.writelines('X90 {}\n'.format(q0))
        qasm_file.writelines(
            'I {}\n'.format(int(wait_during_flux//clock_cycle)))
        qasm_file.writelines('Rphi90 {}\n'.format(q0))
        if case == 'excitation':
            qasm_file.writelines('X180 {}\n'.format(q1))

        qasm_file.writelines('RO {}  \n'.format(RO_target))

    qasm_file.close()
    return qasm_file


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
    filename = join(base_qasm_path, 'CZ_fast_calibration_seq.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0_name, q1_name))

    for i in range(no_of_points):

        if cal_points and (i == no_of_points - 4 or i == no_of_points - 3):
            # Calibration point for |0>
            qasm_file.writelines('\ninit_all\n')
            qasm_file.writelines('RO {}  \n'.format(RO_target))
            pass
        elif cal_points and (i == no_of_points - 2 or i == no_of_points - 1):
            # Calibration point for |1>
            qasm_file.writelines('\ninit_all\n')
            qasm_file.writelines('X180 {} \n'.format(q0_name))
            qasm_file.writelines('X180 {} \n'.format(q1_name))
            qasm_file.writelines('RO {}  \n'.format(RO_target))
        else:
            for case in cases:
                qasm_file.writelines('\ninit_all\n')
                qasm_file.writelines('QWG_trigger_{}\n'.format(i))
                waitTime = wait_after_trigger
                if case == 'excitation':
                    # Decrease wait time because there is an additional pulse
                    waitTime -= mw_pulse_duration
                qasm_file.writelines(
                    'I {}\n'.format(int(waitTime//clock_cycle)))
                if case == 'excitation':
                    qasm_file.writelines('X180 {}\n'.format(q1_name))
                qasm_file.writelines('mX90 {}\n'.format(q0_name))
                qasm_file.writelines(
                    'I {}\n'.format(int(wait_during_flux//clock_cycle)))
                qasm_file.writelines('X90 {}\n'.format(q0_name))
                if case == 'excitation':
                    qasm_file.writelines('X180 {}\n'.format(q1_name))

                qasm_file.writelines('RO {}  \n'.format(RO_target))

    qasm_file.close()
    return qasm_file


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
    filename = join(base_qasm_path, 'chevron_block_seq.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0_name, q1_name))

    for i in range(no_of_points):
        qasm_file.writelines('\ninit_all\n')

        qasm_file.writelines('QWG trigger {}\n'.format(i))
        if excite_q1:
            wait_after_trigger -= mw_pulse_duration
        qasm_file.writelines(
            'I {}\n'.format(int(wait_after_trigger//clock_cycle)))
        qasm_file.writelines('X180 {}\n'.format(q0_name))
        if excite_q1:
            qasm_file.writelines('X180 {}\n'.format(q1_name))
        qasm_file.writelines(
            'I {}\n'.format(int(wait_during_flux//clock_cycle)))
        if excite_q1:
            # q0 is rotated to ground-state to have better contrast
            # (|0> and |2> instead of |1> and |2>)
            qasm_file.writelines('X180 {}\n'.format(q0_name))
        qasm_file.writelines('RO {} \n'.format(RO_target))

    if cal_points:
        # Add calibration pulses
        cal_pulses = []
        for seq in cal_points_2Q:
            cal_pulses += [[seq[0], seq[1], 'RO ' + RO_target + '\n']]

    qasm_file.close()
    return qasm_file


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
    filename = join(base_qasm_path, 'CZ_state_cycling_light.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    U = ''
    U += 'Y90 {} | mY90 {}\n'.format(q0, q1)
    U += 'CZ {} {}\n'.format(q0, q1)
    U += 'Y90 {} | Y90 {}\n'.format(q0, q1)
    U += 'CZ {} {}\n'.format(q0, q1)
    U += 'Y90 {} | Y90 {}\n'.format(q0, q1)

    # Input |00>
    qasm_file.writelines('init_all \n')
    qasm_file.writelines('qwg_trigger_0 {}\n'.format(q0))
    for n in range(N):
        qasm_file.writelines(U)
    qasm_file.writelines('RO {}\n'.format(q0))

    # Input |01>
    qasm_file.writelines('init_all \n')
    qasm_file.writelines('qwg_trigger_0 {}\n'.format(q0))
    qasm_file.writelines('X180 {}\n'.format(q0))
    for n in range(N):
        qasm_file.writelines(U)
    qasm_file.writelines('RO {}\n'.format(q0))

    # Input |10>
    qasm_file.writelines('init_all \n')
    qasm_file.writelines('qwg_trigger_0 {}\n'.format(q0))
    qasm_file.writelines('X180 {}\n'.format(q1))
    for n in range(N):
        qasm_file.writelines(U)
    qasm_file.writelines('RO {}\n'.format(q0))

    # Input |11>
    qasm_file.writelines('init_all \n')
    qasm_file.writelines('qwg_trigger_0 {}\n'.format(q0))
    qasm_file.writelines('X180 {} | X180 {}\n'.format(q0, q1))
    for n in range(N):
        qasm_file.writelines(U)
    qasm_file.writelines('RO {}\n'.format(q0))

    qasm_file.close()
    return qasm_file


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
    filename = join(base_qasm_path, 'CZ_state_cycling_light.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    U = ''
    U += 'Y90 {} | mY90 {}\n'.format(q0, q1)
    U += 'CZ {} {}\n'.format(q0, q1)
    U += 'Y90 {} | Y90 {}\n'.format(q0, q1)
    U += 'CZ {} {}\n'.format(q0, q1)
    U += 'Y90 {} | Y90 {}\n'.format(q0, q1)

    for n in range(N):
        qasm_file.writelines(U)
    qasm_file.writelines('RO {}\n'.format(q0))
