from pycqed.utilities.general import mopen
from os.path import join, dirname
import numpy as np
base_qasm_path = join(dirname(__file__), 'qasm_files')

from pycqed.measurement.waveform_control_CC.multi_qubit_qasm_seqs \
    import cal_points_2Q


def ramZ_flux_latency(q0_name, wait_after_flux=20):
    """
    Sequence designed to calibrate the delay between the
    QWG_trigger and the start of the flux pulse

    Consists of a single point. Intended to change the latency parameter
    in the configuration that is used in compilation.
    """
    filename = join(base_qasm_path, 'RamZ_latency_seq.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(q0_name))

    # simultaneous MW and flux pulse
    qasm_file.writelines('\ninit_all\n')
    qasm_file.writelines('X90 {} \n'.format(q0_name))
    qasm_file.writelines('square {}\n'.format(q0_name))
    qasm_file.writelines('I {}\n'.format(wait_after_flux))
    qasm_file.writelines('X90 {}\n'.format(q0_name))
    qasm_file.writelines('RO {} \n'.format(q0_name))

    qasm_file.close()
    return qasm_file


def chevron_block_seq(q0_name, q1_name, no_of_points,
                      excite_q1=False, wait_after_trigger=40e-9,
                      wait_during_flux=400e-9, clock_cycle=1e-9,
                      RO_target='all', mw_pulse_duration=40e-9,
                      cal_points=True):
    '''
    N.B. this sequence has been edited for compatibility with the XFU compiler
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

        qasm_file.writelines('QWG_trigger_{} {}\n'.format(i, q0_name))
        if excite_q1:
            wait_after_trigger -= mw_pulse_duration

        qasm_file.writelines('X180 {}\n'.format(q0_name))
        if excite_q1:
            qasm_file.writelines('X180 {}\n'.format(q1_name))
        qasm_file.writelines('CZ {} {}\n'.format(q0_name, q1_name))

        if excite_q1:
            # q0 is rotated to ground-state to have better contrast
            # (|0> and |2> instead of |1> and |2>)
            qasm_file.writelines('X180 {}\n'.format(q0_name))
        if RO_target == 'all':
            qasm_file.writelines('RO {} | RO {} \n'.format(q0_name, q1_name))
        else:
            qasm_file.writelines('RO {} \n'.format(RO_target))

    if cal_points:
        # Add calibration pulses
        cal_pulses = []
        for seq in cal_points_2Q:
            cal_pulses += [[seq[0], seq[1], 'RO ' + RO_target + '\n']]

    qasm_file.close()
    return qasm_file


def SWAPN(q0_name, q1_name, nr_pulses: list,
          excite_q1=False,
          RO_target='all',
          cal_points=True):
    '''
    Args:
        q0, q1        (str): names of the addressed qubits.
                             q0 is the pulse that experiences the flux pulse.
        RO_target     (str): can be q0, q1, or 'all'
        excite_q1    (bool): choose whether to excite q1, thus choosing
                             between the |01> <-> |10> and the |11> <-> |20>
                             swap.
        cal_points   (bool): whether to use calibration points or not
    '''
    filename = join(base_qasm_path, 'chevron_block_seq.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0_name, q1_name))

    for i, N in enumerate(nr_pulses):
        qasm_file.writelines('\ninit_all\n')
        qasm_file.writelines('QWG_trigger_{} {}\n'.format(i, q0_name))
        qasm_file.writelines('X180 {}\n'.format(q0_name))
        if excite_q1:
            qasm_file.writelines('X180 {}\n'.format(q1_name))
        for n in range(N):
            qasm_file.writelines('square {} \n'.format(q0_name))

        if excite_q1:
            # q0 is rotated to ground-state to have better contrast
            # (|0> and |2> instead of |1> and |2>)
            qasm_file.writelines('X180 {}\n'.format(q0_name))
        qasm_file.writelines('RO {} \n'.format(RO_target))

    if cal_points:
        # Add calibration pulses
        cal_pulses = []
        for seq in cal_points_2Q:
            cal_pulses += [[seq[0].format(q0_name) +
                            seq[1].format(q1_name) +
                            'RO {} \n'.format(RO_target)]]
    for seq in cal_pulses:
        qasm_file.writelines('\ninit_all\n')
        for p in seq:
            qasm_file.writelines(p)
    qasm_file.close()
    return qasm_file


def CZ_calibration_seq(q0, q1, RO_target='all',
                       vary_single_q_phase=True,
                       cases=('no_excitation', 'excitation')):
    '''
    Sequence used to calibrate flux pulses for CZ gates.

    Timing of the sequence:
    q0:   X90  C-Phase  Rphi90    RO
    q1: (X180)    --    (X180)    RO

    Args:
        q0, q1      (str): names of the addressed qubits
        RO_target   (str): can be q0, q1, or 'all'
        excitations (bool/str): can be True, False, or 'both_cases'
    '''

    filename = join(base_qasm_path, 'CZ_calibration_seq.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    for case in cases:
        qasm_file.writelines('\ninit_all\n')
        if case == 'excitation':
            qasm_file.writelines('X180 {} | '.format(q1))
        qasm_file.writelines('X90 {}\n'.format(q0))

        # temporary workaround to deal with limitation in the QASM config
        # qasm_file.writelines('CZ {} \n'.format(q0))
        qasm_file.writelines('CZ {} {}\n'.format(q0, q1))
        if case == 'excitation':
            qasm_file.writelines('X180 {} | '.format(q1))
        if vary_single_q_phase:
            qasm_file.writelines('Rphi90 {}\n'.format(q0))
        else:
            qasm_file.writelines('mX90 {}\n'.format(q0))
        if 'RO_target' == 'all':
            qasm_file.writelines('RO {} | RO {} \n'.format(q0, q1))
        else:
            qasm_file.writelines('RO {}  \n'.format(RO_target))
    qasm_file.close()
    return qasm_file


def two_qubit_tomo_bell(bell_state, q0, q1, RO_target='all'):
    '''
    Two qubit bell state tomography.

    Args:
        bell_state      (int): index of prepared bell state
                        0 : |00>-|11>
                        1 : |00>+|11>
                        2 : |01>-|10>
                        3 : |01>+|10>
        q0, q1          (str): names of the target qubits
        RO_target   (str): can be q0, q1, or 'all'
    '''

    if RO_target == 'all':
        # This is a bit of a hack as RO all qubits is the same instruction
        # as any specific qubit
        RO_target = q0

    tomo_pulses = ['I ', 'X180 ', 'Y90 ', 'mY90 ', 'X90 ', 'mX90 ']
    tomo_list_q0 = []
    tomo_list_q1 = []
    for tp in tomo_pulses:
        tomo_list_q0 += [tp + q0]
        tomo_list_q1 += [tp + q1]

    tomo_list_q0[0] = 'I {}'.format(q0)
    tomo_list_q1[0] = 'I {}'.format(q1)

    # Choose a bell state and set the corresponding preparation pulses
    if bell_state % 10 == 0:  # |Phi_m>=|00>-|11>
        prep_pulse_q0 = 'Y90 {}'.format(q0)
        prep_pulse_q1 = 'Y90 {}'.format(q1)
    elif bell_state % 10 == 1:  # |Phi_p>=|00>+|11>
        prep_pulse_q0 = 'mY90 {}'.format(q0)
        prep_pulse_q1 = 'Y90 {}'.format(q1)
    elif bell_state % 10 == 2:  # |Psi_m>=|01>-|10>
        prep_pulse_q0 = 'Y90 {}'.format(q0)
        prep_pulse_q1 = 'mY90 {}'.format(q1)
    elif bell_state % 10 == 3:  # |Psi_p>=|01>+|10>
        prep_pulse_q0 = 'mY90 {}'.format(q0)
        prep_pulse_q1 = 'mY90 {}'.format(q1)
    else:
        raise ValueError('Bell state {} is not defined.'.format(bell_state))

    after_pulse = 'mY90 {}\n'.format(q1)

    # Disable preparation pulse on one or the other qubit for debugging
    if bell_state//10 == 1:
        prep_pulse_q1 = 'I {}'.format(q0)
    elif bell_state//10 == 2:
        prep_pulse_q0 = 'I {}'.format(q1)

    # Write tomo sequence

    filename = join(base_qasm_path, 'two_qubit_tomo_bell.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    for p_q1 in tomo_list_q1:
        for p_q0 in tomo_list_q0:
            qasm_file.writelines('\ninit_all\n')
            qasm_file.writelines('{} | {} \n'.format(prep_pulse_q0,
                                                     prep_pulse_q1))
            qasm_file.writelines('CZ {} {} \n'.format(q0, q1))
            qasm_file.writelines(after_pulse)
            qasm_file.writelines('{} | {}\n'.format(p_q1, p_q0))
            qasm_file.writelines('RO ' + RO_target + '  \n')

    # Add calibration pulses
    cal_pulses = []
    # every calibration point is repeated 7 times. This is copied from the
    # script for Tektronix driven qubits. I do not know if this repetition
    # is important or even necessary here.
    for seq in cal_points_2Q:
        cal_pulses += [[seq[0].format(q0), seq[1].format(q1),
                        'RO ' + RO_target + '\n']] * 7

    for seq in cal_pulses:
        qasm_file.writelines('\ninit_all\n')
        for p in seq:
            qasm_file.writelines(p)

    qasm_file.close()
    return qasm_file


def grover_seq(q0_name, q1_name, RO_target='all',
               precompiled_flux=True, cal_points: bool=True):
    '''
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
    '''
    if not precompiled_flux:
        raise NotImplementedError('Currently only precompiled flux pulses '
                                  'are supported.')

    filename = join(base_qasm_path, 'Grover_seq.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(q0_name))
    qasm_file.writelines('qubit {} \n'.format(q1_name))

    if RO_target == 'all':
        RO_line = 'RO {} | RO {}\n'.format(q0_name, q1_name)
    else:
        RO_line = 'RO {} \n'.format(RO_target)

    for G1 in ['Y90', 'mY90']:
        for G0 in ['Y90', 'mY90']:
            qasm_file.writelines('\ninit_all\n')
            qasm_file.writelines('{} {} | {} {}\n'.format(G0, q0_name,
                                                          G1, q1_name))
            qasm_file.writelines('grover_CZ {} {}\n'.format(q0_name, q1_name))
            qasm_file.writelines('Y90 {} | Y90 {}\n'.format(q0_name,
                                                            q1_name))
            qasm_file.writelines('cz {} {}\n'.format(q0_name, q1_name))
            qasm_file.writelines('Y90 {} | Y90 {}\n'.format(q0_name,
                                                            q1_name))

            qasm_file.writelines(RO_line)

    # Add calibration points
    if cal_points:
        cal_pulses = []
        for seq in cal_points_2Q:
            cal_pulses += [[seq[0].format(q0_name), seq[1].format(q1_name),
                            RO_line]]

        for seq in cal_pulses:
            qasm_file.writelines('\ninit_all\n')
            for p in seq:
                qasm_file.writelines(p)

    qasm_file.close()
    return qasm_file


def grover_tomo_seq(q0_name, q1_name, omega, RO_target='all',
                    precompiled_flux=True):
    '''
    Writes the QASM sequence to take a state tomography of the output state
    of Grover's algorithm on two qubits.
    Sequence:
        q0: G0 -       - mY90 -    - mY90  - RO
                 CZ_ij          CZ
        q1: G1 -       - mY90 -    - mY90  - RO
    where (ij) is the binary representation of omega.
    G0 and G1 are Y90 or Y90, depending on the (ij).

    Args:
        q0_name, q1_name (string):
                Names of the qubits to which the sequence is applied.
        omega (int):
                Deterines which (ij) for the CZ_ij.
        RO_target (string):
                Readout target. Can be a qubit name or 'all'.
        precompiled_flux (bool):
                Determies if the full waveform for the flux pulses is
                precompiled, thus only needing one trigger at the start,
                or if every flux pulse should be triggered individually.

    Returns:
        qasm_file: a reference to the new QASM file object.
    '''
    if not precompiled_flux:
        raise NotImplementedError('Currently only precompiled flux pulses '
                                  'are supported.')

    tomo_pulses = ['I ', 'X180 ', 'Y90 ', 'mY90 ', 'X90 ', 'mX90 ']
    tomo_list_q0 = []
    tomo_list_q1 = []
    for tp in tomo_pulses:
        tomo_list_q0 += [tp + q0_name]
        tomo_list_q1 += [tp + q1_name]

    if omega == 0:
        G0 = 'Y90'
        G1 = 'Y90'
    elif omega == 1:
        G0 = 'mY90'
        G1 = 'Y90'
    elif omega == 2:
        G0 = 'Y90'
        G1 = 'mY90'
    elif omega == 3:
        G0 = 'mY90'
        G1 = 'mY90'
    else:
        raise ValueError('omega must be in [0, 3]')

    if RO_target == 'all':
        RO_line = 'RO {} | RO {}\n'.format(q0_name, q1_name)
    else:
        RO_line = 'RO {} \n'.format(RO_target)

    filename = join(base_qasm_path, 'Grover_tomo_seq.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(q0_name))
    qasm_file.writelines('qubit {} \n'.format(q1_name))

    for p_q1 in tomo_list_q1:
        for p_q0 in tomo_list_q0:
            qasm_file.writelines('\ninit_all\n')
            qasm_file.writelines('{} {} | {} {}\n'.format(G0, q0_name,
                                                          G1, q1_name))
            qasm_file.writelines('grover_CZ {} {}\n'.format(q0_name, q1_name))
            qasm_file.writelines('Y90 {} | Y90 {}\n'.format(q0_name,
                                                            q1_name))
            qasm_file.writelines('cz {} {}\n'.format(q0_name, q1_name))
            qasm_file.writelines('Y90 {} | Y90 {}\n'.format(q0_name,
                                                            q1_name))

            qasm_file.writelines('{} | {}\n'.format(p_q1, p_q0))
            qasm_file.writelines(RO_line)

    # Add calibration pulses
    cal_pulses = []
    # every calibration point is repeated 7 times. This is copied from the
    # script for Tektronix driven qubits. I do not know if this repetition
    # is important or even necessary here.
    for seq in cal_points_2Q:
        cal_pulses += [[seq[0].format(q0_name), seq[1].format(q1_name),
                        RO_line]] * 7

    for seq in cal_pulses:
        qasm_file.writelines('\ninit_all\n')
        for p in seq:
            qasm_file.writelines(p)

    qasm_file.close()
    return qasm_file


def grover_test_seq(q0_name, q1_name, RO_target='all',
                    precompiled_flux=True, cal_points: bool=True):
    '''
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
    '''
    if not precompiled_flux:
        raise NotImplementedError('Currently only precompiled flux pulses '
                                  'are supported.')

    filename = join(base_qasm_path, 'Grover_seq.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(q0_name))
    qasm_file.writelines('qubit {} \n'.format(q1_name))

    if RO_target == 'all':
        RO_line = 'RO {} | RO {}\n'.format(q0_name, q1_name)
    else:
        RO_line = 'RO {} \n'.format(RO_target)

    for G1 in ['Y90', 'mY90']:
        for G0 in ['Y90', 'mY90']:
            qasm_file.writelines('\ninit_all\n')
            qasm_file.writelines('{} {} | {} {}\n'.format(G0, q0_name,
                                                          G1, q1_name))
            qasm_file.writelines('grover_CZ {} {}\n'.format(q0_name, q1_name))
            qasm_file.writelines('Y90 {} | Y90 {}\n'.format(q0_name,
                                                              q1_name))
            # qasm_file.writelines('cz {} {}\n'.format(q0_name, q1_name))
            # qasm_file.writelines('Y90 {} | Y90 {}\n'.format(q0_name,
            #                                                   q1_name))

            qasm_file.writelines(RO_line)

    # Add calibration points
    if cal_points:
        cal_pulses = []
        for seq in cal_points_2Q:
            cal_pulses += [[seq[0].format(q0_name), seq[1].format(q1_name),
                            RO_line]]

        for seq in cal_pulses:
            qasm_file.writelines('\ninit_all\n')
            for p in seq:
                qasm_file.writelines(p)

    qasm_file.close()
    return qasm_file


def grover_test_tomo_seq(q0_name, q1_name, omega, RO_target='all',
                         precompiled_flux=True):
    '''
    Test sequence to debug Grover's algorithm.
    '''
    if not precompiled_flux:
        raise NotImplementedError('Currently only precompiled flux pulses '
                                  'are supported.')

    tomo_pulses = ['I ', 'X180 ', 'Y90 ', 'mY90 ', 'X90 ', 'mX90 ']
    tomo_list_q0 = []
    tomo_list_q1 = []
    for tp in tomo_pulses:
        tomo_list_q0 += [tp + q0_name]
        tomo_list_q1 += [tp + q1_name]

    if omega == 0:
        G0 = 'Y90'
        G1 = 'Y90'
    elif omega == 1:
        G0 = 'mY90'
        G1 = 'Y90'
    elif omega == 2:
        G0 = 'Y90'
        G1 = 'mY90'
    elif omega == 3:
        G0 = 'mY90'
        G1 = 'mY90'
    else:
        raise ValueError('omega must be in [0, 3]')

    if RO_target == 'all':
        RO_line = 'RO {} | RO {}\n'.format(q0_name, q1_name)
    else:
        RO_line = 'RO {} \n'.format(RO_target)

    filename = join(base_qasm_path, 'Grover_tomo_seq.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(q0_name))
    qasm_file.writelines('qubit {} \n'.format(q1_name))

    for p_q1 in tomo_list_q1:
        for p_q0 in tomo_list_q0:
            qasm_file.writelines('\ninit_all\n')
            qasm_file.writelines('{} {} | {} {}\n'.format(G0, q0_name,
                                                          G1, q1_name))
            qasm_file.writelines('grover_cz {} {}\n'.format(q0_name, q1_name))
            qasm_file.writelines('Y90 {} | Y90 {}\n'.format(q0_name,
                                                            q1_name))
            qasm_file.writelines('cz {} {}\n'.format(q0_name, q1_name))
            # qasm_file.writelines('Y90 {} | Y90 {}\n'.format(q0_name,
                                                              # q1_name))
            qasm_file.writelines('{} | {}\n'.format(p_q1, p_q0))
            qasm_file.writelines(RO_line)

    # Add calibration pulses
    cal_pulses = []
    # every calibration point is repeated 7 times. This is copied from the
    # script for Tektronix driven qubits. I do not know if this repetition
    # is important or even necessary here.
    for seq in cal_points_2Q:
        cal_pulses += [[seq[0].format(q0_name), seq[1].format(q1_name),
                        RO_line]] * 7

    for seq in cal_pulses:
        qasm_file.writelines('\ninit_all\n')
        for p in seq:
            qasm_file.writelines(p)

    qasm_file.close()
    return qasm_file


def purity_CZ_seq(q0, q1, RO_target='all'):
    """
    Creates the |00> + |11> Bell state and does a partial tomography in
    order to determine the purity of both qubits.
    """

    filename = join(base_qasm_path, 'purity_CZ_seq.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    tomo_list = ['mX90', 'mY90', 'I']

    for p_pulse in tomo_list:
        # Create a Bell state:  |00> + |11>
        qasm_file.writelines('\ninit_all\n')
        qasm_file.writelines('mY90 {} | Y90 {} \n'.format(q0, q1))
        qasm_file.writelines('CZ {} {} \n'.format(q0, q1))
        qasm_file.writelines('mY90 {}\n'.format(q1))

        # Perform pulses to measure the purity of both qubits
        qasm_file.writelines('{} {} | {} {}\n'.format(p_pulse, q0,
                                                      p_pulse, q1))
        if RO_target == 'all':
            qasm_file.writelines('RO {} | RO {} \n'.format(q0, q1))
        else:
            qasm_file.writelines('RO {} \n'.format(RO_target))

    qasm_file.close()
    return qasm_file


def purity_N_CZ_seq(q0, q1, N, RO_target='all'):
    """
    Creates the |00> + |11> Bell state and does a partial tomography in
    order to determine the purity of both qubits.
    """

    filename = join(base_qasm_path, 'purity_CZ_seq.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \nqubit {} \n'.format(q0, q1))

    tomo_list = ['mX90', 'mY90', 'I']

    for p_pulse in tomo_list:
        # Create a Bell state:  |00> + |11>
        qasm_file.writelines('\ninit_all\n')
        qasm_file.writelines('mY90 {} | Y90 {} \n'.format(q0, q1))
        for n in range(N):
            qasm_file.writelines('dummy_CZ {} {} \n'.format(q0, q1))
        qasm_file.writelines('mY90 {}\n'.format(q1))

        # Perform pulses to measure the purity of both qubits
        qasm_file.writelines('{} {} | {} {}\n'.format(p_pulse, q0,
                                                      p_pulse, q1))
        if RO_target == 'all':
            qasm_file.writelines('RO {} | RO {} \n'.format(q0, q1))
        else:
            qasm_file.writelines('RO {} \n'.format(RO_target))

    qasm_file.close()
    return qasm_file

