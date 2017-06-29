from pycqed.utilities.general import mopen
from os.path import join, dirname
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
                       cases=('no_excitation', 'excitation')):
    '''
    Sequence used to calibrate flux pulses for CZ gates.

    Timing of the sequence:
    q0:   --   X90  C-Phase  Rphi90   --      RO
    q1: (X180)  --     --       --   (X180)    RO

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
            qasm_file.writelines('X180 {}\n'.format(q1))
        qasm_file.writelines('X90 {}\n'.format(q0))

        # temporary workaround to deal with limitation in the QASM config
        # qasm_file.writelines('CZ {} \n'.format(q0))
        qasm_file.writelines('CZ {} {}\n'.format(q0, q1))
        qasm_file.writelines('Rphi90 {}\n'.format(q0))
        if case == 'excitation':
            qasm_file.writelines('X180 {}\n'.format(q1))
        if 'RO_target' == 'all':
            qasm_file.writelines('RO {} | RO {} \n'.format(q0, q1))
        else:
            qasm_file.writelines('RO {}  \n'.format(RO_target))
    qasm_file.close()
    return qasm_file