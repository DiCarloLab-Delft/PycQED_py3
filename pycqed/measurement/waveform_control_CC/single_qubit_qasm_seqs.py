'''
Definitions of all the standard single qubit sequences
code that generates the required QASM files

Take a look at http://www.media.mit.edu/quanta/qasm2circ/ for qasm
'''
import numpy as np
from os.path import join, dirname

from pycqed.utilities.general import mopen
from pycqed.measurement.randomized_benchmarking import randomized_benchmarking as rb
base_qasm_path = join(dirname(__file__), 'qasm_files')


def CW_tone():
    filename = join(base_qasm_path, 'CW_tone.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('Pulse \n')
    qasm_file.close()
    return qasm_file


def CW_RO_sequence(qubit_name, trigger_separation, clock_cycle=1e-9):
    # N.B.! this delay is not correct because it does not take the
    # trigger length into account
    delay = np.round(trigger_separation/clock_cycle)

    filename = join(base_qasm_path, 'CW_RO_sequence.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    qasm_file.writelines('I {:d} \n'.format(
        int(delay)))
    qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file


def pulsed_spec_sequence(qubit_name, clock_cycle=1e-9):
    filename = join(base_qasm_path, 'pulsed_spec.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    qasm_file.writelines('SpecPulse {} \n'.format(
                         qubit_name))
    qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file


def T1(qubit_name, times, clock_cycle=1e-9,
       cal_points=True):
    #
    clocks = np.round(times/clock_cycle)
    filename = join(base_qasm_path, 'T1.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    for i, cl in enumerate(clocks):
        qasm_file.writelines('\ninit_all\n')
        if cal_points and (i == (len(clocks)-4) or
                           i == (len(clocks)-3)):
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
        elif cal_points and (i == (len(clocks)-2) or
                             i == (len(clocks)-1)):
            qasm_file.writelines('X180 {} \n'.format(qubit_name))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
        else:
            qasm_file.writelines('X180 {}     # exciting pi pulse\n'.format(
                                 qubit_name))
            qasm_file.writelines('I {:d} \n'.format(int(cl)))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file


def flipping_seq(qubit_name, number_of_flips, clock_cycle=1e-9,
                 equator=False, cal_points=True):
    filename = join(base_qasm_path, 'Flipping.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    for i, n in enumerate(number_of_flips):
        qasm_file.writelines('\ninit_all\n')
        if cal_points and (i == (len(number_of_flips)-4) or
                           i == (len(number_of_flips)-3)):
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
        elif cal_points and (i == (len(number_of_flips)-2) or
                             i == (len(number_of_flips)-1)):
            qasm_file.writelines('X180 {} \n'.format(qubit_name))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
        else:
            if equator:
                qasm_file.writelines('X90 {} \n'.format(qubit_name))
            for j in range(n):
                qasm_file.writelines('X180 {} \n'.format(
                                     qubit_name))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file


def AllXY(qubit_name, double_points=False):
    pulse_combinations = [['I', 'I'], ['X180', 'X180'], ['Y180', 'Y180'],
                          ['X180', 'Y180'], ['Y180', 'X180'],
                          ['X90', 'I'], ['Y90', 'I'], ['X90', 'Y90'],
                          ['Y90', 'X90'], ['X90', 'Y180'], ['Y90', 'X180'],
                          ['X180', 'Y90'], ['Y180', 'X90'], ['X90', 'X180'],
                          ['X180', 'X90'], ['Y90', 'Y180'], ['Y180', 'Y90'],
                          ['X180', 'I'], ['Y180', 'I'], ['X90', 'X90'],
                          ['Y90', 'Y90']]
    if double_points:
        pulse_combinations = [val for val in pulse_combinations
                              for _ in (0, 1)]

    filename = join(base_qasm_path, 'AllXY.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))

    for pulse_comb in pulse_combinations:
        qasm_file.writelines('\ninit_all\n')
        if pulse_comb[0] != 'I':
            qasm_file.writelines('{} {}\n'.format(pulse_comb[0], qubit_name))
        if pulse_comb[1] != 'I':
            qasm_file.writelines('{} {}\n'.format(pulse_comb[1], qubit_name))
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file


def Rabi(qubit_name, amps, n=1):
    filename = join(base_qasm_path, 'Rabi_{}.qasm'.format(n))
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    for amp in amps:
        qasm_file.writelines('\ninit_all\n')
        for i in range(n):
            qasm_file.writelines('Rx {} {} \n'.format(qubit_name, amp))
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file


def Ramsey(qubit_name, times, clock_cycle=1e-9,
           artificial_detuning=None,
           cal_points=True):
    '''
    Ramsey sequence for a single qubit.
    Input pars:
        times:               array of times between (start of) pulses (s)
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters
        artificial_detuning: int,  float or None;
            if int: number of wiggles
            if float: artificial_detuning in (Hz)
            if None: adds no artificial detuning
                implemented using phase of the second pi/2 pulse (R90_phi),
                if None it will use X90 as the recovery pulse
        cal_points:          whether to use calibration points or not
    '''
    # if int interpret artificial detuning as desired nr of wiggles
    if isinstance(artificial_detuning, int):
        phases = (360*np.arange(len(times))/(len(times)-4*cal_points) *
                  artificial_detuning % 360)
    # if float interpret it as artificial detuning in Hz
    elif isinstance(artificial_detuning, float):
        phases = (artificial_detuning*times % 1)*360
    elif artificial_detuning is None:
        phases = np.zeros(len(times))

    clocks = np.round(times/clock_cycle)
    filename = join(base_qasm_path, 'Ramsey.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    for i, cl in enumerate(clocks):
        qasm_file.writelines('\ninit_all\n')
        if cal_points and (i == (len(clocks)-4) or
                           i == (len(clocks)-3)):
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
        elif cal_points and (i == (len(clocks)-2) or
                             i == (len(clocks)-1)):
            qasm_file.writelines('X180 {} \n'.format(qubit_name))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))

        else:
            qasm_file.writelines('X90 {}     \n'.format(
                                 qubit_name))
            qasm_file.writelines('I {:d} \n'.format(int(cl)))
            if artificial_detuning is not None:
                qasm_file.writelines('R90_phi {} {}\n'.format(
                    qubit_name, phases[i]))
            else:
                qasm_file.writelines('X90 {}     \n'.format(
                                     qubit_name))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file


def echo(qubit_name, times, clock_cycle=1e-9,
         artificial_detuning=None,
         cal_points=True):
    '''
    Echo sequence for a single qubit.
    Input pars:
        times:               array of times between (start of) pulses (s)
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters
        artificial_detuning: int,  float or None;
            if int: number of wiggles
            if float: artificial_detuning in (Hz)
            if None: adds no artificial detuning
                implemented using phase of the second pi/2 pulse
        cal_points:          whether to use calibration points or not
    '''
    # if int interpret artificial detuning as desired nr of wiggles
    if isinstance(artificial_detuning, int):
        phases = (360*np.arange(len(times))/(len(times)-4*cal_points) *
                  artificial_detuning % 360)
    # if float interpret it as artificial detuning in Hz
    elif isinstance(artificial_detuning, float):
        phases = (artificial_detuning*times % 1)*360
    elif artificial_detuning is None:
        phases = np.zeros(len(times))

    clocks = np.round(times/clock_cycle)
    filename = join(base_qasm_path, 'echo.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    for i, cl in enumerate(clocks):
        qasm_file.writelines('\ninit_all\n')
        if cal_points and (i == (len(clocks)-4) or
                           i == (len(clocks)-3)):
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
        elif cal_points and (i == (len(clocks)-2) or
                             i == (len(clocks)-1)):
            qasm_file.writelines('X180 {} \n'.format(qubit_name))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
        else:
            qasm_file.writelines('X90 {}     \n'.format(qubit_name))
            qasm_file.writelines('I {:d} \n'.format(int(cl//2)))
            qasm_file.writelines('X180 {}     \n'.format(qubit_name))
            qasm_file.writelines('I {:d} \n'.format(int(cl//2)))
            if artificial_detuning is not None:
                qasm_file.writelines('R90_phi {} {}\n'.format(
                    qubit_name, phases[i]))
            else:
                qasm_file.writelines('X90 {}     \n'.format(qubit_name))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file


def single_elt_on(qubit_name):
    filename = join(base_qasm_path, 'single_elt_on.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    # On
    qasm_file.writelines('\ninit_all\n')
    qasm_file.writelines('X180 {}     # On \n'.format(qubit_name))
    qasm_file.writelines('RO {}  \n'.format(qubit_name))

    qasm_file.close()
    return qasm_file


def two_elt_MotzoiXY(qubit_name):
    '''
    Sequence used for calibrating the motzoi parameter.
    Consists of yX and xY

    needs to reload the points for every data point.
    '''
    filename = join(base_qasm_path, 'Motzoi_XY.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    qasm_file.writelines('\ninit_all\n')
    qasm_file.writelines('Y90 {} \n'.format(qubit_name))
    qasm_file.writelines('X180 {} \n'.format(qubit_name))
    qasm_file.writelines('RO {}  \n'.format(qubit_name))

    qasm_file.writelines('\ninit_all\n')
    qasm_file.writelines('X90 {} \n'.format(qubit_name))
    qasm_file.writelines('Y180 {} \n'.format(qubit_name))
    qasm_file.writelines('RO {}  \n'.format(qubit_name))

    qasm_file.close()
    return qasm_file


def off_on(qubit_name, pulse_comb='off_on'):
    """
    Performs an 'Off_On' sequence on the qubit specified.
    Pulses can be optionally enabled by putting 'off', respectively 'on' in
    the pulse_comb string.
    """
    filename = join(base_qasm_path, 'off_on.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))

    # Off
    if 'off' in pulse_comb.lower():
        qasm_file.writelines('\ninit_all\n')
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
    # On
    if 'on' in pulse_comb.lower():
        qasm_file.writelines('\ninit_all\n')
        qasm_file.writelines('X180 {}     # On \n'.format(qubit_name))
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
    if 'on' not in pulse_comb.lower() and 'off' not in pulse_comb.lower():
        raise ValueError
    qasm_file.close()
    return qasm_file


def butterfly(qubit_name, initialize=False):
    """
    Initialize adds an exta measurement before state preparation to allow
    initialization by post-selection

    The duration of the RO + depletion is specified in the definition of RO
    """
    filename = join(
        base_qasm_path, 'butterfly_init_{}.qasm'.format(initialize))
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    if initialize:
        qasm_file.writelines('\ninit_all\n')
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
        qasm_file.writelines('RO {}  \n'.format(qubit_name))

        qasm_file.writelines('\ninit_all\n')
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
        qasm_file.writelines('X180 {}  \n'.format(qubit_name))
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
    else:
        qasm_file.writelines('\ninit_all\n')
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
        qasm_file.writelines('RO {}  \n'.format(qubit_name))

        qasm_file.writelines('\ninit_all\n')
        qasm_file.writelines('X180 {}  \n'.format(qubit_name))
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file


def randomized_benchmarking(qubit_name, nr_cliffords, nr_seeds,
                            net_clifford=0, restless=False,
                            label='randomized_benchmarking',
                            cal_points=True,
                            double_curves=False):
    '''
    Input pars:
        nr_cliffords:  list nr_cliffords for which to generate RB seqs
        nr_seeds:      int  nr_seeds for which to generate RB seqs
        net_clifford:  int index of net clifford the sequence should perform
                       0 corresponds to Identity and 3 corresponds to X180
        restless:      bool, does not initialize if restless is True
        label:           some string that can be used as a label.
        cal_points:    bool whether to replace the last two elements with
                       calibration points, set to False if you want
                       to measure a single element (for e.g. optimization)

        double_curves: Alternates between net clifford 0 and 3

    returns:
        qasm_file

    generates a qasm file for single qubit Clifford based randomized
    benchmarking.
    '''
    net_cliffords = [0, 3]  # Exists purely for the double curves mode
    filename = join(base_qasm_path, label+'.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    i = 0
    for seed in range(nr_seeds):
        for j, n_cl in enumerate(nr_cliffords):
            if not restless:
                qasm_file.writelines('init_all  \n')
            if cal_points and (j == (len(nr_cliffords)-4) or
                               j == (len(nr_cliffords)-3)):
                qasm_file.writelines('RO {}  \n'.format(qubit_name))
            elif cal_points and (j == (len(nr_cliffords)-2) or
                                 j == (len(nr_cliffords)-1)):
                qasm_file.writelines('X180 {} \n'.format(qubit_name))
                qasm_file.writelines('RO {}  \n'.format(qubit_name))
            else:
                if double_curves:
                    net_clifford = net_cliffords[i % 2]
                    i += 1
                cl_seq = rb.randomized_benchmarking_sequence(
                    n_cl, desired_net_cl=net_clifford)
                pulse_keys = rb.decompose_clifford_seq(cl_seq)
                for pulse in pulse_keys:
                    if pulse != 'I':
                        qasm_file.writelines('{} {}\n'.format(
                            pulse, qubit_name))
                qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file


def MotzoiXY(qubit_name, motzois, cal_points=True):
    '''
    Sequence used for calibrating the motzoi parameter.
    Consists of yX and xY

    Beware that the elements alternate, if you want to measure both Xy and Yx
    at each motzoi you need repeating motzoi parameters. This was chosen
    to be more easily compatible with standard detector functions and sweep pts

    Input pars:
        motzois:             array of motzoi parameters
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters
        cal_points:          if True, replaces the last 2*4 segments with
                             calibration points
    '''
    filename = join(base_qasm_path, 'Motzoi_XY.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    for i, motzoi in enumerate(motzois):
        qasm_file.writelines('\ninit_all\n')
        if cal_points and (i == (len(motzois)-4) or
                           i == (len(motzois)-3)):
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
        elif cal_points and (i == (len(motzois)-2) or
                             i == (len(motzois)-1)):
            qasm_file.writelines('X180 {} \n'.format(qubit_name))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
        if i % 2:
            qasm_file.writelines(
                'Y90_Motz {} {} \n'.format(qubit_name, motzoi))
            qasm_file.writelines(
                'X180_Motz {} {} \n'.format(qubit_name, motzoi))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
        else:
            qasm_file.writelines(
                'X90_Motz {} {} \n'.format(qubit_name, motzoi))
            qasm_file.writelines(
                'Y180_Motz {} {} \n'.format(qubit_name, motzoi))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file


def Ram_Z(qubit_name,
          wait_before=150e-9, wait_between=200e-9, clock_cycle=1e-9):
    '''
    Performs a Ram-Z sequence similar to a conventional echo sequence.

    Timing of sequence:
        trigger flux pulse -- wait_before -- mX90 -- wait_between -- X90 -- RO

    Args:
        qubit_name      (str): name of the targeted qubit
        wait_before     (float): delay time in seconds between triggering the
                                 AWG and the first pi/2 pulse
        wait_between    (float): delay time in seconds between the two pi/2
                                 pulses
        clock_cycle     (float): period of the internal AWG clock
    '''
    filename = join(base_qasm_path, 'Ram-Z.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))

    qasm_file.writelines('\ninit_all\n')

    qasm_file.writelines('QWG trigger \n')
    qasm_file.writelines('I {}\n'.format(int(wait_before//clock_cycle)))
    qasm_file.writelines('mX90 {}\n'.format(qubit_name))
    qasm_file.writelines('I {}\n'.format(int(wait_between//clock_cycle)))
    qasm_file.writelines('X90 {}\n'.format(qubit_name))
    qasm_file.writelines('RO {}  \n'.format(qubit_name))

    qasm_file.close()
    return qasm_file
