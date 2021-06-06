'''
Definitions of all the standard single qubit sequences
code that generates the required QASM files

Take a look at http://www.media.mit.edu/quanta/qasm2circ/ for qasm
'''
import numpy as np
from os.path import join, dirname

from pycqed.utilities.general import mopen
from pycqed.measurement.randomized_benchmarking import \
    randomized_benchmarking as rb
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
    qasm_file.writelines('Idx {:d} \n'.format(int(delay)))
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
            qasm_file.writelines('Idx {:d} \n'.format(int(cl)))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file


def flipping_seq(qubit_name: str, number_of_flips: list,
                 equator: bool=False, cal_points: bool=True,
                 restless: bool=False):
    filename = join(base_qasm_path, 'Flipping.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    for i, n in enumerate(number_of_flips):
        if not restless:
            qasm_file.writelines('\ninit_all\n')
        if cal_points and (i == (len(number_of_flips)-4) or
                           i == (len(number_of_flips)-3)):
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
        elif cal_points and (i == (len(number_of_flips)-2) or
                             i == (len(number_of_flips)-1)):
            qasm_file.writelines('X180 {} \n'.format(qubit_name))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
        else:
            for j in range(n):
                qasm_file.writelines('X180 {} \n'.format(
                                     qubit_name))
            if equator:
                qasm_file.writelines('X90 {} \n'.format(qubit_name))
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
            qasm_file.writelines('Idx {:d} \n'.format(int(cl)))
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
            qasm_file.writelines('Idx {:d} \n'.format(int(cl//2)))
            qasm_file.writelines('X180 {}     \n'.format(qubit_name))
            qasm_file.writelines('Idx {:d} \n'.format(int(cl//2)))
            if artificial_detuning is not None:
                qasm_file.writelines('R90_phi {} {}\n'.format(
                    qubit_name, phases[i]))
            else:
                qasm_file.writelines('X90 {}     \n'.format(qubit_name))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.close()
    return qasm_file


def single_elt_on(qubit_name, n=1):
    filename = join(base_qasm_path, 'single_elt_on.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))
    # On
    qasm_file.writelines('\ninit_all\n')
    for i in range(n):
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
    # simulatneous on
    if 'sim_on' in pulse_comb.lower():
        qasm_file.writelines('\ninit_all\n')
        qasm_file.writelines('X180 {} | RO {}  \n'.format(qubit_name, qubit_name))
    # On
    elif 'on' in pulse_comb.lower():
        qasm_file.writelines('\ninit_all\n')
        qasm_file.writelines('X180 {}     # On \n'.format(qubit_name))
        qasm_file.writelines('RO {}  \n'.format(qubit_name))

    if 'on' not in pulse_comb.lower() and 'off' not in pulse_comb.lower():
        raise ValueError('pulse_comb must contain "off" or "on" (is {})'
                         .format(pulse_comb))
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

    qasm_file.writelines('\ninit_all\n')
    if initialize:
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.writelines('RO {}  \n'.format(qubit_name))
    qasm_file.writelines('RO {}  \n'.format(qubit_name))

    qasm_file.writelines('\ninit_all\n')
    if initialize:
        qasm_file.writelines('RO {}  \n'.format(qubit_name))
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


def Ram_Z(qubit_name, no_of_points, cal_points=True,
          case='interleaved'):
    '''
    Creates QASM sequence for an entire Ram-Z experiment, including
    calibration points.

    sequence:
        mX90 -- flux_pulse -- X90 -- RO

    Args:
        qubit_name (str):
            Name of the targeted qubit
        no_of_points (int):
            Number of points in the hard sweep. This is
                               limited by the QWG waveform memory and number of
                               codeword channels used.
        case (str):
            'sin', 'cos', or 'interleaved.
            'cos': use X90 as the last mw pulse.
            'sin': use Y90 as the last mw pulse.
            'interleaved': use both cases, first 'cos', then 'sin'.

    '''
    filename = join(base_qasm_path, 'Ram_Z.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))

    if case == 'sin':
        recPulses = ['Y90']
    elif case == 'cos':
        recPulses = ['X90']
    elif case == 'interleaved':
        recPulses = ['X90', 'Y90']
    else:
        raise ValueError('Unknown case "{}".'.format(case))

    # Write  measurement sequence no_of_points times
    for i in range(no_of_points):
        for recPulse in recPulses:
            qasm_file.writelines('\ninit_all\n')

            if cal_points and (i == no_of_points - 4 or i == no_of_points - 3):
                # Calibration point for |0>
                pass
            elif cal_points and (i == no_of_points - 2 or i == no_of_points - 1):
                # Calibration point for |1>
                qasm_file.writelines('X180 {} \n'.format(qubit_name))
            else:
                qasm_file.writelines('mX90 {}\n'.format(qubit_name))
                qasm_file.writelines('square_{} {}\n'.format(i, qubit_name))
                qasm_file.writelines('{} {}\n'.format(recPulse, qubit_name))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))

    qasm_file.close()
    return qasm_file


def Ram_Z_single(qubit_name,
                 wait_before=150e-9, wait_between=200e-9, clock_cycle=1e-9):
    '''
    Creates the QASM sequence for a single point in a Ram-Z experiment.

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
    filename = join(base_qasm_path, 'Ram-Z-single.qasm')
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


def Ram_Z_echo(qubit_name: str , nr_of_points: int, cal_points: bool=True):
    '''
    Creates QASM sequence for an entire Ram-Z experiment, including
    calibration points.

    sequence:
        mX90 -- flux(T) -- X180 -- flux(T+dt) -- X90 -- RO

    Args:
        qubit_name      (str): name of the targeted qubit
        nr_of_points    (int): number of points in the hard sweep. This is
                               limited by the QWG waveform memory and number of
                               codeword channels used.
    '''
    filename = join(base_qasm_path, 'Ram_Z_echo.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))

    # Write  measurement sequence nr_of_points times
    for i in range(nr_of_points):
        qasm_file.writelines('\ninit_all\n')

        if cal_points and (i == nr_of_points - 4 or i == nr_of_points - 3):
            # Calibration point for |0>
            pass
        elif cal_points and (i == nr_of_points - 2 or i == nr_of_points - 1):
            # Calibration point for |1>
            qasm_file.writelines('X180 {} \n'.format(qubit_name))
        else:
            qasm_file.writelines('mX90 {}\n'.format(qubit_name))
            qasm_file.writelines('square_{} {}\n'.format(i, qubit_name))
            qasm_file.writelines('X180 {}\n'.format(qubit_name))
            qasm_file.writelines('square_dummy {}\n'.format(qubit_name))
            qasm_file.writelines('X90 {}\n'.format(qubit_name))
        qasm_file.writelines('RO {}  \n'.format(qubit_name))

    qasm_file.close()
    return qasm_file


def distortion_scope_fine_seq(qubit_name: str, nr_of_points: int,
                              case: str='interleaved'):
    '''
    Create QASM sequence for the enhanced cryo-scope using two flux pulses.

    Sequence:
        long flux -- X90 -- scoping flux -- X90 or Y90 -- RO

    Args:
        qubit_name (str):
                name of the targeted qubit
        nr_of_points (int):
                number of points in the hard sweep. This is limited by the QWG
                waveform memory and number of codeword channels used.
        case (str):
                Can be 'cos', 'sin', or 'interleaved.
                'cos' uses X90 for the last mw pulse.
                'sin' uses Y90 for the last mw pulse.
                'interleaved' uses both cases for every point.
    '''
    filename = join(base_qasm_path, 'distortion_scope_fine.qasm')
    qasm_file = mopen(filename, mode='w')

    qasm_file.writelines('qubit {} \n'.format(qubit_name))

    if case == 'interleaved' or case == 'background':
        recPulses = ['X90', 'Y90']
    elif case == 'cos':
        recPulses = ['X90']
    elif case == 'sin':
        recPulses = ['Y90']
    else:
        raise ValueError('Unknown case "{}".'.format(case))

    for i in range(nr_of_points):
        for recPulse in recPulses:
            qasm_file.writelines('\ninit_all\n')

            qasm_file.writelines('long_square_{} {}\n'.format(i, qubit_name))
            qasm_file.writelines('X90 {}\n'.format(qubit_name))
            qasm_file.writelines('scoping_flux {}\n'.format(qubit_name))
            qasm_file.writelines('{} {}\n'.format(recPulse, qubit_name))
            qasm_file.writelines('RO {}  \n'.format(qubit_name))

    qasm_file.close()
    return qasm_file


def flux_timing_seq(qubit_name, taus,
                    wait_between=220e-9, clock_cycle=1e-9, cal_points=True):
    '''
    Creates the QASM sequence to calibrate the timing of the flux pulse
    relative to microwave pulses.

    Timing of the sequence:
        trigger flux pulse -- tau -- X90 -- wait_between -- X90 -- RO

    Args:
        qubit_name (str):       name of the targeted qubit
        taus (array of floats): delays between the flux trigger and the first
                                pi-half pulse (the sweep points).
        wait_between (float):   delay between the two pi-half pulses
        clock_cycle (float):    period of the internal AWG clock
        cal_points (bool):      whether to include calibration points
    '''
    filename = join(base_qasm_path, 'flux_timing.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))

    for i, tau in enumerate(taus):
        qasm_file.writelines('\ninit_all\n')

        if cal_points and (i == len(taus) - 4 or i == len(taus) - 3):
            # Calibration point for |0>
            pass
        elif cal_points and (i == len(taus) - 2 or i == len(taus) - 1):
            # Calibration point for |1>
            qasm_file.writelines('X180 {} \n'.format(qubit_name))
        else:
            qasm_file.writelines('flux square {}\n'.format(qubit_name))
            qasm_file.writelines(
                'I {}\n'.format(int(round(tau/clock_cycle))))
            qasm_file.writelines('X90 {}\n'.format(qubit_name))
            qasm_file.writelines(
                'I {}\n'.format(int(round(wait_between/clock_cycle))))
            qasm_file.writelines('X90 {}\n'.format(qubit_name))

        qasm_file.writelines('RO {}  \n'.format(qubit_name))

    qasm_file.close()
    return qasm_file


def flux_resonator_shift_seq(qubit_name):
    '''
    Creates the QASM sequence for a flux resonator shift experiment. This
    experiment is used to measure the response of the qubit to a flux pulse via
    the resonator shift: sweep RO frequency, measure transient at every
    frequency -> 2D plot looks similar to what you would measure on the scope
    (step response).
    Note: delay in the flux pulse must be set, such that it is played during
    the readout.
    '''
    filename = join(base_qasm_path, 'flux_resonator_shift.qasm')
    qasm_file = mopen(filename, mode='w')
    qasm_file.writelines('qubit {} \n'.format(qubit_name))

    qasm_file.writelines('\ninit_all\n')

    qasm_file.writelines('QWG trigger square\n')
    qasm_file.writelines('RO {}  \n'.format(qubit_name))

    qasm_file.close()
    return qasm_file
