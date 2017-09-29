'''
'''
import numpy as np
from os.path import join, dirname
import openql.openql as ql
from openql.openql import Program, Kernel, Platform

from pycqed.utilities.general import mopen
from pycqed.measurement.randomized_benchmarking import randomized_benchmarking as rb


base_qasm_path = join(dirname(__file__), 'qasm_files')
output_dir = join(dirname(__file__), 'output')
ql.set_output_dir(output_dir)


def CW_tone():
    pass


def vsm_timing_cal_sequence(qubit_idx: int, marker_idx: int, platf_cfg: str):
    """
    A sequence for calibrating the VSM timing delay.

    The marker idx is a qubit number for which a dummy pulse is played.
    This can be used as a reference.

    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="vsm_timing_cal_sequence",
                nqubits=platf.get_qubit_number(),
                p=platf)

    k = Kernel("main", p=platf)
    k.prepz(qubit_idx)  # to ensure enough separation in timing
    k.gate('CW_00', marker_idx)  # to trigger on
    k.gate('spec', qubit_idx)
    p.add_kernel(k)
    p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def CW_RO_sequence(qubit_idx: int, platf_cfg: str):
    """
    A sequence that performs readout back to back without initialization.
    The separation of the readout triggers is done by specifying the duration
    of the readout parameter in the configuration file used for compilation.
    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="CW_RO_sequence", nqubits=platf.get_qubit_number(),
                p=platf)

    k = Kernel("main", p=platf)
    k.measure(qubit_idx)
    p.add_kernel(k)
    p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def pulsed_spec_sequence(qubit_idx: int, spec_pulse_length: float,
                         platf_cfg: str):
    """
    Sequence for a pulsed spectroscopy.
    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="CW_RO_sequence", nqubits=platf.get_qubit_number(),
                p=platf)
    k = Kernel("main", p=platf)

    nr_clocks = int(spec_pulse_length/20e-9)

    for i in range(nr_clocks):
        # The spec pulse is a pulse that lasts 20ns, because of the way the VSM
        # control works. By repeating it the duration can be controlled.
        k.gate('spec', qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)
    p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def T1(qubit_name, times, clock_cycle=1e-9,
       cal_points=True):
    pass


def flipping(qubit_idx: int, number_of_flips, platf_cfg: str,
             equator: bool=False, cal_points: bool=True):
    """
    Generates a flipping sequence that performs multiple pi-pulses
    Basic sequence:
        - (X)^n - RO

    Input pars:
        qubit_idx:      int specifying the target qubit (starting at 0)
        number_of_flips: array of ints specifying the sweep points
        platf_cfg:      filename of the platform config file
        equator:        if True add an extra pi/2 pulse at the end to
                        make the state end at the equator.
        cal_points:     replaces last 4 points by calibration points

    Returns:
        p:              OpenQL Program object
    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="Flipping", nqubits=platf.get_qubit_number(),
                p=platf)

    for i, n in enumerate(number_of_flips):
        k = Kernel("Flipping_"+str(i), p=platf)
        k.prepz(qubit_idx)
        if cal_points and (i == (len(number_of_flips)-4) or
                           i == (len(number_of_flips)-3)):
            k.measure(qubit_idx)
        elif cal_points and (i == (len(number_of_flips)-2) or
                             i == (len(number_of_flips)-1)):
            k.x(qubit_idx)
            k.measure(qubit_idx)
        else:
            if equator:
                k.gate('rx90', qubit_idx)
            for j in range(n):
                k.x(qubit_idx)
            k.measure(qubit_idx)
        p.add_kernel(k)

    p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def AllXY(qubit_idx: int, platf_cfg: str, double_points: bool=True):
    """
    Single qubit AllXY sequence.
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
        double_points:  if true repeats every element twice
    Returns:
        p:              OpenQL Program object containing


    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="AllXY", nqubits=platf.get_qubit_number(),
                p=platf)

    allXY = [['i', 'i'], ['rx180', 'rx180'], ['ry180', 'ry180'],
             ['rx180', 'ry180'], ['ry180', 'rx180'],
             ['rx90', 'i'], ['ry90', 'i'], ['rx90', 'ry90'],
             ['ry90', 'rx90'], ['rx90', 'ry180'], ['ry90', 'rx180'],
             ['rx180', 'ry90'], ['ry180', 'rx90'], ['rx90', 'rx180'],
             ['rx180', 'rx90'], ['ry90', 'ry180'], ['ry180', 'ry90'],
             ['rx180', 'i'], ['ry180', 'i'], ['rx90', 'rx90'],
             ['ry90', 'ry90']]

    # this should be implicit
    p.set_sweep_points(np.arange(len(allXY), dtype=float), len(allXY))

    for i, xy in enumerate(allXY):
        k = Kernel("AllXY_"+str(i), p=platf)
        k.prepz(qubit_idx)
        k.gate(xy[0], qubit_idx)
        k.gate(xy[1], qubit_idx)
        k.measure(qubit_idx)
        p.add_kernel(k)

    p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


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
    pass
    # if isinstance(artificial_detuning, int):
    #     phases = (360*np.arange(len(times))/(len(times)-4*cal_points) *
    #               artificial_detuning % 360)
    # # if float interpret it as artificial detuning in Hz
    # elif isinstance(artificial_detuning, float):
    #     phases = (artificial_detuning*times % 1)*360
    # elif artificial_detuning is None:
    #     phases = np.zeros(len(times))

    # clocks = np.round(times/clock_cycle)
    # filename = join(base_qasm_path, 'Ramsey.qasm')
    # qasm_file = mopen(filename, mode='w')
    # qasm_file.writelines('qubit {} \n'.format(qubit_name))
    # for i, cl in enumerate(clocks):
    #     qasm_file.writelines('\ninit_all\n')
    #     if cal_points and (i == (len(clocks)-4) or
    #                        i == (len(clocks)-3)):
    #         qasm_file.writelines('RO {}  \n'.format(qubit_name))
    #     elif cal_points and (i == (len(clocks)-2) or
    #                          i == (len(clocks)-1)):
    #         qasm_file.writelines('X180 {} \n'.format(qubit_name))
    #         qasm_file.writelines('RO {}  \n'.format(qubit_name))

    #     else:
    #         qasm_file.writelines('X90 {}     \n'.format(
    #                              qubit_name))
    #         qasm_file.writelines('I {:d} \n'.format(int(cl)))
    #         if artificial_detuning is not None:
    #             qasm_file.writelines('R90_phi {} {}\n'.format(
    #                 qubit_name, phases[i]))
    #         else:
    #             qasm_file.writelines('X90 {}     \n'.format(
    #                                  qubit_name))
    #         qasm_file.writelines('RO {}  \n'.format(qubit_name))
    # qasm_file.close()
    # return qasm_file


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
    pass
    # if isinstance(artificial_detuning, int):
    #     phases = (360*np.arange(len(times))/(len(times)-4*cal_points) *
    #               artificial_detuning % 360)
    # # if float interpret it as artificial detuning in Hz
    # elif isinstance(artificial_detuning, float):
    #     phases = (artificial_detuning*times % 1)*360
    # elif artificial_detuning is None:
    #     phases = np.zeros(len(times))

    # clocks = np.round(times/clock_cycle)
    # filename = join(base_qasm_path, 'echo.qasm')
    # qasm_file = mopen(filename, mode='w')
    # qasm_file.writelines('qubit {} \n'.format(qubit_name))
    # for i, cl in enumerate(clocks):
    #     qasm_file.writelines('\ninit_all\n')
    #     if cal_points and (i == (len(clocks)-4) or
    #                        i == (len(clocks)-3)):
    #         qasm_file.writelines('RO {}  \n'.format(qubit_name))
    #     elif cal_points and (i == (len(clocks)-2) or
    #                          i == (len(clocks)-1)):
    #         qasm_file.writelines('X180 {} \n'.format(qubit_name))
    #         qasm_file.writelines('RO {}  \n'.format(qubit_name))
    #     else:
    #         qasm_file.writelines('X90 {}     \n'.format(qubit_name))
    #         qasm_file.writelines('I {:d} \n'.format(int(cl//2)))
    #         qasm_file.writelines('X180 {}     \n'.format(qubit_name))
    #         qasm_file.writelines('I {:d} \n'.format(int(cl//2)))
    #         if artificial_detuning is not None:
    #             qasm_file.writelines('R90_phi {} {}\n'.format(
    #                 qubit_name, phases[i]))
    #         else:
    #             qasm_file.writelines('X90 {}     \n'.format(qubit_name))
    #         qasm_file.writelines('RO {}  \n'.format(qubit_name))
    # qasm_file.close()
    # return qasm_file


def single_elt_on(qubit_idx: int, platf_cfg: str):
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="Single_elt_on", nqubits=platf.get_qubit_number(),
                p=platf)
    k = Kernel("main", p=platf)
    k.prepz(qubit_idx)
    k.x(qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def two_elt_MotzoiXY(qubit_name):
    '''
    Sequence used for calibrating the motzoi parameter.
    Consists of yX and xY

    needs to reload the points for every data point.
    '''
    pass
    # filename = join(base_qasm_path, 'Motzoi_XY.qasm')
    # qasm_file = mopen(filename, mode='w')
    # qasm_file.writelines('qubit {} \n'.format(qubit_name))
    # qasm_file.writelines('\ninit_all\n')
    # qasm_file.writelines('Y90 {} \n'.format(qubit_name))
    # qasm_file.writelines('X180 {} \n'.format(qubit_name))
    # qasm_file.writelines('RO {}  \n'.format(qubit_name))

    # qasm_file.writelines('\ninit_all\n')
    # qasm_file.writelines('X90 {} \n'.format(qubit_name))
    # qasm_file.writelines('Y180 {} \n'.format(qubit_name))
    # qasm_file.writelines('RO {}  \n'.format(qubit_name))

    # qasm_file.close()
    # return qasm_file


def off_on(qubit_idx: int, pulse_comb: str, platf_cfg: str):
    """
    Performs an 'off_on' sequence on the qubit specified.
        off: prepz -      - RO
        on:  prepz - x180 - RO
    Args:
        qubit_idx (int) :
        pulse_comb (str): What pulses to play valid options are
            "off", "on", "off_on"
        platf_cfg (str) : filepath of OpenQL platform config file

    Pulses can be optionally enabled by putting 'off', respectively 'on' in
    the pulse_comb string.
    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="CW_RO_sequence", nqubits=platf.get_qubit_number(),
                p=platf)
    # # Off
    if 'off' in pulse_comb.lower():
        k = Kernel("off", p=platf)
        k.prepz(qubit_idx)
        k.measure(qubit_idx)
        p.add_kernel(k)

    if 'on' in pulse_comb.lower():
        k = Kernel("on", p=platf)
        k.prepz(qubit_idx)
        k.gate('rx180', qubit_idx)
        k.measure(qubit_idx)
        p.add_kernel(k)

    if ('on' not in pulse_comb.lower()) and ('off' not in pulse_comb.lower()):
        raise ValueError()

    p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def butterfly(qubit_idx: int, initialize: bool, platf_cfg: str):
    """

    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="Butterfly", nqubits=platf.get_qubit_number(),
                p=platf)

    k = Kernel('0', p=platf)
    k.prepz(qubit_idx)
    if initialize:
        k.measure(qubit_idx)
    k.measure(qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    k = Kernel('1', p=platf)
    k.prepz(qubit_idx)
    if initialize:
        k.measure(qubit_idx)
    k.x(qubit_idx)
    k.measure(qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    p.compile()
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


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
    pass
    # net_cliffords = [0, 3]  # Exists purely for the double curves mode
    # filename = join(base_qasm_path, label+'.qasm')
    # qasm_file = mopen(filename, mode='w')
    # qasm_file.writelines('qubit {} \n'.format(qubit_name))
    # i = 0
    # for seed in range(nr_seeds):
    #     for j, n_cl in enumerate(nr_cliffords):
    #         if not restless:
    #             qasm_file.writelines('init_all  \n')
    #         if cal_points and (j == (len(nr_cliffords)-4) or
    #                            j == (len(nr_cliffords)-3)):
    #             qasm_file.writelines('RO {}  \n'.format(qubit_name))
    #         elif cal_points and (j == (len(nr_cliffords)-2) or
    #                              j == (len(nr_cliffords)-1)):
    #             qasm_file.writelines('X180 {} \n'.format(qubit_name))
    #             qasm_file.writelines('RO {}  \n'.format(qubit_name))
    #         else:
    #             if double_curves:
    #                 net_clifford = net_cliffords[i % 2]
    #                 i += 1
    #             cl_seq = rb.randomized_benchmarking_sequence(
    #                 n_cl, desired_net_cl=net_clifford)
    #             pulse_keys = rb.decompose_clifford_seq(cl_seq)
    #             for pulse in pulse_keys:
    #                 if pulse != 'I':
    #                     qasm_file.writelines('{} {}\n'.format(
    #                         pulse, qubit_name))
    #             qasm_file.writelines('RO {}  \n'.format(qubit_name))
    # qasm_file.close()
    # return qasm_file


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
    pass


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
    pass
