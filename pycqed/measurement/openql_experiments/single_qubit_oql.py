'''
'''
import numpy as np
from os.path import join, dirname
import openql.openql as ql
from openql.openql import Program, Kernel, Platform
from pycqed.utilities.general import suppress_stdout
# from pycqed.utilities.general import mopen
from pycqed.measurement.randomized_benchmarking import randomized_benchmarking as rb


base_qasm_path = join(dirname(__file__), 'qasm_files')
output_dir = join(dirname(__file__), 'output')
ql.set_output_dir(output_dir)


def CW_tone():
    pass


def vsm_timing_cal_sequence(qubit_idx: int, platf_cfg: str):
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
    k.gate('spec', qubit_idx)
    p.add_kernel(k)
    with suppress_stdout():
        p.compile(verbose=False)
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
    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def pulsed_spec_seq(qubit_idx: int, spec_pulse_length: float,
                    platf_cfg: str):
    """
    Sequence for pulsed spectroscopy.

    Important notes: because of the way the CCL functions this sequence is
    made by repeating multiple "spec" pulses of 20ns back to back.
    As such the spec_pulse_lenght must be a multiple of 20e-9. If
    this is not the case the spec_pulse_length will be rounded.

    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="pulsed_spec_seq", nqubits=platf.get_qubit_number(),
                p=platf)
    k = Kernel("main", p=platf)

    nr_clocks = int(spec_pulse_length/20e-9)

    for i in range(nr_clocks):
        # The spec pulse is a pulse that lasts 20ns, because of the way the VSM
        # control works. By repeating it the duration can be controlled.
        k.gate('spec', qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)
    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


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

    with suppress_stdout():
        p.compile(verbose=False)
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
        if double_points:
            js = 2
        else:
            js = 1
        for j in range(js):
            k = Kernel("AllXY_"+str(i+j/2), p=platf)
            k.prepz(qubit_idx)
            k.gate(xy[0], qubit_idx)
            k.gate(xy[1], qubit_idx)
            k.measure(qubit_idx)
            p.add_kernel(k)

    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def T1(times, qubit_idx: int, platf_cfg: str):
    """
    Single qubit T1 sequence.
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        times:          the list of waiting times for each T1 element
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
    Returns:
        p:              OpenQL Program object containing


    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="T1", nqubits=platf.get_qubit_number(),
                p=platf)

    for i, time in enumerate(times[:-4]):
        k = Kernel("T1_"+str(i), p=platf)
        k.prepz(qubit_idx)
        wait_nanoseconds = int(round(time/1e-9))
        k.gate('rx180', qubit_idx)
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        k.measure(qubit_idx)
        p.add_kernel(k)

    # adding the calibration points
    add_single_qubit_cal_points(p, platf=platf, qubit_idx=qubit_idx)

    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def Ramsey(times, qubit_idx: int, platf_cfg: str):
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
    p = Program(pname="Ramsey", nqubits=platf.get_qubit_number(),
                p=platf)

    for i, time in enumerate(times[:-4]):
        k = Kernel("Ramsey_"+str(i), p=platf)
        k.prepz(qubit_idx)
        wait_nanoseconds = int(round(time/1e-9))
        k.gate('rx90', qubit_idx)
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        k.gate('rx90', qubit_idx)
        k.measure(qubit_idx)
        p.add_kernel(k)

    # adding the calibration points
    add_single_qubit_cal_points(p, platf=platf, qubit_idx=qubit_idx)

    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def echo(times, qubit_idx: int, platf_cfg: str):
    """
    Single qubit Echo sequence.
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        times:          the list of waiting times for each Echo element
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
    Returns:
        p:              OpenQL Program object containing

    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="echo", nqubits=platf.get_qubit_number(),
                p=platf)

    for i, time in enumerate(times[:-4]):
        k = Kernel("echo_"+str(i), p=platf)
        k.prepz(qubit_idx)
        # nr_clocks = int(time/20e-9/2)
        wait_nanoseconds = int(round(time/1e-9/2))
        k.gate('rx90', qubit_idx)
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        k.gate('rx180', qubit_idx)
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        k.gate('rx90', qubit_idx)
        k.measure(qubit_idx)
        p.add_kernel(k)

    # adding the calibration points
    add_single_qubit_cal_points(p, platf=platf, qubit_idx=qubit_idx)

    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def idle_error_rate_seq(nr_of_idle_gates,
                        states: list,
                        gate_duration_ns: int,
                        echo: bool,
                        qubit_idx: int, platf_cfg: str,
                        post_select=True):
    """
    Sequence to perform the idle_error_rate_sequence.
    Virtually identical to a T1 experiment (Z-basis)
                        or a ramsey/echo experiment (X-basis)

    Input pars:
        nr_of_idle_gates : list of integers specifying the number of idle gates
            corresponding to each data point.
        gate_duration_ns : integer specifying the duration of the wait gate.
        states  :       list of states to prepare
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
    Returns:
        p:              OpenQL Program object containing


    """
    allowed_states = ['0', '1', '+']
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="idle_error_rate", nqubits=platf.get_qubit_number(),
                p=platf)
    sweep_points = []
    for N in nr_of_idle_gates:
        for state in states:
            if state not in allowed_states:
                raise ValueError('State must be in {}'.format(allowed_states))
            k = Kernel("idle_prep{}_N{}".format(state, N), p=platf)
            # 1. Preparing in the right basis
            k.prepz(qubit_idx)
            if post_select:
                # adds an initialization measurement used to post-select
                k.measure(qubit_idx)
            if state =='1':
                k.gate('rx180', qubit_idx)
            elif state == '+':
                k.gate('rym90', qubit_idx)
            # 2. The "waiting" gates
            wait_nanoseconds = N*gate_duration_ns
            if state == '+' and echo:
                k.gate("wait", [qubit_idx], wait_nanoseconds//2)
                k.gate('rx180', qubit_idx)
                k.gate("wait", [qubit_idx], wait_nanoseconds//2)
            else:
                k.gate("wait", [qubit_idx], wait_nanoseconds)
            # 3. Reading out in the proper basis
            if state == '+' and echo:
                k.gate('rym90', qubit_idx)
            elif state =='+':
                k.gate('ry90', qubit_idx)
            k.measure(qubit_idx)
            p.add_kernel(k)
        sweep_points.append(N)

    p.set_sweep_points(sweep_points, num_sweep_points=len(sweep_points))
    p.sweep_points = sweep_points
    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def single_elt_on(qubit_idx: int, platf_cfg: str):
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="Single_elt_on", nqubits=platf.get_qubit_number(),
                p=platf)
    k = Kernel("main", p=platf)
    k.prepz(qubit_idx)
    k.x(qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p



def off_on(qubit_idx: int, pulse_comb: str, initialize: bool, platf_cfg: str):
    """
    Performs an 'off_on' sequence on the qubit specified.
        off: (RO) - prepz -      - RO
        on:  (RO) - prepz - x180 - RO
    Args:
        qubit_idx (int) :
        pulse_comb (str): What pulses to play valid options are
            "off", "on", "off_on"
        initialize (bool): if True does an extra initial measurement to
            post select data.
        platf_cfg (str) : filepath of OpenQL platform config file

    Pulses can be optionally enabled by putting 'off', respectively 'on' in
    the pulse_comb string.
    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="OffOn_RO_sequence", nqubits=platf.get_qubit_number(),
                p=platf)
    # # Off
    if 'off' in pulse_comb.lower():
        k = Kernel("off", p=platf)
        k.prepz(qubit_idx)
        if initialize:
            k.measure(qubit_idx)
        k.measure(qubit_idx)
        p.add_kernel(k)

    if 'on' in pulse_comb.lower():
        k = Kernel("on", p=platf)
        k.prepz(qubit_idx)
        if initialize:
            k.measure(qubit_idx)
        k.gate('rx180', qubit_idx)
        k.measure(qubit_idx)
        p.add_kernel(k)

    if ('on' not in pulse_comb.lower()) and ('off' not in pulse_comb.lower()):
        raise ValueError()

    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def butterfly(qubit_idx: int, initialize: bool, platf_cfg: str):
    """
    Performs a 'butterfly' sequence on the qubit specified.
        0:  prepz (RO) -      - RO - RO
        1:  prepz (RO) - x180 - RO - RO

    Args:
        qubit_idx (int)  : index of the qubit
        initialize (bool): if True does an extra initial measurement to
            post select data.
        platf_cfg (str)  : openql config used for setup.

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

    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def RTE(qubit_idx: int, sequence_type: str, platf_cfg: str,
        net_gate: str, feedback=False):
    """
    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="RTE", nqubits=platf.get_qubit_number(), p=platf)

    k = Kernel('RTE', p=platf)
    if sequence_type == 'echo':
        k.gate('rx90', qubit_idx)
        k.gate('i', qubit_idx)
        k.gate('i', qubit_idx)
        k.gate('i', qubit_idx)
        k.gate('i', qubit_idx)
        k.gate('rx180', qubit_idx)
        k.gate('i', qubit_idx)
        k.gate('i', qubit_idx)
        k.gate('i', qubit_idx)
        k.gate('i', qubit_idx)
        if net_gate == 'pi':
            k.gate('rxm90', qubit_idx)
        elif net_gate == 'i':
            k.gate('rx90', qubit_idx)
        else:
            raise ValueError('net_gate ({})should be "i" or "pi"'.format(
                net_gate))
        if feedback:
            k.gate('Crx180', qubit_idx)
    elif sequence_type == 'pi':
        if net_gate == 'pi':
            k.gate('rx180', qubit_idx)
        elif net_gate == 'i':
            pass
        else:
            raise ValueError('net_gate ({})should be "i" or "pi"'.format(
                net_gate))
        if feedback:
            k.gate('Crx180', qubit_idx)
    else:
        raise ValueError('sequence_type ({})should be "echo" or "pi"'.format(
            sequence_type))
    k.measure(qubit_idx)
    p.add_kernel(k)

    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def randomized_benchmarking(qubit_idx: int, platf_cfg: str,
                            nr_cliffords, nr_seeds: int,
                            net_clifford: int=0, restless: bool=False,
                            program_name: str='randomized_benchmarking',
                            cal_points: bool=True,
                            double_curves: bool=False):
    '''
    Input pars:
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
        nr_cliffords:   list nr_cliffords for which to generate RB seqs
        nr_seeds:       int  nr_seeds for which to generate RB seqs
        net_clifford:   int index of net clifford the sequence should perform
                            0 -> Idx
                            3 -> rx180
        restless:       bool, does not initialize if restless is True
        program_name:           some string that can be used as a label.
        cal_points:     bool whether to replace the last two elements with
                        calibration points, set to False if you want
                        to measure a single element (for e.g. optimization)

        double_curves: Alternates between net clifford 0 and 3

    Returns:
        p:              OpenQL Program object

    generates a program for single qubit Clifford based randomized
    benchmarking.
    '''
    net_cliffords = [0, 3]  # Exists purely for the double curves mode
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname=program_name, nqubits=platf.get_qubit_number(),
                p=platf)

    i = 0
    for seed in range(nr_seeds):
        for j, n_cl in enumerate(nr_cliffords):
            k = Kernel('RB_{}Cl_s{}'.format(n_cl, seed), p=platf)
            if not restless:
                k.prepz(qubit_idx)
            if cal_points and (j == (len(nr_cliffords)-4) or
                               j == (len(nr_cliffords)-3)):
                k.measure(qubit_idx)

            elif cal_points and (j == (len(nr_cliffords)-2) or
                                 j == (len(nr_cliffords)-1)):
                k.x(qubit_idx)
                k.measure(qubit_idx)
            else:
                if double_curves:
                    net_clifford = net_cliffords[i % 2]
                    i += 1
                cl_seq = rb.randomized_benchmarking_sequence(
                    n_cl, desired_net_cl=net_clifford)
                # pulse_keys = rb.decompose_clifford_seq(cl_seq)
                for cl in cl_seq:
                    k.gate('cl_{}'.format(cl), qubit_idx)
                k.measure(qubit_idx)
            p.add_kernel(k)
    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def motzoi_XY(qubit_idx:int, platf_cfg: str,
             program_name: str='motzoi_XY'):
    '''
    Sequence used for calibrating the motzoi parameter.
    Consists of yX and xY

    Beware that the elements alternate, if you want to measure both Xy and Yx
    at each motzoi you need repeating motzoi parameters. This was chosen
    to be more easily compatible with standard detector functions and sweep pts

    '''
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname=program_name, nqubits=platf.get_qubit_number(),
                p=platf)
    k = Kernel("yX", p=platf)
    k.prepz(qubit_idx)
    k.gate('ry90', qubit_idx)
    k.gate('rx180', qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    k = Kernel("xY", p=platf)
    k.prepz(qubit_idx)
    k.gate('rx90', qubit_idx)
    k.gate('ry180', qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p

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


def add_single_qubit_cal_points(p, platf, qubit_idx):
    for i in np.arange(2):
        k = Kernel("cal_gr_"+str(i), p=platf)
        k.prepz(qubit_idx)
        k.measure(qubit_idx)
        p.add_kernel(k)

    for i in np.arange(2):
        k = Kernel("cal_ex_"+str(i), p=platf)
        k.prepz(qubit_idx)
        k.gate('rx180', qubit_idx)
        k.measure(qubit_idx)
        p.add_kernel(k)
    return p


def FluxTimingCalibration(qubit_idx: int, buffer_time1, times, platf_cfg: str,
                          cal_points: bool=True):
    """
    A Ramsey sequence with varying waiting times `times` around a flux pulse.
    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="FluxTimingCalibration", nqubits=platf.get_qubit_number(),
                p=platf)

    buffer_nanoseconds1 = int(round(buffer_time1/1e-9))
    # don't use last 4 points if calibration points are used
    if cal_points:
        times= times[:-4]
    for t in times:
        t_nanoseconds = int(round(t/1e-9))

        k = Kernel("pifluxpi", p=platf)
        k.prepz(qubit_idx)
        k.gate('rx90', qubit_idx)
        if buffer_nanoseconds1 > 10:
            k.gate("wait", [qubit_idx], buffer_nanoseconds1)
        k.gate('fl_cw_02', 2, 0)
        if t_nanoseconds > 10:
            k.gate("wait", [qubit_idx], t_nanoseconds)
        k.gate('rx90', qubit_idx)
        k.measure(qubit_idx)
        p.add_kernel(k)

    if cal_points:
        add_single_qubit_cal_points(p, platf=platf, qubit_idx=qubit_idx)

    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def FluxTimingCalibration_2q(q0, q1, buffer_time1, times, platf_cfg: str):
    """
    A Ramsey sequence with varying waiting times `times` around a flux pulse.
    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="FluxTimingCalibration2q", nqubits=platf.get_qubit_number(),
                p=platf)

    buffer_nanoseconds1 = int(round(buffer_time1/1e-9))

    for t in times:

        t_nanoseconds = int(round(t/1e-9))

        k = Kernel("pifluxpi", p=platf)
        k.prepz(q0)
        k.prepz(q1)

        k.gate('rx180', q0)
        k.gate('rx180', q1)

        if buffer_nanoseconds1 > 10:
            k.gate("wait", [2, 0], buffer_nanoseconds1)
        k.gate('fl_cw_02', 2, 0)
        if t_nanoseconds > 10:
            k.gate("wait", [2, 0 ], t_nanoseconds)
        #k.gate('rx180', q0)
        #k.gate('rx180', q1)
        k.gate("wait",[2, 0 ], 1)
        k.measure(q0)
        k.gate("wait",[2, 0 ], 1)

        p.add_kernel(k)


    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p


def FastFeedbackControl(lantecy, qubit_idx: int, platf_cfg: str):
    """
    Single qubit sequence to test fast feedback control (fast conditional
    execution).
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        lantecy:        the waiting time between measurement and the feedback
                          pulse, which should be longer than the feedback
                          latency.
        feedback:       if apply
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
    Returns:
        p:              OpenQL Program object containing


    """
    platf = Platform('OpenQL_Platform', platf_cfg)
    p = Program(pname="FastFdbkCtrl", nqubits=platf.get_qubit_number(),
                p=platf)

    k = Kernel("FastFdbkCtrl_nofb", p=platf)
    k.prepz(qubit_idx)
    k.gate('rx90', qubit_idx)
    k.measure(qubit_idx)
    wait_nanoseconds = int(round(lantecy/1e-9))
    k.gate("wait", [qubit_idx], wait_nanoseconds)
    k.gate("i", qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    k = Kernel("FastFdbkCtrl_fb", p=platf)
    k.prepz(qubit_idx)
    k.gate('rx90', qubit_idx)
    k.measure(qubit_idx)
    wait_nanoseconds = int(round(lantecy/1e-9))
    k.gate("wait", [qubit_idx], wait_nanoseconds)
    k.gate('Crx180', qubit_idx) # fast feedback control here
    k.measure(qubit_idx)
    p.add_kernel(k)

    # adding the calibration points
    add_single_qubit_cal_points(p, platf=platf, qubit_idx=qubit_idx)

    with suppress_stdout():
        p.compile(verbose=False)
    # attribute get's added to program to help finding the output files
    p.output_dir = ql.get_output_dir()
    p.filename = join(p.output_dir, p.name + '.qisa')
    return p

