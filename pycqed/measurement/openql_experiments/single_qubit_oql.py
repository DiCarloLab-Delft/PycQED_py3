import numpy as np
from pycqed.measurement.randomized_benchmarking import \
    randomized_benchmarking as rb

import pycqed.measurement.openql_experiments.openql_helpers as oqh


def CW_tone(qubit_idx: int, platf_cfg: str):
    """
    Sequence to generate an "always on" pulse or "ContinuousWave" (CW) tone.
    This is a sequence that goes a bit against the paradigm of openql.
    """
    p = oqh.create_program('CW_tone', platf_cfg)

    k = oqh.create_kernel("Main", p)
    for i in range(40):
        k.gate('square', [qubit_idx])
    p.add_kernel(k)

    p = oqh.compile(p)
    return p


def vsm_timing_cal_sequence(qubit_idx: int, platf_cfg: str):
    """
    A sequence for calibrating the VSM timing delay.

    The marker idx is a qubit number for which a dummy pulse is played.
    This can be used as a reference.

    """
    p = oqh.create_program('vsm_timing_cal_sequence', platf_cfg)

    k = oqh.create_kernel("Main", p)
    k.prepz(qubit_idx)  # to ensure enough separation in timing
    k.gate('spec', [qubit_idx])
    p.add_kernel(k)

    p = oqh.compile(p)
    return p


def CW_RO_sequence(qubit_idx: int, platf_cfg: str):
    """
    A sequence that performs readout back to back without initialization.
    The separation of the readout triggers is done by specifying the duration
    of the readout parameter in the configuration file used for compilation.

    args:
        qubit_idx (int/list) :  the qubit(s) to be read out, can be either an
            int or a list of integers.
        platf_cfg (str)     :
    """
    p = oqh.create_program('CW_RO_sequence', platf_cfg=platf_cfg)

    k = oqh.create_kernel("main", p)
    if not hasattr(qubit_idx, "__iter__"):
        qubit_idx = [qubit_idx]
    k.gate('wait', qubit_idx, 0)
    for qi in qubit_idx:
        k.measure(qi)
    k.gate('wait', qubit_idx, 0)
    p.add_kernel(k)
    p = oqh.compile(p)
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
    p = oqh.create_program("pulsed_spec_seq", platf_cfg)
    k = oqh.create_kernel("main", p)

    nr_clocks = int(spec_pulse_length/20e-9)

    for i in range(nr_clocks):
        # The spec pulse is a pulse that lasts 20ns, because of the way the VSM
        # control works. By repeating it the duration can be controlled.
        k.gate('spec', [qubit_idx])
    k.measure(qubit_idx)
    p.add_kernel(k)

    p = oqh.compile(p)
    return p


def pulsed_spec_seq_marked(qubit_idx: int, spec_pulse_length: float,
                           platf_cfg: str, trigger_idx: int,
                           wait_time_ns: int = 0, cc: str = 'CCL'):
    """
    Sequence for pulsed spectroscopy, similar to old version. Difference is that
    this one triggers the 0th trigger port of the CCLight and uses the zeroth
    wave output on the AWG (currently hardcoded, should be improved)
    FIXME: comment outdated
    """
    p = oqh.create_program("pulsed_spec_seq_marked", platf_cfg)
    k = oqh.create_kernel("main", p)

    nr_clocks = int(spec_pulse_length/20e-9)
    print('Adding {} [ns] to spec seq'.format(wait_time_ns))
    if cc.upper() == 'CCL':
        spec_instr = 'spec'
    elif cc.upper() == 'QCC':
        spec_instr = 'sf_square'
    elif cc.lower() == 'cc':
        spec_instr = 'spec'
    else:
        raise ValueError('CC type not understood: {}'.format(cc))

    for i in range(nr_clocks):
        # The spec pulse is a pulse that lasts 20ns, because of the way the VSM
        # control works. By repeating it the duration can be controlled.
        k.gate(spec_instr, [trigger_idx])
    if trigger_idx != qubit_idx:
        k.wait([trigger_idx, qubit_idx], 0)
    k.wait([qubit_idx], wait_time_ns)
    k.measure(qubit_idx)
    p.add_kernel(k)

    p = oqh.compile(p)
    return p


def pulsed_spec_seq_v2(qubit_idx: int, spec_pulse_length: float,
                       platf_cfg: str, trigger_idx: int):
    """
    Sequence for pulsed spectroscopy, similar to old version. Difference is that
    this one triggers the 0th trigger port of the CCLight and usus the zeroth
    wave output on the AWG (currently hardcoded, should be improved)

    """
    p = oqh.create_program("pulsed_spec_seq_v2", platf_cfg)
    k = oqh.create_kernel("main", p)

    nr_clocks = int(spec_pulse_length/20e-9)

    for i in range(nr_clocks):
        # The spec pulse is a pulse that lasts 20ns, because of the way the VSM
        # control works. By repeating it the duration can be controlled.
        k.gate('spec', [trigger_idx])
    if trigger_idx != qubit_idx:
        k.wait([trigger_idx, qubit_idx], 0)

    k.measure(qubit_idx)
    p.add_kernel(k)

    p = oqh.compile(p)
    return p


def flipping(qubit_idx: int, number_of_flips, platf_cfg: str,
             equator: bool = False, cal_points: bool = True,
             ax: str = 'x', angle: str = '180'):
    """
    Generates a flipping sequence that performs multiple pi-pulses
    Basic sequence:
        - (X)^n - RO
        or
        - (Y)^n - RO
        or
        - (X90)^2n - RO
        or
        - (Y90)^2n - RO


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
    p = oqh.create_program("flipping", platf_cfg)

    for i, n in enumerate(number_of_flips):
        k = oqh.create_kernel('flipping_{}'.format(i), p)
        k.prepz(qubit_idx)
        if cal_points and (i == (len(number_of_flips)-4) or
                           i == (len(number_of_flips)-3)):
            k.measure(qubit_idx)
        elif cal_points and (i == (len(number_of_flips)-2) or
                             i == (len(number_of_flips)-1)):
            if ax == 'y':
                k.y(qubit_idx)
            else:
                k.x(qubit_idx)
            k.measure(qubit_idx)
        else:
            if equator:
                if ax == 'y':
                    k.gate('ry90', [qubit_idx])
                else:
                    k.gate('rx90', [qubit_idx])
            for j in range(n):
                if ax == 'y' and angle == '90':
                    k.gate('ry90', [qubit_idx])
                    k.gate('ry90', [qubit_idx])
                elif ax == 'y' and angle == '180':
                    k.y(qubit_idx)
                elif angle == '90':
                    k.gate('rx90', [qubit_idx])
                    k.gate('rx90', [qubit_idx])
                else:
                    k.x(qubit_idx)
            k.measure(qubit_idx)
        p.add_kernel(k)

    p = oqh.compile(p)
    return p


def AllXY(qubit_idx: int, platf_cfg: str, double_points: bool = True):
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
    p = oqh.create_program("AllXY", platf_cfg)

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
            k.prepz(qubit_idx)
            k.gate(xy[0], [qubit_idx])
            k.gate(xy[1], [qubit_idx])
            k.measure(qubit_idx)
            p.add_kernel(k)

    p = oqh.compile(p)
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
    p = oqh.create_program('T1', platf_cfg)

    for i, time in enumerate(times[:-4]):
        k = oqh.create_kernel('T1_{}'.format(i), p)
        k.prepz(qubit_idx)
        wait_nanoseconds = int(round(time/1e-9))
        k.gate('rx180', [qubit_idx])
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        k.measure(qubit_idx)
        p.add_kernel(k)

    # adding the calibration points
    oqh.add_single_qubit_cal_points(p,  qubit_idx=qubit_idx)

    p = oqh.compile(p)
    return p


def T1_second_excited_state(times, qubit_idx: int, platf_cfg: str):
    """
    Single qubit T1 sequence for the second excited states.
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        times:          the list of waiting times for each T1 element
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
    Returns:
        p:              OpenQL Program object containing


    """
    p = oqh.create_program("T1_2nd_exc", platf_cfg)

    for i, time in enumerate(times):
        for j in range(2):
            k = oqh.create_kernel("T1_2nd_exc_{}_{}".format(i, j), p)
            k.prepz(qubit_idx)
            wait_nanoseconds = int(round(time/1e-9))
            k.gate('rx180', [qubit_idx])
            k.gate('rx12', [qubit_idx])
            k.gate("wait", [qubit_idx], wait_nanoseconds)
            if j == 1:
                k.gate('rx180', [qubit_idx])
            k.measure(qubit_idx)
            p.add_kernel(k)

    # adding the calibration points
    oqh.add_single_qubit_cal_points(p,  qubit_idx=qubit_idx,
                                    f_state_cal_pts=True)

    dt = times[1] - times[0]
    sweep_points = np.concatenate([np.repeat(times, 2),
                                   times[-1]+dt*np.arange(6)+dt])
    # attribute get's added to program to help finding the output files
    p.sweep_points = sweep_points

    p = oqh.compile(p)
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
    p = oqh.create_program("Ramsey", platf_cfg)

    for i, time in enumerate(times[:-4]):
        k = oqh.create_kernel("Ramsey_{}".format(i), p)
        k.prepz(qubit_idx)
        wait_nanoseconds = int(round(time/1e-9))
        k.gate('rx90', [qubit_idx])
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        k.gate('rx90', [qubit_idx])
        k.measure(qubit_idx)
        p.add_kernel(k)

    # adding the calibration points
    oqh.add_single_qubit_cal_points(p,  qubit_idx=qubit_idx)

    p = oqh.compile(p)
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
    p = oqh.create_program("echo", platf_cfg)

    for i, time in enumerate(times[:-4]):

        k = oqh.create_kernel("echo_{}".format(i), p)
        k.prepz(qubit_idx)
        # nr_clocks = int(time/20e-9/2)
        wait_nanoseconds = int(round(time/1e-9/2))
        k.gate('rx90', [qubit_idx])
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        k.gate('rx180', [qubit_idx])
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        #k.gate('rx90', [qubit_idx])
        angle = (i*40) % 360
        cw_idx = angle//20 + 9
        if angle == 0:
            k.gate('rx90', [qubit_idx])
        else:
            k.gate('cw_{:02}'.format(cw_idx), [qubit_idx])

        k.measure(qubit_idx)
        p.add_kernel(k)

    # adding the calibration points
    oqh.add_single_qubit_cal_points(p,  qubit_idx=qubit_idx)

    p = oqh.compile(p)
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

    p = oqh.create_program("idle_error_rate", platf_cfg)

    sweep_points = []
    for N in nr_of_idle_gates:
        for state in states:
            if state not in allowed_states:
                raise ValueError('State must be in {}'.format(allowed_states))
            k = oqh.create_kernel("idle_prep{}_N{}".format(state, N), p)
            # 1. Preparing in the right basis
            k.prepz(qubit_idx)
            if post_select:
                # adds an initialization measurement used to post-select
                k.measure(qubit_idx)
            if state == '1':
                k.gate('rx180', [qubit_idx])
            elif state == '+':
                k.gate('rym90', [qubit_idx])
            # 2. The "waiting" gates
            wait_nanoseconds = N*gate_duration_ns
            if state == '+' and echo:
                k.gate("wait", [qubit_idx], wait_nanoseconds//2)
                k.gate('rx180', [qubit_idx])
                k.gate("wait", [qubit_idx], wait_nanoseconds//2)
            else:
                k.gate("wait", [qubit_idx], wait_nanoseconds)
            # 3. Reading out in the proper basis
            if state == '+' and echo:
                k.gate('rym90', [qubit_idx])
            elif state == '+':
                k.gate('ry90', [qubit_idx])
            k.measure(qubit_idx)
            p.add_kernel(k)
        sweep_points.append(N)

    # FIXME: remove try-except, when we depend hardly on >=openql-0.6
    try:
        p.set_sweep_points(sweep_points)
    except TypeError:
        # openql-0.5 compatibility
        p.set_sweep_points(sweep_points, num_sweep_points=len(sweep_points))
    p.sweep_points = sweep_points
    p = oqh.compile(p)
    return p


def single_elt_on(qubit_idx: int, platf_cfg: str):
    p = oqh.create_program('single_elt_on', platf_cfg)

    k = oqh.create_kernel('main', p)

    k.prepz(qubit_idx)
    k.x(qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    p = oqh.compile(p)
    return p


def off_on(qubit_idx: int, pulse_comb: str, initialize: bool, platf_cfg: str):
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
    p = oqh.create_program('off_on', platf_cfg)

    # # Off
    if 'off' in pulse_comb.lower():
        k = oqh.create_kernel("off", p)
        k.prepz(qubit_idx)
        if initialize:
            k.measure(qubit_idx)
        k.measure(qubit_idx)
        p.add_kernel(k)

    if 'on' in pulse_comb.lower():
        k = oqh.create_kernel("on", p)
        k.prepz(qubit_idx)
        if initialize:
            k.measure(qubit_idx)
        k.gate('rx180', [qubit_idx])
        k.measure(qubit_idx)
        p.add_kernel(k)

    if ('on' not in pulse_comb.lower()) and ('off' not in pulse_comb.lower()):
        raise ValueError()

    p = oqh.compile(p)
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
    p = oqh.create_program('butterfly', platf_cfg)

    k = oqh.create_kernel('0', p)
    k.prepz(qubit_idx)
    if initialize:
        k.measure(qubit_idx)
    k.measure(qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    k = oqh.create_kernel('1', p)
    k.prepz(qubit_idx)
    if initialize:
        k.measure(qubit_idx)
    k.x(qubit_idx)
    k.measure(qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    p = oqh.compile(p)

    return p


def RTE(qubit_idx: int, sequence_type: str, platf_cfg: str,
        net_gate: str, feedback=False):
    """
    Creates a sequence for the rounds to event (RTE) experiment

    Args:
        qubit_idx             (int) :
        sequence_type ['echo'|'pi'] :
        net_gate         ['i'|'pi'] :
        feedback             (bool) : if last measurement == 1, then apply
            an extra pi-pulse. N.B. more options for fast feedback should be
            added.

    N.B. there is some hardcoded stuff in here (such as rest times).
    It should be better documented what this is and what it does.
    """
    p = oqh.create_program('RTE', platf_cfg)

    k = oqh.create_kernel('RTE', p)
    if sequence_type == 'echo':
        k.gate('rx90', [qubit_idx])
        k.gate('i', [qubit_idx])
        k.gate('i', [qubit_idx])
        k.gate('i', [qubit_idx])
        #k.gate('i', [qubit_idx])
        k.gate('rx180', [qubit_idx])
        k.gate('i', [qubit_idx])
        k.gate('i', [qubit_idx])
        k.gate('i', [qubit_idx])
        #k.gate('i', [qubit_idx])
        if net_gate == 'pi':
            k.gate('rxm90', [qubit_idx])
        elif net_gate == 'i':
            k.gate('rx90', [qubit_idx])
        else:
            raise ValueError('net_gate ({})should be "i" or "pi"'.format(
                net_gate))
        if feedback:
            k.gate("wait", [qubit_idx], 20)
            k.gate('C1rx180', [qubit_idx])
    elif sequence_type == 'pi':
        if net_gate == 'pi':
            k.gate('rx180', [qubit_idx])
        elif net_gate == 'i':
            pass
        else:
            raise ValueError('net_gate ({})should be "i" or "pi"'.format(
                net_gate))
        if feedback:
            k.gate("wait", [qubit_idx], 20)
            k.gate('C1rx180', [qubit_idx])
    else:
        raise ValueError('sequence_type ({})should be "echo" or "pi"'.format(
            sequence_type))
    k.measure(qubit_idx)
    p.add_kernel(k)

    p = oqh.compile(p)
    return p


def randomized_benchmarking(qubit_idx: int, platf_cfg: str,
                            nr_cliffords, nr_seeds: int,
                            net_clifford: int = 0, restless: bool = False,
                            program_name: str = 'randomized_benchmarking',
                            cal_points: bool = True,
                            double_curves: bool = False):
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
    p = oqh.create_program(program_name, platf_cfg)

    i = 0
    for seed in range(nr_seeds):
        for j, n_cl in enumerate(nr_cliffords):
            k = oqh.create_kernel('RB_{}Cl_s{}_{}'.format(n_cl, seed, j), p)

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
                    k.gate('cl_{}'.format(cl), [qubit_idx])
                k.measure(qubit_idx)
            p.add_kernel(k)

    p = oqh.compile(p)
    return p


def motzoi_XY(qubit_idx: int, platf_cfg: str,
              program_name: str = 'motzoi_XY'):
    '''
    Sequence used for calibrating the motzoi parameter.
    Consists of yX and xY

    Beware that the elements alternate, if you want to measure both Xy and Yx
    at each motzoi you need repeating motzoi parameters. This was chosen
    to be more easily compatible with standard detector functions and sweep pts

    '''
    p = oqh.create_program(program_name, platf_cfg)

    k = oqh.create_kernel("yX", p)
    k.prepz(qubit_idx)
    k.gate('ry90', [qubit_idx])
    k.gate('rx180', [qubit_idx])
    k.measure(qubit_idx)
    p.add_kernel(k)

    k = oqh.create_kernel("xY", p)
    k.prepz(qubit_idx)
    k.gate('rx90', [qubit_idx])
    k.gate('ry180', [qubit_idx])
    k.measure(qubit_idx)
    p.add_kernel(k)

    p = oqh.compile(p)
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


def FluxTimingCalibration(qubit_idx: int, times, platf_cfg: str,
                          flux_cw: str = 'fl_cw_02',
                          cal_points: bool = True,
                          mw_gate: str = "rx90"):
    """
    A Ramsey sequence with varying waiting times `times` around a flux pulse.
    """
    p = oqh.create_program('FluxTimingCalibration', platf_cfg)

    # don't use last 4 points if calibration points are used
    if cal_points:
        times = times[:-4]
    for i_t, t in enumerate(times):
        t_nanoseconds = int(round(t/1e-9))
        k = oqh.create_kernel('pi_flux_pi_{}'.format(i_t), p)
        k.prepz(qubit_idx)
        k.gate(mw_gate, [qubit_idx])
        # k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        k.gate("wait", [], 0)  # alignment workaround
        # k.gate(flux_cw, [2, 0])
        k.gate('sf_square', [qubit_idx])
        if t_nanoseconds > 10:
            # k.gate("wait", [0, 1, 2, 3, 4, 5, 6], t_nanoseconds)
            k.gate("wait", [], t_nanoseconds)  # alignment workaround
            # k.gate("wait", [qubit_idx], t_nanoseconds)
        k.gate(mw_gate, [qubit_idx])
        k.measure(qubit_idx)
        p.add_kernel(k)

    if cal_points:
        oqh.add_single_qubit_cal_points(p,  qubit_idx=qubit_idx)
    p = oqh.compile(p)
    return p


def TimingCalibration_1D(qubit_idx: int, times, platf_cfg: str,
                         # flux_cw: str = 'fl_cw_02',
                         cal_points: bool = True):
    """
    A Ramsey sequence with varying waiting times `times`in between.
    It calibrates the timing between spec and measurement pulse.
    """
    p = oqh.create_program('TimingCalibration1D', platf_cfg)

    # don't use last 4 points if calibration points are used
    if cal_points:
        times = times[:-4]
    for i_t, t in enumerate(times):
        t_nanoseconds = int(round(t/1e-9))
        k = oqh.create_kernel('pi_times_pi_{}'.format(i_t), p)
        k.prepz(qubit_idx)
        k.gate('rx90', [qubit_idx])
        # k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        k.gate("wait", [], 0)  # alignment workaround
        # k.gate(flux_cw, [2, 0])
        # k.gate('sf_square', [qubit_idx])
        if t_nanoseconds > 10:
            # k.gate("wait", [0, 1, 2, 3, 4, 5, 6], t_nanoseconds)
            k.gate("wait", [], t_nanoseconds)  # alignment workaround
            # k.gate("wait", [qubit_idx], t_nanoseconds)
        k.gate('rx90', [qubit_idx])
        k.measure(qubit_idx)
        p.add_kernel(k)

    if cal_points:
        oqh.add_single_qubit_cal_points(p,  qubit_idx=qubit_idx)
    p = oqh.compile(p)
    return p


def FluxTimingCalibration_2q(q0, q1, buffer_time1, times, platf_cfg: str):
    """
    A Ramsey sequence with varying waiting times `times` around a flux pulse.

    N.B. this function is not consistent with "FluxTimingCalibration".
    This should be fixed
    """
    p = oqh.create_program("FluxTimingCalibration_2q", platf_cfg)

    buffer_nanoseconds1 = int(round(buffer_time1/1e-9))

    for i_t, t in enumerate(times):

        t_nanoseconds = int(round(t/1e-9))
        k = oqh.create_kernel("pi-flux-pi_{}".format(i_t), p)
        k.prepz(q0)
        k.prepz(q1)

        k.gate('rx180', [q0])
        k.gate('rx180', [q1])

        if buffer_nanoseconds1 > 10:
            k.gate("wait", [2, 0], buffer_nanoseconds1)
        k.gate('fl_cw_02', [2, 0])
        if t_nanoseconds > 10:
            k.gate("wait", [2, 0], t_nanoseconds)
        #k.gate('rx180', [q0])
        #k.gate('rx180', [q1])
        k.gate("wait", [2, 0], 1)
        k.measure(q0)
        k.gate("wait", [2, 0], 1)

        p.add_kernel(k)

    p = oqh.compile(p)
    return p


def FastFeedbackControl(latency, qubit_idx: int, platf_cfg: str):
    """
    Single qubit sequence to test fast feedback control (fast conditional
    execution).
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        latency:        the waiting time between measurement and the feedback
                          pulse, which should be longer than the feedback
                          latency.
        feedback:       if apply
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
    Returns:
        p:              OpenQL Program object containing


    """
    p = oqh.create_program("FastFeedbackControl", platf_cfg)

    k = oqh.create_kernel("FastFdbkCtrl_nofb", p)
    k.prepz(qubit_idx)
    k.gate('rx90', [qubit_idx])
    # k.gate('rx180', [qubit_idx])
    k.measure(qubit_idx)
    wait_nanoseconds = int(round(latency/1e-9))
    k.gate("wait", [qubit_idx], wait_nanoseconds)
    k.gate("i", [qubit_idx])
    k.measure(qubit_idx)

    p.add_kernel(k)

    k = oqh.create_kernel("FastFdbkCtrl_fb0", p)
    k.prepz(qubit_idx)
    k.gate('rx90', [qubit_idx])
    # k.gate('rx180', [qubit_idx])
    k.measure(qubit_idx)
    wait_nanoseconds = int(round(latency/1e-9))
    k.gate("wait", [qubit_idx], wait_nanoseconds)
    k.gate('C0rx180', [qubit_idx])  # fast feedback control here
    k.measure(qubit_idx)
    p.add_kernel(k)

    k = oqh.create_kernel("FastFdbkCtrl_fb1", p)
    k.prepz(qubit_idx)
    k.gate('rx90', [qubit_idx])
    # k.gate('rx180', [qubit_idx])
    k.measure(qubit_idx)
    wait_nanoseconds = int(round(latency/1e-9))
    k.gate("wait", [qubit_idx], wait_nanoseconds)
    k.gate('C1rx180', [qubit_idx])  # fast feedback control here
    k.measure(qubit_idx)
    p.add_kernel(k)

    # adding the calibration points
    oqh.add_single_qubit_cal_points(p,  qubit_idx=qubit_idx)

    p = oqh.compile(p)
    return p


def ef_rabi_seq(q0: int,
                amps: list,
                platf_cfg: str,
                recovery_pulse: bool = True,
                add_cal_points: bool = True):
    """
    Sequence used to calibrate pulses for 2nd excited state (ef/12 transition)

    Timing of the sequence:
    q0:   --   X180 -- X12 -- (X180) -- RO

    Args:
        q0      (str): name of the addressed qubit
        amps   (list): amps for the two state pulse, note that these are only
            used to label the kernels. Load the pulse in the LutMan
        recovery_pulse (bool): if True adds a recovery pulse to enhance
            contrast in the measured signal.
    """
    if len(amps) > 18:
        raise ValueError('Only 18 free codewords available for amp pulses')

    p = oqh.create_program("ef_rabi_seq", platf_cfg)
    # These angles correspond to special pi/2 pulses in the lutman
    for i, amp in enumerate(amps):
        # cw_idx corresponds to special hardcoded pulses in the lutman
        cw_idx = i + 9

        k = oqh.create_kernel("ef_A{}_{}".format(int(abs(1000*amp)),i), p)
        k.prepz(q0)
        k.gate('rx180', [q0])
        k.gate('cw_{:02}'.format(cw_idx), [q0])
        if recovery_pulse:
            k.gate('rx180', [q0])
        k.measure(q0)
        p.add_kernel(k)
    if add_cal_points:
        p = oqh.add_single_qubit_cal_points(p, qubit_idx=q0)

    p = oqh.compile(p)

    if add_cal_points:
        cal_pts_idx = [amps[-1] + .1, amps[-1] + .15,
                       amps[-1] + .2, amps[-1] + .25]
    else:
        cal_pts_idx = []

    p.sweep_points = np.concatenate([amps, cal_pts_idx])
    # FIXME: remove try-except, when we depend hardly on >=openql-0.6
    try:
        p.set_sweep_points(p.sweep_points)
    except TypeError:
        # openql-0.5 compatibility
        p.set_sweep_points(p.sweep_points, len(p.sweep_points))
    return p


def Depletion(time, qubit_idx: int, platf_cfg: str, double_points: bool):
    """
    Input pars:
        times:          the list of waiting times for each ALLXY element
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
    Returns:
        p:              OpenQL Program object containing
    """

    allXY = [['i', 'i'], ['rx180', 'rx180'], ['ry180', 'ry180'],
             ['rx180', 'ry180'], ['ry180', 'rx180'],
             ['rx90', 'i'], ['ry90', 'i'], ['rx90', 'ry90'],
             ['ry90', 'rx90'], ['rx90', 'ry180'], ['ry90', 'rx180'],
             ['rx180', 'ry90'], ['ry180', 'rx90'], ['rx90', 'rx180'],
             ['rx180', 'rx90'], ['ry90', 'ry180'], ['ry180', 'ry90'],
             ['rx180', 'i'], ['ry180', 'i'], ['rx90', 'rx90'],
             ['ry90', 'ry90']]

    p = oqh.create_program('Depletion', platf_cfg)

    try:
        p.set_sweep_points(np.arange(len(allXY), dtype=float))
    except TypeError:
        # openql-0.5 compatibility
        p.set_sweep_points(np.arange(len(allXY), dtype=float), len(allXY))

    if double_points:
        js=2
    else:
        js=1

    for i, xy in enumerate(allXY):
        for j in range(js):
            k = oqh.create_kernel('Depletion_{}_{}'.format(i, j), p)
            # Prepare qubit
            k.prepz(qubit_idx)
            # Initial measurement
            k.measure(qubit_idx)
            # Wait time
            wait_nanoseconds = int(round(time/1e-9))
            k.gate("wait", [qubit_idx], wait_nanoseconds)
            # AllXY pulse
            k.gate(xy[0], [qubit_idx])
            k.gate(xy[1], [qubit_idx])
            # Final measurement
            k.measure(qubit_idx)
            p.add_kernel(k)

    p = oqh.compile(p)
    return p

def TEST_RTE(qubit_idx: int, platf_cfg: str,
             measurements:int):
    """

    """
    p = oqh.create_program('RTE', platf_cfg)

    k = oqh.create_kernel('RTE', p)
    k.prepz(qubit_idx)
    ######################
    # Parity check
    ######################
    for m in range(measurements):
        # Superposition
        k.gate('rx90', [qubit_idx])
        # CZ emulation
        k.gate('i', [qubit_idx])
        k.gate('i', [qubit_idx])
        k.gate('i', [qubit_idx])
        # Refocus
        k.gate('rx180', [qubit_idx])
        # CZ emulation
        k.gate('i', [qubit_idx])
        k.gate('i', [qubit_idx])
        k.gate('i', [qubit_idx])
        # Recovery pulse
        k.gate('rx90', [qubit_idx])
        k.measure(qubit_idx)

    p.add_kernel(k)

    p = oqh.compile(p)
    return p