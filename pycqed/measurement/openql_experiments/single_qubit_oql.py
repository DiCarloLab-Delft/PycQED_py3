import numpy as np
from pycqed.measurement.randomized_benchmarking import \
    randomized_benchmarking as rb

from pycqed.measurement.openql_experiments.openql_helpers import OqlProgram


def CW_tone(qubit_idx: int, platf_cfg: str):
    """
    Sequence to generate an "always on" pulse or "ContinuousWave" (CW) tone.
    This is a sequence that goes a bit against the paradigm of openql.
    """
    p = OqlProgram('CW_tone', platf_cfg)

    k = p.create_kernel("Main")
    for i in range(40):
        k.gate('square', [qubit_idx])
    p.add_kernel(k)

    p.compile()
    return p


def vsm_timing_cal_sequence(qubit_idx: int, platf_cfg: str):
    """
    A sequence for calibrating the VSM timing delay.

    The marker idx is a qubit number for which a dummy pulse is played.
    This can be used as a reference.

    """
    p = OqlProgram('vsm_timing_cal_sequence', platf_cfg)

    k = p.create_kernel("Main")
    k.prepz(qubit_idx)  # to ensure enough separation in timing
    k.gate('spec', [qubit_idx])
    p.add_kernel(k)

    p.compile()
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
    p = OqlProgram('CW_RO_sequence', platf_cfg=platf_cfg)

    k = p.create_kernel("main")
    if not hasattr(qubit_idx, "__iter__"):
        qubit_idx = [qubit_idx]
    k.barrier(qubit_idx)
    for qi in qubit_idx:
        k.measure(qi)
    k.barrier(qubit_idx)
    p.add_kernel(k)
    p.compile()
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
    p = OqlProgram("pulsed_spec_seq", platf_cfg)
    k = p.create_kernel("main")

    nr_clocks = int(spec_pulse_length/20e-9)

    for i in range(nr_clocks):
        # The spec pulse is a pulse that lasts 20ns, because of the way the VSM
        # control works. By repeating it the duration can be controlled.
        k.gate('spec', [qubit_idx])
    k.measure(qubit_idx)
    p.add_kernel(k)

    p.compile()
    return p


def pulsed_spec_seq_marked(qubit_idx: int, spec_pulse_length: float,
                           platf_cfg: str, trigger_idx: int, trigger_idx_2: int = None,
                           wait_time_ns: int = 0, cc: str = 'CCL'):
    """
    Sequence for pulsed spectroscopy, similar to old version. Difference is that
    this one triggers the 0th trigger port of the CCLight and uses the zeroth
    wave output on the AWG (currently hardcoded, should be improved)
    FIXME: comment outdated
    """
    p = OqlProgram("pulsed_spec_seq_marked", platf_cfg)
    k = p.create_kernel("main")

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

    # k.prepz(qubit_idx)
    for i in range(nr_clocks):
        # The spec pulse is a pulse that lasts 20ns, because of the way the VSM
        # control works. By repeating it the duration can be controlled.
        k.gate(spec_instr, [trigger_idx])
        if trigger_idx_2 is not None:
            k.gate(spec_instr, [trigger_idx_2])
            k.barrier([trigger_idx, trigger_idx_2])

    if trigger_idx != qubit_idx:
        k.barrier([trigger_idx, qubit_idx])
        if trigger_idx_2 is not None:
            k.barier([trigger_idx_2])
    k.wait([qubit_idx], wait_time_ns)
    k.measure(qubit_idx)
    p.add_kernel(k)

    p.compile()
    return p


def pulsed_spec_seq_v2(qubit_idx: int, spec_pulse_length: float,
                       platf_cfg: str, trigger_idx: int):
    """
    Sequence for pulsed spectroscopy, similar to old version. Difference is that
    this one triggers the 0th trigger port of the CCLight and usus the zeroth
    wave output on the AWG (currently hardcoded, should be improved)

    """
    p = OqlProgram("pulsed_spec_seq_v2", platf_cfg)
    k = p.create_kernel("main")

    nr_clocks = int(spec_pulse_length/20e-9)

    for i in range(nr_clocks):
        # The spec pulse is a pulse that lasts 20ns, because of the way the VSM
        # control works. By repeating it the duration can be controlled.
        k.gate('spec', [trigger_idx])
    if trigger_idx != qubit_idx:
        k.barrier([trigger_idx, qubit_idx])

    k.measure(qubit_idx)
    p.add_kernel(k)

    p.compile()
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
    p = OqlProgram("flipping", platf_cfg)

    for i, n in enumerate(number_of_flips):
        k = p.create_kernel('flipping_{}'.format(i))
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

    p.compile()
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
    p = OqlProgram("AllXY", platf_cfg)

    allXY = [['i', 'i'], ['rx180', 'rx180'], ['ry180', 'ry180'],
             ['rx180', 'ry180'], ['ry180', 'rx180'],
             ['rx90', 'i'], ['ry90', 'i'], ['rx90', 'ry90'],
             ['ry90', 'rx90'], ['rx90', 'ry180'], ['ry90', 'rx180'],
             ['rx180', 'ry90'], ['ry180', 'rx90'], ['rx90', 'rx180'],
             ['rx180', 'rx90'], ['ry90', 'ry180'], ['ry180', 'ry90'],
             ['rx180', 'i'], ['ry180', 'i'], ['rx90', 'rx90'],
             ['ry90', 'ry90']]

    # this should be implicit
    if 0: # FIXME: p.set_sweep_points has been replaced by p.sweep_points, since that was missing here they are probably not necessary for this function
        p.set_sweep_points(np.arange(len(allXY), dtype=float))

    for i, xy in enumerate(allXY):
        if double_points:
            js = 2
        else:
            js = 1
        for j in range(js):
            k = p.create_kernel("AllXY_{}_{}".format(i, j))
            k.prepz(qubit_idx)
            k.gate(xy[0], [qubit_idx])
            k.gate(xy[1], [qubit_idx])
            k.measure(qubit_idx)
            p.add_kernel(k)

    p.compile()
    return p


def T1(
        qubit_idx: int,
        platf_cfg: str, 
        times: list, 
        nr_cz_instead_of_idle_time: list=None,
        qb_cz_idx: int=None, 
        nr_flux_dance: float=None, 
        wait_time_after_flux_dance: float=0
        ):
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
    p = OqlProgram('T1', platf_cfg)

    for i, time in enumerate(times[:-4]):
        k = p.create_kernel('T1_{}'.format(i))
        k.prepz(qubit_idx)

        if nr_flux_dance:
            for _ in range(int(nr_flux_dance)):
                for step in [1,2,3,4]:
                    # if refocusing:
                    #     k.gate(f'flux-dance-{step}-refocus', [0])
                    # else:
                    k.gate(f'flux-dance-{step}', [0])
                k.barrier([])  # alignment 
            k.gate("wait", [], wait_time_after_flux_dance)

        k.gate('rx180', [qubit_idx])

        if nr_cz_instead_of_idle_time is not None:
            for n in range(nr_cz_instead_of_idle_time[i]):
                k.gate("cz", [qubit_idx, qb_cz_idx])
            k.barrier([])  # alignment 
            k.gate("wait", [], wait_time_after_flux_dance)
        else:
            wait_nanoseconds = int(round(time/1e-9))
            k.gate("wait", [qubit_idx], wait_nanoseconds)
        
        k.measure(qubit_idx)
        p.add_kernel(k)

    # adding the calibration points
    p.add_single_qubit_cal_points(qubit_idx=qubit_idx)

    p.compile()
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
    p = OqlProgram("T1_2nd_exc", platf_cfg)

    for i, time in enumerate(times):
        for j in range(2):
            k = p.create_kernel("T1_2nd_exc_{}_{}".format(i, j))
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
    p.add_single_qubit_cal_points(qubit_idx=qubit_idx,
                                    f_state_cal_pts=True)

    dt = times[1] - times[0]
    sweep_points = np.concatenate([np.repeat(times, 2),
                                   times[-1]+dt*np.arange(6)+dt])
    # attribute get's added to program to help finding the output files
    p.sweep_points = sweep_points

    p.compile()
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
    p = OqlProgram("Ramsey", platf_cfg)

    for i, time in enumerate(times[:-4]):
        k = p.create_kernel("Ramsey_{}".format(i))
        k.prepz(qubit_idx)
        wait_nanoseconds = int(round(time/1e-9))
        k.gate('rx90', [qubit_idx])
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        k.gate('rx90', [qubit_idx])
        k.measure(qubit_idx)
        p.add_kernel(k)

    # adding the calibration points
    p.add_single_qubit_cal_points(qubit_idx=qubit_idx)

    p.compile()
    return p


def complex_Ramsey(times, qubit_idx: int, platf_cfg: str):
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
    p = OqlProgram("complex_Ramsey", platf_cfg)

    prerotations = ['rx90','rym90']
    timeloop = times[:-4][::2]
    for i, time in enumerate(timeloop):
        for rot in prerotations:
            k = p.create_kernel("Ramsey_" + rot + "_{}".format(i))
            k.prepz(qubit_idx)
            wait_nanoseconds = int(round(time/1e-9))
            k.gate('rx90', [qubit_idx])
            k.gate("wait", [qubit_idx], wait_nanoseconds)
            k.gate(rot, [qubit_idx])
            k.measure(qubit_idx)
            p.add_kernel(k)

    # adding the calibration points
    p.add_single_qubit_cal_points(qubit_idx=qubit_idx)

    p.compile()
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
    p = OqlProgram("echo", platf_cfg)

    for i, time in enumerate(times[:-4]):

        k = p.create_kernel("echo_{}".format(i))
        k.prepz(qubit_idx)
        # nr_clocks = int(time/20e-9/2)
        wait_nanoseconds = int(round(time/1e-9/2))
        k.gate('rx90', [qubit_idx])
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        k.gate('rx180', [qubit_idx])
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        # k.gate('rx90', [qubit_idx])
        angle = (i*40) % 360
        cw_idx = angle//20 + 9
        if angle == 0:
            k.gate('rx90', [qubit_idx])
        else:
            k.gate('cw_{:02}'.format(cw_idx), [qubit_idx])

        k.measure(qubit_idx)
        p.add_kernel(k)

    # adding the calibration points
    p.add_single_qubit_cal_points(qubit_idx=qubit_idx)

    p.compile()
    return p

def CPMG(times, order: int, qubit_idx: int, platf_cfg: str):
    """
    Single qubit CPMG sequence.
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        times:          the list of waiting times for each Echo element
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
    Returns:
        p:              OpenQL Program object containing

    """
    p = OqlProgram("CPMG", platf_cfg)

    for i, time in enumerate(times[:-4]):

        k = p.create_kernel("CPMG_{}".format(i))
        k.prepz(qubit_idx)
        # nr_clocks = int(time/20e-9/2)

        wait_nanoseconds = int(round((time/1e-9)/(2*order)))
        k.gate('rx90', [qubit_idx])
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        for j in range(order-1):
            k.gate('ry180', [qubit_idx])
            k.gate("wait", [qubit_idx], 2*wait_nanoseconds)
        k.gate('ry180', [qubit_idx])
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        # angle = (i*40)%360
        # cw_idx = angle//20 + 9
        # if angle == 0:
        k.gate('rx90', [qubit_idx])
        # else:
        #     k.gate('cw_{:02}'.format(cw_idx), [qubit_idx])

        k.measure(qubit_idx)
        p.add_kernel(k)



    # adding the calibration points
    p.add_single_qubit_cal_points(qubit_idx=qubit_idx)

    p.compile()
    return p


def CPMG_SO(orders, tauN: int, qubit_idx: int, platf_cfg: str):
    """
    Single qubit CPMG sequence.
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        times:          the list of waiting times for each Echo element
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
    Returns:
        p:              OpenQL Program object containing

    """
    p = OqlProgram("CPMG_SO", platf_cfg)

    for i, order in enumerate(orders[:-4]):

        k = p.create_kernel("CPMG_SO_{}".format(i))
        k.prepz(qubit_idx)
        # nr_clocks = int(time/20e-9/2)

        wait_nanoseconds = int(round((tauN/1e-9)/2))
        k.gate('rx90', [qubit_idx])
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        for j in range(order-1):
            k.gate('ry180', [qubit_idx])
            k.gate("wait", [qubit_idx], 2*wait_nanoseconds)
        k.gate('ry180', [qubit_idx])
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        # angle = (i*40)%360
        # cw_idx = angle//20 + 9
        # if angle == 0:
        k.gate('rx90', [qubit_idx])
        # else:
        #     k.gate('cw_{:02}'.format(cw_idx), [qubit_idx])

        k.measure(qubit_idx)
        p.add_kernel(k)

    # adding the calibration points
    p.add_single_qubit_cal_points(qubit_idx=qubit_idx)

    p.compile()
    return p

def spin_lock_simple(times, qubit_idx: int, platf_cfg: str, 
                     mw_gate_duration: float = 40e-9, 
                     tomo: bool = False):
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
    p = OqlProgram("spin_lock_simple", platf_cfg)
    # Poor mans tomography:
    if tomo:
        tomo_gates = ['I','rX180','rX12']
    else:
        tomo_gates = ['I']

    if tomo:
        timeloop = times[:-6][::3]
    else:
        timeloop = times[:-4]

    for i, time in enumerate(timeloop):
        for tomo_gate in tomo_gates:
            k = p.create_kernel("spin_lock_simple" + "_tomo_" + tomo_gate + "_{}".format(i))
            k.prepz(qubit_idx)
            # nr_clocks = int(time/20e-9/2)
            square_us_cycles = np.floor(time/1e-6).astype(int)
            square_ns_cycles = np.round((time%1e-6)/mw_gate_duration).astype(int)
            # print("square_us_cycles", square_us_cycles)
            # print("square_us_cycles", square_ns_cycles)
            k.gate('rYm90', [qubit_idx])
            for suc in range(square_us_cycles):
                k.gate('cw_10', [qubit_idx]) # make sure that the square pulse lasts 1us
            for snc in range(square_ns_cycles):
                k.gate('cw_11', [qubit_idx]) # make sure that the square pulse lasts mw_gate_duration ns
            k.gate('rYm90', [qubit_idx])
            if tomo:
                k.gate(tomo_gate,[qubit_idx])
            k.measure(qubit_idx)
            p.add_kernel(k)

    # adding the calibration points
    p.add_single_qubit_cal_points(qubit_idx=qubit_idx, f_state_cal_pts=tomo)
    p.compile()
    return p


def rabi_frequency(times, qubit_idx: int, platf_cfg: str, 
                    mw_gate_duration: float = 40e-9,
                    tomo: bool = False):
    """
    Rabi Sequence consising out of sequence of square pulses
    Writes output files to the directory specified in openql.
    Output directory is set as an attribute to the program for convenience.

    Input pars:
        times:          the list of waiting times for each Echo element
        qubit_idx:      int specifying the target qubit (starting at 0)
        platf_cfg:      filename of the platform config file
    Returns:
        p:              OpenQL Program object containing

    """
    p = OqlProgram("rabi_frequency", platf_cfg)

    if tomo:
        tomo_gates = ['I','rX180','rX12']
    else:
        tomo_gates = ['I']

    if tomo:
        timeloop = times[:-6][::3]
    else:
        timeloop = times[:-4]

    for i, time in enumerate(timeloop):
        for tomo_gate in tomo_gates:
            k = p.create_kernel("rabi_frequency"+ "_tomo_" + tomo_gate + "{}".format(i))
            k.prepz(qubit_idx)
            # nr_clocks = int(time/20e-9/2)
            square_us_cycles = np.floor((time+1e-10)/1e-6).astype(int)
            leftover_us = (time-square_us_cycles*1e-6)
            square_ns_cycles = np.floor((leftover_us+1e-10)/mw_gate_duration).astype(int)
            leftover_ns = (leftover_us-square_ns_cycles*mw_gate_duration)
            print(leftover_us)
            print(leftover_ns)
            mwlutman_index = np.round((leftover_ns+1e-10)/4e-9).astype(int)
            print(mwlutman_index)
            print("square_us_cycles", square_us_cycles)
            print("square_ns_cycles", square_ns_cycles)
            for suc in range(square_us_cycles):
                k.gate('cw_10', [qubit_idx]) # make sure that the square pulse lasts 1us
            for snc in range(square_ns_cycles):
                k.gate('cw_11', [qubit_idx]) # make sure that the square pulse lasts mw_gate_duration ns
            k.gate('cw_{}'.format(mwlutman_index+11), [qubit_idx])
            if tomo:
                k.gate(tomo_gate,[qubit_idx])
            k.measure(qubit_idx)
            p.add_kernel(k)

    # adding the calibration points
    p.add_single_qubit_cal_points(qubit_idx=qubit_idx, f_state_cal_pts=tomo)

    p.compile()
    return p


def spin_lock_echo(times, qubit_idx: int, platf_cfg: str):
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
    p = OqlProgram("spin_lock_echo", platf_cfg)

    for i, time in enumerate(times[:-4]):

        k = p.create_kernel("spin_lock_echo{}".format(i))
        k.prepz(qubit_idx)
        # nr_clocks = int(time/20e-9/2)
        square_us_cycles = np.floor(time/1e-6).astype(int)
        square_ns_cycles = np.round((time%1e-6)/mw_gate_duration).astype(int) # FIXME: unresolved
        wait_nanoseconds = 1
        # print("square_us_cycles", square_us_cycles)
        # print("square_us_cycles", square_ns_cycles)
        k.gate('rYm90', [qubit_idx])
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        k.gate('rx180', [qubit_idx])
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        for suc in range(square_us_cycles):
            k.gate('cw_10', [qubit_idx]) # make sure that the square pulse lasts 1us
        for snc in range(square_ns_cycles):
            k.gate('cw_11', [qubit_idx]) # make sure that the square pulse lasts mw_gate_duration ns
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        k.gate('rx180', [qubit_idx])
        k.gate("wait", [qubit_idx], wait_nanoseconds)
        k.gate('rYm90', [qubit_idx])
        k.measure(qubit_idx)
        p.add_kernel(k)

    # adding the calibration points
    p.add_single_qubit_cal_points(qubit_idx=qubit_idx)

    p.compile()
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

    p = OqlProgram("idle_error_rate", platf_cfg)

    sweep_points = []
    for N in nr_of_idle_gates:
        for state in states:
            if state not in allowed_states:
                raise ValueError('State must be in {}'.format(allowed_states))
            k = p.create_kernel("idle_prep{}_N{}".format(state, N))
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

    p.sweep_points = sweep_points
    p.compile()
    return p


def single_elt_on(qubit_idx: int, platf_cfg: str):
    p = OqlProgram('single_elt_on', platf_cfg)

    k = p.create_kernel('main')

    k.prepz(qubit_idx)
    k.x(qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    p.compile()
    return p


def off_on(qubit_idx: int, pulse_comb: str, initialize: bool, platf_cfg: str,nr_flux_dance:float=None,wait_time:float=None):
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
    p = OqlProgram('off_on', platf_cfg)

    # # Off
    if 'off' in pulse_comb.lower():
        k = p.create_kernel("off")
        k.prepz(qubit_idx)
        if initialize:
            k.measure(qubit_idx)

        if nr_flux_dance:
            for i in range(int(nr_flux_dance)):
                for step in [1,2,3,4]:
                    # if refocusing:
                    #     k.gate(f'flux-dance-{step}-refocus', [0])
                    # else:
                    k.gate(f'flux-dance-{step}', [0])
                k.barrier([])  # alignment 
            k.gate("wait", [], wait_time)

        k.measure(qubit_idx)
        p.add_kernel(k)

    if 'on' in pulse_comb.lower():
        k = p.create_kernel("on")
        k.prepz(qubit_idx)
        if initialize:
            k.measure(qubit_idx)

        if nr_flux_dance:
            for i in range(int(nr_flux_dance)):
                for step in [1,2,3,4]:
                    # if refocusing:
                    #     k.gate(f'flux-dance-{step}-refocus', [0])
                    # else:
                    k.gate(f'flux-dance-{step}', [0])
                k.barrier([])  # alignment 
            k.gate("wait", [], wait_time) 

        k.gate('rx180', [qubit_idx])
        k.measure(qubit_idx)
        p.add_kernel(k)

    if ('on' not in pulse_comb.lower()) and ('off' not in pulse_comb.lower()):
        raise ValueError()

    p.compile()
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
    p = OqlProgram('butterfly', platf_cfg)

    k = p.create_kernel('0')
    k.prepz(qubit_idx)
    if initialize:
        k.measure(qubit_idx)
    k.measure(qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    k = p.create_kernel('1')
    k.prepz(qubit_idx)
    if initialize:
        k.measure(qubit_idx)
    k.x(qubit_idx)
    k.measure(qubit_idx)
    k.measure(qubit_idx)
    p.add_kernel(k)

    p.compile()

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
    p = OqlProgram('RTE', platf_cfg)

    k = p.create_kernel('RTE')
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

    p.compile()
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
    p = OqlProgram(program_name, platf_cfg)

    i = 0
    for seed in range(nr_seeds):
        for j, n_cl in enumerate(nr_cliffords):
            k = p.create_kernel('RB_{}Cl_s{}_{}'.format(n_cl, seed, j))

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

    p.compile()
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
    p = OqlProgram(program_name, platf_cfg)

    k = p.create_kernel("yX")
    k.prepz(qubit_idx)
    k.gate('ry90', [qubit_idx])
    k.gate('rx180', [qubit_idx])
    k.measure(qubit_idx)
    p.add_kernel(k)

    k = p.create_kernel("xY")
    k.prepz(qubit_idx)
    k.gate('rx90', [qubit_idx])
    k.gate('ry180', [qubit_idx])
    k.measure(qubit_idx)
    p.add_kernel(k)

    p.compile()
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
                          flux_cw: str = 'fl_cw_02', # FIXME: unused
                          cal_points: bool = True,
                          mw_gate: str = "rx90"):
    """
    A Ramsey sequence with varying waiting times `times` around a flux pulse.
    """
    p = OqlProgram('FluxTimingCalibration', platf_cfg)

    # don't use last 4 points if calibration points are used
    if cal_points:
        times = times[:-4]
    for i_t, t in enumerate(times):
        t_nanoseconds = int(round(t/1e-9))
        k = p.create_kernel('pi_flux_pi_{}'.format(i_t))
        k.prepz(qubit_idx)
        k.gate(mw_gate, [qubit_idx])
        # k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        k.barrier([])  # alignment workaround
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
        p.add_single_qubit_cal_points(qubit_idx=qubit_idx)
    p.compile()
    return p


def TimingCalibration_1D(qubit_idx: int, times, platf_cfg: str,
                         # flux_cw: str = 'fl_cw_02', # FIXME: unused
                         cal_points: bool = True):
    """
    A Ramsey sequence with varying waiting times `times`in between.
    It calibrates the timing between spec and measurement pulse.
    """
    p = OqlProgram('TimingCalibration1D', platf_cfg)

    # don't use last 4 points if calibration points are used
    if cal_points:
        times = times[:-4]
    for i_t, t in enumerate(times):
        t_nanoseconds = int(round(t/1e-9))
        k = p.create_kernel('pi_times_pi_{}'.format(i_t))
        k.prepz(qubit_idx)
        k.gate('rx90', [qubit_idx])
        # k.gate("wait", [0, 1, 2, 3, 4, 5, 6], 0) #alignment workaround
        k.barrier([])  # alignment workaround
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
        p.add_single_qubit_cal_points(qubit_idx=qubit_idx)
    p.compile()
    return p


def FluxTimingCalibration_2q(q0, q1, buffer_time1, times, platf_cfg: str):
    """
    A Ramsey sequence with varying waiting times `times` around a flux pulse.

    N.B. this function is not consistent with "FluxTimingCalibration".
    This should be fixed
    """
    p = OqlProgram("FluxTimingCalibration_2q", platf_cfg)

    buffer_nanoseconds1 = int(round(buffer_time1/1e-9))

    for i_t, t in enumerate(times):

        t_nanoseconds = int(round(t/1e-9))
        k = p.create_kernel("pi-flux-pi_{}".format(i_t))
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

    p.compile()
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
    p = OqlProgram("FastFeedbackControl", platf_cfg)

    k = p.create_kernel("FastFdbkCtrl_nofb")
    k.prepz(qubit_idx)
    k.gate('rx90', [qubit_idx])
    # k.gate('rx180', [qubit_idx])
    k.measure(qubit_idx)
    wait_nanoseconds = int(round(latency/1e-9))
    k.gate("wait", [qubit_idx], wait_nanoseconds)
    k.gate("i", [qubit_idx])
    k.measure(qubit_idx)

    p.add_kernel(k)

    k = p.create_kernel("FastFdbkCtrl_fb0")
    k.prepz(qubit_idx)
    k.gate('rx90', [qubit_idx])
    # k.gate('rx180', [qubit_idx])
    k.measure(qubit_idx)
    wait_nanoseconds = int(round(latency/1e-9))
    k.gate("wait", [qubit_idx], wait_nanoseconds)
    k.gate('C0rx180', [qubit_idx])  # fast feedback control here
    k.measure(qubit_idx)
    p.add_kernel(k)

    k = p.create_kernel("FastFdbkCtrl_fb1")
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
    p.add_single_qubit_cal_points(qubit_idx=qubit_idx)

    p.compile()
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

    p = OqlProgram("ef_rabi_seq", platf_cfg)
    # These angles correspond to special pi/2 pulses in the lutman
    for i, amp in enumerate(amps):
        # cw_idx corresponds to special hardcoded pulses in the lutman
        cw_idx = i + 9

        k = p.create_kernel("ef_A{}_{}".format(int(abs(1000*amp)),i))
        k.prepz(q0)
        k.gate('rx180', [q0])
        k.gate('cw_{:02}'.format(cw_idx), [q0])
        if recovery_pulse:
            k.gate('rx180', [q0])
        k.measure(q0)
        p.add_kernel(k)
    if add_cal_points:
        p.add_single_qubit_cal_points(qubit_idx=q0)

    p.compile()

    if add_cal_points:
        cal_pts_idx = [amps[-1] + .1, amps[-1] + .15,
                       amps[-1] + .2, amps[-1] + .25]
    else:
        cal_pts_idx = []

    p.sweep_points = np.concatenate([amps, cal_pts_idx])
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

    p = OqlProgram('Depletion', platf_cfg)

    if 0: # FIXME: p.set_sweep_points has been replaced by p.sweep_points, since that was missing here they are probably not necessary for this function
        p.set_sweep_points(np.arange(len(allXY), dtype=float))

    if double_points:
        js=2
    else:
        js=1

    for i, xy in enumerate(allXY):
        for j in range(js):
            k = p.create_kernel('Depletion_{}_{}'.format(i, j))
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

    p.compile()
    return p

def TEST_RTE(qubit_idx: int, platf_cfg: str,
             measurements:int):
    """

    """
    p = OqlProgram('RTE', platf_cfg)

    k = p.create_kernel('RTE')
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
    p.compile()
    return p