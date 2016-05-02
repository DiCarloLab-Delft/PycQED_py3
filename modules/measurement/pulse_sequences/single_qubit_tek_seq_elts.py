import logging
import numpy as np
from copy import deepcopy
try:
    from math import gcd
except:  # Moved to math in python 3.5, this is to be 3.4 compatible
    from fractions import gcd
from ..waveform_control import pulsar
from ..waveform_control import element
from ..waveform_control.element import calculate_time_corr
from ..waveform_control import pulse
from ..waveform_control.pulse_library import MW_IQmod_pulse, SSB_DRAG_pulse, \
    Mux_DRAG_pulse
from ..waveform_control import sequence
from modules.measurement.randomized_benchmarking import randomized_benchmarking as rb
from importlib import reload
reload(pulse)
from ..waveform_control import pulse_library
reload(pulse_library)

station = None
reload(element)
# You need to explicitly set this before running any functions from this module
# I guess there are cleaner solutions :)


def Rabi_seq(amps, pulse_pars, RO_pars, n=1, post_msmt_delay=3e-6,
             verbose=False):
    '''
    Rabi sequence for a single qubit using the tektronix.
    Input pars:
        amps:            array of pulse amplitudes (V)
        pulse_pars:      dict containing the pulse parameters
        RO_pars:         dict containing the RO parameters
        n:               number of pulses (1 is conventional Rabi)
        post_msmt_delay: extra wait time for resetless compatibility
    '''
    seq_name = 'Rabi_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    for i, amp in enumerate(amps):  # seq has to have at least 2 elts
        pulses['X180']['amplitude'] = amp
        pulse_list = n*[pulses['X180']]+[RO_pars]

        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += post_msmt_delay
        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name


def T1_seq(times,
           pulse_pars, RO_pars,
           cal_points=True,
           verbose=False):
    '''
    Rabi sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    Input pars:
        times:       array of times to wait after the initial pi-pulse
        pulse_pars:  dict containing the pulse parameters
        RO_pars:     dict containing the RO parameters
    '''
    seq_name = 'T1_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    RO_pulse_delay = RO_pars['pulse_delay']
    RO_pars = deepcopy(RO_pars)  # Prevents overwriting of the dict
    pulses = get_pulse_dict_from_pars(pulse_pars)

    for i, tau in enumerate(times):  # seq has to have at least 2 elts
        RO_pars['pulse_delay'] = RO_pulse_delay + tau
        if cal_points:
            if (i == (len(times)-4) or i == (len(times)-3)):
                el = multi_pulse_elt(i, station, [pulses['I'], RO_pars])
            elif(i == (len(times)-2) or i == (len(times)-1)):
                RO_pars['pulse_delay'] = RO_pulse_delay
                el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
            else:
                el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name


def Ramsey_seq(times, pulse_pars, RO_pars,
               artificial_detuning=None,
               cal_points=True,
               verbose=False):
    '''
    Ramsey sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    Input pars:
        times:               array of times between (start of) pulses (s)
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters
        artificial_detuning: artificial_detuning (Hz) implemented using phase
        cal_points:          whether to use calibration points or not
    '''
    seq_name = 'Ramsey_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    # First extract values from input, later overwrite when generating waveforms
    pulses = get_pulse_dict_from_pars(pulse_pars)

    pulse_pars_x2 = deepcopy(pulses['X90'])
    for i, tau in enumerate(times):
        pulse_pars_x2['pulse_delay'] = tau

        if artificial_detuning is not None:
            pulse_pars_x2['phase'] = tau * artificial_detuning * 360

        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
                el = multi_pulse_elt(i, station, [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
                el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        else:
            el = multi_pulse_elt(i, station,
                                 [pulses['X90'], pulse_pars_x2, RO_pars])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name


def Echo_seq(times, pulse_pars, RO_pars,
             cal_points=True,
             verbose=False):
    '''
    Echo sequence for a single qubit using the tektronix.
    Input pars:
        times:          array of times between (start of) pulses (s)
        pulse_pars:     dict containing the pulse parameters
        RO_pars:        dict containing the RO parameters
        cal_points:     whether to use calibration points or not
    '''
    seq_name = 'Echo_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    pulses = get_pulse_dict_from_pars(pulse_pars)
    center_X180 = deepcopy(pulses['X180'])
    final_X90 = deepcopy(pulses['X90'])
    for i, tau in enumerate(times):
        center_X180['pulse_delay'] = tau/2
        final_X90['pulse_delay'] = tau/2

        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
                el = multi_pulse_elt(i, station, [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
                el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        else:
            el = multi_pulse_elt(i, station,
                                 [pulses['X90'], center_X180,
                                  final_X90, RO_pars])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name


def AllXY_seq(pulse_pars, RO_pars, double_points=False,
              verbose=False):
    '''
    AllXY sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    Input pars:
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters

    '''
    seq_name = 'AllXY_seq'
    seq = sequence.Sequence(seq_name)
    el_list = []
    # Create a dict with the parameters for all the pulses
    pulses = get_pulse_dict_from_pars(pulse_pars)

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

    for i, pulse_comb in enumerate(pulse_combinations):
        pulse_list = [pulses[pulse_comb[0]],
                      pulses[pulse_comb[1]],
                      RO_pars]
        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name


def OffOn_seq(pulse_pars, RO_pars,
              verbose=False, pulse_comb='OffOn'):
    '''
    OffOn sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    Input pars:
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters
    '''
    seq_name = 'OffOn_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    # Create a dict with the parameters for all the pulses
    pulses = get_pulse_dict_from_pars(pulse_pars)

    if pulse_comb == 'OffOn':
        pulse_combinations = ['I', 'X180']
    elif pulse_comb == 'OnOn':
        pulse_combinations = ['X180', 'X180']
    elif pulse_comb == 'OffOff':
        pulse_combinations = ['I', 'I']

    for i, pulse_comb in enumerate(pulse_combinations):
        el = multi_pulse_elt(i, station, [pulses[pulse_comb], RO_pars])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name


def Butterfly_seq(pulse_pars, RO_pars, initialize=False,
                  post_msmt_delay=2000e-9, verbose=False):
    '''
    Butterfly sequence to measure single shot readout fidelity to the
    pre-and post-measurement state. This is the way to veify the QND-ness off
    the measurement.
    - Initialize adds an exta measurement before state preparation to allow
    initialization by post-selection
    - Post-measurement delay can be varied to correct data for Tone effects.
    Post-measurement delay should be sufficiently long to avoid photon-induced gate
    errors when post-selecting.
    '''
    seq_name = 'Butterfly_seq'
    seq = sequence.Sequence(seq_name)
    el_list = []
    # Create a dict with the parameters for all the pulses
    pulses = get_pulse_dict_from_pars(pulse_pars)
    fixed_point_freq = RO_pars['fixed_point_frequency']
    RO_pars['fixed_point_frequency'] = None

    pulses['RO'] = RO_pars
    pulse_lists = ['', '']
    pulses['I_wait'] = deepcopy(pulses['I'])
    pulses['I_wait']['pulse_delay'] = post_msmt_delay
    if initialize:
        pulse_lists[0] = [['I', 'RO'], ['I', 'RO'], ['I', 'RO']]
        pulse_lists[1] = [['I', 'RO'], ['X180', 'RO'], ['I', 'RO']]
    else:
        pulse_lists[0] = [['I', 'RO'], ['I', 'RO']]
        pulse_lists[1] = [['X180', 'RO'], ['I', 'RO']]
    for i, pulse_keys_sub_list in enumerate(pulse_lists):
        pulse_list = []
        # sub list exist to make sure the RO-s are in phase
        for pulse_keys in pulse_keys_sub_list:
            pulse_sub_list = [pulses[key] for key in pulse_keys]
            sub_seq_duration = sum([p['pulse_delay'] for p in pulse_sub_list])
            extra_delay = calculate_time_corr(
                sub_seq_duration+post_msmt_delay, fixed_point_freq)
            initial_pulse_delay = post_msmt_delay + extra_delay
            start_pulse = deepcopy(pulse_sub_list[0])
            start_pulse['pulse_delay'] += initial_pulse_delay
            pulse_sub_list[0] = start_pulse
            pulse_list += pulse_sub_list

        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name


def Randomized_Benchmarking_seq(pulse_pars, RO_pars,
                                nr_cliffords,
                                nr_seeds,
                                post_msmt_delay=3e-6,
                                cal_points=True,
                                resetless=False,
                                verbose=False):
    '''
    Input pars:
        pulse_pars:    dict containing pulse pars
        RO_pars:       dict containing RO pars
        nr_cliffords:  list nr_cliffords for which to generate RB seqs
        nr_seeds:      int  nr_seeds for which to generate RB seqs
        post_msmt_delay:
        cal_points:    bool whether to replace the last two elements with
                       calibration points, set to False if you want
                       to measure a single element (for e.g. optimization)
        resetless:     bool if False will append extra Id element if seq
                       is longer than 50us to ensure proper initialization

    Creates a randomized benchmarking sequence where 1 seed is loaded
    per element.

    Conventional use:
        nr_cliffords = [n1, n2, n3 ....]
        cal_points = True
    Optimization use (resetless or not):
        nr_cliffords = [n] is a list with a single entry
        cal_points = False
        post_msmt_delay is set (optional)
        resetless (optional)
    '''

    seq_name = 'RandomizedBenchmarking_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    i = 0
    for seed in range(nr_seeds):
            for j, n_cl in enumerate(nr_cliffords):
                i += 1  # only used for ensuring unique elt names
                if cal_points and (j == (len(nr_cliffords)-4) or
                                   j == (len(nr_cliffords)-3)):
                    el = multi_pulse_elt(i, station,
                                         [pulses['I'], RO_pars])
                elif cal_points and (j == (len(nr_cliffords)-2) or
                                     j == (len(nr_cliffords)-1)):
                    el = multi_pulse_elt(i, station,
                                         [pulses['X180'], RO_pars])
                else:
                    cl_seq = rb.randomized_benchmarking_sequence(n_cl)
                    pulse_keys = rb.decompose_clifford_seq(cl_seq)
                    pulse_list = [pulses[x] for x in pulse_keys]
                    pulse_list += [RO_pars]
                    # copy first element and set extra wait
                    pulse_list[0] = deepcopy(pulse_list[0])
                    pulse_list[0]['pulse_delay'] += post_msmt_delay
                    el = multi_pulse_elt(i, station, pulse_list)
                el_list.append(el)
                seq.append_element(el, trigger_wait=True)

                if n_cl*pulse_pars['pulse_delay']*1.875 > 50e-6:
                    # If the element is too long, add in an extra wait elt
                    # to skip a trigger
                    el = multi_pulse_elt(i, station, [pulses['I']])
                    el_list.append(el)
                    seq.append_element(el, trigger_wait=True)

    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name


def resetless_RB_seq(pulse_pars, RO_pars,
                     nr_cliffords,
                     nr_seeds=10,
                     post_msmt_delay=3e-6,
                     verbose=False, repetitions=2**16):
    '''
    Consists of 1 very long element that interleaves RB-sequences with
    measurement.

    Takes care all RO pulses are in phase by waiting a post_msmt_delay
    + a correction time before starting each block of pulses.

    Appends empty samples to ensure the total length of the sequence is a
    multiple of the modulation (fix_point) frequency.
    '''

    logging.warning('Deprecated, use Randomized_Benchmarking_seq')
    seq_name = 'Resetless_RB_seq'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    pulse_list = []
    # Extracting and fixed point freq, taking care of by hand here
    fixed_point_freq = RO_pars['fixed_point_frequency']
    RO_pars['fixed_point_frequency'] = None

    # You can think of every seed as it's own little "block"
    # the idea is to shift whole blocks to ensure the RO is in-phase
    for seed in range(nr_seeds):
        cl_seq = rb.randomized_benchmarking_sequence(nr_cliffords,
                                                     desired_net_cl=3)
        pulse_keys = rb.decompose_clifford_seq(cl_seq)
        pulse_sub_list = [pulses[x] for x in pulse_keys]
        pulse_sub_list += [RO_pars]

        # Calculate the time correction to ensure RO pulse starts at a
        # multiple of the fixed_point_freq
        sub_seq_duration = sum([p['pulse_delay'] for p in pulse_sub_list])
        extra_delay = calculate_time_corr(
            sub_seq_duration+post_msmt_delay, fixed_point_freq)
        initial_pulse_delay = post_msmt_delay + extra_delay

        # Replace the initial element to wait for an extended period of time
        start_pulse = deepcopy(pulse_sub_list[0])
        start_pulse['pulse_delay'] += initial_pulse_delay
        pulse_sub_list[0] = start_pulse
        pulse_list += pulse_sub_list

    el = multi_pulse_elt(1, station, pulse_list)
    extra_delay = calculate_time_corr(
        el.length(), fixed_point_freq)
    el.min_samples = el.samples() + int(extra_delay*el.clock)
    el_list.append(el)
    seq.append_element(el, trigger_wait=True, repetitions=repetitions)

    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name


def resetless_rabi_seq(pulse_pars, RO_pars,
                       n=1,
                       post_msmt_delay=3e-6,
                       verbose=False):
    '''
    Interleaves n-rabi pulses with measurements, waits at least
    post_msmt_delay after (the start of) each RO before starting a new
    pulse.

    Appends empty samples to ensure the total length of the sequence is a
    multiple of the modulation (fix_point) frequency.
    '''
    logging.warning('Deprecated use normal Rabi seq')
    seq_name = 'Resetless_rabi_seq'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    pulse_list = []
    # Extracting and fixed point freq, taking care of by hand here
    fixed_point_freq = RO_pars['fixed_point_frequency']
    RO_pars['fixed_point_frequency'] = None

    # You can think of every seed as it's own little "block"
    # the idea is to shift whole blocks to ensure the RO is in-phase
    # Makes 10 "block" to ensure we fill up the minimum element size
    for i in range(2):

        pulse_sub_list = n*[pulses['X180']]+[RO_pars]

        # TODO: This is a block common to all resetless sequences, I should
        # make it into a function.
        # Calculate the time correction to ensure RO pulse starts at a
        # multiple of the fixed_point_freq
        sub_seq_duration = sum([p['pulse_delay'] for p in pulse_sub_list])
        extra_delay = calculate_time_corr(
            sub_seq_duration+post_msmt_delay, fixed_point_freq)
        initial_pulse_delay = post_msmt_delay + extra_delay

        # Replace the initial element to wait for an extended period of time
        pulse_sub_list[0] = deepcopy(pulse_sub_list[0])
        pulse_sub_list[0]['pulse_delay'] += initial_pulse_delay
        pulse_list += pulse_sub_list

    el = multi_pulse_elt(1, station, pulse_list)
    extra_delay = calculate_time_corr(
        el.length(), fixed_point_freq)
    el.min_samples = el.samples() + int(extra_delay*el.clock)
    el_list.append(el)

    seq.append_element(el, trigger_wait=True, repetitions=2**16)
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name


def Motzoi_XY(motzois, pulse_pars, RO_pars,
              cal_points=True, verbose=False):
    '''
    Sequence used for calibrating the motzoi parameter.
    Consists of Xy and Yx

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
    seq_name = 'MotzoiXY'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulse_combinations = [['X180', 'Y90'], ['Y180', 'X90']]
    pulses = get_pulse_dict_from_pars(pulse_pars)
    for i, motzoi in enumerate(motzois):
        pulse_keys = pulse_combinations[i % 2]

        for p_name in ['X180', 'Y180', 'X90', 'Y90']:
            pulses[p_name]['motzoi'] = motzoi

        if cal_points and (i == (len(motzois)-4) or
                           i == (len(motzois)-3)):
            el = multi_pulse_elt(i, station, [pulses['I'], RO_pars])
        elif cal_points and (i == (len(motzois)-2) or
                             i == (len(motzois)-1)):
            # pick motzoi for calpoint in the middle of the range
            pulses['X180']['motzoi'] = np.mean(motzois)
            el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        else:
            pulse_list = [pulses[x] for x in pulse_keys]
            pulse_list += [RO_pars]
            el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name

# sequence element generating functions


def multi_pulse_elt(i, station, pulse_list):
        '''
        Input args
            station:    qcodes station object, contains AWG etc
            pulse_list: list of pulse_dicts containing pulse parameters
        Returns:
            element:    for use with the pulsar sequencer

        Currently works with two types of pulses, 'SSB_DRAG_pulse' and
        'MW_IQmod_pulse' from pulselib. The idea is to make this function
        work with arbitrary pulses that have a function in the pulselib.
        The other idea is to have one 'pulse' in pulselib per 'pulse' that
        also include markers in the RO this is still added by hand in the if
        statement.

        If you want to add extra pulses to this function please talk to me
        (Adriaan) as I would like to keep it clean and prevent a  big if, elif
        loop. (I have some ideas on how to implement this).

        Note: this function could be the template for the most standard
        element we use.
        '''
        el = element.Element(
            name='{}-pulse-elt_{}'.format(len(pulse_list), i),
            pulsar=station.pulsar)
        # exitst to ensure that channel is not high when waiting for trigger
        last_pulse = el.add(pulse.SquarePulse(name='refpulse_0', channel='ch1',
                                              amplitude=0, length=1e-9))
        for i in range(3):  # Exist to ensure there are no empty channels
            el.add(pulse.SquarePulse(name='refpulse_0',
                                     channel='ch{}'.format(i+1),
                                     amplitude=0, length=1e-9))

        for i, pulse_pars in enumerate(pulse_list):
            if pulse_pars['pulse_type'] == 'SSB_DRAG_pulse':
                last_pulse = el.add(
                    SSB_DRAG_pulse(name='pulse_{}'.format(i),
                                   I_channel=pulse_pars['I_channel'],
                                   Q_channel=pulse_pars['Q_channel'],
                                   amplitude=pulse_pars['amplitude'],
                                   sigma=pulse_pars['sigma'],
                                   nr_sigma=pulse_pars['nr_sigma'],
                                   motzoi=pulse_pars['motzoi'],
                                   mod_frequency=pulse_pars['mod_frequency'],
                                   phase=pulse_pars['phase'],
                                   phi_skew=pulse_pars['phi_skew'],
                                   alpha=pulse_pars['alpha']),
                    start=pulse_pars['pulse_delay'],
                    refpulse=last_pulse, refpoint='start')
            elif pulse_pars['pulse_type'] == 'Mux_DRAG_pulse':
                # pulse_pars.pop('pulse_type')
                last_pulse = el.add(Mux_DRAG_pulse(name='pulse_{}'.format(i),
                                                   **pulse_pars),
                                    start=pulse_pars['pulse_delay'],
                                    refpulse=last_pulse, refpoint='start')

            elif (pulse_pars['pulse_type'] == 'MW_IQmod_pulse' or
                  pulse_pars['pulse_type'] == 'Gated_MW_RO_pulse'):
                # Does more than just call the function as it also adds the
                # markers. Ideally we combine both in one function in pulselib
                if pulse_pars['pulse_type'] == 'MW_IQmod_pulse':
                    last_pulse = el.add(MW_IQmod_pulse(
                            name='RO_tone',
                            I_channel=pulse_pars['I_channel'],
                            Q_channel=pulse_pars['Q_channel'],
                            length=pulse_pars['length'],
                            amplitude=pulse_pars['amplitude'],
                            mod_frequency=pulse_pars['mod_frequency']),
                        start=pulse_pars['pulse_delay'],
                        refpulse=last_pulse, refpoint='start',
                        fixed_point_freq=pulse_pars['fixed_point_frequency'])
                else:
                    last_pulse = el.add(pulse.SquarePulse(
                            name='RO_marker', amplitude=1,
                            length=pulse_pars['length'],
                            channel=pulse_pars['RO_pulse_marker_channel']),
                        start=pulse_pars['pulse_delay'], refpulse=last_pulse,
                        refpoint='start',
                        fixed_point_freq=pulse_pars['fixed_point_frequency'])
                # Start Acquisition marker
                Acq_marker = pulse.SquarePulse(
                    name='Acq-trigger', amplitude=1, length=20e-9,
                    channel=pulse_pars['acq_marker_channel'])
                el.add(
                    Acq_marker, start=pulse_pars['acq_marker_delay'],
                    refpulse=last_pulse, refpoint='start')

            else:
                raise KeyError('pulse_type {} not recognized'.format(
                    pulse_pars['pulse_type']))

        # This pulse ensures that the sequence always ends at zero amp
        last_pulse = el.add(pulse.SquarePulse(name='final_empty_pulse',
                                              channel='ch1',
                                              amplitude=0, length=1e-9),
                            refpulse=last_pulse, refpoint='end')

        return el


# Helper functions


def get_pulse_dict_from_pars(pulse_pars):
    '''
    Returns a dictionary containing pulse_pars for all the primitive pulses
    based on a single set of pulse_pars.
    Using this function deepcopies the pulse parameters preventing accidently
    editing the input dictionary.

    input args:
        pulse_pars: dictionary containing pulse_parameters
    return:
        pulses: dictionary of pulse_pars dictionaries
    '''
    pi_amp = pulse_pars['amplitude']
    pi2_amp = pulse_pars['amplitude']/2

    pulses = {'I': deepcopy(pulse_pars),
              'X180': deepcopy(pulse_pars),
              'mX180': deepcopy(pulse_pars),
              'X90': deepcopy(pulse_pars),
              'mX90': deepcopy(pulse_pars),
              'Y180': deepcopy(pulse_pars),
              'mY180': deepcopy(pulse_pars),
              'Y90': deepcopy(pulse_pars),
              'mY90': deepcopy(pulse_pars)}

    pulses['I']['amplitude'] = 0
    pulses['mX180']['amplitude'] = -pi_amp
    pulses['X90']['amplitude'] = pi2_amp
    pulses['mX90']['amplitude'] = -pi2_amp
    pulses['Y180']['phase'] = 90
    pulses['mY180']['phase'] = 90
    pulses['mY180']['amplitude'] = -pi_amp

    pulses['Y90']['amplitude'] = pi2_amp
    pulses['Y90']['phase'] = 90
    pulses['mY90']['amplitude'] = -pi2_amp
    pulses['mY90']['phase'] = 90

    return pulses
