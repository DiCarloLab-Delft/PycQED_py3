import logging
import numpy as np
from copy import deepcopy
from ..waveform_control import element
from ..waveform_control.element import calculate_time_correction
from ..waveform_control import pulse
from ..waveform_control import sequence
from pycqed.measurement.randomized_benchmarking import randomized_benchmarking as rb
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt

from importlib import reload
reload(pulse)
from ..waveform_control import pulse_library
reload(pulse_library)

station = None
reload(element)
# You need to explicitly set this before running any functions from this module
# I guess there are cleaner solutions :)


def Pulsed_spec_seq(spec_pars, RO_pars, upload=True, return_seq=False):
    '''
    Pulsed spectroscopy sequence using the tektronix.
    Input pars:
        spec_pars:      dict containing spectroscopy pars
        RO_pars:        dict containing RO pars
    '''
    period = spec_pars['pulse_delay'] + RO_pars['pulse_delay']
    f_RO_mod = RO_pars['mod_frequency']
    if f_RO_mod == None:
        remainder = 0.0
    else:
        remainder = period % (1/RO_pars['mod_frequency'])

    if (remainder != 0.0):
        msg = ('Period of spec seq ({})'.format(period) +
               'must be multiple of RO modulation period ({})'.format(
               1/RO_pars['mod_frequency']) +
               "\nAdding {}s to spec_pars['pulse_delay']".format(
            1/RO_pars['mod_frequency'] - remainder) +
            '\nConsider updating parameter')
        logging.warning(msg)
        print(msg)
        spec_pars['pulse_delay'] += 1 / \
            RO_pars['mod_frequency'] - remainder

    # Nr of pulse reps is set to ensure max nr of pulses and end 10us before
    # next trigger comes in. Assumes 200us trigger period, also works for
    # faster trigger rates.
    period = spec_pars['pulse_delay'] + RO_pars['pulse_delay']
    nr_of_pulse_reps = int((200e-6-10e-6)//period)

    seq_name = 'Pulsed_spec'
    seq = sequence.Sequence(seq_name)
    el_list = []

    pulse_dict = {'spec_pulse': spec_pars, 'RO': RO_pars}
    pulse_list = [pulse_dict['spec_pulse'], pulse_dict['RO']]*nr_of_pulse_reps
    for i in range(1):
        el = multi_pulse_elt(
            i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=False)
    return seq


def photon_number_splitting_seq(spec_pars, RO_pars, disp_pars, upload=True, return_seq=False):
    '''
    Pulsed spectroscopy sequence using the tektronix.
    Input pars:
        spec_pars:      dict containing spectroscopy pars
        RO_pars:        dict containing RO pars
    '''
    period = spec_pars['pulse_delay'] + RO_pars['pulse_delay']

    msg = ('Period of spec seq ({})'.format(period) +
           'must be multiple of RO modulation period ({})'.format(
           1/RO_pars['f_RO_mod']))

    if (period % (1/RO_pars['f_RO_mod'])) != 0.0:
        raise ValueError(msg)

    # Nr of pulse reps is set to ensure max nr of pulses and end 10us before
    # next trigger comes in. Assumes 200us trigger period, also works for
    # faster trigger rates.
    nr_of_pulse_reps = int((200e-6-10e-6)//period)

    seq_name = 'photon_number_spliting'
    seq = sequence.Sequence(seq_name)
    el_list = []

    pulse_dict = {'disp': disp_pars, 'spec_pulse': spec_pars, 'RO': RO_pars}
    pulse_list = [pulse_dict['disp'], pulse_dict[
        'spec_pulse'], pulse_dict['RO']]*nr_of_pulse_reps
    for i in range(2):
        el = multi_pulse_elt(
            i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=False)
    return seq


def Rabi_seq(amps, pulse_pars, RO_pars, n=1, post_msmt_delay=3e-6, no_cal_points=2,
             cal_points=True, verbose=False, upload=True, return_seq=False):
    '''
    Rabi sequence for a single qubit using the tektronix.
    Input pars:
        amps:            array of pulse amplitudes (V)
        pulse_pars:      dict containing the pulse parameters
        RO_pars:         dict containing the RO parameters
        n:               number of pulses (1 is conventional Rabi)
        post_msmt_delay: extra wait time for resetless compatibility
        cal_points:      whether to use calibration points or not
        upload:          whether to upload sequence to instrument or not
    '''
    seq_name = 'Rabi_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)

    for i, amp in enumerate(amps):  # seq has to have at least 2 elts
        if cal_points and no_cal_points==4 and \
                (i == (len(amps)-4) or i == (len(amps)-3)):
            el = multi_pulse_elt(i, station,[pulses['I'], RO_pars])
        elif cal_points and no_cal_points==4 and \
                (i == (len(amps)-2) or i == (len(amps)-1)):
            el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        elif cal_points and no_cal_points==2 and \
                (i == (len(amps)-2) or i == (len(amps)-1)):
            el = multi_pulse_elt(i, station,[pulses['I'], RO_pars])
        else:
            pulses['X180']['amplitude'] = amp
            pulse_list = n*[pulses['X180']]+[RO_pars]

            # copy first element and set extra wait
            pulse_list[0] = deepcopy(pulse_list[0])
            pulse_list[0]['pulse_delay'] += post_msmt_delay

            el = multi_pulse_elt(i, station, pulse_list)

        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq


def Flipping_seq(pulse_pars, RO_pars, n=1, post_msmt_delay=10e-9,
                 verbose=False, upload=True, return_seq=False):
    '''
    Flipping sequence for a single qubit using the tektronix.
    Input pars:
        pulse_pars:      dict containing the pulse parameters
        RO_pars:         dict containing the RO parameters
        n:               iterations (up to 2n+1 pulses)
        post_msmt_delay: extra wait time for resetless compatibility
    '''
    seq_name = 'Flipping_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    RO_pulse_delay = RO_pars['pulse_delay']
    for i in range(n+4):  # seq has to have at least 2 elts

        if (i == (n+1) or i == (n)):
            el = multi_pulse_elt(i, station, [pulses['I'], RO_pars])
        elif(i == (n+3) or i == (n+2)):
            RO_pars['pulse_delay'] = RO_pulse_delay
            el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        else:
            pulse_list = [pulses['X90']]+(2*i+1)*[pulses['X180']]+[RO_pars]
            # # copy first element and set extra wait
            # pulse_list[0] = deepcopy(pulse_list[0])
            # pulse_list[0]['pulse_delay'] += post_msmt_delay
            el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq


def Rabi_amp90_seq(scales, pulse_pars, RO_pars, n=1, post_msmt_delay=3e-6,
                   verbose=False, upload=True, return_seq=False):
    '''
    Rabi sequence to determine amp90 scaling factor for a single qubit using the tektronix.
    Input pars:
        scales:            array of amp90 scaling parameters
        pulse_pars:      dict containing the pulse parameters
        RO_pars:         dict containing the RO parameters
        n:               number of pulses (1 is 2 pi half pulses)
        post_msmt_delay: extra wait time for resetless compatibility
    '''
    seq_name = 'Rabi_amp90_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    for i, scale in enumerate(scales):  # seq has to have at least 2 elts
        pulses['X90']['amplitude'] = pulses['X180']['amplitude'] * scale
        pulse_list = 2*n*[pulses['X90']]+[RO_pars]

        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += post_msmt_delay
        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def T1_seq(times,
           pulse_pars, RO_pars,
           cal_points=True,
           verbose=False, upload=True, return_seq=False):
    '''
    Rabi sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modulation used for RO
    Input pars:
        times:       array of times to wait after the initial pi-pulse
        pulse_pars:  dict containing the pulse parameters
        RO_pars:     dict containing the RO parameters
    '''
    if np.any(times>1e-3):
        logging.warning('The values in the times array might be too large.'
                        'The units should be seconds.')
    seq_name = 'T1_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    RO_pulse_delay = RO_pars['pulse_delay']
    RO_pars = deepcopy(RO_pars)  # Prevents overwriting of the dict
    pulses = get_pulse_dict_from_pars(pulse_pars)

    for i, tau in enumerate(times):  # seq has to have at least 2 elts
        RO_pars['pulse_delay'] = RO_pulse_delay + tau
        #RO_pars['refpoint'] = 'start'  # time defined between start of ops
        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
            el = multi_pulse_elt(i, station, [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
            RO_pars['pulse_delay'] = RO_pulse_delay
            el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        else:
            el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def Ramsey_seq(times, pulse_pars, RO_pars,
               artificial_detuning=None,
               cal_points=True,
               verbose=False,
               upload=True, return_seq=False):
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
    if np.any(times>1e-3):
        logging.warning('The values in the times array might be too large.'
                        'The units should be seconds.')

    seq_name = 'Ramsey_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    # First extract values from input, later overwrite when generating
    # waveforms
    pulses = get_pulse_dict_from_pars(pulse_pars)

    pulse_pars_x2 = deepcopy(pulses['X90'])
    pulse_pars_x2['refpoint'] = 'start'
    for i, tau in enumerate(times):
        pulse_pars_x2['pulse_delay'] = tau

        if artificial_detuning is not None:
            Dphase = ((tau-times[0]) * artificial_detuning * 360) % 360
            pulse_pars_x2['phase'] = Dphase

        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
            el = multi_pulse_elt(i, station, [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
            el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        else:
            el = multi_pulse_elt(i, station,
                                 [pulses['X90'], pulse_pars_x2, RO_pars])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)


    if return_seq:
        return seq, el_list
    else:
        return seq_name

def Ramsey_seq_multiple_detunings(times, pulse_pars, RO_pars,
               artificial_detunings=None,
               cal_points=True,
               verbose=False,
               upload=True, return_seq=False):
    '''
    Ramsey sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    !!! Each value in the times array must be repeated len(artificial_detunings)
    times!!!
    Input pars:
        times:               array of times between (start of) pulses (s)
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters
        artificial_detunings: list of artificial_detunings (Hz) implemented
                              using phase
        cal_points:          whether to use calibration points or not
    '''
    seq_name = 'Ramsey_sequence_multiple_detunings'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    # First extract values from input, later overwrite when generating
    # waveforms
    pulses = get_pulse_dict_from_pars(pulse_pars)

    pulse_pars_x2 = deepcopy(pulses['X90'])
    pulse_pars_x2['refpoint'] = 'start'
    for i, tau in enumerate(times):
        pulse_pars_x2['pulse_delay'] = tau
        art_det = artificial_detunings[i % len(artificial_detunings)]

        if art_det is not None:
            Dphase = ((tau-times[0]) * art_det * 360) % 360
            pulse_pars_x2['phase'] = Dphase

        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
            el = multi_pulse_elt(i, station, [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
            el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        else:
            el = multi_pulse_elt(i, station,
                                 [pulses['X90'], pulse_pars_x2, RO_pars])
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        print('uploading')
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
        print('upload finished')

    if return_seq:
        return seq, el_list
    else:
        return seq_name


def Echo_seq(times, pulse_pars, RO_pars,
             artificial_detuning=None,
             cal_points=True,
             verbose=False,
             upload=True, return_seq=False):
    '''
    Echo sequence for a single qubit using the tektronix.
    Input pars:
        times:          array of times between (start of) pulses (s)
        artificial_detuning: artificial_detuning (Hz) implemented using phase
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
    center_X180['refpoint'] = 'start'
    final_X90['refpoint'] = 'start'

    for i, tau in enumerate(times):
        center_X180['pulse_delay'] = tau/2
        final_X90['pulse_delay'] = tau/2
        if artificial_detuning is not None:
            final_X90['phase'] = (tau-times[0]) * artificial_detuning * 360
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
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def AllXY_seq(pulse_pars, RO_pars, double_points=False,
              verbose=False, upload=True, return_seq=False):
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

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def OffOn_seq(pulse_pars, RO_pars, verbose=False, pulse_comb='OffOn',
              upload=True, return_seq=False,  preselection=False):
    '''
    OffOn sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    Input pars:
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters
        Initialize:          adds an exta measurement before state preparation
                             to allow initialization by post-selection
        Post-measurement delay:  should be sufficiently long to avoid
                             photon-induced gate errors when post-selecting.
        pulse_comb:          OffOn/OnOn/OffOff cobmination of pulses to play
        preselection:        adds an extra readout pulse before other pulses.
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

    spacer = {'pulse_type': 'SquarePulse',
              'channel': RO_pars['acq_marker_channel'],
              'amplitude': 0.0,
              'length': max(0, 300e-9 - pulse_pars['pulse_delay'] -
                            pulse_pars['nr_sigma']*pulse_pars['sigma']),
              'pulse_delay': 0}

    for i, pulse_comb in enumerate(pulse_combinations):
        if preselection:
            pulse_list = [RO_pars, spacer, pulses[pulse_comb], RO_pars]
        else:
            pulse_list = [pulses[pulse_comb], RO_pars]
        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name

def Butterfly_seq(pulse_pars, RO_pars, initialize=False,
                  post_msmt_delay=2000e-9, upload=True, verbose=False):
    '''
    Butterfly sequence to measure single shot readout fidelity to the
    pre-and post-measurement state. This is the way to veify the QND-ness off
    the measurement.
    - Initialize adds an exta measurement before state preparation to allow
    initialization by post-selection
    - Post-measurement delay can be varied to correct data for Tone effects.
    Post-measurement delay should be sufficiently long to avoid photon-induced
    gate errors when post-selecting.
    '''
    seq_name = 'Butterfly_seq'
    seq = sequence.Sequence(seq_name)
    el_list = []
    # Create a dict with the parameters for all the pulses
    pulses = get_pulse_dict_from_pars(pulse_pars)

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
            if RO_pars['mod_frequency'] == None:
                extra_delay = 0
            else:
                extra_delay = calculate_time_correction(
                    sub_seq_duration+post_msmt_delay, 1/RO_pars['mod_frequency'])
            initial_pulse_delay = post_msmt_delay + extra_delay
            start_pulse = deepcopy(pulse_sub_list[0])
            start_pulse['pulse_delay'] += initial_pulse_delay
            pulse_sub_list[0] = start_pulse
            pulse_list += pulse_sub_list

        el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    return seq_name


def Randomized_Benchmarking_seq(pulse_pars, RO_pars,
                                nr_cliffords,
                                nr_seeds,
                                net_clifford=0,
                                post_msmt_delay=3e-6,
                                cal_points=True,
                                resetless=False,
                                double_curves=False,
                                seq_name=None,
                                verbose=False, upload=True):
    '''
    Input pars:
        pulse_pars:    dict containing pulse pars
        RO_pars:       dict containing RO pars
        nr_cliffords:  list nr_cliffords for which to generate RB seqs
        nr_seeds:      int  nr_seeds for which to generate RB seqs
        net_clifford:  int index of net clifford the sequence should perform
                       0 corresponds to Identity and 3 corresponds to X180
        post_msmt_delay:
        cal_points:    bool whether to replace the last two elements with
                       calibration points, set to False if you want
                       to measure a single element (for e.g. optimization)
        resetless:     bool if False will append extra Id element if seq
                       is longer than 50us to ensure proper initialization
        double_curves: Alternates between net clifford 0 and 3
        upload:        Upload to the AWG

    returns:
        seq, elements_list


    Conventional use:
        nr_cliffords = [n1, n2, n3 ....]
        cal_points = True
    Optimization use (resetless or not):
        nr_cliffords = [n] is a list with a single entry
        cal_points = False
        net_clifford = 3 (optional) make sure it does a net pi-pulse
        post_msmt_delay is set (optional)
        resetless (optional)
    '''
    if seq_name is None:
        seq_name = 'RandomizedBenchmarking_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    net_cliffords = [0, 3]  # Exists purely for the double curves mode
    i = 0
    for seed in range(nr_seeds):
        for j, n_cl in enumerate(nr_cliffords):
            if double_curves:
                net_clifford = net_cliffords[i % 2]
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
                cl_seq = rb.randomized_benchmarking_sequence(
                    n_cl, desired_net_cl=net_clifford)
                pulse_keys = rb.decompose_clifford_seq(cl_seq)
                pulse_list = [pulses[x] for x in pulse_keys]
                pulse_list += [RO_pars]
                # copy first element and set extra wait
                pulse_list[0] = deepcopy(pulse_list[0])
                pulse_list[0]['pulse_delay'] += post_msmt_delay
                el = multi_pulse_elt(i, station, pulse_list)
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)

            # If the element is too long, add in an extra wait elt
            # to skip a trigger
            if resetless and n_cl*pulse_pars['pulse_delay']*1.875 > 50e-6:
                el = multi_pulse_elt(i, station, [pulses['I']])
                el_list.append(el)
                seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
        return seq, el_list
    else:
        return seq, el_list


def Freq_XY(freqs, pulse_pars, RO_pars,
            cal_points=True, verbose=False, return_seq=False):
    '''
    Sequence used for calibrating the frequency.
    Consists of Xy and Yx

    Beware that the elements alternate, if you want to measure both Xy and Yx
    at each motzoi you need repeating motzoi parameters. This was chosen
    to be more easily compatible with standard detector functions and sweep pts

    Input pars:
        freqs:               array of frequency parameters
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
    for i, ff in enumerate(freqs):
        pulse_keys = pulse_combinations[i % 2]
        for p_name in ['X180', 'Y180', 'X90', 'Y90']:
            pulses[p_name]['mod_frequency'] = ff
        if cal_points and (i == (len(freqs)-4) or
                           i == (len(freqs)-3)):
            el = multi_pulse_elt(i, station, [pulses['I'], RO_pars])
        elif cal_points and (i == (len(freqs)-2) or
                             i == (len(freqs)-1)):
            # pick motzoi for calpoint in the middle of the range
            pulses['X180']['mod_frequency'] = np.mean(freqs)
            el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        else:
            pulse_list = [pulses[x] for x in pulse_keys]
            pulse_list += [RO_pars]
            el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name


def Motzoi_XY(motzois, pulse_pars, RO_pars,
              cal_points=True, verbose=False, upload=True, return_seq=False):
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

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name

def QScale(qscales, pulse_pars, RO_pars,
              cal_points=True, verbose=False, upload=True, return_seq=False):
    '''
    Sequence used for calibrating the QScale factor used in the DRAG pulses.
    Applies X(pi/2)X(pi), X(pi/2)Y(pi), X(pi/2)Y(-pi) for each value of
    QScale factor.

    Beware that the elements alternate, in order to perform these 3
    measurements per QScale factor, the qscales sweep values must be
    repeated 3 times. This was chosen to be more easily compatible with
    standard detector functions and sweep pts.

    Input pars:
        qscales:             array of qscale factors
        pulse_pars:          dict containing the DRAG pulse parameters
        RO_pars:             dict containing the RO parameters
        cal_points:          if True, replaces the last 3*4 segments with
                             calibration points
    '''
    seq_name = 'QScale'
    seq = sequence.Sequence(seq_name)
    station.pulsar.update_channel_settings()
    el_list = []
    pulse_combinations=[['X90','X180'],['X90','Y180'],['X90','mY180']]
    pulses = get_pulse_dict_from_pars(pulse_pars)
    for i, motzoi in enumerate(qscales):
        pulse_keys = pulse_combinations[i % 3]
        for p_name in ['X180', 'Y180', 'X90', 'mY180']:
            pulses[p_name]['motzoi'] = motzoi
        if cal_points and (i == (len(qscales)-4) or
                                   i == (len(qscales)-3)):
            el = multi_pulse_elt(i, station, [pulses['I'], RO_pars])
        elif cal_points and (i == (len(qscales)-2) or
                                     i == (len(qscales)-1)):
            # pick motzoi for calpoint in the middle of the range
            pulses['X180']['motzoi'] = np.mean(qscales)
            el = multi_pulse_elt(i, station, [pulses['X180'], RO_pars])
        else:
            pulse_list = [pulses[x] for x in pulse_keys]
            pulse_list += [RO_pars]
            el = multi_pulse_elt(i, station, pulse_list)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name

# Sequences involving the second excited state


# Testing sequences


def Rising_seq(amps, pulse_pars, RO_pars, n=1, post_msmt_delay=3e-6,
               verbose=False, upload=True, return_seq=False):
    '''
    Rabi sequence for a single qubit using the tektronix.
    Input pars:
        amps:            array of pulse amplitudes (V)
        pulse_pars:      dict containing the pulse parameters
        RO_pars:         dict containing the RO parameters
        n:               number of pulses (1 is conventional Rabi)
        post_msmt_delay: extra wait time for resetless compatibility
    '''
    seq_name = 'Rising_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulse_pars = {'pulse_type': 'RisingPulse'}
    pulse_list = [pulse_pars]
    el = multi_pulse_elt(0, station, pulse_list)
    el_list.append(el)
    seq.append_element(el, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq


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
    pi2_amp = pulse_pars['amplitude']*pulse_pars['amp90_scale']

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
    pulses['Y180']['phase'] += 90
    pulses['mY180']['phase'] += 90
    pulses['mY180']['amplitude'] = -pi_amp

    pulses['Y90']['amplitude'] = pi2_amp
    pulses['Y90']['phase'] += 90
    pulses['mY90']['amplitude'] = -pi2_amp
    pulses['mY90']['phase'] += 90

    return pulses
