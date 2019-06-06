import logging
import numpy as np
from copy import deepcopy
from pycqed.measurement.waveform_control import element
from pycqed.measurement.waveform_control.element import calculate_time_correction
from pycqed.measurement.waveform_control import pulse
from pycqed.measurement.waveform_control import pulsar as ps
from pycqed.measurement.waveform_control import sequence as sequence
from pycqed.measurement.waveform_control import segment as segment
from pycqed.measurement.randomized_benchmarking import randomized_benchmarking as rb
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt

from importlib import reload
reload(pulse)
from ..waveform_control import pulse_library
reload(pulse_library)

def pulse_list_list_seq(pulse_list_list, name='pulse_list_list_sequence', 
                        upload=True):
    seq = sequence.Sequence(name)
    for i, pulse_list in enumerate(pulse_list_list):
        seq.add(segment.Segment('segment_{}'.format(i), pulse_list))
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    return seq

def Rabi_seq(amps, pulse_pars, RO_pars, active_reset=False, n=1,
             post_msmt_delay=0, no_cal_points=2,
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
    seg_list = []
    pulses_unmodified = get_pulse_dict_from_pars(pulse_pars)
    # FIXME: Nathan 2019.05.07: I believe next line is useless since deepcopy is already done
    # in get_pulse_dict_from_pars()
    pulses = deepcopy(pulses_unmodified)
    for i, amp in enumerate(amps):  # seq has to have at least 2 elts
        if cal_points and no_cal_points==4 and \
                (i == (len(amps)-4) or i == (len(amps)-3)):
            seg = segment.Segment('segment_{}'.format(i), 
                [pulses_unmodified['I'], RO_pars])
        elif cal_points and no_cal_points==4 and \
                (i == (len(amps)-2) or i == (len(amps)-1)):
            seg = segment.Segment('segment_{}'.format(i), 
                [pulses_unmodified['X180'], RO_pars])
        elif cal_points and no_cal_points==2 and \
                (i == (len(amps)-2) or i == (len(amps)-1)):
            seg = segment.Segment('segment_{}'.format(i), 
                [pulses_unmodified['I'], RO_pars])
        else:
            pulses['X180']['amplitude'] = amp
            pulse_list = n*[pulses['X180']]+[RO_pars]

            # copy first element and set extra wait
            pulse_list[0] = deepcopy(pulse_list[0])
            pulse_list[0]['pulse_delay'] += post_msmt_delay

            seg = segment.Segment('segment_{}'.format(i), pulse_list)

        seg_list.append(seg)
        seq.add(seg)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

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
    seg_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    RO_pulse_delay = RO_pars['pulse_delay']
    for i in range(n+4):  # seq has to have at least 2 elts

        if (i == (n+1) or i == (n)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['I'], RO_pars])
        elif(i == (n+3) or i == (n+2)):
            RO_pars['pulse_delay'] = RO_pulse_delay
            seg = segment.Segment('segment_{}'.format(i), [pulses['X180'], RO_pars])
        else:
            pulse_list = [pulses['X90']]+(2*i+1)*[pulses['X180']]+[RO_pars]
            # # copy first element and set extra wait
            # pulse_list[0] = deepcopy(pulse_list[0])
            # pulse_list[0]['pulse_delay'] += post_msmt_delay
            seg = segment.Segment('segment_{}'.format(i), pulse_list)
        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    if return_seq:
        return seq, seg_list
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
    seg_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    for i, scale in enumerate(scales):  # seq has to have at least 2 elts
        pulses['X90']['amplitude'] = pulses['X180']['amplitude'] * scale
        pulse_list = 2*n*[pulses['X90']]+[RO_pars]

        # copy first element and set extra wait
        pulse_list[0] = deepcopy(pulse_list[0])
        pulse_list[0]['pulse_delay'] += post_msmt_delay
        seg = segment.Segment('segment_{}'.format(i), pulse_list)
        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    if return_seq:
        return seq, seg_list
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
    seg_list = []
    RO_pulse_delay = RO_pars['pulse_delay']
    RO_pars = deepcopy(RO_pars)  # Prevents overwriting of the dict
    pulses = get_pulse_dict_from_pars(pulse_pars)

    for i, tau in enumerate(times):  # seq has to have at least 2 elts
        RO_pars['pulse_delay'] = RO_pulse_delay + tau
        #RO_pars['refpoint'] = 'start'  # time defined between start of ops
        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
            RO_pars['pulse_delay'] = RO_pulse_delay
            seg = segment.Segment('segment_{}'.format(i), [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
            RO_pars['pulse_delay'] = RO_pulse_delay
            seg = segment.Segment('segment_{}'.format(i), [pulses['X180'], RO_pars])
        else:
            seg = segment.Segment('segment_{}'.format(i), [pulses['X180'], RO_pars])
        seg_list.append(seg)
        seq.add(seg)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    if return_seq:
        return seq, seg_list
    else:
        return seq_name


def Ramsey_seq_Echo(times, pulse_pars, RO_pars, nr_echo_pulses=4,
               artificial_detuning=None,
               cal_points=True, cpmg_scheme=True,
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
    if np.any(times > 1e-3):
        logging.warning('The values in the times array might be too large.'
                        'The units should be seconds.')

    seq_name = 'Ramsey_sequence'
    seq = sequence.Sequence(seq_name)
    seg_list = []
    # First extract values from input, later overwrite when generating
    # waveforms
    pulses = get_pulse_dict_from_pars(pulse_pars)

    pulse_pars_x2 = deepcopy(pulses['X90'])
    pulse_pars_x2['refpoint'] = 'start'

    X180_pulse = deepcopy(pulses['X180'])
    Echo_pulses = nr_echo_pulses*[X180_pulse]
    DRAG_length = pulse_pars['nr_sigma']*pulse_pars['sigma']

    for i, tau in enumerate(times):
        if artificial_detuning is not None:
            Dphase = ((tau-times[0]) * artificial_detuning * 360) % 360
            pulse_pars_x2['phase'] = Dphase

        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['X180'], RO_pars])
        else:
            X90_separation = tau - DRAG_length
            if cpmg_scheme:
                if i == 0:
                    print('cpmg')
                echo_pulse_delay = (X90_separation -
                                    nr_echo_pulses*DRAG_length) / \
                                    nr_echo_pulses
                if echo_pulse_delay < 0:
                    pulse_pars_x2['pulse_delay'] = tau
                    pulse_dict_list = [pulses['X90'], pulse_pars_x2, RO_pars]
                else:
                    pulse_dict_list = [pulses['X90']]
                    start_end_delay = echo_pulse_delay/2
                    for p_nr, pulse_dict in enumerate(Echo_pulses):
                        pd = deepcopy(pulse_dict)
                        pd['refpoint'] = 'end'
                        pd['pulse_delay'] = \
                            (start_end_delay if p_nr == 0 else echo_pulse_delay)
                        pulse_dict_list.append(pd)

                    pulse_pars_x2['refpoint'] = 'end'
                    pulse_pars_x2['pulse_delay'] = start_end_delay
                    pulse_dict_list += [pulse_pars_x2, RO_pars]
            else:
                if i == 0:
                    print('UDD')
                pulse_positions_func = \
                    lambda idx, N: np.sin(np.pi*idx/(2*N+2))**2
                pulse_delays_func = (lambda idx, N: X90_separation*(
                    pulse_positions_func(idx, N) -
                    pulse_positions_func(idx-1, N)) -
                                ((0.5 if idx == 1 else 1)*DRAG_length))

                if nr_echo_pulses*DRAG_length > X90_separation:
                    pulse_pars_x2['pulse_delay'] = tau
                    pulse_dict_list = [pulses['X90'], pulse_pars_x2, RO_pars]
                else:
                    pulse_dict_list = [pulses['X90']]
                    for p_nr, pulse_dict in enumerate(Echo_pulses):
                        pd = deepcopy(pulse_dict)
                        pd['refpoint'] = 'end'
                        pd['pulse_delay'] = pulse_delays_func(
                            p_nr+1, nr_echo_pulses)
                        pulse_dict_list.append(pd)

                    pulse_pars_x2['refpoint'] = 'end'
                    pulse_pars_x2['pulse_delay'] = pulse_delays_func(
                        1, nr_echo_pulses)
                    pulse_dict_list += [pulse_pars_x2, RO_pars]

            seg = segment.Segment('segment_{}'.format(i), pulse_dict_list)

        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    if return_seq:
        return seq, seg_list
    else:
        return seq_name


def Ramsey_seq_cont_drive(times, pulse_pars, RO_pars,
                          artificial_detuning=None,
                          cal_points=True,
                          verbose=False,
                          upload=True, return_seq=False, **kw):
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
    if np.any(times > 1e-3):
        logging.warning('The values in the times array might be too large.'
                        'The units should be seconds.')

    seq_name = 'Ramsey_sequence'
    seq = sequence.Sequence(seq_name)
    seg_list = []
    # First extract values from input, later overwrite when generating
    # waveforms
    pulses = get_pulse_dict_from_pars(pulse_pars)

    pulse_pars_x2 = deepcopy(pulses['X90'])

    DRAG_length = pulse_pars['nr_sigma']*pulse_pars['sigma']
    cont_drive_ampl = 0.1 * pulse_pars['amplitude']
    X180_pulse = deepcopy(pulses['X180'])
    cos_pulse = {'pulse_type': 'CosPulse_gauss_rise',
                 'channel': X180_pulse['I_channel'],
                 'frequency': X180_pulse['mod_frequency'],
                 'length': 0,
                 'phase': X180_pulse['phi_skew'],
                 'amplitude': cont_drive_ampl * X180_pulse['alpha'],
                 'pulse_delay': 0,
                 'refpoint': 'end'}
    sin_pulse = {'pulse_type': 'CosPulse_gauss_rise',
                 'channel': X180_pulse['Q_channel'],
                 'frequency': X180_pulse['mod_frequency'],
                 'length': 0,
                 'phase': 90,
                 'amplitude': cont_drive_ampl * X180_pulse['alpha'],
                 'pulse_delay': 0,
                 'refpoint': 'simultaneous'}

    for i, tau in enumerate(times):

        if artificial_detuning is not None:
            Dphase = ((tau-times[0]) * artificial_detuning * 360) % 360
            pulse_pars_x2['phase'] = Dphase

        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['X180'], RO_pars])
        else:
            X90_separation = tau - DRAG_length
            if X90_separation > 0:
                pulse_pars_x2['refpoint'] = 'end'
                cos_pls1 = deepcopy(cos_pulse)
                sin_pls1 = deepcopy(sin_pulse)
                cos_pls1['length'] = X90_separation/2
                sin_pls1['length'] = X90_separation/2
                cos_pls2 = deepcopy(cos_pls1)
                sin_pls2 = deepcopy(sin_pls1)
                cos_pls2['amplitude'] = -cos_pls1['amplitude']
                cos_pls2['pulse_type'] = 'CosPulse_gauss_fall'
                sin_pls2['amplitude'] = -sin_pls1['amplitude']
                sin_pls2['pulse_type'] = 'CosPulse_gauss_fall'

                pulse_dict_list = [pulses['X90'], cos_pls1, sin_pls1,
                                   cos_pls2, sin_pls2, pulse_pars_x2, RO_pars]
            else:
                pulse_pars_x2['refpoint'] = 'start'
                pulse_pars_x2['pulse_delay'] = tau
                pulse_dict_list = [pulses['X90'], pulse_pars_x2, RO_pars]

            seg = segment.Segment('segment_{}'.format(i), pulse_dict_list)

        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    if return_seq:
        return seq, seg_list
    else:
        return seq_name


def Ramsey_seq(times, pulse_pars, RO_pars,
               artificial_detuning=None,
               cal_points=True,
               verbose=False,
               upload=True, return_seq=False, **kw):
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
    if np.any(times > 1e-3):
        logging.warning('The values in the times array might be too large.'
                        'The units should be seconds.')

    seq_name = 'Ramsey_sequence'
    seq = sequence.Sequence(seq_name)
    seg_list = []
    # First extract values from input, later overwrite when generating
    # waveforms
    pulses = get_pulse_dict_from_pars(pulse_pars)

    # pulses['I_fb'] = {
    #     'pulse_type': 'SquarePulse',
    #     'channel': RO_pars['acq_marker_channel'],
    #     'amplitude': 0.0,
    #     'length': 1e-6,
    #     'pulse_delay': 0}

    pulse_pars_x2 = deepcopy(pulses['X90'])
    pulse_pars_x2['refpoint'] = 'start'
    for i, tau in enumerate(times):
        pulse_pars_x2['pulse_delay'] = tau

        if artificial_detuning is not None:
            Dphase = ((tau-times[0]) * artificial_detuning * 360) % 360
            pulse_pars_x2['phase'] = Dphase

        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['X180'], RO_pars])
        else:
             seg = segment.Segment('segment_{}'.format(i),
                                 [pulses['X90'], pulse_pars_x2, RO_pars])

        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    if return_seq:
        return seq, seg_list
    else:
        return seq_name


def Ramsey_seq_VZ(times, pulse_pars, RO_pars,
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
    seg_list = []
    # First extract values from input, later overwrite when generating
    # waveforms
    pulses = get_pulse_dict_from_pars(pulse_pars)

    pulse_pars_x2 = deepcopy(pulses['X90'])
    pulse_pars_x2['refpoint'] = 'start'
    for i, tau in enumerate(times):
        pulse_pars_x2['pulse_delay'] = tau

        if artificial_detuning is not None:
            Dphase = ((tau-times[0]) * artificial_detuning * 360) % 360
        else:
            Dphase = ((tau-times[0]) * 1e6 * 360) % 360
        Z_gate = Z(Dphase, pulse_pars)

        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['X180'], RO_pars])
        else:
            pulse_list = [pulses['X90'], Z_gate, pulse_pars_x2, RO_pars]
            seg = segment.Segment('segment_{}'.format(i), pulse_list)

            #a = [j['phase'] for j in pulse_list]

        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    if return_seq:
        return seq, seg_list
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
    ps.Pulsar.get_instance().update_channel_settings()
    seg_list = []
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
             seg = segment.Segment('segment_{}'.format(i), [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['X180'], RO_pars])
        else:
             seg = segment.Segment('segment_{}'.format(i),
                                 [pulses['X90'], pulse_pars_x2, RO_pars])
        seg_list.append(seg)
        seq.add(seg)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    if return_seq:
        return seq, seg_list
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
    seg_list = []

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
             seg = segment.Segment('segment_{}'.format(i), [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['X180'], RO_pars])
        else:
             seg = segment.Segment('segment_{}'.format(i),
                                 [pulses['X90'], center_X180,
                                  final_X90, RO_pars])
        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    if return_seq:
        return seq, seg_list
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
    seg_list = []
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
        seg = segment.Segment('segment_{}'.format(i), pulse_list)
        seg_list.append(seg)
        seq.add(seg)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    if return_seq:
        return seq, seg_list
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
    seg_list = []
    # Create a dict with the parameters for all the pulses
    pulses = get_pulse_dict_from_pars(pulse_pars)

    if pulse_comb == 'OffOn':
        pulse_combinations = ['I', 'X180']
    elif pulse_comb == 'OnOn':
        pulse_combinations = ['X180', 'X180']
    elif pulse_comb == 'OffOff':
        pulse_combinations = ['I', 'I']

    for i, pulse_comb in enumerate(pulse_combinations):
        if preselection:
            pulse = deepcopy(pulses[pulse_comb])
            pulse['pulse_delay'] = 300e-9
            pulse_list = [RO_pars, pulse, RO_pars]
        else:
            pulse_list = [pulses[pulse_comb], RO_pars]
        seg = segment.Segment('segment_{}'.format(i), pulse_list)
        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    if return_seq:
        return seq, seg_list
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
    seg_list = []
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

        seg = segment.Segment('segment_{}'.format(i), pulse_list)
        seg_list.append(seg)
        seq.add(seg)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
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
    seg_list = []
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
                 seg = segment.Segment('segment_{}'.format(i),
                                     [pulses['I'], RO_pars])
            elif cal_points and (j == (len(nr_cliffords)-2) or
                                 j == (len(nr_cliffords)-1)):
                 seg = segment.Segment('segment_{}'.format(i),
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
                seg = segment.Segment('segment_{}'.format(i), pulse_list)
            seg_list.append(seg)
            seq.add(seg)

            # If the element is too long, add in an extra wait elt
            # to skip a trigger
            if resetless and n_cl*pulse_pars['pulse_delay']*1.875 > 50e-6:
                seg = segment.Segment('segment_{}'.format(i), [pulses['I']])
                seg_list.append(seg)
                seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
        return seq, seg_list
    else:
        return seq, seg_list

def Randomized_Benchmarking_seq_one_length(pulse_pars, RO_pars,
                                            nr_cliffords_value, #scalar
                                            nr_seeds,           #array
                                            net_clifford=0,
                                            gate_decomposition='HZ',
                                            interleaved_gate=None,
                                            cal_points=False,
                                            resetless=False,
                                            seq_name=None,
                                            verbose=False,
                                            upload=True, upload_all=True):

    if seq_name is None:
        seq_name = 'RandomizedBenchmarking_sequence_one_length'
    seq = sequence.Sequence(seq_name)
    seg_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)

    if upload_all:
        upload_AWGs = 'all'
    else:
        upload_AWGs = ['AWG1']
        upload_AWGs += [ps.Pulsar.get_instance().get(pulse_pars['I_channel'] + '_AWG'),
                        ps.Pulsar.get_instance().get(pulse_pars['Q_channel'] + '_AWG')]
        upload_AWGs = list(set(upload_AWGs))
    print(upload_AWGs)

    # pulse_keys_list = []
    for i in nr_seeds:
        if cal_points and (i == (len(nr_seeds)-4) or
                                   i == (len(nr_seeds)-3)):
             seg = segment.Segment('segment_{}'.format(i),
                                 [pulses['I'], RO_pars])
        elif cal_points and (i == (len(nr_seeds)-2) or
                                     i == (len(nr_seeds)-1)):
             seg = segment.Segment('segment_{}'.format(i),
                                 [pulses['X180'], RO_pars])
        else:
            cl_seq = rb.randomized_benchmarking_sequence(
                nr_cliffords_value, desired_net_cl=net_clifford,
                interleaved_gate=interleaved_gate)
            if i == 0:
                if interleaved_gate == 'X90':
                    print(np.any(np.array([cl_seq[1::2]]) != 16))
                elif interleaved_gate == 'Y90':
                    print(np.any(np.array([cl_seq[1::2]]) != 21))
            pulse_keys = rb.decompose_clifford_seq(
                cl_seq,
                gate_decomp=gate_decomposition)
            pulse_list = [pulses[x] for x in pulse_keys]
            pulse_list += [RO_pars]
            seg = segment.Segment('segment_{}'.format(i), pulse_list)
        seg_list.append(seg)
        seq.add(seg)

        # If the element is too long, add in an extra wait elt
        # to skip a trigger
        if resetless and nr_cliffords_value*pulse_pars[
            'pulse_delay']*1.875 > 50e-6:
            seg = segment.Segment('segment_{}'.format(i), [pulses['I']])
            seg_list.append(seg)
            seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
        return seq, seg_list
    else:
        return seq, seg_list


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
    seg_list = []
    pulse_combinations = [['X180', 'Y90'], ['Y180', 'X90']]
    pulses = get_pulse_dict_from_pars(pulse_pars)
    for i, ff in enumerate(freqs):
        pulse_keys = pulse_combinations[i % 2]
        for p_name in ['X180', 'Y180', 'X90', 'Y90']:
            pulses[p_name]['mod_frequency'] = ff
        if cal_points and (i == (len(freqs)-4) or
                           i == (len(freqs)-3)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['I'], RO_pars])
        elif cal_points and (i == (len(freqs)-2) or
                             i == (len(freqs)-1)):
            # pick motzoi for calpoint in the middle of the range
            pulses['X180']['mod_frequency'] = np.mean(freqs)
            seg = segment.Segment('segment_{}'.format(i), [pulses['X180'], RO_pars])
        else:
            pulse_list = [pulses[x] for x in pulse_keys]
            pulse_list += [RO_pars]
            seg = segment.Segment('segment_{}'.format(i), pulse_list)
        seg_list.append(seg)
        seq.add(seg)

    ps.Pulsar.get_instance().program_awgs(seq)
    if return_seq:
        return seq, seg_list
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
    seg_list = []
    pulse_combinations = [['X180', 'Y90'], ['Y180', 'X90']]
    pulses = get_pulse_dict_from_pars(pulse_pars)
    for i, motzoi in enumerate(motzois):
        pulse_keys = pulse_combinations[i % 2]
        for p_name in ['X180', 'Y180', 'X90', 'Y90']:
            pulses[p_name]['motzoi'] = motzoi
        if cal_points and (i == (len(motzois)-4) or
                           i == (len(motzois)-3)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['I'], RO_pars])
        elif cal_points and (i == (len(motzois)-2) or
                             i == (len(motzois)-1)):
            # pick motzoi for calpoint in the middle of the range
            pulses['X180']['motzoi'] = np.mean(motzois)
            seg = segment.Segment('segment_{}'.format(i), [pulses['X180'], RO_pars])
        else:
            pulse_list = [pulses[x] for x in pulse_keys]
            pulse_list += [RO_pars]
            seg = segment.Segment('segment_{}'.format(i), pulse_list)
        seg_list.append(seg)
        seq.add(seg)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    if return_seq:
        return seq, seg_list
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
    seg_list = []
    pulse_combinations=[['X90','X180'],['X90','Y180'],['X90','mY180']]
    pulses = get_pulse_dict_from_pars(pulse_pars)
    for i, motzoi in enumerate(qscales):
        pulse_keys = pulse_combinations[i % 3]
        for p_name in ['X180', 'Y180', 'X90', 'mY180']:
            pulses[p_name]['motzoi'] = motzoi
        if cal_points and (i == (len(qscales)-4) or
                                   i == (len(qscales)-3)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['I'], RO_pars])
        elif cal_points and (i == (len(qscales)-2) or
                                     i == (len(qscales)-1)):
            # pick motzoi for calpoint in the middle of the range
            pulses['X180']['motzoi'] = np.mean(qscales)
            seg = segment.Segment('segment_{}'.format(i), [pulses['X180'], RO_pars])
        else:
            pulse_list = [pulses[x] for x in pulse_keys]
            pulse_list += [RO_pars]
            seg = segment.Segment('segment_{}'.format(i), pulse_list)
        seg_list.append(seg)
        seq.add(seg)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    if return_seq:
        return seq, seg_list
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
    seg_list = []
    pulse_pars = {'pulse_type': 'RisingPulse'}
    pulse_list = [pulse_pars]
    seg = segment.Segment('segment_0', pulse_list)
    seg_list.append(seg)
    seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    if return_seq:
        return seq, seg_list
    else:
        return seq


def custom_seq(seq_func, sweep_points, pulse_pars, RO_pars,
               upload=True, return_seq=False, cal_points=True):

    seq_name = 'Custom_sequence'
    seq = sequence.Sequence(seq_name)
    seg_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)

    for i, sp in enumerate(sweep_points):
        pulse_keys_list = seq_func(i, sp)
        pulse_list = [pulses[key] for key in pulse_keys_list]
        pulse_list += [RO_pars]

        if cal_points and (i == (len(sweep_points)-4) or
                                   i == (len(sweep_points)-3)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['I'], RO_pars])
        elif cal_points and (i == (len(sweep_points)-2) or
                                     i == (len(sweep_points)-1)):
             seg = segment.Segment('segment_{}'.format(i), [pulses['X180'], RO_pars])
        else:
             seg = segment.Segment('segment_{}'.format(i), pulse_list)
        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    if return_seq:
        return seq, seg_list
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

    pulses = {'I': deepcopy(pulse_pars),
              'X180': deepcopy(pulse_pars),
              'mX180': deepcopy(pulse_pars),
              'X90': deepcopy(pulse_pars),
              'mX90': deepcopy(pulse_pars),
              'Y180': deepcopy(pulse_pars),
              'mY180': deepcopy(pulse_pars),
              'Y90': deepcopy(pulse_pars),
              'mY90': deepcopy(pulse_pars)}

    pi_amp = pulse_pars['amplitude']
    pi2_amp = pulse_pars['amplitude'] * pulse_pars['amp90_scale']

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

    pulses_sim = {key + 's': deepcopy(val) for key, val in pulses.items()}
    for val in pulses_sim.values():
        val['refpoint'] = 'simultaneous'

    pulses.update(pulses_sim)

    # Software Z-gate: apply phase offset to all subsequent X and Y pulses
    target_qubit = pulse_pars.get('target_qubit', None)
    if target_qubit is not None:
        Z180 = {'pulse_type': 'Z_pulse',
                'basis_rotation': {target_qubit: 0},
                'target_qubit': target_qubit,
                'operation_type': 'Virtual',
                'pulse_delay': 0}
        pulses.update({'Z180': Z180,
                       'mZ180': deepcopy(Z180),
                       'Z90': deepcopy(Z180),
                       'mZ90': deepcopy(Z180)})
        pulses['Z180']['basis_rotation'][target_qubit] += 180
        pulses['mZ180']['basis_rotation'][target_qubit] += -180
        pulses['Z90']['basis_rotation'][target_qubit] += 90
        pulses['mZ90']['basis_rotation'][target_qubit] += -90

    return pulses

# def multi_elem_segment_timing_seq(phases, qbn, op_dict, ramsey_time,
#                                   nr_wait_elems, elem_type='interleaved',
#                                   cal_points=((-4, -3), (-2, -1)),
#                                   return_seq=True, upload=True):
#     """
#     Args:
#         phases: the phases for the second pi/2 pulse (in rad)
#         qbn: qubit name
#         op_dict: operation dictionaty
#         ramsey_time: delay between the two pi/2 pulses
#         nr_wait_elems: the number of waiting elements between the readout
#                        pulses
#         elem_type: 'fixed'/'codeword'/'interleaved'
#     """
#     # convert cal elems to correct range:
#     cal_points = (
#         tuple(i % len(phases) for i in cal_points[0]),
#         tuple(i % len(phases) for i in cal_points[1])
#     )

#     ## Create elements
#     seg_list = []

#     idle_pulse = deepcopy(op_dict['I ' + qbn])
#     idle_pulse['nr_sigma'] = 1
#     idle_pulse['sigma'] = 2e-6
#     start_elem = multi_pulse_elt(0, station,
#                                  [idle_pulse, op_dict['X90 ' + qbn]], name='s',
#                                  trigger=True)
#     seg_list.append(start_elem)

#     wait_pulse = deepcopy(op_dict['I ' + qbn])
#     wait_pulse['nr_sigma'] = 1
#     wait_pulse['sigma'] = ramsey_time/nr_wait_elems
#     wait_pulse['sigma'] -= ps.Pulsar.get_instance().inter_element_spacing()
#     wait_samples_tek = ramsey_time/nr_wait_elems*1.2e9
#     dramsey_time = wait_samples_tek - 4*int(wait_samples_tek/4)
#     dramsey_time *= nr_wait_elems/1.2e9
#     print('wait_elem length {} Tektronix samples. Reduce ramsey time by {} s'
#           .format(wait_samples_tek, dramsey_time) + ' to satisfy granularity '
#           'constraint')
#     wait_elem = multi_pulse_elt(1, station, [wait_pulse], name='w',
#                                 trigger=False, previous_element=start_elem)
#     el_list.append(wait_elem)

#     # check that no phase is acquired over the wait element
#     ifreq = op_dict['X180 ' + qbn]['mod_frequency']
#     phase_from_if = 360*ifreq*wait_elem.ideal_length()
#     dynamic_phase = wait_elem.drive_phase_offsets.get(qbn, 0)
#     total_phase = phase_from_if + dynamic_phase
#     total_mod_phase = total_phase - 360*(total_phase//360)
#     print(qbn + ' aquires a phase of {} ≡ {} (mod 360)'.format(
#         total_phase, total_mod_phase) + ' degrees each correction ' +
#           'cycle. You should reduce the intermediate frequency by {} Hz.' \
#           .format(total_mod_phase/wait_elem.ideal_length()/360))

#     cal0_elem = multi_pulse_elt(2, station, [op_dict['I ' + qbn],
#                                              op_dict['RO ' + qbn]], name='c0',
#                                 trigger=True)
#     el_list.append(cal0_elem)
#     cal1_elem = multi_pulse_elt(3, station, [op_dict['X180 ' + qbn],
#                                              op_dict['RO ' + qbn]], name='c1',
#                                 trigger=True)
#     el_list.append(cal1_elem)

#     for i, phase in enumerate(phases):
#         if i in cal_points[0] or i in cal_points[1]:
#             continue
#         x90_pulse_mes = deepcopy(op_dict['X90 ' + qbn])
#         x90_pulse_mes['phase'] = phase*180/np.pi
#         # multi-element-segment end element

#         mes_end_pulses = [x90_pulse_mes, op_dict['RO ' + qbn]]
#         mes_end_elem = multi_pulse_elt(4+2*i, station, mes_end_pulses,
#                                        name='e{}'.format(i), trigger=False,
#                                        previous_element=wait_elem)
#         el_list.append(mes_end_elem)

#         x90_pulse_ses = deepcopy(x90_pulse_mes)
#         x90_pulse_ses['pulse_delay'] = ramsey_time
#         x90_pulse_ses['pulse_delay'] += ps.Pulsar.get_instance().inter_element_spacing()
#         ses_pulses = [idle_pulse, op_dict['X90 ' + qbn], x90_pulse_ses,
#                       op_dict['RO ' + qbn]]
#         ses_elem = multi_pulse_elt(5+2*i, station, ses_pulses,
#                                        name='a{}'.format(i), trigger=True)
#         el_list.append(ses_elem)

#     ## Create sequence
#     seq_name = 'Multi_elem_segment_timing_seq'
#     seq = sequence.Sequence(seq_name)
#     seq.codewords[0] = 'w'
#     seq.codewords[1] = 'w'
#     for i, phase in enumerate(phases):
#         if i in cal_points[0]:
#             seq.append('c0s{}'.format(i), 'c0', trigger_wait=True)
#             seq.append('c0m{}'.format(i), 'c0', trigger_wait=True)
#         elif i in cal_points[1]:
#             seq.append('c0s{}'.format(i), 'c1', trigger_wait=True)
#             seq.append('c0m{}'.format(i), 'c1', trigger_wait=True)
#         else:
#             seq.append('a{}'.format(i), 'a{}'.format(i), trigger_wait=True)
#             seq.append('s{}'.format(i), 's', trigger_wait=True)
#             for j in range(nr_wait_elems):
#                 if elem_type == 'fixed':
#                     wfname = 'w'
#                 elif elem_type == 'codeword':
#                     wfname = 'codeword'
#                 elif elem_type == 'interleaved':
#                     wfname = ['w', 'codeword'][j%2]
#                 else:
#                     raise ValueError('Invalid elem_type {}'.format(elem_type))
#                 seq.append('w{}_{}'.format(j, i), wfname, trigger_wait=False)
#             seq.append('e{}'.format(i), 'e{}'.format(i), trigger_wait=False)

#     if upload:
#         ps.Pulsar.get_instance().program_awgs(seq)
#     if return_seq:
#         return seq, seg_list
#     else:
#         return seq

def Z(theta=0, pulse_pars=None):

    """
    Software Z-gate of arbitrary rotation.

    :param theta:           rotation angle
    :param pulse_pars:      pulse parameters (dict)

    :return: Pulse dict of the Z-gate
    """
    if pulse_pars is None:
        raise ValueError('Pulse_pars is None.')
    else:
        pulses = get_pulse_dict_from_pars(pulse_pars)

    Z_gate = deepcopy(pulses['Z180'])
    Z_gate['phase'] = theta

    return Z_gate


def over_under_rotation_seq(qb_name, nr_pi_pulses_array, operation_dict,
                            pi_pulse_amp=None, cal_points=True, upload=True):
    seq_name = 'Over-under rotation sequence'
    seq = sequence.Sequence(seq_name)
    seg_list = []
    X90 = deepcopy(operation_dict['X90 ' + qb_name])
    X180 = deepcopy(operation_dict['X180 ' + qb_name])
    if pi_pulse_amp is not None:
        X90['amplitude'] = pi_pulse_amp/2
        X180['amplitude'] = pi_pulse_amp

    for i, N in enumerate(nr_pi_pulses_array):
        if cal_points and (i == (len(nr_pi_pulses_array)-4) or
                                   i == (len(nr_pi_pulses_array)-3)):
             seg = segment.Segment('segment_{}'.format(i),
                                 [operation_dict['I ' + qb_name],
                                  operation_dict['RO ' + qb_name]])
        elif cal_points and (i == (len(nr_pi_pulses_array)-2) or
                                     i == (len(nr_pi_pulses_array)-1)):
             seg = segment.Segment('segment_{}'.format(i),
                                 [operation_dict['X180 ' + qb_name],
                                  operation_dict['RO ' + qb_name]])
        else:
            pulse_list = [X90]
            pulse_list += N*[X180]
            pulse_list += [operation_dict['RO ' + qb_name]]
            seg = segment.Segment('segment_{}'.format(i), pulse_list)

        seg_list.append(seg)
        seq.add(seg)
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    return
