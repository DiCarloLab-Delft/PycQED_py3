import logging
import numpy as np
from copy import deepcopy
from pycqed.measurement.waveform_control.element import calculate_time_correction
from pycqed.measurement.waveform_control import pulse
from pycqed.measurement.waveform_control import pulsar as ps
from pycqed.measurement.waveform_control import sequence as sequence
from pycqed.measurement.waveform_control import segment as segment
from pycqed.measurement.randomized_benchmarking import randomized_benchmarking as rb

from importlib import reload
reload(pulse)
from ..waveform_control import pulse_library
reload(pulse_library)
import logging
log = logging.getLogger(__name__)

def pulse_list_list_seq(pulse_list_list, name='pulse_list_list_sequence', 
                        upload=True):
    seq = sequence.Sequence(name)
    for i, pulse_list in enumerate(pulse_list_list):
        seq.add(segment.Segment('segment_{}'.format(i), pulse_list))
    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    return seq

def rabi_seq_active_reset(amps, qb_name, operation_dict, cal_points_obj,
                          cal_points=True, no_cal_points=4, upload=True, n=1,
                          preparation_type='wait', post_ro_wait=1e-6,
                          reset_reps=1, final_reset_pulse=True, for_ef=False,
                          last_ge_pulse=False):
    '''
    Rabi sequence for a single qubit using the tektronix.
    Input pars:
        amps:            array of pulse amplitudes (V)
        pulse_pars:      dict containing the pulse parameters
        RO_pars:         dict containing the RO parameters
        active_reset:    boolean flag specifying if active reset is used
        n:               number of pulses (1 is conventional Rabi)
        post_msmt_delay: extra wait time for resetless compatibility
        cal_points:      whether to use calibration points or not
        upload:          whether to upload sequence to instrument or not
    '''

    seq_name = 'Rabi_sequence'
    prep_params = dict(preparation_type=preparation_type,
                       post_ro_wait=post_ro_wait,
                       repetitions=reset_reps,
                       final_reset_pulse=final_reset_pulse)

    # add Rabi amplitudes segments
    rabi_ops = ["X180_ef " + qb_name if for_ef else "X180 " + qb_name] * n

    if for_ef:
        rabi_ops = ["X180 " + qb_name] + rabi_ops
        if last_ge_pulse:
            rabi_ops += ["X180 " + qb_name]
    rabi_ops += ["RO " + qb_name]
    rabi_pulses = [operation_dict[op] for op in rabi_ops]
    for i in range(1 if for_ef else 0, n + 1 if for_ef else n):
        rabi_pulses[i]["name"] = f"Rabi_{i - 1 if for_ef else i}"

    swept_pulses = \
        sweep_pulse_params(rabi_pulses,
                           {f'Rabi_{i}.amplitude': amps for i in range(n)})
    swept_pulses_with_prep = \
        [add_preparation_pulses(p, operation_dict, [qb_name], **prep_params)
         for p in swept_pulses]
    seq = pulse_list_list_seq(swept_pulses_with_prep, seq_name, upload=False)

    # add calibration segments
    cal_segs = cal_points_obj.create_segments(operation_dict, **prep_params)
    for seg in cal_segs:
        seq.add(seg)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    return seq

def add_preparation_pulses(pulse_list, operation_dict, qb_names,
                           preparation_type='wait', post_ro_wait=1e-6,
                           repetitions=3, final_reset_pulse=True):
    """
    Prepends to pulse_list the preparation pulses corresponding to preparation

    preparation:
        for active reset on |e>: ('active_reset_e', nr_resets)
        for active reset on |e> and |f>: ('active_reset_ef', nr_resets)
        for preselection: ('preselection', nr_readouts)
    """

    # Calculate the length of a ge pulse, assumed the same for all qubits
    ge_pulse = operation_dict['X180 ' + qb_names[0]]
    ge_length = ge_pulse['nr_sigma']*ge_pulse['sigma']

    if preparation_type == 'wait':
        return pulse_list

    elif 'active_reset' in preparation_type:
        reset_ro_pulses = []
        for i, qbn in enumerate(qb_names):
            reset_ro_pulses.append(deepcopy(operation_dict['RO ' + qbn]))
            reset_ro_pulses[-1]['ref_point'] = 'start' if i != 0 else 'end'

        if preparation_type == 'active_reset_e':
            ops_and_codewords = [(['I '], 0), (['X180 '], 1)]
            ef_length = 0
        elif preparation_type == 'active_reset_ef':
            ops_and_codewords = [(['I '], 0), (['X180 '], 1),
                (['X180 ', 'X180_ef '], 2), (['X180 ', 'X180_ef '], 3)]
            ef_pulse = operation_dict['X180_ef ' + qb_names[0]]
            ef_length = ef_pulse['nr_sigma'] * ef_pulse['sigma']
        else:
            raise ValueError(f'Invalid preparation type {preparation_type}')

        reset_pulses = []
        for i, qbn in enumerate(qb_names):
            for ops, codeword in ops_and_codewords:
                for j, op in enumerate(ops):
                    reset_pulses.append(deepcopy(operation_dict[op + qbn]))
                    reset_pulses[-1]['codeword'] = codeword
                    if j == 0:
                        reset_pulses[-1]['ref_point'] = 'start'
                        reset_pulses[-1]['pulse_delay'] = post_ro_wait

        prep_pulse_list = []
        for rep in range(repetitions):
            ro_list = deepcopy(reset_ro_pulses)
            for pulse in ro_list:
                pulse['element_name'] = 'reset_ro_element_{}'.format(rep)
            if rep == 0:
                ro_list[0]['ref_pulse'] = 'segment_start'
                ro_list[0]['pulse_delay'] = -repetitions*(post_ro_wait +
                                                          ge_length + ef_length)
            ro_list[0]['pulse_name'] = 'refpulse_reset_element_{}'.format(rep)
            rp_list = deepcopy(reset_pulses)
            for pulse in rp_list:
                pulse['element_name'] = 'reset_pulse_element_{}'.format(rep)
                pulse['refpulse'] = 'refpulse_reset_element_{}'.format(rep)
            prep_pulse_list += ro_list
            prep_pulse_list += rp_list

        if final_reset_pulse:
            rp_list = deepcopy(reset_pulses)
            for pulse in rp_list:
                pulse['element_name'] = f'reset_pulse_element_{repetitions}'
            pulse_list += rp_list

        return prep_pulse_list + pulse_list

    elif preparation_type == 'preselection':
        preparation_pulses = []
        for i, qbn in enumerate(qb_names):
            preparation_pulses.append(deepcopy(operation_dict['RO ' + qbn]))
            preparation_pulses[-1]['ref_point'] = 'start'
            preparation_pulses[-1]['element_name'] = 'preselection_element'
        preparation_pulses[0]['ref_pulse'] = 'segment_start'
        preparation_pulses[0]['pulse_delay'] = -post_ro_wait

        return preparation_pulses + pulse_list

def sweep_pulse_params(pulses, params):
    """
    Sweeps a list of pulses over specified parameters.
    Args:
        pulses (list): All pulses. Pulses which have to be swept over need to
            have a 'name' key.
        params (dict):  keys in format <pulse_name>.<pulse_param_name>,
            values are the sweep values.

    Returns: a list of pulses_lists where each element is to be used
        for a single segment

    """
    swept_pulses = []
    assert len(params.keys()) > 0, "No params to sweep"
    n_sweep_points = len(list(params.values())[0])

    assert np.all([len(v) == n_sweep_points for v in params.values()]), \
        "Parameter sweep values are not of all of the same length: {}" \
        .format({n: len(v) for n, v in params.items()})

    for i in range(n_sweep_points):
        pulses_cp = deepcopy(pulses)

        for name, sweep_values in params.items():
            pulse_name, param_name = name.split('.')
            pulse_indices = [i for i, p in enumerate(pulses)
                             if p.get('name', False) == pulse_name]
            if len(pulse_indices) == 0:
                raise ValueError(f"No pulse with name {pulse_name} found.")
            if len(pulse_indices) > 1:
                log.warning(f"Pulse name is not unique. Found "
                            f"{len(pulse_indices)} with name {pulse_name}")
            for p_idx in pulse_indices:
                pulses_cp[p_idx][param_name] = sweep_values[i]

        swept_pulses.append(pulses_cp)

    return swept_pulses

def rabi_seq(amps, pulse_pars, RO_pars, n=1, no_cal_points=2,
             cal_points=True, upload=True, return_seq=False):
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
    pulses_unmodified = get_pulse_dict_from_pars(pulse_pars)
    pulses = deepcopy(pulses_unmodified)
    seg_list = []
    for i, amp in enumerate(amps):  # seq has to have at least 2 elts
        if cal_points and no_cal_points == 4 and \
                (i == (len(amps)-4) or i == (len(amps)-3)):
            seg = segment.Segment('segment_{}'.format(i),
                                  [pulses_unmodified['I'], RO_pars])
        elif cal_points and no_cal_points == 4 and \
                (i == (len(amps)-2) or i == (len(amps)-1)):
            seg = segment.Segment('segment_{}'.format(i),
                                  [pulses_unmodified['X180'], RO_pars])
        elif cal_points and no_cal_points == 2 and \
                (i == (len(amps)-2) or i == (len(amps)-1)):
            seg = segment.Segment('segment_{}'.format(i),
                                  [pulses_unmodified['I'], RO_pars])
        else:
            pulses['X180']['amplitude'] = amp
            pulse_list = n*[pulses['X180']]+[RO_pars]
            seg = segment.Segment('segment_{}'.format(i), pulse_list)
        seg_list += [seg]
        seq.add(seg)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    if return_seq:
        return seq, seg_list
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


def rabi_amp90_seq(scales, pulse_pars, RO_pars, n=1, post_msmt_delay=3e-6,
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
           cal_points=True, upload=True, return_seq=False):
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
        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
            RO_pars['pulse_delay'] = RO_pulse_delay
            seg = segment.Segment('segment_{}'.format(i),
                                  [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
            RO_pars['pulse_delay'] = RO_pulse_delay
            seg = segment.Segment('segment_{}'.format(i),
                                  [pulses['X180'], RO_pars])
        else:
            seg = segment.Segment('segment_{}'.format(i),
                                  [pulses['X180'], RO_pars])
        seg_list.append(seg)
        seq.add(seg)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)

    if return_seq:
        return seq, seg_list
    else:
        return seq_name


def ramsey_seq_Echo(times, pulse_pars, RO_pars, nr_echo_pulses=4,
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
    pulse_pars_x2['ref_point'] = 'start'

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
                        pd['ref_point'] = 'end'
                        pd['pulse_delay'] = \
                            (start_end_delay if p_nr == 0 else echo_pulse_delay)
                        pulse_dict_list.append(pd)

                    pulse_pars_x2['ref_point'] = 'end'
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
                        pd['ref_point'] = 'end'
                        pd['pulse_delay'] = pulse_delays_func(
                            p_nr+1, nr_echo_pulses)
                        pulse_dict_list.append(pd)

                    pulse_pars_x2['ref_point'] = 'end'
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


def ramsey_seq_cont_drive(times, pulse_pars, RO_pars,
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
                 'ref_point': 'end'}
    sin_pulse = {'pulse_type': 'CosPulse_gauss_rise',
                 'channel': X180_pulse['Q_channel'],
                 'frequency': X180_pulse['mod_frequency'],
                 'length': 0,
                 'phase': 90,
                 'amplitude': cont_drive_ampl * X180_pulse['alpha'],
                 'pulse_delay': 0,
                 'ref_point': 'simultaneous'}

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
                pulse_pars_x2['ref_point'] = 'end'
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
                pulse_pars_x2['ref_point'] = 'start'
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


def ramsey_seq(times, pulse_pars, RO_pars,
               artificial_detuning=None,
               cal_points=True,
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
    pulse_pars_x2['ref_point'] = 'start'
    for i, tau in enumerate(times):
        pulse_pars_x2['pulse_delay'] = tau
        if artificial_detuning is not None:
            Dphase = ((tau-times[0]) * artificial_detuning * 360) % 360
            pulse_pars_x2['phase'] = Dphase

        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
             seg = segment.Segment('segment_{}'.format(i),
                                   [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
             seg = segment.Segment('segment_{}'.format(i),
                                   [pulses['X180'], RO_pars])
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


def ramsey_seq_VZ(times, pulse_pars, RO_pars,
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
    pulse_pars_x2['ref_point'] = 'start'
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

def ramsey_seq_multiple_detunings(times, pulse_pars, RO_pars,
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
    pulse_pars_x2['ref_point'] = 'start'
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


def echo_seq(times, pulse_pars, RO_pars,
             artificial_detuning=None,
             cal_points=True, upload=True, return_seq=False):
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
    center_X180['ref_point'] = 'start'
    final_X90['ref_point'] = 'start'

    for i, tau in enumerate(times):
        center_X180['pulse_delay'] = tau/2
        final_X90['pulse_delay'] = tau/2
        if artificial_detuning is not None:
            final_X90['phase'] = (tau-times[0]) * artificial_detuning * 360
        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
             seg = segment.Segment('segment_{}'.format(i),
                                   [pulses['I'], RO_pars])
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
             seg = segment.Segment('segment_{}'.format(i),
                                   [pulses['X180'], RO_pars])
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


def single_level_seq(pulse_pars, RO_pars, pulse_pars_2nd=None, verbose=False,
                     level='e', RO_spacing=300e-9,
                     upload=True, return_seq=False, preselection=False):
    '''
    OffOn sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    Input pars:
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters
        pulse_pars_2nd:      dict containing the pulse parameters of ef transition.
                             Required if level is 'f'.
        Initialize:          adds an exta measurement before state preparation
                             to allow initialization by post-selection
        Post-measurement delay:  should be sufficiently long to avoid
                             photon-induced gate errors when post-selecting.
        level:               specifies for which level a pulse should be generated (g,e,f)
        preselection:        adds an extra readout pulse before other pulses.
    '''
    seq_name = 'single_level_sequence'
    seq = sequence.Sequence(seq_name)
    seg_list = []
    # Create dicts with the parameters for all the pulses
    pulse_1st = get_pulse_dict_from_pars(pulse_pars)
    if level == 'g':
        pulse_combination = [pulse_1st['I']]
    elif level == 'e':
        pulse_combination = [pulse_1st['X180']]
    elif level == 'f':
        assert pulse_pars_2nd is not None, \
            "pulse_pars_2nd is a required parameter for f-level pulse."
        pulse_2nd = get_pulse_dict_from_pars(pulse_pars_2nd)
        pulse_combination = [pulse_1st['X180'], pulse_2nd['X180']]
    else:
        raise ValueError("Unrecognized Level: {}. Should be g, e or f.\n"
                         "Note: Naming levels 'on' and 'off' is now deprecated "
                         "to ensure clear denomination for 3 level readout. "
                         "Please adapt your code:\n 'off' --> 'g'\n'on' --> 'e'"
                         "\n'f' for f-level .")

    
    if preselection:
        pulse_list = [RO_pars]
        pulse = deepcopy(pulse_combination[0])
        pulse['pulse_delay'] = 500e-9
        pulse_list += [pulse]
        if len(pulse_combination) > 1:
            pulse_list += pulse_combination[1:]
        pulse_list += [RO_pars]    
    else:
        pulse_list = pulse_combination+[RO_pars]
    print(pulse_list)
    seg = segment.Segment('segment_{}_level'.format(level), pulse_list)
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

def qscale(qscales, pulse_pars, RO_pars,
           cal_points=True, upload=True, return_seq=False):
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
    pulse_combinations = [['X90', 'X180'], ['X90', 'Y180'], ['X90', 'mY180']]
    pulses = get_pulse_dict_from_pars(pulse_pars)
    for i, motzoi in enumerate(qscales):
        pulse_keys = pulse_combinations[i % 3]
        for p_name in ['X180', 'Y180', 'X90', 'mY180']:
            pulses[p_name]['motzoi'] = motzoi
        if cal_points and (i == (len(qscales)-4) or i == (len(qscales)-3)):
            seg = segment.Segment('segment_{}'.format(i),
                                  [pulses['I'], RO_pars])
        elif cal_points and (i == (len(qscales)-2) or i == (len(qscales)-1)):
            # pick motzoi for calpoint in the middle of the range
            pulses['X180']['motzoi'] = np.mean(qscales)
            seg = segment.Segment('segment_{}'.format(i),
                                  [pulses['X180'], RO_pars])
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
        val['ref_point'] = 'start'

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
