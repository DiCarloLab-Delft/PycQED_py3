import numpy as np
from copy import deepcopy
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from pycqed.measurement.waveform_control import pulsar as ps
from pycqed.measurement.waveform_control import sequence as sequence
from pycqed.measurement.waveform_control import segment as segment
from pycqed.measurement.pulse_sequences.single_qubit_tek_seq_elts import \
    get_pulse_dict_from_pars


station = None

def cos_seq(amplitude, frequency, channels, phases,
            marker_channels=None, marker_lenght=20e-9,
            verbose=False, alphas=[1], phi_skews=[0], ):
    '''
    Cosine  sequence, plays a continuous cos on the specified channel


    Input pars:
        amplitude (float): amplitude in Vpp
        frequency (float): frequency in Hz
        channels (list[(str)]: channels on which to set a cos
        phases (list[(float)]: phases in degree

        marker_channels (list[(str)]: optionally specify markers to play

    '''
    seq_name = 'ModSquare'
    seq = sequence.Sequence(seq_name)
    el_list = []

    base_pars = {'pulse_type': 'ModSquare',
                'mod_frequency': frequency,
                'length': 2e-6,
                'amplitude': amplitude,
                'pulse_delay': 0}
    marker_pars = {'pulse_type': 'SquarePulse',
                   'length': marker_lenght,
                   'amplitude': 1,
                   'pulse_delay': 10e-9}

    pulse_list = []
    for i, phase in enumerate(phases):
        pulse = deepcopy(base_pars)
        pulse['I_channel'] = channels[i*2]
        pulse['Q_channel'] = channels[i*2+1]
        pulse['phase'] = phase
        pulse['alpha'] = alphas[i]
        pulse['phi_skew'] = phi_skews[i]
        pulse_list.append(pulse)
        # copy first element and set extra wait
    if marker_channels !=None:
        for i, marker_channel in enumerate(marker_channels):
            pulse = deepcopy(marker_pars)
            pulse['channel'] = marker_channel
            if i != 0:
                pulse['pulse_delay'] = 0
            pulse_list.append(pulse)

    el = multi_pulse_elt(0, station, pulse_list)
    el_list.append(el)
    seq.append_element(el, trigger_wait=False)
    # seq.append_element(el, trigger_wait=False)
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    return seq_name

def mixer_calibration_sequence(trigger_separation, amplitude, trigger_channel=None,
                               RO_pars = None,
                               pulse_I_channel='AWG_ch1', pulse_Q_channel='AWG_ch2',
                               f_pulse_mod=0, phi_skew=0, alpha=1, upload=True):
    if trigger_channel is not None:
        RO_trigger = {'pulse_type': 'SquarePulse',
                      'channel': trigger_channel,
                      'length': 20e-9,
                      'amplitude': 1.,
                      'pulse_delay': 0}
    elif RO_pars is not None:
        RO_trigger = RO_pars
        trigger_channel = RO_pars['acq_marker_channel']
    else:
        raise ValueError('Set either RO_pars or trigger_channel')

    pulses = [RO_trigger]
    channels = [trigger_channel]
    channels += station.sequencer_config['slave_AWG_trig_channels']
    if pulse_I_channel is not None:
        cos_pulse = {'pulse_type': 'CosPulse',
                     'channel': pulse_I_channel,
                     'frequency': f_pulse_mod,
                     'length': trigger_separation -
                               station.pulsar.inter_element_spacing(),
                     'phase': phi_skew,
                     'amplitude': amplitude * alpha,
                     'pulse_delay': 0,
                     'refpoint': 'simultaneous'}
        pulses.append(cos_pulse)
        channels.append(pulse_I_channel)
    if pulse_Q_channel is not None:
        sin_pulse = {'pulse_type': 'CosPulse',
                     'channel': pulse_Q_channel,
                     'frequency': f_pulse_mod,
                     'length': trigger_separation -
                               station.pulsar.inter_element_spacing(),
                     'phase': 90,
                     'amplitude': amplitude,
                     'pulse_delay': 0,
                     'refpoint': 'simultaneous'}
        pulses.append(sin_pulse)
        channels.append(pulse_Q_channel)
    if pulse_I_channel is None and pulse_Q_channel is None:
        empty_pulse = {'pulse_type': 'SquarePulse',
                       'channel': trigger_channel,
                       'length': trigger_separation -
                                 station.pulsar.inter_element_spacing(),
                       'amplitude': 0.,
                       'pulse_delay': 0,
                       'refpoint': 'simultaneous'}
        pulses.append(empty_pulse)
    el = multi_pulse_elt(0, station, pulses, trigger=True)
    seq = sequence.Sequence('Sideband_modulation_seq')
    seq.append(name='SSB_modulation_el', wfname=el.name, trigger_wait=True)
    if upload:
        station.pulsar.program_awgs(seq, el, channels=channels)
    return seq, [el]





def readout_pulse_scope_seq(delays, pulse_pars, RO_pars, RO_separation,
                            cal_points=((-4, -3), (-2, -1)), comm_freq=225e6,
                            upload=True, return_seq=False, prep_pulses=None,
                            verbose=False):
    """
    Prepares the AWGs for a readout pulse shape and timing measurement.

    The sequence consists of two readout pulses where the drive pulse start
    time is swept through the first readout pulse. Because the photons in the
    readout resonator induce an ac-Stark shift of the qubit frequency, we can
    determine the readout pulse shape by sweeping the drive frequency in an
    outer loop to determine the qubit frequency.

    Important: This sequence includes two readouts per segment. For this reason
    the calibration points are also duplicated.

    Args:
        delays: A list of delays between the start of the first readout pulse
                and the end of the drive pulse.
        pulse_pars: Pulse dictionary for the drive pulse.
        RO_pars: Pulse dictionary for the readout pulse.
        RO_separation: Separation between the starts of the two readout pulses.
                       If the comm_freq parameter is not None, the used value
                       is increased to satisfy the commensurability constraint.
        cal_points: True for default calibration points, False for no
                          calibration points or a list of two lists, containing
                          the indices of the calibration segments for the ground
                          and excited state.
        comm_freq: The readout pulse separation will be a multiple of
                   1/comm_freq
    Returns:
        The sequence object and the element list if return_seq is True. Else
        return the sequence name.
    """
    if cal_points:
        cal_points = ((-4, -3), (-2, -1))
    elif not cal_points or cal_points is None:
        cal_points = ((), ())
    if prep_pulses is None:
        prep_pulses = []
    if comm_freq:
        RO_separation -= RO_separation % (-1/comm_freq)

    seq_name = 'readout_pulse_scope_sequence'
    seq = sequence.Sequence(seq_name)
    seg_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    min_delay = min(delays)
    readout_x1 = deepcopy(RO_pars)
    readout_x1['ref_point'] = 'end'
    readout_x2 = deepcopy(RO_pars)
    readout_x2['pulse_delay'] = RO_separation
    readout_x2['ref_point'] = 'start'
    probe_pulse = deepcopy(pulses['X180'])
    prep_pulses = [deepcopy(pulses[pulse_name]) for pulse_name in prep_pulses]
    for pulse in prep_pulses:
        pulse['pulse_delay'] = -2*np.abs(min_delay)
    for i, tau in enumerate(delays):
        if i in cal_points[0] or i - len(delays) in cal_points[0]:
            seg = segment.Segment('segment_{}'.format(2*i),
                                  [pulses['I'], RO_pars])
            seg_list.append(seg)
            seq.add(seg)
            seg = segment.Segment('segment_{}'.format(2*i+1),
                                  [pulses['I'], RO_pars])
            seg_list.append(seg)
            seq.add(seg)
        elif i in cal_points[1] or i - len(delays) in cal_points[1]:
            seg = segment.Segment('segment_{}'.format(2*i),
                                  [pulses['X180'], RO_pars])
            seg_list.append(seg)
            seq.add(seg)
            seg = segment.Segment('segment_{}'.format(2*i+1),
                                  [pulses['X180'], RO_pars])
            seg_list.append(seg)
            seq.add(seg)
        else:
            probe_pulse['pulse_delay'] = tau - min_delay
            readout_x1['pulse_delay'] = -tau
            # probe_pulse.update({
            #     'reference_pulse': 'segment_start',
            #     'name': 'probe_pulse{}'.format(2 * i),
            #     'element_name': 'probe_elt{}'.format(2 * i),
            # })
            # readout_x1.update({
            #     'reference_pulse': 'probe_pulse{}'.format(2 * i),
            #     'name': 'probe_ro{}'.format(2 * i),
            #     'element_name': 'probe_elt{}'.format(2 * i),
            # })
            # readout_x2.update({
            #     'reference_pulse': 'probe_ro{}'.format(2 * i),
            #     'name': 'measure_ro{}'.format(2 * i),
            #     'element_name': 'measure_elt{}'.format(2 * i),
            # })
            # from pprint import pprint
            # for p in [probe_pulse, readout_x1, readout_x2]:
            #     pprint(p)
            seg = segment.Segment('segment_{}'.format(2*i), prep_pulses +
                                  [probe_pulse, readout_x1, readout_x2])
            seg_list.append(seg)
            seq.add(seg)

    if upload:
        ps.Pulsar.get_instance().program_awgs(seq)
    if return_seq:
        return seq, seg_list
    else:
        return seq_name

def readout_photons_in_resonator_seq(delay_to_relax, delay_buffer, ramsey_times,
                            RO_pars, pulse_pars, cal_points=((-4, -3), (-2, -1)),
                            verbose=False, upload=True, return_seq=False,
                            artificial_detuning=None):
    """
    The sequence consists of two readout pulses sandwitching two ramsey pulses
    inbetween. The delay between the first readout pulse and first ramsey pulse
    is swept, to measure the ac stark shift and dephasing from any residual
    photons.

    Important: This sequence includes two readouts per segment. For this reason
    the calibration points are also duplicated.

    Args:
        delay_ro_relax: delay between the end of the first readout
                        pulse and the start of the first ramsey pulse.

        pulse_pars: Pulse dictionary for the ramsey pulse.
        RO_pars: Pulse dictionary for the readout pulse.
        delay_buffer: delay between the start of the last ramsey pulse and the
                      start of the second readout pulse.
        ramsey_times: delays between ramsey pulses
        cal_points: True for default calibration points, False for no
                          calibration points or a list of two lists, containing
                          the indices of the calibration segments for the ground
                          and excited state.

    Returns:
        The sequence object and the element list if return_seq is True. Else
        return the sequence name.
    """
    if cal_points is True: cal_points = ((-4, -3), (-2, -1))
    elif cal_points is False or cal_points is None: cal_points = ((), ())

    seq_name = 'readout_photons_in_resonator_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    ramsey_x1 = deepcopy(pulses['X90'])
    ramsey_x1['pulse_delay'] = delay_to_relax
    readout_x2 = deepcopy(RO_pars)
    readout_x2['pulse_delay'] = delay_buffer

    for i, tau in enumerate(ramsey_times):
        if i in cal_points[0] or i - len(ramsey_times) in cal_points[0]:
            el = multi_pulse_elt(2 * i, station, [pulses['I'], RO_pars])
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)
            el = multi_pulse_elt(2 * i + 1, station, [pulses['I'], RO_pars])
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)
        elif i in cal_points[1] or i - len(ramsey_times) in cal_points[1]:
            el = multi_pulse_elt(2 * i, station, [pulses['X180'], RO_pars])
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)
            el = multi_pulse_elt(2 * i + 1, station, [pulses['X180'], RO_pars])
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)
        else:
            ramsey_x2 = deepcopy(pulses['X90'])
            ramsey_x2['refpoint'] = 'start'
            ramsey_x2['pulse_delay'] = tau
            if artificial_detuning is not None:
                Dphase = (tau * artificial_detuning * 360) % 360
                ramsey_x2['phase'] += Dphase

            prep_pulse = [pulses['I'], pulses['X180']][i % 2]
            el = multi_pulse_elt(2 * i, station, [prep_pulse, RO_pars,
                                              ramsey_x1, ramsey_x2, readout_x2])
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name

