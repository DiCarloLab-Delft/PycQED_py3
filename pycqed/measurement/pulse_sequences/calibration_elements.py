from copy import deepcopy
from pycqed.measurement.pulse_sequences.standard_elements import multi_pulse_elt
from pycqed.measurement.waveform_control import sequence
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

def mixer_calibration_sequence(trigger_separation, amplitude, trigger_channel,
                               pulse_I_channel=None, pulse_Q_channel=None,
                               f_pulse_mod=0, phi_skew=0, alpha=1, upload=True):
    RO_trigger = {'pulse_type': 'SquarePulse',
                  'channel': trigger_channel,
                  'length': 20e-9,
                  'amplitude': 1.,
                  'pulse_delay': 0}
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


def mixer_calibration_sequence_NN(trigger_separation, amplitude, trigger_channel,
                                  pulse_I_channel=None, pulse_Q_channel=None,
                                  f_pulse_mod=0, phi_skew=[0], alpha=[1], upload=True):


    if not isinstance(phi_skew,list):
        phi_skew = [phi_skew]
    if not isinstance(alpha,list):
        alpha = [alpha]
    if not len(alpha) == len(phi_skew):
        raise ValueError('pulse properties have to be of equal size. Received: ',
                         'len(phi): ',len(phi_skew),'. len(alpha): ',len(alpha))
    seq = []
    el = []
    channels = [trigger_channel]
    channels += station.sequencer_config['slave_AWG_trig_channels']

    if pulse_I_channel is not None:
        channels.append(pulse_I_channel)
    if pulse_Q_channel is not None:
        channels.append(pulse_Q_channel)

    for it in range(len(alpha)):
        new_seq, new_el = mixer_calibration_sequence(trigger_separation, amplitude,
                                              trigger_channel,
                                              pulse_I_channel=pulse_I_channel,
                                              pulse_Q_channel=pulse_Q_channel,
                                              f_pulse_mod=f_pulse_mod,
                                              phi_skew=phi_skew[it],
                                              alpha=alpha[it],
                                              upload=False)
        seq.append(new_seq)
        el += new_el
    if upload:
        station.pulsar.program_awgs(seq, el, channels=channels)
    return seq,el



def readout_pulse_scope_seq(delays, pulse_pars, RO_pars, RO_separation,
                            cal_points=((-4, -3), (-2, -1)), comm_freq=225e6,
                            verbose=False, upload=True, return_seq=False,
                            prep_pulses=None):
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
    if cal_points is True: cal_points = ((-4, -3), (-2, -1))
    elif cal_points is False or cal_points is None: cal_points = ((), ())
    if prep_pulses is None: prep_pulses = []
    if comm_freq: RO_separation -= RO_separation % (-1/comm_freq)

    seq_name = 'readout_pulse_scope_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    min_delay = min(delays)
    readout_x1 = deepcopy(RO_pars)
    readout_x1['refpoint'] = 'end'
    readout_x2 = deepcopy(RO_pars)
    readout_x2['pulse_delay'] = RO_separation
    readout_x2['refpoint'] = 'start'
    probe_pulse = deepcopy(pulses['X180'])
    prep_pulses = [pulses[pulse_name] for pulse_name in prep_pulses]
    for i, tau in enumerate(delays):
        if i in cal_points[0] or i - len(delays) in cal_points[0]:
            el = multi_pulse_elt(2 * i, station, [pulses['I'], RO_pars])
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)
            el = multi_pulse_elt(2 * i + 1, station, [pulses['I'], RO_pars])
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)
        elif i in cal_points[1] or i - len(delays) in cal_points[1]:
            el = multi_pulse_elt(2 * i, station, [pulses['X180'], RO_pars])
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)
            el = multi_pulse_elt(2 * i + 1, station, [pulses['X180'], RO_pars])
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)
        else:

            probe_pulse['pulse_delay'] = tau - min_delay
            readout_x1['pulse_delay'] = -tau
            el = multi_pulse_elt(2 * i, station, prep_pulses +
                                 [probe_pulse, readout_x1, readout_x2])
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name
