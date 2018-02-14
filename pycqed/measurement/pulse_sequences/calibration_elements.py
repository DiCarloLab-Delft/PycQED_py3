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
                               pulse_I_channel, pulse_Q_channel, f_pulse_mod=0,
                               phi_skew=0, alpha=1):
    RO_trigger = {'pulse_type': 'SquarePulse',
                  'channel': trigger_channel,
                  'length': 20e-9,
                  'amplitude': 1.,
                  'pulse_delay': 0}
    cos_pulse = {'pulse_type': 'CosPulse',
                 'channel': pulse_I_channel,
                 'frequency': f_pulse_mod,
                 'length': trigger_separation,
                 'phase': phi_skew,
                 'amplitude': amplitude * alpha,
                 'pulse_delay': 0,
                 'refpoint': 'simultaneous'}
    sin_pulse = {'pulse_type': 'CosPulse',
                 'channel': pulse_Q_channel,
                 'frequency': f_pulse_mod,
                 'length': trigger_separation,
                 'phase': 90,
                 'amplitude': amplitude,
                 'pulse_delay': 0,
                 'refpoint': 'simultaneous'}
    el = multi_pulse_elt(0, station, [RO_trigger, cos_pulse, sin_pulse],
                         trigger=False)
    seq = sequence.Sequence('Sideband_modulation_seq')
    seq.append(name='SSB_modulation_el', wfname=el.name, trigger_wait=False)
    station.pulsar.program_awgs(seq, el, channels=[pulse_I_channel,
                                                   pulse_Q_channel,
                                                   trigger_channel])


def readout_pulse_scope_seq(delays, pulse_pars, RO_pars, RO_separation,
                            cal_points=((-4, -3), (-2, -1)), comm_freq=225e6,
                            verbose=False, upload=True, return_seq=False):
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
                and the center of the drive pulse.
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

    if comm_freq:
        RO_separation -= RO_separation % (-1/comm_freq)

    seq_name = 'readout_pulse_scope_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    pulses = get_pulse_dict_from_pars(pulse_pars)
    readout_x1 = deepcopy(RO_pars)
    readout_x1['refpoint'] = 'center'
    readout_x2 = deepcopy(RO_pars)
    readout_x2['pulse_delay'] = RO_separation
    readout_x2['refpoint'] = 'start'

    for i, tau in enumerate(delays):
        readout_x1['pulse_delay'] = -tau
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
            el = multi_pulse_elt(2 * i, station, [pulses['X180'], readout_x1,
                                              readout_x2])
            el_list.append(el)
            seq.append_element(el, trigger_wait=True)

    if upload:
        station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    if return_seq:
        return seq, el_list
    else:
        return seq_name
