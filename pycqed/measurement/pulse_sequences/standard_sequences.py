from ..waveform_control import pulse
from ..waveform_control import element
from ..waveform_control import sequence
from importlib import reload
station = None
reload(element)
# You need to explicitly set this before running any functions from this module
# I guess there are cleaner solutions :)

'''
Update (12-4-2016) this module has grown out to be very CBox specific.
Almost any sequence here can be made with the "multi_pulse_elt" that exists
in single_qubit_tek_seq_elts.

Will not move what is here now for backwards compatibility.
Standard sequences. Currently (17-1-2016) all sequences have the channels
hardcoded. It would be better if these use the named channels and have the
mapping defined on the pulsar.

'''


def generate_marker_element(
        i, marker_length, marker_interval,
        acq_marker_channels='ch4_marker1,ch4_marker2,ch3_marker2,ch3_marker1'):
    # NOTE: This should be in standard elements
    # make sure tau is a multiple of 1 ns, if it is not the fixed point will
    # not be able to be computed.
    channels = list(acq_marker_channels.split(','))
    marker_element = element.Element(name=('marker_element %s' % i),
                                     pulsar=station.pulsar)
    marker_element.add(pulse.SquarePulse(name='refpulse_0', channel=channels[0],
                       amplitude=0, length=100e-9))
    number_of_pulses = int(200*1e-6/marker_interval)
    for i in range(number_of_pulses):
        for channel in channels:
            marker_element.add(pulse.SquarePulse(channel=channel, amplitude=1,
                                                 length=marker_length),
                               name='Readout tone {}_{}'.format(channel, i),
                               start=marker_interval*(i+1)-marker_length,
                               refpulse='refpulse_0-0', refpoint='start')
    return marker_element


def generate_marker_element_with_RF_mod(
        i, marker_length, marker_interval, IF, mod_amp=.5,
        acq_marker_channels='ch4_marker1,ch4_marker2,ch3_marker2,ch3_marker1',
        I_channel='ch3', Q_channel='ch4'):
    # NOTE: This should be in standard elements
    # make sure tau is a multiple of 1 ns, if it is not the fixed point will
    # not be able to be computed.
    channels = list(acq_marker_channels.split(','))
    marker_element = element.Element(name=('marker_element %s' % i),
                                     pulsar=station.pulsar)
    marker_element.add(pulse.SquarePulse(name='refpulse_0', channel=channels[0],
                       amplitude=0, length=100e-9))
    number_of_pulses = int(200*1e-6/marker_interval)
    marker_element.add(pulse.CosPulse(name='cosI', channel=I_channel,
                       amplitude=mod_amp, frequency=IF, length=200e-6))
    marker_element.add(pulse.CosPulse(name='sinQ', channel=Q_channel,
                       amplitude=mod_amp, frequency=IF,
                       length=200e-6, phase=90))

    for i in range(number_of_pulses):
        for channel in channels:
            marker_element.add(pulse.SquarePulse(channel=channel, amplitude=1,
                                                 length=marker_length),
                               name='Readout tone {}_{}'.format(channel, i),
                               start=marker_interval*(i+1)-marker_length,
                               refpulse='refpulse_0-0', refpoint='start')
    return marker_element


def generate_and_upload_marker_sequence(
        marker_length, marker_interval, RF_mod=False, IF=None, mod_amp=None,
        acq_marker_channels='ch4_marker1,ch4_marker2,ch3_marker2,ch3_marker1',
        I_channel='ch3', Q_channel='ch4'):
    seq_name = 'Heterodyne_marker_seq_RF_mod'
    seq = sequence.Sequence(seq_name)
    el_list = []
    for i in range(2):
        if RF_mod:
            el = generate_marker_element_with_RF_mod(
                i, marker_length, marker_interval, IF=IF, mod_amp=mod_amp,
                acq_marker_channels=acq_marker_channels,
                I_channel=I_channel, Q_channel=Q_channel)
        else:
            el = generate_marker_element(
                i, marker_length, marker_interval,
                acq_marker_channels=acq_marker_channels)
        el_list.append(el)
        seq.append_element(el, trigger_wait=False) # Ensures a continuously running sequence
    station.pulsar.program_awgs(seq, *el_list, verbose=False)
    return seq_name


def Pulsed_spec_seq_RF_mod(IF, spec_pulse_length=1e-6,
                           RO_pulse_length=1e-6,
                           RO_pulse_delay=100e-9,
                           RO_trigger_delay=0,
                           marker_interval=10e-6,
                           mod_amp=0.5):
    seq_name = 'Pulsed_spec_with_RF_mod'
    seq = sequence.Sequence(seq_name)
    el_list = []
    for i in range(2):
        el = st_elts.pulsed_spec_elt_with_RF_mod(
            i, station, IF,
            spec_pulse_length=spec_pulse_length,
            RO_pulse_length=RO_pulse_length,
            RO_pulse_delay=RO_pulse_delay,
            RO_trigger_delay=RO_trigger_delay,
            marker_interval=marker_interval,
            mod_amp=mod_amp)
        el_list.append(el)
        seq.append_element(el, trigger_wait=False) # Ensures a continuously running sequence
    station.pulsar.program_awgs(seq, *el_list, verbose=False)


def single_marker_seq(verbose=False):
    seq_name = 'Single_marker_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    for i in range(2):  # seq has to have at least 2 elts
        el = st_elts.single_marker_elt(i, station)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    return seq_name


def CBox_single_pulse_seq(IF, RO_pulse_delay, RO_trigger_delay,
                          RO_pulse_length, verbose=False):
    seq_name = 'Single_pulse_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    for i in range(2):  # seq has to have at least 2 elts
        el = st_elts.single_pulse_elt(i, station, IF,
                                      RO_pulse_delay,
                                      RO_trigger_delay,
                                      RO_pulse_length)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    return seq_name


def CBox_two_pulse_seq(IF, pulse_delay, RO_pulse_length,
                       RO_pulse_delay, RO_trigger_delay,
                       verbose=False):
    seq_name = 'CBox_two_pulse_seq'
    seq = sequence.Sequence(seq_name)
    el_list = []

    for i in range(2):  # seq has to have at least 2 elts
        el = st_elts.two_pulse_elt(i, station, IF,
                                   pulse_delay=pulse_delay,
                                   RO_pulse_delay=RO_pulse_delay,
                                   RO_trigger_delay=RO_trigger_delay,
                                   RO_pulse_length=RO_pulse_length)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    return seq_name


def CBox_multi_pulse_seq(IF, n_pulses,
                         pulse_delay,
                         RO_pulse_delay,
                         RO_pulse_length,
                         RO_trigger_delay,
                         verbose=False):
    seq_name = 'CBox_%s_pulse_seq' % n_pulses
    seq = sequence.Sequence(seq_name)
    el_list = []
    for i in range(2):  # seq has to have at least 2 elts
        el = st_elts.multi_pulse_elt(i, station, IF,
                                     pulse_delay=pulse_delay,
                                     RO_pulse_delay=RO_pulse_delay,
                                     RO_pulse_length=RO_pulse_length,
                                     RO_trigger_delay=RO_trigger_delay,
                                     n_pulses=n_pulses)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
    return seq_name


def CBox_resetless_multi_pulse_seq(IF, n_pulses,
                                   pulse_delay=60e-9,
                                   RO_pulse_delay=0,
                                   RO_pulse_length=1e-6,
                                   RO_trigger_delay=0,
                                   resetless_interval=10e-6,
                                   verbose=False):
    seq_name = 'CBox_resetless_{}_pulse_seq'.format(n_pulses)
    seq = sequence.Sequence(seq_name)
    el_list = []
    for i in range(3):  # seq has to have at least 2 elts
        el = st_elts.CBox_resetless_multi_pulse_elt(
            i, station, IF,
            pulse_delay=pulse_delay,
            RO_pulse_delay=RO_pulse_delay,
            RO_trigger_delay=RO_trigger_delay,
            RO_pulse_length=RO_pulse_length,
            resetless_interval=resetless_interval,
            n_pulses=n_pulses)
        el_list.append(el)

    # trick to make sure the sequence runs continously while still requiring
    # a trigger
    seq.append_element(el_list[0], trigger_wait=True)
    seq.append_element(el_list[1], trigger_wait=False,
                       goto_target=el_list[1].name)
    # Extra element is needed because otherwise last elt will always goto 1
    seq.append_element(el_list[2], trigger_wait=False,
                       goto_target=el_list[1].name)
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)


def CBox_T1_marker_seq(IF, times, RO_pulse_delay,
                       RO_trigger_delay=0, verbose=False):
    '''
    Beware, replaces the last 4 points with calibration points
    '''
    seq_name = 'CBox_T1'
    seq = sequence.Sequence(seq_name)
    el_list = []

    for i, tau in enumerate(times[:-4]):
        el = st_elts.single_pulse_elt(i, station, IF, RO_pulse_delay,
                                      RO_trigger_delay, tau=tau)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    cal_0_elt0 = st_elts.no_pulse_elt(i+1, station, IF, RO_trigger_delay)
    cal_0_elt1 = st_elts.no_pulse_elt(i+2, station, IF, RO_trigger_delay)
    cal_1_elt0 = st_elts.single_pulse_elt(i+3, station, IF, RO_pulse_delay,
                                          RO_trigger_delay)
    cal_1_elt1 = st_elts.single_pulse_elt(i+4, station, IF, RO_pulse_delay,
                                          RO_trigger_delay)
    seq.append_element(cal_0_elt0, trigger_wait=True)
    seq.append_element(cal_0_elt1, trigger_wait=True)
    seq.append_element(cal_1_elt0, trigger_wait=True)
    seq.append_element(cal_1_elt1, trigger_wait=True)
    el_list.append(cal_0_elt0)
    el_list.append(cal_0_elt1)
    el_list.append(cal_1_elt0)
    el_list.append(cal_1_elt1)
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)


def CBox_Ramsey_marker_seq(IF, times, RO_pulse_delay, RO_pulse_length,
                           RO_trigger_delay, pulse_delay,
                           verbose=False):
    '''
    If cal_points, replaces the last 4 elements with calibration points
    '''
    seq_name = 'CBox_Ramsey'
    seq = sequence.Sequence(seq_name)
    el_list = []

    for i, tau in enumerate(times):
        el = st_elts.two_pulse_elt(i, station, IF, RO_pulse_delay,
                                   RO_trigger_delay=RO_trigger_delay,
                                   RO_pulse_length=RO_pulse_length,
                                   pulse_delay=pulse_delay,
                                   tau=tau)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)

    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)


def CBox_marker_train_seq(marker_separation=100e-9,
                          verbose=False):
    seq_name = 'CBox_marker_train_seq'
    seq = sequence.Sequence(seq_name)
    el_list = []
    for i in range(3):  # seq has to have at least 2 elts
        el = st_elts.CBox_marker_sequence(
            i, station, marker_separation)
        el_list.append(el)

    # trick to make sure the sequence runs continously while still requiring
    # a trigger
    seq.append_element(el_list[0], trigger_wait=True)
    seq.append_element(el_list[1], trigger_wait=False,
                       goto_target=el_list[1].name)
    # Extra element is needed because otherwise last elt will always goto 1
    seq.append_element(el_list[2], trigger_wait=False,
                       goto_target=el_list[1].name)
    station.pulsar.program_awgs(seq, *el_list, verbose=verbose)
