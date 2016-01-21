from ..waveform_control import pulsar
from ..waveform_control import pulse
from ..waveform_control import element
from ..waveform_control import sequence
from ..waveform_control.viewer import show_element, show_wf
from ..waveform_control import pulse_library as pl

from . import standard_elements as st_elts
from importlib import reload
reload(st_elts)
station = None
reload(element)
# You need to explicitly set this before running any functions from this module
# I guess there are cleaner solutions :)

'''
Standard sequences. Currently (17-1-2016) all sequences have the channels
hardcoded. It would be better if these use the named channels and have the
mapping defined on the pulsar.
'''


def generate_marker_element(i, marker_length, marker_interval):
    # make sure tau is a multiple of 1 ns, if it is not the fixed point will
    # not be able to be computed.
    marker_element = element.Element(name=('marker_element %s' % i),
                                     pulsar=station.pulsar)
    marker_element.add(pulse.SquarePulse(name='refpulse_0', channel='ch1',
                       amplitude=0, length=100e-9))
    number_of_pulses = int(200*1e-6/marker_interval)
    for i in range(number_of_pulses):
        for channel in  ['ch4_marker1', 'ch4_marker2', 'ch3_marker2', 'ch3_marker1']:
            marker_element.add(pulse.SquarePulse(
                               channel=channel,
                               amplitude=1,
                               length=marker_length),
                               name='Readout tone {}_{}'.format(channel, i),
                               start=marker_interval*(i+1)-marker_length,
                               refpulse='refpulse_0-0', refpoint='start')
    return marker_element


def generate_marker_element_with_RF_mod(i, marker_length, marker_interval, IF,
                                        mod_amp=.5):
    # make sure tau is a multiple of 1 ns, if it is not the fixed point will
    # not be able to be computed.
    marker_element = element.Element(name=('marker_element %s' % i),
                                     pulsar=station.pulsar)
    marker_element.add(pulse.SquarePulse(name='refpulse_0', channel='ch1',
                       amplitude=0, length=100e-9))
    number_of_pulses = int(200*1e-6/marker_interval)
    marker_element.add(pulse.CosPulse(name='cosI', channel='ch3',
                       amplitude=mod_amp, frequency=IF, length=200e-6))
    marker_element.add(pulse.CosPulse(name='sinQ', channel='ch4',
                       amplitude=mod_amp, frequency=IF,
                       length=200e-6, phase=90))
    for i in range(number_of_pulses):
        for channel in ['ch4_marker1', 'ch4_marker2', 'ch3_marker2', 'ch3_marker1']:
            marker_element.add(pulse.SquarePulse(
                           channel=channel,
                           amplitude=1,
                           length=marker_length),
                           name='Readout tone {}_{}'.format(channel, i),
                           start=marker_interval*(i+1)-marker_length, refpulse='refpulse_0-0', refpoint='start')
    return marker_element


def generate_and_upload_marker_sequence(marker_length, marker_interval,
                                        RF_mod=False, IF=None, mod_amp=None):
    seq_name = 'Heterodyne_marker_seq_RF_mod'
    seq = sequence.Sequence(seq_name)
    el_list = []
    for i in range(2):
        if RF_mod:
            el = generate_marker_element_with_RF_mod(i, marker_length,
                                                     marker_interval,
                                                     IF=IF, mod_amp=mod_amp)
        else:
            el = generate_marker_element(i, marker_length, marker_interval)
        el_list.append(el)
        seq.append_element(el, trigger_wait=False) # Ensures a continuously running sequence
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=False)
    return seq_name


def CBox_single_pulse_seq(IF, meas_pulse_delay=0, RO_trigger_delay=0,
                          verbose=False):
    seq_name = 'Single_pulse_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    for i in range(2):
        el = st_elts.single_pulse_elt(i, station, IF, meas_pulse_delay,
                                      RO_trigger_delay)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name

def CBox_two_pulse_seq(IF, meas_pulse_delay=0, RO_trigger_delay=0,
                       verbose=False):
    seq_name = 'Single_pulse_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    for i in range(2):
        el = st_elts.single_pulse_elt(i, station, IF, meas_pulse_delay,
                                      RO_trigger_delay)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name



def CBox_T1_marker_seq(IF, times, meas_pulse_delay,
                       RO_trigger_delay=0, verbose=False):
    '''
    Beware, replaces the last 4 points with calibration points
    '''
    seq_name = 'CBox_T1'
    seq = sequence.Sequence(seq_name)
    el_list = []

    for i, tau in enumerate(times[:-4]):
        el = st_elts.single_pulse_elt(i, station, IF, meas_pulse_delay,
                                      RO_trigger_delay, tau=tau)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    cal_0_elt0 = st_elts.no_pulse_elt(i+1, station, IF, RO_trigger_delay)
    cal_0_elt1 = st_elts.no_pulse_elt(i+2, station, IF, RO_trigger_delay)
    cal_1_elt0 = st_elts.single_pulse_elt(i+3, station, IF, meas_pulse_delay,
                                          RO_trigger_delay)
    cal_1_elt1 = st_elts.single_pulse_elt(i+4, station, IF, meas_pulse_delay,
                                          RO_trigger_delay)
    seq.append_element(cal_0_elt0, trigger_wait=True)
    seq.append_element(cal_0_elt1, trigger_wait=True)
    seq.append_element(cal_1_elt0, trigger_wait=True)
    seq.append_element(cal_1_elt1, trigger_wait=True)
    el_list.append(cal_0_elt0)
    el_list.append(cal_0_elt1)
    el_list.append(cal_1_elt0)
    el_list.append(cal_1_elt1)
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
