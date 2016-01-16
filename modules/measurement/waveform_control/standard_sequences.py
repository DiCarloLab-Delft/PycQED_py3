from . import pulsar
from . import pulse
from . import element
from . import sequence
from .viewer import show_element, show_wf
from . import pulse_library as pl

station = None
# You need to explicitly set this before running any functions from this module
# I guess there are cleaner solutions :)


def generate_marker_element(i, marker_length, marker_interval):
    # make sure tau is a multiple of 1 ns, if it is not the fixed point will
    # not be able to be computed.
    marker_element = element.Element(name=('marker_element %s' % i), pulsar=station.pulsar)
    marker_element.add(pulse.SquarePulse(name='refpulse_0', channel='ch1', amplitude=0, length=100e-9))
    number_of_pulses = int(200*1e-6/marker_interval)
    for i in range(number_of_pulses):
        for channel in ['ch2_marker1', 'ch2_marker2', 'ch1_marker2', 'ch1_marker1']:
            marker_element.add(pulse.SquarePulse(
                           channel=channel,
                           amplitude=1,
                           length=marker_length),
                           name='Readout tone {}_{}'.format(channel, i),
                           start=marker_interval*(i+1)-marker_length, refpulse='refpulse_0-0', refpoint='start')
    return marker_element


def generate_marker_element_with_RF_mod(i, marker_length, marker_interval, IF):
    # make sure tau is a multiple of 1 ns, if it is not the fixed point will
    # not be able to be computed.

    marker_element = element.Element(name=('marker_element %s' % i),
                                     pulsar=station.pulsar)
    marker_element.add(pulse.SquarePulse(name='refpulse_0', channel='ch1',
                       amplitude=0, length=100e-9))
    number_of_pulses = int(200*1e-6/marker_interval)
    marker_element.add(pulse.CosPulse(name='cosI', channel='ch3',
                       amplitude=.5, frequency=IF, length=200e-6))
    marker_element.add(pulse.CosPulse(name='sinQ', channel='ch4',
                       amplitude=.5, frequency=IF, length=200e-6, phase=90))
    for i in range(number_of_pulses):
        for channel in ['ch2_marker1', 'ch2_marker2', 'ch1_marker2', 'ch1_marker1']:
            marker_element.add(pulse.SquarePulse(
                           channel=channel,
                           amplitude=1,
                           length=marker_length),
                           name='Readout tone {}_{}'.format(channel, i),
                           start=marker_interval*(i+1)-marker_length, refpulse='refpulse_0-0', refpoint='start')
    return marker_element


def generate_and_upload_marker_sequence(marker_length, marker_interval,
                                        RF_mod=False, IF=None):
    seq = sequence.Sequence('Heterodyne marker sequence with modulation')
    el_list = []
    for i in range(2):
        if RF_mod:
            el = generate_marker_element_with_RF_mod(i, marker_length,
                                                     marker_interval, IF=IF)
        else:
            el = generate_marker_element(i, marker_length, marker_interval)
        el_list.append(el)
        seq.append_element(el, trigger_wait=False) # Ensures a continuously running sequence
    station.instruments['AWG'].stop()
    awg_file = station.pulsar.program_awg(seq, *el_list, verbose=False)
    return
