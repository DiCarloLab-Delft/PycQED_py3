from . import pulsar
from . import pulse
from . import element
from . import sequence
from .viewer import show_element, show_wf
from . import pulse_library as pl

station = None
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


def gen_and_upl_CBox_single_pulse_seq(IF, mod_amp,
                                      meas_pulse_delay=0, CBox_RO_trigger_delay=0,
                                      verbose=False):
    '''
    '''
    seq_name = 'Single_pulse_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    def generate_element(i):
        el = element.Element(name='single-pulse-el_%s' % i,
                             pulsar=station.pulsar)

        dummy_el = el.add(pulse.SquarePulse(name='refpulse_0', channel='ch1',
                           amplitude=0, length=100e-9))

        sqp = pulse.SquarePulse(name='CBox-pulse-trigger',
                                channel='ch1_marker1',
                                amplitude=1, length=5e-9)
        el.add(sqp, name='CBox-pulse-trigger', start=10e-9, refpulse=dummy_el)
        el.add(pulse.cp(sqp, channel='ch1_marker2'),
               refpulse='CBox-pulse-trigger', refpoint='start', start=0)
        # The Readout tone
        cosP = pulse.CosPulse(name='cosI', channel='ch3',
                              amplitude=mod_amp, frequency=IF, length=2e-6)
        ROpulse = el.add(cosP, start=meas_pulse_delay,
                         refpulse='CBox-pulse-trigger')#, fixed_point_freq=IF)
        el.add(pulse.CosPulse(name='sinQ', channel='ch4',
                              amplitude=mod_amp, frequency=IF,
                              length=2e-6, phase=90),
                              start=0, refpulse=ROpulse, refpoint='start')

        ROm = pulse.SquarePulse(name='RO-marker',
                                amplitude=1, length=100e-9,
                                channel='ch4_marker1')
        ROm_name = el.add(ROm, start=CBox_RO_trigger_delay,
                          refpulse=ROpulse, refpoint='start')
        el.add(pulse.cp(ROm, channel='ch4_marker2'),
               refpulse=ROm_name, refpoint='start', start=0)
        return el
    for i in range(2):
        el = generate_element(i)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name
