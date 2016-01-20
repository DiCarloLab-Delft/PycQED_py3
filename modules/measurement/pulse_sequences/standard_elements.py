from ..waveform_control import pulsar
from ..waveform_control import pulse
from ..waveform_control import element
from ..waveform_control import sequence
from ..waveform_control.viewer import show_element, show_wf
from ..waveform_control import pulse_library as pl


def single_pulse_elt(i, station, IF, meas_pulse_delay=0, RO_trigger_delay=0,
                     tau=0):
        el = element.Element(name='single-pulse-el_%s' % i,
                             pulsar=station.pulsar)

        # exitst to ensure that channel is not high when waiting for trigger
        dummy_el = el.add(pulse.SquarePulse(name='refpulse_0', channel='ch1',
                          amplitude=0, length=100e-9))

        sqp = pulse.SquarePulse(name='CBox-pulse-trigger',
                                channel='ch1_marker1',
                                amplitude=1, length=15e-9)
        el.add(sqp, name='CBox-pulse-trigger', start=10e-9, refpulse=dummy_el)
        el.add(pulse.cp(sqp, channel='ch1_marker2'),
               refpulse='CBox-pulse-trigger', refpoint='start', start=0)
        # The Readout tone
        cosP = pulse.CosPulse(name='cosI', channel='ch3',
                              amplitude=0.5, frequency=IF, length=2e-6)
        ROpulse = el.add(cosP, start=tau+meas_pulse_delay,
                         refpulse='CBox-pulse-trigger')#, fixed_point_freq=IF)
        el.add(pulse.CosPulse(name='sinQ', channel='ch4',
                              amplitude=0.5, frequency=IF,
                              length=2e-6, phase=90),
                              start=0, refpulse=ROpulse, refpoint='start')

        ROm = pulse.SquarePulse(name='RO-marker',
                                amplitude=1, length=100e-9,
                                channel='ch4_marker1')
        ROm_name = el.add(ROm, start=RO_trigger_delay,
                          refpulse=ROpulse, refpoint='start')
        el.add(pulse.cp(ROm, channel='ch4_marker2'),
               refpulse=ROm_name, refpoint='start', start=0)
        return el