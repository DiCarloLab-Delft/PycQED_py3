import logging
import numpy as np
from ..waveform_control import pulsar
from ..waveform_control import element
from ..waveform_control import pulse
from ..waveform_control import pulse_library as pl


def single_pulse_elt(i, station, IF, meas_pulse_delay=0, RO_trigger_delay=0,
                     tau=0):
        '''
        Single pulse element for triggering the CBox.
        Plays a single marker (15ns) on ch1m1 and ch1m2 followed by
        cos that modulates the LO for Readout and a RO marker.

        The RO-tone is fixed in phase with respect to the RO-trigger
        The RO trigger is delayed by RO-trigger delay.
        '''
        el = element.Element(name='single-pulse-elt_%s' % i,
                             pulsar=station.pulsar)

        # exitst to ensure that channel is not high when waiting for trigger
        ref_elt = el.add(pulse.SquarePulse(name='refpulse_0', channel='ch1',
                         amplitude=0, length=1e-9))
        for i in range(3): # Exist to ensure there are no empty channels
            el.add(pulse.SquarePulse(name='refpulse_0',
                                     channel='ch{}'.format(i+1),
                                     amplitude=0, length=1e-9))

        # Pulse trigger
        sqp = pulse.SquarePulse(name='CBox-pulse-trigger',
                                channel='ch1_marker1',
                                amplitude=1, length=15e-9)
        el.add(sqp, name='CBox-pulse-trigger', start=20e-9, refpulse=ref_elt)
        el.add(pulse.cp(sqp, channel='ch1_marker2'),
               refpulse='CBox-pulse-trigger', refpoint='start', start=0)

        # Readout modulation tone
        cosP = pulse.CosPulse(name='cosI', channel='ch3',
                              amplitude=0.5, frequency=IF, length=2e-6)
        ROpulse = el.add(cosP, start=tau+meas_pulse_delay,
                         refpulse='CBox-pulse-trigger',
                         fixed_point_freq=10e6)
        el.add(pulse.CosPulse(name='sinQ', channel='ch4',
                              amplitude=0.5, frequency=IF,
                              length=2e-6, phase=90),
               start=0, refpulse=ROpulse, refpoint='start')

        # Start Acquisition marker
        ROm = pulse.SquarePulse(name='RO-marker',
                                amplitude=1, length=20e-9,
                                channel='ch4_marker1')
        ROm_name = el.add(ROm, start=RO_trigger_delay,
                          refpulse=ROpulse, refpoint='start')
        el.add(pulse.cp(ROm, channel='ch4_marker2'),
               refpulse=ROm_name, refpoint='start', start=0)
        return el


def no_pulse_elt(i, station, IF, RO_trigger_delay=0):
        '''
        Element that contains a RO modulating tone and a marker to
        trigger acquisiton.

        The RO-tone is fixed in phase with respect to the RO-trigger
        The RO trigger is delayed by RO-trigger delay.
        '''
        el = element.Element(name='no-pulse-elt_%s' % i,
                             pulsar=station.pulsar)

        # exitst to ensure that channel is not high when waiting for trigger
        ref_elt = el.add(pulse.SquarePulse(name='refpulse_0', channel='ch1',
                         amplitude=0, length=1e-9))
        for i in range(3): # Exist to ensure there are no empty channels
            el.add(pulse.SquarePulse(name='refpulse_{}'.format(i),
                                     channel='ch{}'.format(i+1),
                                     amplitude=0, length=1e-9))
        # Readout modulation tone
        cosP = pulse.CosPulse(name='cosI', channel='ch3',
                              amplitude=0.5, frequency=IF, length=2e-6)
        ROpulse = el.add(cosP, start=10e-6,
                         refpulse=ref_elt, fixed_point_freq=IF)
        el.add(pulse.CosPulse(name='sinQ', channel='ch4',
                              amplitude=0.5, frequency=IF,
                              length=2e-6, phase=90),
               start=0, refpulse=ROpulse, refpoint='start')

        # Start Acquisition marker
        ROm = pulse.SquarePulse(name='RO-marker',
                                amplitude=1, length=20e-9,
                                channel='ch4_marker1')
        ROm_name = el.add(ROm, start=RO_trigger_delay,
                          refpulse=ROpulse, refpoint='start')
        el.add(pulse.cp(ROm, channel='ch4_marker2'),
               refpulse=ROm_name, refpoint='start', start=0)
        return el


def two_pulse_elt(i, station, IF, meas_pulse_delay=0, RO_trigger_delay=0,
                  interpulse_delay=40e-9, tau=0):
        '''
        two pulse element for triggering the CBox.
        Plays two markers of (15ns) on ch1m1 and ch1m2 separated by
        the interpulse_delay and tau, followed by a cos that modulates the LO
        for Readout and a RO marker.

        The RO-tone is fixed in phase with respect to the RO-trigger
        The RO trigger is delayed by RO-trigger delay.
        '''
        if round(tau*1e10) % 50 != 0.0:  # round to 5 ns
            # actually anything not a multiple of 1/mod_freq can give errors
            # but that info is not available here
            logging.warning(
                'tau ({} ns) is not a multiple '.format(tau*1e9) +
                'of 5ns this can cause phase errors in CBox pulses')
        el = element.Element(name='two-pulse-elt_%s' % i,
                             pulsar=station.pulsar)

        # exitst to ensure that channel is not high when waiting for trigger
        ref_elt = el.add(pulse.SquarePulse(name='refpulse_0', channel='ch1',
                         amplitude=0, length=1e-9))
        for i in range(3):  # Exist to ensure there are no empty channels
            el.add(pulse.SquarePulse(name='refpulse_0',
                                     channel='ch{}'.format(i+1),
                                     amplitude=0, length=1e-9))

        # Pulse trigger
        sqp = pulse.SquarePulse(name='CBox-pulse-trigger',
                                channel='ch1_marker1',
                                amplitude=1, length=15e-9)
        el.add(sqp, name='CBox-pulse-trigger-1', start=20e-9, refpulse=ref_elt)
        el.add(pulse.cp(sqp, channel='ch1_marker2'),
               refpulse='CBox-pulse-trigger-1', refpoint='start', start=0)

        el.add(pulse.cp(sqp, channel='ch1_marker1'),
               name='CBox-pulse-trigger-2',
               refpulse='CBox-pulse-trigger-1', refpoint='start',
               start=interpulse_delay+tau)
        el.add(pulse.cp(sqp, channel='ch1_marker2'),
               refpulse='CBox-pulse-trigger-2', refpoint='start', start=0)

        # Readout modulation tone
        cosP = pulse.CosPulse(name='cosI', channel='ch3',
                              amplitude=0.5, frequency=IF, length=2e-6)
        ROpulse = el.add(cosP, start=meas_pulse_delay,
                         refpulse='CBox-pulse-trigger-2',
                         fixed_point_freq=10e6)

        el.add(pulse.CosPulse(name='sinQ', channel='ch4',
                              amplitude=0.5, frequency=IF,
                              length=2e-6, phase=90),
               start=0, refpulse=ROpulse, refpoint='start')

        # Start Acquisition marker
        ROm = pulse.SquarePulse(name='RO-marker',
                                amplitude=1, length=20e-9,
                                channel='ch4_marker1')
        ROm_name = el.add(ROm, start=RO_trigger_delay,
                          refpulse=ROpulse, refpoint='start')
        el.add(pulse.cp(ROm, channel='ch4_marker2'),
               refpulse=ROm_name, refpoint='start', start=0)
        return el


def multi_pulse_elt(i, station, IF, meas_pulse_delay=0, RO_trigger_delay=0,
                    interpulse_delay=40e-9,
                    n_pulses=3,
                    taus=None):
        '''


        '''
        logging.warning('FUnction not tested')
        if taus is None:
            taus = np.zeros(n_pulses)
        else:
            # prepend a 0 to the list of inter-pulse wait times tau to
            # make the length of the wait times add up
            taus = np.concatenate([[0], taus])
        # prepend a 0 to the list of taus
        for tau in taus:  #this statement can probably be reduced to 1 line
            if tau % 5e-9 != 0:
                logging.warning('tau is not a multiple of 5ns this can cause' +
                                'phase errors in CBox pulses')
        el = element.Element(name='%s-pulse-elt_%s' % (n_pulses, i),
                             pulsar=station.pulsar)

        # exitst to ensure that channel is not high when waiting for trigger
        ref_elt = el.add(pulse.SquarePulse(name='refpulse_0', channel='ch1',
                         amplitude=0, length=1e-9))
        for i in range(3):  # Exist to ensure there are no empty channels
            el.add(pulse.SquarePulse(name='refpulse_0',
                                     channel='ch{}'.format(i+1),
                                     amplitude=0, length=1e-9))
        marker_ref = ref_elt
        sqp = pulse.SquarePulse(name='CBox-pulse-trigger',
                                channel='ch1_marker1',
                                amplitude=1, length=15e-9)
        for j in range(n_pulses):
            # overwrite the reference with the latest added marker
            marker_ref = el.add(pulse.cp(sqp, channel='ch1_marker1'),
                                name='CBox-pulse-trigger-%s' % j,
                                start=interpulse_delay+taus[j],
                                refpulse=marker_ref)
            el.add(pulse.cp(sqp, channel='ch1_marker2'),
                   refpulse=marker_ref,
                   refpoint='start', start=0)

        # Readout modulation tone
        cosP = pulse.CosPulse(name='cosI', channel='ch3',
                              amplitude=0.5, frequency=IF, length=2e-6)
        ROpulse = el.add(cosP, start=meas_pulse_delay,
                         refpulse=marker_ref, fixed_point_freq=10e6)
        el.add(pulse.CosPulse(name='sinQ', channel='ch4',
                              amplitude=0.5, frequency=IF,
                              length=2e-6, phase=90),
               start=0, refpulse=ROpulse, refpoint='start')

        # Start Acquisition marker
        ROm = pulse.SquarePulse(name='RO-marker',
                                amplitude=1, length=20e-9,
                                channel='ch4_marker1')
        ROm_name = el.add(ROm, start=RO_trigger_delay,
                          refpulse=ROpulse, refpoint='start')
        el.add(pulse.cp(ROm, channel='ch4_marker2'),
               refpulse=ROm_name, refpoint='start', start=0)
        return el
