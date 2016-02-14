import logging
import numpy as np
from ..waveform_control import pulsar
from ..waveform_control import element
from ..waveform_control import pulse
from ..waveform_control import pulse_library as pl
from importlib import reload
reload(pulse)


def single_pulse_elt(i, station, IF, RO_pulse_delay=0, RO_trigger_delay=0,
                     RO_pulse_length=1e-6, tau=0):
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
        for i in range(3):  # Exist to ensure there are no empty channels
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
                              amplitude=0.5, frequency=IF, length=RO_pulse_length)
        ROpulse = el.add(cosP, start=tau+RO_pulse_delay,
                         refpulse='CBox-pulse-trigger',
                         fixed_point_freq=10e6)
        el.add(pulse.CosPulse(name='sinQ', channel='ch4',
                              amplitude=0.5, frequency=IF,
                              length=RO_pulse_length, phase=90),
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


def no_pulse_elt(i, station, IF, RO_trigger_delay=0, RO_pulse_length=1e-6):
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
                              amplitude=0.5, frequency=IF, length=RO_pulse_length)
        ROpulse = el.add(cosP, start=10e-6,
                         refpulse=ref_elt, fixed_point_freq=IF)
        el.add(pulse.CosPulse(name='sinQ', channel='ch4',
                              amplitude=0.5, frequency=IF,
                              length=RO_pulse_length, phase=90),
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


def two_pulse_elt(i, station, IF, RO_pulse_delay, RO_trigger_delay,
                  RO_pulse_length,
                  pulse_separation, tau=0):
        '''
        two pulse element for triggering the CBox.
        Plays two markers of (15ns) on ch1m1 and ch1m2 separated by
        the pulse_separation and tau, followed by a cos that modulates the LO
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
               start=pulse_separation+tau)
        el.add(pulse.cp(sqp, channel='ch1_marker2'),
               refpulse='CBox-pulse-trigger-2', refpoint='start', start=0)

        # Readout modulation tone
        cosP = pulse.CosPulse(name='cosI', channel='ch3',
                              amplitude=0.5, frequency=IF, length=RO_pulse_length)
        ROpulse = el.add(cosP, start=RO_pulse_delay,
                         refpulse='CBox-pulse-trigger-2',
                         fixed_point_freq=10e6)

        el.add(pulse.CosPulse(name='sinQ', channel='ch4',
                              amplitude=0.5, frequency=IF,
                              length=RO_pulse_length, phase=90),
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


def multi_pulse_elt(i, station, IF, RO_pulse_delay=0,
                    RO_pulse_length=1e-6, RO_trigger_delay=0,
                    pulse_separation=40e-9,
                    n_pulses=3,
                    taus=None):

        if taus is None:
            taus = np.zeros(n_pulses)
        else:
            # prepend a 0 to the list of inter-pulse wait times tau to
            # make the length of the wait times add up
            taus = np.concatenate([[0], taus])
        # prepend a 0 to the list of taus
        for tau in taus:  # this statement can probably be reduced to 1 line
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
        sqp = pulse.SquarePulse(name='CBox-pulse-trigger',
                                channel='ch1_marker1',
                                amplitude=1, length=15e-9)

        for j in range(n_pulses):
            el.add(pulse.cp(sqp, channel='ch1_marker1'),
                   name='CBox-pulse-trigger-ch1_{}.{}'.format(i, j),
                   start=j*pulse_separation+taus[j],
                   refpulse=ref_elt, refpoint='end')
            el.add(pulse.cp(sqp, channel='ch1_marker2'),
                   refpulse='CBox-pulse-trigger-ch1_{}.{}'.format(i, j),
                   name='CBox-pulse-trigger-ch2_{}.{}'.format(i, j),
                   refpoint='start', start=0)

        # Readout modulation tone
        cosP = pulse.CosPulse(name='cosI', channel='ch3',
                              amplitude=0.5, frequency=IF,
                              length=RO_pulse_length)
        ROpulse = el.add(cosP, start=RO_pulse_delay,
                         refpulse='CBox-pulse-trigger-ch1_{}.{}'.format(i, j),
                         fixed_point_freq=10e6)
        el.add(pulse.CosPulse(name='sinQ', channel='ch4',
                              amplitude=0.5, frequency=IF,
                              length=RO_pulse_length, phase=90),
               start=0, refpulse=ROpulse, refpoint='start')

        # Start Acquisition marker
        ROm = pulse.SquarePulse(name='RO-marker',
                                amplitude=1, length=20e-9,
                                channel='ch4_marker1')
        ROm_name = el.add(ROm, start=RO_trigger_delay,
                          refpulse=ROpulse,
                          refpoint='start')
        el.add(pulse.cp(ROm, channel='ch4_marker2'),
               refpulse=ROm_name, refpoint='start', start=0)
        return el


def CBox_resetless_multi_pulse_elt(
        i, station, IF,
        RO_pulse_length=1e-6, RO_trigger_delay=0,
        RO_pulse_delay=100e-9,
        pulse_separation=60e-9,
        resetless_interval=10e-6,
        n_pulses=3,
        mod_amp=.5):
    el = element.Element(name=('el_{}'.format(i)),
                         pulsar=station.pulsar)

    # Thispulse ensures that the total length of the element is exactly 200us
    el.add(pulse.SquarePulse(name='refpulse_200us',
                             channel='ch2',
                             amplitude=0, length=200e-6,
                             start=0))
    # This pulse is used as a reference
    refpulse = el.add(pulse.SquarePulse(name='refpulse_0', channel='ch1',
                      amplitude=0, length=100e-9,
                      start=10e-9))
    # a marker pulse
    sqp = pulse.SquarePulse(name='CBox-pulse-trigger',
                            channel='ch1_marker1',
                            amplitude=1, length=15e-9)
    CosP = pulse.CosPulse(name='cosI', channel='ch3',
                          amplitude=mod_amp, frequency=IF,
                          length=RO_pulse_length)
    SinP = pulse.CosPulse(name='sinQ', channel='ch4',
                          amplitude=mod_amp, frequency=IF,
                          length=RO_pulse_length, phase=90)

    # I multiply the interval by 1e6 instead of dividing 200 to prevent
    # rounding errors
    number_of_resetless_sequences = int(200/(resetless_interval*1e6))

    if number_of_resetless_sequences < 1:
        logging.warning('Number of resetless seqs <1 ')
    if ((n_pulses * pulse_separation + RO_pulse_length+RO_pulse_delay) >
            resetless_interval):
        logging.warning('Sequence does not fit in the resetless interval')
    if number_of_resetless_sequences > 200:
        logging.warning('More than 200 iterations, probably some typo')

    marker_tr_p = pulse.marker_train(name='CBox-pulse_marker',
                                     channel='ch1_marker1', amplitude=1,
                                     marker_length=15e-9,
                                     marker_separation=pulse_separation,
                                     nr_markers=n_pulses)

    for i in range(number_of_resetless_sequences):
        el.add(pulse.cp(marker_tr_p, channel='ch1_marker1'),
               name='CBox-pulse-trigger-ch1_{}'.format(i),
               start=i*resetless_interval,
               refpulse=refpulse, refpoint='end')
        el.add(pulse.cp(marker_tr_p, channel='ch1_marker2'),
               name='CBox-pulse-trigger-ch2_{}'.format(i),
               refpulse='CBox-pulse-trigger-ch1_{}'.format(i),
               refpoint='start', start=0)

        # for j in range(n_pulses):
        #     # overwrite the reference with the latest added marker
        #     el.add(pulse.cp(sqp, channel='ch1_marker2'),
        #            refpulse='CBox-pulse-trigger-ch1_{}.{}'.format(i, j),
        #            name='CBox-pulse-trigger-ch2_{}.{}'.format(i, j),
        #            refpoint='start', start=0)
        # RO modulation tone
        el.add(pulse.cp(CosP), name='RO-Cos-{}'.format(i),
               start=RO_pulse_delay, refpoint='end',
               refpulse='CBox-pulse-trigger-ch1_{}'.format(i))
        el.add(pulse.cp(SinP), name='RO-Sin-{}'.format(i),
               start=0, refpoint='start', refpulse='RO-Cos-{}'.format(i))
        for k in range(2):
            # RO acquisition marker
            el.add(pulse.cp(sqp, channel='ch4_marker{}'.format(k+1)),
                   name='RO-marker-{}{}'.format(i, k),
                   start=RO_trigger_delay,
                   refpulse='RO-Cos-{}'.format(i), refpoint='start')
    return el


def pulsed_spec_elt_with_RF_mod(i, station, IF,
                                spec_pulse_length=1e-6,
                                RO_pulse_length=1e-6,
                                RO_pulse_delay=100e-9,
                                RO_trigger_delay=0,
                                marker_interval=4e-6,
                                mod_amp=0.5):
    el = element.Element(name=('el %s' % i),
                         pulsar=station.pulsar)

    # Thispulse ensures that the total length of the element is exactly 200us
    ref_length_pulse = el.add(pulse.SquarePulse(name='refpulse_0',
                              channel='ch2',
                              amplitude=0, length=200e-6,
                              start=0))
    # This pulse is used as a reference
    refpulse = el.add(pulse.SquarePulse(name='refpulse_0', channel='ch1',
                      amplitude=0, length=100e-9,
                      start=10e-9))


    # a marker pulse
    sqp = pulse.SquarePulse(name='CBox-pulse-trigger',
                            channel='ch1_marker1',
                            amplitude=1, length=15e-9)
    CosP = pulse.CosPulse(name='cosI', channel='ch3',
                          amplitude=mod_amp, frequency=IF,
                          length=RO_pulse_length)
    SinP = pulse.CosPulse(name='sinQ', channel='ch4',
                          amplitude=mod_amp, frequency=IF,
                          length=RO_pulse_length, phase=90)

    number_of_pulses = int(200*1e-6/marker_interval)

    for i in range(number_of_pulses):
        for j in range(2):
            # spec pulse marker
            el.add(pulse.cp(sqp, channel='ch2_marker{}'.format(j+1),
                            length=spec_pulse_length),
                   name='Spec-marker-{}{}'.format(i, j),
                   start=i*marker_interval,
                   refpulse=refpulse, refpoint='start')
        # RO modulation tone
        el.add(pulse.cp(CosP), name='RO-Cos-{}'.format(i),
               start=RO_pulse_delay, refpoint='end',
               refpulse='Spec-marker-{}{}'.format(i, j))
        el.add(pulse.cp(SinP), name='RO-Sin-{}'.format(i),
               start=0, refpoint='start', refpulse='RO-Cos-{}'.format(i))
        for j in range(2):
            # RO acquisition marker
            el.add(pulse.cp(sqp, channel='ch4_marker{}'.format(j+1)),
                   name='RO-marker-{}{}'.format(i, j),
                   start=RO_trigger_delay,
                   refpulse='RO-Cos-{}'.format(i), refpoint='start')
    return el


def CBox_marker_sequence(i, station, marker_separation):
    el = element.Element(name=('el_{}'.format(i)),
                         pulsar=station.pulsar)

    # Somehow I got errors whenever I picked a marker separation>63 ns
    # 14-2-2016 MAR -> Update quite sure it is a ns rounding error!

    # Thispulse ensures that the total length of the element is exactly 20us
    refpulse = el.add(pulse.SquarePulse(name='refpulse_20us',
                                        channel='ch2',
                                        amplitude=0, length=20e-6,
                                        start=0))
    # ensures complete seq is filled with markers
    nr_markers = int(20e-6//marker_separation)
    marker_tr_p_a = pulse.marker_train(name='CBox-pulse_marker',
                                       channel='ch1_marker1', amplitude=1,
                                       marker_length=15e-9,
                                       marker_separation=marker_separation,
                                       nr_markers=nr_markers)
    marker_tr_p_b = pulse.marker_train(name='CBox-pulse_marker',
                                       channel='ch1_marker2', amplitude=1,
                                       marker_length=15e-9,
                                       marker_separation=marker_separation,
                                       nr_markers=nr_markers)
    el.add(pulse.cp(marker_tr_p_a),
           name='CBox-pulse-trigger-ch1_{}'.format(i),
           start=0,
           refpulse=refpulse, refpoint='start')

    el.add(pulse.cp(marker_tr_p_b),
           name='CBox-pulse-trigger-ch2_{}'.format(i),
           start=0,
           refpulse=refpulse, refpoint='start')
    return el
