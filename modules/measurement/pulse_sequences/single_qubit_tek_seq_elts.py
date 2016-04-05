import logging
import numpy as np
from ..waveform_control import pulsar
from ..waveform_control import element
from ..waveform_control import pulse
from ..waveform_control.pulse_library import MW_IQmod_pulse, SSB_DRAG_pulse
from ..waveform_control import sequence
from importlib import reload
reload(pulse)


station = None
reload(element)
# You need to explicitly set this before running any functions from this module
# I guess there are cleaner solutions :)


def Rabi_seq(amps,
             pulse_pars, RO_pars,
             verbose=False):
    '''
    Rabi sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    Input pars:
        amps:        array of pulse amplitudes (V)
        pulse_pars:  dict containing the pulse parameters
        RO_pars:     dict containing the RO parameters
    '''
    seq_name = 'Rabi_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []

    for amp, i in enumerate(amps):  # seq has to have at least 2 elts
        pulse_pars['amplitude'] = amp
        el = single_SSB_DRAG_pulse_elt(i, station,
                                       pulse_pars,
                                       RO_pars)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name


def single_SSB_DRAG_pulse_elt(i, station,
                              pulse_pars,
                              RO_pars):
        '''
        Single pulse on channels 1 and 2, followed by a modulation of the LO
        on Ch3 and 4.
        Additonally spits out a marker for the RO
        The RO-tone is fixed in phase with respect to the RO-trigger
        The RO trigger is delayed by RO-trigger delay.

        Input args
            station:    qcodes station object, contains AWG etc
            pulse_pars: dictionary containing parameters for the qubit pulse
            RO_pars:    dictionary containing the parameters for the RO tone
                        and markers
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
        rabi_pulse = SSB_DRAG_pulse(name='Rabi_pulse',
                                    I_channel=pulse_pars['I_channel'],
                                    Q_channel=pulse_pars['Q_channel'],
                                    amplitude=pulse_pars['amplitude'],
                                    sigma=pulse_pars['sigma'],
                                    nr_sigma=pulse_pars['nr_sigma'],
                                    motzoi=pulse_pars['motzoi'],
                                    mod_frequency=pulse_pars['mod_frequency'],
                                    phase=0)

        el.add(rabi_pulse, name='Rabi_pulse', start=20e-9, refpulse=ref_elt)

        # Readout modulation tone
        # TODO: add option to use same sequence but spit out only marker
        # instead of a RO tone.
        RO_tone = MW_IQmod_pulse(name='RO_tone',
                                 I_channel=RO_pars['I_channel'],
                                 Q_channel=RO_pars['Q_channel'],
                                 length=RO_pars['length'],
                                 amplitude=RO_pars['amplitude'],
                                 mod_frequency=RO_pars['mod_frequency'])

        el.add(RO_tone, start=RO_pars['pulse_delay'],
               refpulse='Rabi_pulse',
               fixed_point_freq=RO_pars['mod_frequency'])

        # Start Acquisition marker
        ROm = pulse.SquarePulse(name='RO-marker',
                                amplitude=1, length=20e-9,
                                channel=RO_pars['marker_ch1'])
        ROm_name = el.add(ROm, start=RO_pars['trigger_delay'],
                          refpulse=RO_tone, refpoint='start')
        el.add(pulse.cp(ROm, channel=RO_pars['marker_ch2']),
               refpulse=ROm_name, refpoint='start', start=0)

        return el