import logging
import numpy as np
from copy import deepcopy
from math import gcd
from ..waveform_control import pulsar
from ..waveform_control import element
from ..waveform_control import pulse
from ..waveform_control.pulse_library import MW_IQmod_pulse, SSB_DRAG_pulse
from ..waveform_control import sequence
from importlib import reload
reload(pulse)
from ..waveform_control import pulse_library
reload(pulse_library)

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
    pulses = get_pulse_dict_from_pars(pulse_pars)
    for i, amp in enumerate(amps):  # seq has to have at least 2 elts
        pulses['X']['amplitude'] = amp
        el = single_SSB_DRAG_pulse_elt(i, station,
                                       pulses['X'],
                                       RO_pars)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name


def T1_seq(times,
           pulse_pars, RO_pars,
           cal_points=True,
           verbose=False):
    '''
    Rabi sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    Input pars:
        times:       array of times to wait after the initial pi-pulse
        pulse_pars:  dict containing the pulse parameters
        RO_pars:     dict containing the RO parameters
    '''
    seq_name = 'T1_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    RO_pulse_delay = RO_pars['pulse_delay']
    RO_pars = deepcopy(RO_pars)  # Prevents overwriting of the dict
    pulses = get_pulse_dict_from_pars(pulse_pars)

    for i, tau in enumerate(times):  # seq has to have at least 2 elts
        RO_pars['pulse_delay'] = RO_pulse_delay + tau
        if cal_points:
            if (i == (len(times)-4) or i == (len(times)-3)):
                el = single_SSB_DRAG_pulse_elt(i, station,
                                               pulses['I'],
                                               RO_pars)
            elif(i == (len(times)-2) or i == (len(times)-1)):
                RO_pars['pulse_delay'] = RO_pulse_delay
                el = single_SSB_DRAG_pulse_elt(i, station,
                                               pulses['X'],
                                               RO_pars)
            else:
                el = single_SSB_DRAG_pulse_elt(i, station,
                                               pulses['X'],
                                               RO_pars)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name


def Ramsey_seq(times, pulse_pars, RO_pars,
               artificial_detuning=None,
               cal_points=True,
               verbose=False):
    '''
    Ramsey sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    Input pars:
        times:               array of times between (start of) pulses (s)
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters
        artificial_detuning: artificial_detuning (Hz) implemented using phase
        cal_points:          whether to use calibration points or not
    '''
    seq_name = 'Ramsey_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    # First extract values from input, later overwrite when generating waveforms
    pulses = get_pulse_dict_from_pars(pulse_pars)

    pulse_pars_x2 = deepcopy(pulses['x'])
    for i, tau in enumerate(times):
        pulse_pars_x2['pulse_separation'] = tau

        if artificial_detuning is not None:
            pulse_pars_x2['phase'] = tau * artificial_detuning * 360

        if cal_points and (i == (len(times)-4) or i == (len(times)-3)):
                el = single_SSB_DRAG_pulse_elt(i, station,
                                               pulses['I'],
                                               RO_pars)
        elif cal_points and (i == (len(times)-2) or i == (len(times)-1)):
                el = single_SSB_DRAG_pulse_elt(i, station,
                                               pulses['X'],
                                               RO_pars)
        else:
            el = double_SSB_DRAG_pulse_elt(i, station,
                                           pulses['x'],
                                           pulse_pars_x2,
                                           RO_pars)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name


def AllXY_seq(pulse_pars, RO_pars, double_points=False,
              verbose=False):
    '''
    AllXY sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    Input pars:
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters

    '''
    seq_name = 'Ramsey_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    # Create a dict with the parameters for all the pulses
    pulses = get_pulse_dict_from_pars(pulse_pars)

    pulse_combinations = ['II', 'XX', 'YY', 'XY', 'YX',
                          'xI', 'yI', 'xy', 'yx', 'xY', 'yX',
                          'Xy', 'Yx', 'xX', 'Xx', 'yY', 'Yy',
                          'XI', 'YI', 'xx', 'yy']
    if double_points:
        pulse_combinations = [val for val in pulse_combinations
                              for _ in (0, 1)]

    for i, pulse_comb in enumerate(pulse_combinations):
        el = double_SSB_DRAG_pulse_elt(i, station,
                                       pulses[pulse_comb[0]],
                                       pulses[pulse_comb[1]],
                                       RO_pars)
        el_list.append(el)
        seq.append_element(el, trigger_wait=True)
    station.instruments['AWG'].stop()
    station.pulsar.program_awg(seq, *el_list, verbose=verbose)
    return seq_name

def OffOn_seq(pulse_pars, RO_pars,
              verbose=False):
    '''
    OffOn sequence for a single qubit using the tektronix.
    SSB_Drag pulse is used for driving, simple modualtion used for RO
    Input pars:
        pulse_pars:          dict containing the pulse parameters
        RO_pars:             dict containing the RO parameters
    '''
    seq_name = 'OffOn_sequence'
    seq = sequence.Sequence(seq_name)
    el_list = []
    # Create a dict with the parameters for all the pulses
    pulses = get_pulse_dict_from_pars(pulse_pars)

    pulse_combinations = ['I', 'X']

    for i, pulse_comb in enumerate(pulse_combinations):
        el = single_SSB_DRAG_pulse_elt(i, station,
                                       pulses[pulse_comb],
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
        Single SSB_DRAG pulse, a RO-tone and a RO-marker.

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
        pulse1 = SSB_DRAG_pulse(name='pulse1',
                                I_channel=pulse_pars['I_channel'],
                                Q_channel=pulse_pars['Q_channel'],
                                amplitude=pulse_pars['amplitude'],
                                sigma=pulse_pars['sigma'],
                                nr_sigma=pulse_pars['nr_sigma'],
                                motzoi=pulse_pars['motzoi'],
                                mod_frequency=pulse_pars['mod_frequency'],
                                phase=pulse_pars['phase'])

        el.add(pulse1, name='pulse1', start=20e-9, refpulse=ref_elt)

        # Readout modulation tone
        # TODO: add option to use same sequence but spit out only marker
        # instead of a RO tone.
        RO_tone = el.add(MW_IQmod_pulse(name='RO_tone',
                                        I_channel=RO_pars['I_channel'],
                                        Q_channel=RO_pars['Q_channel'],
                                        length=RO_pars['length'],
                                        amplitude=RO_pars['amplitude'],
                                        mod_frequency=RO_pars['mod_frequency']),
                         start=RO_pars['pulse_delay'],
                         refpulse='pulse1',
                         fixed_point_freq=gcd(int(RO_pars['mod_frequency']),
                                              int(20e6)))
        # Hardcoded gcd to ensure fixed point always a multiple of 5ns for CBox

        # Start Acquisition marker
        ROm = pulse.SquarePulse(name='RO-marker',
                                amplitude=1, length=20e-9,
                                channel=RO_pars['marker_ch1'])
        ROm_name = el.add(ROm, start=RO_pars['trigger_delay'],
                          refpulse=RO_tone, refpoint='start')
        el.add(pulse.cp(ROm, channel=RO_pars['marker_ch2']),
               refpulse=ROm_name, refpoint='start', start=0)
        return el


def double_SSB_DRAG_pulse_elt(i, station,
                              pulse_pars_1,
                              pulse_pars_2,
                              RO_pars):
        '''
        Two SSB_DRAG pulses, a RO-tone and a RO-marker.

        The RO-tone is fixed in phase with respect to the RO-trigger
        The RO trigger is delayed by RO-trigger delay.

        Input args
            station:    qcodes station object, contains AWG etc
            pulse_pars: dictionary containing parameters for the qubit pulse
            RO_pars:    dictionary containing the parameters for the RO tone
                        and markers
        '''
        el = element.Element(name='double-pulse-elt_%s' % i,
                             pulsar=station.pulsar)

        # exitst to ensure that channel is not high when waiting for trigger
        ref_elt = el.add(pulse.SquarePulse(name='refpulse_0', channel='ch1',
                         amplitude=0, length=1e-9))
        for i in range(3):  # Exist to ensure there are no empty channels
            el.add(pulse.SquarePulse(name='refpulse_0',
                                     channel='ch{}'.format(i+1),
                                     amplitude=0, length=1e-9))

        # Pulse trigger
        pulse_1 = SSB_DRAG_pulse(name='pulse_1',
                                 I_channel=pulse_pars_1['I_channel'],
                                 Q_channel=pulse_pars_1['Q_channel'],
                                 amplitude=pulse_pars_1['amplitude'],
                                 sigma=pulse_pars_1['sigma'],
                                 nr_sigma=pulse_pars_1['nr_sigma'],
                                 motzoi=pulse_pars_1['motzoi'],
                                 mod_frequency=pulse_pars_1['mod_frequency'],
                                 phase=pulse_pars_1['phase'])

        el.add(pulse_1, name='pulse_1', start=20e-9, refpulse=ref_elt)

        pulse_2 = SSB_DRAG_pulse(name='pulse_2',
                                 I_channel=pulse_pars_2['I_channel'],
                                 Q_channel=pulse_pars_2['Q_channel'],
                                 amplitude=pulse_pars_2['amplitude'],
                                 sigma=pulse_pars_2['sigma'],
                                 nr_sigma=pulse_pars_2['nr_sigma'],
                                 motzoi=pulse_pars_2['motzoi'],
                                 mod_frequency=pulse_pars_2['mod_frequency'],
                                 phase=pulse_pars_2['phase'])

        el.add(pulse_2, name='pulse_2', start=pulse_pars_2['pulse_separation'],
               refpulse='pulse_1')

        # Readout modulation tone
        # TODO: add option to use same sequence but spit out only marker
        # instead of a RO tone.
        RO_tone = el.add(MW_IQmod_pulse(name='RO_tone',
                                        I_channel=RO_pars['I_channel'],
                                        Q_channel=RO_pars['Q_channel'],
                                        length=RO_pars['length'],
                                        amplitude=RO_pars['amplitude'],
                                        mod_frequency=RO_pars['mod_frequency']),
                         start=RO_pars['pulse_delay'],
                         refpulse='pulse_2',
                         fixed_point_freq=gcd(int(RO_pars['mod_frequency']),
                                              int(20e6)))
        # Hardcoded gcd to ensure fixed point always a multiple of 5ns for CBox

        # Start Acquisition marker
        ROm = pulse.SquarePulse(name='RO-marker',
                                amplitude=1, length=20e-9,
                                channel=RO_pars['marker_ch1'])
        ROm_name = el.add(ROm, start=RO_pars['trigger_delay'],
                          refpulse=RO_tone, refpoint='start')
        el.add(pulse.cp(ROm, channel=RO_pars['marker_ch2']),
               refpulse=ROm_name, refpoint='start', start=0)
        return el

# Helper functions


def get_pulse_dict_from_pars(pulse_pars):
    '''
    Returns a dictionary containing pulse_pars for all the primitive pulses
    based on a single set of pulse_pars.
    Using this function deepcopies the pulse parameters preventing accidently
    editing to input dictionary.

    input args:
        pulse_pars: dictionary containing pulse_parameters
    return:
        pulses: dictionary of pulse_pars dictionaries
    '''
    pi2_amp = pulse_pars['amplitude']/2

    pulses = {'I': deepcopy(pulse_pars),
              'X': deepcopy(pulse_pars),
              'x': deepcopy(pulse_pars),
              'Y': deepcopy(pulse_pars),
              'y': deepcopy(pulse_pars)}

    pulses['I']['amplitude'] = 0
    pulses['x']['amplitude'] = pi2_amp
    pulses['Y']['phase'] = 90
    pulses['y']['amplitude'] = pi2_amp
    pulses['y']['phase'] = 90

    return pulses
