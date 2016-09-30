import logging
import numpy as np
from copy import deepcopy
try:
    from math import gcd
except:  # Moved to math in python 3.5, this is to be 3.4 compatible
    from fractions import gcd
from ..waveform_control import pulsar
from ..waveform_control import element
from ..waveform_control import pulse
from ..waveform_control.pulse_library import MW_IQmod_pulse, SSB_DRAG_pulse, \
    Mux_DRAG_pulse
from ..waveform_control.pulse import CosPulse, SquarePulse
from pycqed.measurement.randomized_benchmarking import randomized_benchmarking as rb

from importlib import reload
reload(pulse)
from ..waveform_control import pulse_library
reload(pulse_library)

reload(element)


def multi_pulse_elt(i, station, pulse_list):
        '''
        Input args
            i:          index of the element, ensures unique element name
            station:    qcodes station object, contains AWG etc
            pulse_list: list of pulse_dicts containing pulse parameters
        Returns:
            element:    for use with the pulsar sequencer

        Currently works with two types of pulses, 'SSB_DRAG_pulse' and
        'MW_IQmod_pulse' from pulselib. The idea is to make this function
        work with arbitrary pulses that have a function in the pulselib.
        The other idea is to have one 'pulse' in pulselib per 'pulse' that
        also include markers in the RO this is still added by hand in the if
        statement.

        If you want to add extra pulses to this function please talk to me
        (Adriaan) as I would like to keep it clean and prevent a  big if, elif
        loop. (I have some ideas on how to implement this).

        Note: this function could be the template for the most standard
        element we use.
        '''
        el = element.Element(
            name='{}-pulse-elt_{}'.format(len(pulse_list), i),
            pulsar=station.pulsar)
        # exitst to ensure that channel is not high when waiting for trigger
        last_pulse = el.add(pulse.SquarePulse(name='refpulse_0', channel='ch1',
                                              amplitude=0, length=1e-9))
        for i in range(3):  # Exist to ensure there are no empty channels
            el.add(pulse.SquarePulse(name='refpulse_0',
                                     channel='ch{}'.format(i+1),
                                     amplitude=0, length=1e-9))

        for i, pulse_pars in enumerate(pulse_list):
            if pulse_pars['pulse_type'] == 'SSB_DRAG_pulse':
                last_pulse = el.add(
                    SSB_DRAG_pulse(name='pulse_{}'.format(i),
                                   I_channel=pulse_pars['I_channel'],
                                   Q_channel=pulse_pars['Q_channel'],
                                   amplitude=pulse_pars['amplitude'],
                                   sigma=pulse_pars['sigma'],
                                   nr_sigma=pulse_pars['nr_sigma'],
                                   motzoi=pulse_pars['motzoi'],
                                   mod_frequency=pulse_pars['mod_frequency'],
                                   phase=pulse_pars['phase'],
                                   phi_skew=pulse_pars['phi_skew'],
                                   alpha=pulse_pars['alpha']),
                    start=pulse_pars['pulse_delay'],
                    refpulse=last_pulse, refpoint='start')
            elif pulse_pars['pulse_type'] == 'Mux_DRAG_pulse':
                # pulse_pars.pop('pulse_type')
                last_pulse = el.add(Mux_DRAG_pulse(name='pulse_{}'.format(i),
                                                   **pulse_pars),
                                    start=pulse_pars['pulse_delay'],
                                    refpulse=last_pulse, refpoint='start')
            elif pulse_pars['pulse_type'] == 'CosPulse':
                last_pulse = el.add(CosPulse(name='pulse_{}'.format(i),
                                             **pulse_pars),
                                    start=pulse_pars['pulse_delay'],
                                    refpulse=last_pulse, refpoint='start')
            elif pulse_pars['pulse_type'] == 'SquarePulse':
                last_pulse = el.add(SquarePulse(name='pulse_{}'.format(i),
                                                **pulse_pars),
                                    start=pulse_pars['pulse_delay'],
                                    refpulse=last_pulse, refpoint='start')

            elif pulse_pars['pulse_type'] == 'ModSquare':
                last_pulse = el.add(MW_IQmod_pulse(name='pulse_{}'.format(i),
                                                   **pulse_pars),
                                    start=pulse_pars['pulse_delay'],
                                    refpulse=last_pulse, refpoint='start')

            elif (pulse_pars['pulse_type'] == 'MW_IQmod_pulse' or
                  pulse_pars['pulse_type'] == 'Gated_MW_RO_pulse'):
                # Does more than just call the function as it also adds the
                # markers. Ideally we combine both in one function in pulselib
                if pulse_pars['pulse_type'] == 'MW_IQmod_pulse':
                    last_pulse = el.add(MW_IQmod_pulse(
                            name='RO_tone',
                            I_channel=pulse_pars['I_channel'],
                            Q_channel=pulse_pars['Q_channel'],
                            length=pulse_pars['length'],
                            amplitude=pulse_pars['amplitude'],
                            mod_frequency=pulse_pars['mod_frequency']),
                        start=pulse_pars['pulse_delay'],
                        refpulse=last_pulse, refpoint='start',
                        fixed_point_freq=pulse_pars['fixed_point_frequency'])
                else:
                    last_pulse=el.add(pulse.SquarePulse(
                            name='RO_marker', amplitude=1,
                            length=pulse_pars['length'],
                            channel=pulse_pars['RO_pulse_marker_channel']),
                        start=pulse_pars['pulse_delay'], refpulse=last_pulse,
                        refpoint='start',
                        fixed_point_freq=pulse_pars['fixed_point_frequency'])
                # Start Acquisition marker
                if type(pulse_pars['acq_marker_channel']) is str:
                    Acq_marker = pulse.SquarePulse(
                        name='Acq-trigger', amplitude=1, length=20e-9,
                        channel=pulse_pars['acq_marker_channel'])
                    el.add(
                        Acq_marker, start=pulse_pars['acq_marker_delay'],
                        refpulse=last_pulse, refpoint='start')
                else:
                    # TODO: remove hacked in second marker and support list
                    # functionality
                    # Want to implement compatibiilty with a list of marker
                    # channels here to allow copies of the pulse
                    raise TypeError()

            else:
                raise KeyError('pulse_type {} not recognized'.format(
                    pulse_pars['pulse_type']))

        # This pulse ensures that the sequence always ends at zero amp
        last_pulse = el.add(pulse.SquarePulse(name='final_empty_pulse',
                                              channel='ch1',
                                              amplitude=0, length=1e-9),
                            refpulse=last_pulse, refpoint='end')

        return el
