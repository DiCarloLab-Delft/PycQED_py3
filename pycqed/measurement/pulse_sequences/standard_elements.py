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
    Mux_DRAG_pulse, SquareFluxPulse, MartinisFluxPulse
from ..waveform_control.pulse import CosPulse, SquarePulse
from pycqed.measurement.randomized_benchmarking import randomized_benchmarking as rb

from importlib import reload
reload(pulse)
import pycqed.measurement.waveform_control.pulse_library as pl
import pycqed.measurement.waveform_control.pulse as bpl  # base pulse lib

from ..waveform_control import pulse_library
reload(pulse_library)

reload(element)


def multi_pulse_elt_old(i, station, pulse_list):
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
    for i in range(4):  # Exist to ensure there are no empty channels
        el.add(pulse.SquarePulse(name='refpulse_0',
                                 channel='ch{}'.format(i+1),
                                 amplitude=0, length=1e-9))
    # exists to ensure that channel is not high when waiting for trigger
    # and to allow negavtive pulse delay of elements up to 300 ns
    last_pulse = el.add(pulse.SquarePulse(name='refpulse_0', channel='ch1',
                                          amplitude=0, length=1e-9,),
                        start=300e-9)
    for i, pulse_pars in enumerate(pulse_list):
        # print(i)
        if 'refpoint' not in pulse_pars.keys():
            # default refpoint for backwards compatibility
            pulse_pars['refpoint'] = 'end'

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
                refpulse=last_pulse, refpoint=pulse_pars['refpoint'])
        elif pulse_pars['pulse_type'] == 'Mux_DRAG_pulse':
            # pulse_pars.pop('pulse_type')
            last_pulse = el.add(Mux_DRAG_pulse(name='pulse_{}'.format(i),
                                               **pulse_pars),
                                start=pulse_pars['pulse_delay'],
                                refpulse=last_pulse, refpoint=pulse_pars['refpoint'])
        elif pulse_pars['pulse_type'] == 'CosPulse':
            last_pulse = el.add(CosPulse(name='pulse_{}'.format(i),
                                         **pulse_pars),
                                start=pulse_pars['pulse_delay'],
                                refpulse=last_pulse, refpoint=pulse_pars['refpoint'])
        elif pulse_pars['pulse_type'] == 'SquarePulse':
            last_pulse = el.add(SquarePulse(name='pulse_{}'.format(i),
                                            **pulse_pars),
                                start=pulse_pars['pulse_delay'],
                                refpulse=last_pulse, refpoint=pulse_pars['refpoint'])
        elif pulse_pars['pulse_type'] == 'SquareFluxPulse':
            last_pulse = el.add(SquareFluxPulse(name='pulse_{}'.format(i),
                                                **pulse_pars),
                                start=pulse_pars['pulse_delay'],
                                refpulse=last_pulse, refpoint=pulse_pars['refpoint'])
        elif pulse_pars['pulse_type'] == 'MartinisFluxPulse':
            last_pulse = el.add(MartinisFluxPulse(name='pulse_{}'.format(i),
                                                  **pulse_pars),
                                start=pulse_pars['pulse_delay'],
                                refpulse=last_pulse, refpoint=pulse_pars['refpoint'])

        elif pulse_pars['pulse_type'] == 'ModSquare':
            last_pulse = el.add(MW_IQmod_pulse(name='pulse_{}'.format(i),
                                               **pulse_pars),
                                start=pulse_pars['pulse_delay'],
                                refpulse=last_pulse, refpoint=pulse_pars['refpoint'])

        elif (pulse_pars['pulse_type'] == 'MW_IQmod_pulse_tek' or
              pulse_pars['pulse_type'] == 'MW_IQmod_pulse_UHFQC' or
              pulse_pars['pulse_type'] == 'Gated_MW_RO_pulse'):
            # Does more than just call the function as it also adds the
            # markers. Ideally we combine both in one function in pulselib
            if pulse_pars['pulse_type'] == 'MW_IQmod_pulse_tek':
                last_pulse = el.add(MW_IQmod_pulse(
                    name='RO_tone',
                    I_channel=pulse_pars['I_channel'],
                    Q_channel=pulse_pars['Q_channel'],
                    length=pulse_pars['length'],
                    amplitude=pulse_pars['amplitude'],
                    mod_frequency=pulse_pars['mod_frequency']),
                    start=pulse_pars['pulse_delay'],
                    refpulse=last_pulse, refpoint=pulse_pars['refpoint'],
                    fixed_point_freq=pulse_pars['fixed_point_frequency'])
            elif pulse_pars['pulse_type'] == 'Gated_MW_RO_pulse':
                last_pulse = el.add(pulse.SquarePulse(
                    name='RO_marker', amplitude=1,
                    length=pulse_pars['length'],
                    channel=pulse_pars['RO_pulse_marker_channel']),
                    start=pulse_pars['pulse_delay'], refpulse=last_pulse,
                    refpoint=pulse_pars['refpoint'],
                    fixed_point_freq=pulse_pars['fixed_point_frequency'])
            elif pulse_pars['pulse_type'] == 'MW_IQmod_pulse_UHFQC':
                #"adding a 0 amp pulse because the sequencer needs an element for timing
                last_pulse = el.add(pulse.SquarePulse(
                    name='RO_marker', amplitude=0,
                    length=pulse_pars['length'],
                    channel=pulse_pars['RO_pulse_marker_channel']),
                    start=pulse_pars['pulse_delay'], refpulse=last_pulse,
                    refpoint=pulse_pars['refpoint'],
                    fixed_point_freq=pulse_pars['fixed_point_frequency'])
            # Start Acquisition marker
            if type(pulse_pars['acq_marker_channel']) is str:
                channels = pulse_pars['acq_marker_channel']
                channels = list(channels.split(','))
                for channel in channels:
                    Acq_marker = pulse.SquarePulse(
                        name='Acq-trigger', amplitude=1, length=20e-9,
                        channel=channel)
                    # Note that refpoint here is an exception because the
                    # refpulse is always the RO pulse that is added in the
                    # block just above here.
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


def multi_pulse_elt(i, station, pulse_list, sequencer_config=None):
    '''
    Input args
        i:          index of the element, ensures unique element name
        station:    qcodes station object, contains AWG etc
        pulse_list: list of pulse_dicts containing pulse parameters
        sequencer_config:   configuration containg values like pulse buffers
                            and fixed point
    Returns:
        element:    for use with the pulsar sequencer

    Note: this function is used to generate most standard elements we use.
    '''

    last_op_type = 'other'  # used for determining relevant buffers
    flux_compensation_pulse_list = []

    if sequencer_config is None:
        logging.warning('No sequencer config detected, using default config ' +
                        'from station')
        if hasattr(station, 'sequencer_config'):
            sequencer_config = station.sequencer_config
        else:
            sequencer_config = {'RO_fixed_point': 1e-6,
                                'Buffer_Flux_Flux': 0,
                                'Buffer_Flux_MW': 0,
                                'Buffer_Flux_RO': 0,
                                'Buffer_MW_Flux': 0,
                                'Buffer_MW_MW': 0,
                                'Buffer_MW_RO': 0,
                                'Buffer_RO_Flux': 0,
                                'Buffer_RO_MW': 0,
                                'Buffer_RO_RO': 0,
                                'Flux_comp_dead_time': 3e-6,
                                }

    ##########################
    # Instantiate an element #
    ##########################
    el = element.Element(
        name='{}-pulse-elt_{}'.format(len(pulse_list), i),
        pulsar=station.pulsar,
        readout_fixed_point=sequencer_config['RO_fixed_point'])
    for i in range(4):  # Exists to ensure there are no empty channels
        el.add(pulse.SquarePulse(name='refpulse_0',
                                 channel='ch{}'.format(i+1),
                                 amplitude=0, length=1e-9))
    # exists to ensure that channel is not high when waiting for trigger
    # and to allow negavtive pulse delay of elements up to 300 ns
    last_pulse = el.add(
        pulse.SquarePulse(name='refpulse_0', channel='ch1', amplitude=0,
                          length=1e-9,), start=300e-9)

    ##############################
    # Add all pulses one by one  #
    ##############################

    for i, pulse_pars in enumerate(pulse_list):
        ###################################
        # Determine timings of the pulses #
        ###################################
        if 'operation_type' in pulse_pars.keys():
            cur_op_type = pulse_pars['operation_type']
        else:
            cur_op_type = 'other'

        try:
            buffer_delay = sequencer_config[
                'Buffer_{}_{}'.format(last_op_type, cur_op_type)]
        except KeyError:
            buffer_delay = 0
        last_op_type = cur_op_type

        t0 = pulse_pars['pulse_delay'] + buffer_delay

        if 'refpoint' not in pulse_pars.keys():
            # default refpoint for backwards compatibility
            pulse_pars['refpoint'] = 'end'
        if 'operation_type' not in pulse_pars.keys():
            # default operation_type for backwards compatibility
            pulse_pars['operation_type'] = 'other'

        if cur_op_type == 'Flux':
            # Adds flux pulses to a list for automatic compensation pulses
            flux_compensation_pulse_list += [deepcopy(pulse_pars)]

        ###################################
        #       Pulses get added here     #
        ###################################
        # Adding any non special (composite) pulse
        if (pulse_pars['pulse_type'] not in ['MW_IQmod_pulse_tek',
                                             'MW_IQmod_pulse_UHFQC',
                                             'Gated_MW_RO_pulse']):
            try:
                # Look for the function in pl = pulse_lib
                pulse_func = getattr(pl, pulse_pars['pulse_type'])
            except AttributeError:
                try:
                    # Look for the function in bpl = pulse
                    pulse_func = getattr(bpl, pulse_pars['pulse_type'])
                except AttributeError:
                    raise KeyError('pulse_type {} not recognized'.format(
                        pulse_pars['pulse_type']))

            last_pulse = el.add(
                pulse_func(name=pulse_pars['pulse_type']+'_'+str(i),
                           **pulse_pars),
                start=t0, refpulse=last_pulse,
                refpoint=pulse_pars['refpoint'],
                operation_type=pulse_pars['operation_type'])

        else:
            # Composite "special pulses"
            # Ideally this should be combined in one function in pulselib
            # Does more than just call the function as it also adds the
            # markers.
            if pulse_pars['pulse_type'] == 'MW_IQmod_pulse_tek':
                last_pulse = el.add(MW_IQmod_pulse(
                    name='RO_tone',
                    I_channel=pulse_pars['I_channel'],
                    Q_channel=pulse_pars['Q_channel'],
                    length=pulse_pars['length'],
                    amplitude=pulse_pars['amplitude'],
                    mod_frequency=pulse_pars['mod_frequency']),
                    operation_type=pulse_pars['operation_type'],
                    start=t0,
                    refpulse=last_pulse, refpoint=pulse_pars['refpoint'])
            elif pulse_pars['pulse_type'] == 'Gated_MW_RO_pulse':
                last_pulse = el.add(pulse.SquarePulse(
                    name='RO_marker', amplitude=1,
                    length=pulse_pars['length'],
                    channel=pulse_pars['RO_pulse_marker_channel']),
                    operation_type=pulse_pars['operation_type'],
                    start=t0, refpulse=last_pulse,
                    refpoint=pulse_pars['refpoint'])

            elif pulse_pars['pulse_type'] == 'MW_IQmod_pulse_UHFQC':
                # "adding a 0 amp pulse because the sequencer needs an element for timing
                last_pulse = el.add(pulse.SquarePulse(
                    name='RO_marker', amplitude=0,
                    length=pulse_pars['length'],
                    channel=pulse_pars['RO_pulse_marker_channel']),
                    operation_type=pulse_pars['operation_type'],
                    start=t0, refpulse=last_pulse,
                    refpoint=pulse_pars['refpoint'])
            # Start Acquisition marker
            if type(pulse_pars['acq_marker_channel']) is str:
                channels = pulse_pars['acq_marker_channel']
                channels = list(channels.split(','))
                for channel in channels:
                    Acq_marker = pulse.SquarePulse(
                        name='Acq-trigger', amplitude=1, length=20e-9,
                        channel=channel)
                    # Note that refpoint here is an exception because the
                    # refpulse is always the RO pulse that is added in the
                    # block just above here.
                    el.add(
                        Acq_marker, start=pulse_pars['acq_marker_delay'],
                        refpulse=last_pulse, refpoint='start')

    ####################################################################
    # Adding Flux compensation pulses
    ####################################################################
    for j, pulse_pars in enumerate(flux_compensation_pulse_list):
        pulse_pars['amplitude'] *= -1
        if j == 0:
            t0 = sequencer_config['Flux_comp_dead_time']
        else:
            t0 = sequencer_config['Buffer_Flux_Flux']
        try:
            # Look for the function in pl = pulse_lib
            pulse_func = getattr(pl, pulse_pars['pulse_type'])
        except AttributeError:
            try:
                # Look for the function in bpl = pulse
                pulse_func = getattr(bpl, pulse_pars['pulse_type'])
            except AttributeError:
                raise KeyError('pulse_type {} not recognized'.format(
                    pulse_pars['pulse_type']))

        last_pulse = el.add(
            pulse_func(name=pulse_pars['pulse_type']+'_'+str(i+j),
                       **pulse_pars),
            start=t0, refpulse=last_pulse,
            refpoint=pulse_pars['refpoint'],
            operation_type=pulse_pars['operation_type'])

    # This pulse ensures that the sequence always ends at zero amp
    last_pulse = el.add(pulse.SquarePulse(name='final_empty_pulse',
                                          channel='ch1',
                                          amplitude=0, length=1e-9),
                        refpulse=last_pulse, refpoint='end')

    return el
