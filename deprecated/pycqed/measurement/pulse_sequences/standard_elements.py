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
import pycqed.measurement.waveform_control.pulse as bpl  # base pulse library

from ..waveform_control import pulse_library
reload(pulse_library)

reload(element)


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
    # Prevents accidently overwriting pulse pars in this list
    pulse_list = deepcopy(pulse_list)
    last_op_type = 'other'  # used for determining relevant buffers
    flux_compensation_pulse_list = []

    if sequencer_config is None:
        if hasattr(station, 'sequencer_config'):
            sequencer_config = station.sequencer_config
        else:
            logging.warning('No sequencer config detected, using default ' +
                            '\n you can specify one as station.sequencer_config')
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
                                'slave_AWG_trig_channels': [],
                                }

    ##########################
    # Instantiate an element #
    ##########################
    el = element.Element(
        name='{}-pulse-elt_{}'.format(len(pulse_list), i),
        pulsar=station.pulsar,
        readout_fixed_point=sequencer_config['RO_fixed_point'])
    for cname in station.pulsar.channels:  # Exists to ensure there are no empty channels
        el.add(pulse.SquarePulse(name='refpulse_0', channel=cname,
                                 amplitude=0, length=1e-9))


    # exists to ensure that channel is not high when waiting for trigger
    # and to allow negavtive pulse delay of elements up to 300 ns
    # last_pulse = el.add(
    #     pulse.SquarePulse(name='refpulse_0', channel='ch1', amplitude=0,
    #                       length=1e-9,), start=300e-9)
    last_pulse = None

    ##############################
    # Add all pulses one by one  #
    ##############################

    for i, pulse_pars in enumerate(pulse_list):
        # Default values for backwards compatibility
        if 'refpoint' not in pulse_pars.keys():
            # default refpoint for backwards compatibility
            pulse_pars['refpoint'] = 'end'
        if 'operation_type' not in pulse_pars.keys():
            # default operation_type for backwards compatibility
            pulse_pars['operation_type'] = 'other'

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

        # Buffers and delays get overwritten if two pulses have to be executed
        # simultaneous
        if pulse_pars['refpoint'] == 'simultaneous':
            pulse_pars['refpoint'] = 'start'
            t0 = 0

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
    if len(flux_compensation_pulse_list) > 0:
        final_len = 500e-9
    else:
        final_len = 1e-9
    for cname in station.pulsar.channels:
        el.add(pulse.SquarePulse(name='final_empty_pulse', channel=cname,
                                 amplitude=0, length=final_len),
               refpulse=last_pulse, refpoint='end')

    # Trigger the slave AWG-s
    slave_triggers = sequencer_config.get('slave_AWG_trig_channels', [])
    for cname in slave_triggers:
        el.add(pulse.SquarePulse(name='slave_trigger', channel=cname,
                                 amplitude=1, length=20e-9), start=10e-9)

    return el


def distort_and_compensate(element, distortion_dict):
    """
    Distorts an element using the contenst of a distortion dictionary.
    The distortion dictionary should be formatted as follows.

    dist_dict{'ch_list': ['chx', 'chy'],
              'chx': np.array(.....),
              'chy': np.array(.....)}
    """
    t_vals, outputs_dict = element.waveforms()
    for ch in distortion_dict['ch_list']:
        element._channels[ch]['distorted'] = True
        length = len(outputs_dict[ch])
        kernelvec = distortion_dict[ch]
        outputs_dict[ch] = np.convolve(
            outputs_dict[ch], kernelvec)[:length]
        element.distorted_wfs[ch] = outputs_dict[ch][:len(t_vals)]
    return element
