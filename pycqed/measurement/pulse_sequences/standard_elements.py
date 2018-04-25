import logging
import numpy as np
from copy import deepcopy
try:
    from math import gcd
except:  # Moved to math in python 3.5, this is to be 3.4 compatible
    from fractions import gcd
from pycqed.measurement.waveform_control import pulsar as pulsar
from pycqed.measurement.waveform_control import element as element

import pycqed.measurement.waveform_control.pulse_library as pl
import pycqed.measurement.waveform_control.pulse as bpl  # base pulse lib

def multi_pulse_elt(i, station, pulse_list, sequencer_config=None, name=None,
                    trigger=True, previous_element=None):
    """
    Input args
        i:          index of the element, ensures unique element name
        station:    qcodes station object, contains AWG etc
        pulse_list: list of pulse_dicts containing pulse parameters
        sequencer_config:   configuration containg values like pulse buffers
                            and fixed point
    Returns:
        element:    for use with the pulsar sequencer

    Note: this function is used to generate most standard elements we use.
    """
    # Prevents accidently overwriting pulse pars in this list
    pulse_list = [deepcopy(pulse) for pulse in pulse_list]
    last_op_type = 'other'  # used for determining relevant buffers

    if sequencer_config is None:
        if hasattr(station, 'sequencer_config'):
            sequencer_config = station.sequencer_config
        else:
            logging.warning('No sequencer config detected, using default \n you'
                            ' can specify one as station.sequencer_config')
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
    
    # don't count the Z_pulses when specifying number of pulses in element name
    count_z = 0
    for pls in pulse_list:
        if 'Z_pulse' in pls['pulse_type']:
            count_z += 1
    no_of_pulses = len(pulse_list) - count_z

    if name is None:
        name = '{}-pulse-elt_{}'.format(no_of_pulses, i)
    
    el = element.Element(
        name=name,
        pulsar=station.pulsar,
        readout_fixed_point=sequencer_config['RO_fixed_point'],
        ignore_delays=not trigger)
    if not trigger:
        el.fixed_point_applied = True
        el.ignore_offset_correction = True
    if previous_element is not None:
        el.time_offset = previous_element.time_offset + \
                         previous_element.ideal_length()

    last_pulse = None

    ##############################
    # Add all pulses one by one  #
    ##############################
    # used for software Z-gates
    if previous_element is None or not hasattr(previous_element,
                                               'drive_phase_offsets'):
        phase_offset = {}
    else:
        phase_offset = previous_element.drive_phase_offsets.copy()
    j = 0
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

        ###################################
        #       Pulses get added here     #
        ###################################
        # Adding any non special (composite) pulse
        if (pulse_pars['pulse_type'] not in ['MW_IQmod_pulse_tek',
                                             'MW_IQmod_pulse_UHFQC',
                                             'Gated_MW_RO_pulse',
                                             'Multiplexed_UHFQC_pulse',
                                             'Z_pulse']):
            # only add phase_offset if the pulse is a qubit drive pulse
            if pulse_pars.get('operation_type', None) == 'MW':
                target_qubit = pulse_pars.get('target_qubit', None)
                if target_qubit is not None:
                    if 'phase' in pulse_pars.keys():
                        if target_qubit in phase_offset:
                            pulse_pars['phase'] += phase_offset[target_qubit]
                            pulse_pars['phase'] %= 360
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
                pulse_func(name=pulse_pars['pulse_type']+'_'+str(j),
                           **pulse_pars),
                start=t0, refpulse=last_pulse,
                refpoint=pulse_pars['refpoint'],
                operation_type=pulse_pars['operation_type'])
            j += 1
        else:
            if pulse_pars['pulse_type'] == 'Z_pulse':
                pass
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
                    last_pulse = el.add(bpl.SquarePulse(
                        name='RO_marker', amplitude=pulse_pars['amplitude'],
                        length=pulse_pars['length'],
                        channel=pulse_pars['RO_pulse_marker_channel']),
                        operation_type=pulse_pars['operation_type'],
                        start=t0, refpulse=last_pulse,
                        refpoint=pulse_pars['refpoint'])

                elif (pulse_pars['pulse_type'] == 'MW_IQmod_pulse_UHFQC' or
                      pulse_pars['pulse_type'] == 'Multiplexed_UHFQC_pulse'):
                    # "adding a 0 amp pulse because the sequencer needs an element
                    # for timing
                    last_pulse = el.add(bpl.SquarePulse(
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
                        Acq_marker = bpl.SquarePulse(
                            name='Acq-trigger', amplitude=1, length=20e-9,
                            channel=channel)
                        # Note that refpoint here is an exception because the
                        # refpulse is always the RO pulse that is added in the
                        # block just above here.
                        el.add(
                            Acq_marker, start=pulse_pars['acq_marker_delay'],
                            refpulse=last_pulse, refpoint='start')
        # apply any virtual-z rotations in the pulse
        basis_rotation = pulse_pars.get('basis_rotation', {})
        for qubit_name, offset in basis_rotation.items():
            if qubit_name not in phase_offset:
                phase_offset[qubit_name] = 0
            phase_offset[qubit_name] -= offset
    el.drive_phase_offsets = phase_offset.copy()

    # make sure that the waveforms on all the channels end at the same time
    # in case the next element is ran back to back with this one.
    for cname in station.pulsar.channels:
        el.add(bpl.SquarePulse(name='empty_pulse', channel=cname, amplitude=0,
                               length=station.pulsar.get(
                                   '{}_min_length'.format(cname))),
               refpulse=last_pulse, refpoint='end', refpoint_new='end')


    # switch to global timing for adding the trigger pulses.
    el.shift_all_pulses(-el.offset())
    el.ignore_offset_correction = True

    # make sure that the channels that have a precompiled FIR filter have
    # enough zeros at the start such that the filtered waveform will not be cut
    # off.
    if trigger:
        max_advance_len = 0
        for cname in station.pulsar.channels:
            if station.pulsar.get(cname + '_type') != 'analog':
                continue
            if station.pulsar.get(cname + '_distortion') != 'precalculate':
                continue
            distortion_dictionary = station.pulsar.get(
                '{}_distortion_dict'.format(cname))
            fir_kernels = distortion_dictionary.get('FIR', None)
            kernel_length = 1
            if fir_kernels is not None:
                if hasattr(fir_kernels, '__iter__') and not \
                        hasattr(fir_kernels[0], '__iter__'): # 1 kernel only
                    kernel_length += len(fir_kernels) - 1
                else:
                    for kernel in fir_kernels:
                        kernel_length += len(kernel) - 1
            advance_len = ((kernel_length + 1)//2)/station.pulsar.clock(cname)
            max_advance_len = max(advance_len, max_advance_len)
        max_advance_len += element.calculate_time_correction(
            max_advance_len, sequencer_config['RO_fixed_point'])
        el.shift_all_pulses(max_advance_len)

    # Trigger the slave AWG-s
    if trigger:
        slave_triggers = sequencer_config.get('slave_AWG_trig_channels', [])
        for cname in slave_triggers:
            el.add(bpl.SquarePulse(name='slave_trigger', channel=cname,
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
