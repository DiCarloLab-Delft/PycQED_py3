# Originally by Wolfgang Pfaff
# Modified by Adriaan Rol 9/2015
# Modified by Ants Remm 5/2017
# Modified by Michael Kerschbaum 5/2019
import os
import shutil
import ctypes
import numpy as np
import logging
from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import (
    ManualParameter, InstrumentRefParameter)
import qcodes.utils.validators as vals
import pycqed.measurement.waveform_control.element as element
import string
import time
import pprint
import base64
import hashlib

from pycqed.instrument_drivers.virtual_instruments.virtual_awg5014 import \
    VirtualAWG5014
from pycqed.instrument_drivers.virtual_instruments.virtual_AWG8 import \
    VirtualAWG8
# exception catching removed because it does not work in python versions before
# 3.6
try:
    from qcodes.instrument_drivers.tektronix.AWG5014 import Tektronix_AWG5014
except Exception:
    Tektronix_AWG5014 = type(None)
try:
    from pycqed.instrument_drivers.physical_instruments.ZurichInstruments.\
        UHFQuantumController import UHFQC
except Exception:
    UHFQC = type(None)
try:
    from pycqed.instrument_drivers.physical_instruments.ZurichInstruments. \
        ZI_HDAWG8 import ZI_HDAWG8
except Exception:
    ZI_HDAWG8 = type(None)
log = logging.getLogger(__name__)

from pycqed.instrument_drivers.physical_instruments.ZurichInstruments. \
        dummy_UHFQC import dummy_UHFQC

class UHFQCPulsar:
    """
    Defines the Zurich Instruments UHFQC specific functionality for the Pulsar
    class
    """
    _supportedAWGtypes = (UHFQC, dummy_UHFQC)
    
    _uhf_sequence_string_template = (
        "const WINT_EN   = 0x01ff0000;\n"
        "const WINT_TRIG = 0x00000010;\n"
        "const IAVG_TRIG = 0x00000020;\n"
        "var RO_TRIG;\n"
        "if (getUserReg(1)) {{\n"
        "  RO_TRIG = WINT_EN + IAVG_TRIG;\n"
        "}} else {{\n"
        "  RO_TRIG = WINT_EN + WINT_TRIG;\n"
        "}}\n"
        "setTrigger(WINT_EN);\n"
        "\n"
        "{wave_definitions}\n"
        "\n"
        "var loop_cnt = getUserReg(0);\n"
        "\n"
        "repeat (loop_cnt) {{\n"
        "  {playback_string}\n"
        "}}\n"
    )

    def _create_awg_parameters(self, awg, channel_name_map):
        if not isinstance(awg, UHFQCPulsar._supportedAWGtypes):
            return super()._create_awg_parameters(awg, channel_name_map)
        
        name = awg.name

        self.add_parameter('{}_reuse_waveforms'.format(awg.name),
                           initial_value=True, vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_minimize_sequencer_memory'.format(awg.name),
                           initial_value=True, vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           docstring="Minimizes the sequencer "
                                     "memory by repeating specific sequence "
                                     "patterns (eg. readout) passed in "
                                     "'repeat dictionary'")
        self.add_parameter('{}_granularity'.format(awg.name),
                           get_cmd=lambda: 16)
        self.add_parameter('{}_element_start_granularity'.format(awg.name),
                           initial_value=8/(1.8e9),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_min_length'.format(awg.name),
                           get_cmd=lambda: 16 /(1.8e9))
        self.add_parameter('{}_inter_element_deadtime'.format(awg.name),
                           # get_cmd=lambda: 80 / 2.4e9)
                           get_cmd=lambda: 8 / (1.8e9))
                           # get_cmd=lambda: 0 / 2.4e9)
        self.add_parameter('{}_precompile'.format(awg.name), 
                           initial_value=False, vals=vals.Bool(),
                           label='{} precompile segments'.format(awg.name),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_delay'.format(awg.name), 
                           initial_value=0, label='{} delay'.format(name), 
                           unit='s', parameter_class=ManualParameter,
                           docstring='Global delay applied to this '
                                     'channel. Positive values move pulses'
                                     ' on this channel forward in time')
        self.add_parameter('{}_trigger_channels'.format(awg.name), 
                           initial_value=None, 
                           label='{} trigger channel'.format(awg.name), 
                           parameter_class=ManualParameter)
        self.add_parameter('{}_active'.format(awg.name), initial_value=True,
                           label='{} active'.format(awg.name),
                           vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_compensation_pulse_min_length'.format(name), 
                           initial_value=0, unit='s',
                           parameter_class=ManualParameter)

        for ch_nr in range(2):
            id = 'ch{}'.format(ch_nr + 1)
            name = channel_name_map.get(id, awg.name + '_' + id)
            self._uhfqc_create_channel_parameters(id, name, awg)
            self.channels.add(name)

    def _uhfqc_create_channel_parameters(self, id, name, awg):
        self.add_parameter('{}_id'.format(name), get_cmd=lambda _=id: _)
        self.add_parameter('{}_awg'.format(name), get_cmd=lambda _=awg.name: _)
        self.add_parameter('{}_type'.format(name), get_cmd=lambda: 'analog')
        self.add_parameter('{}_amp'.format(name),
                            label='{} amplitude'.format(name), unit='V',
                            set_cmd=self._uhfqc_setter(awg, id, 'amp'),
                            get_cmd=self._uhfqc_getter(awg, id, 'amp'),
                            vals=vals.Numbers(0.075, 1.5),
                            initial_value=0.75)
        self.add_parameter('{}_offset'.format(name),
                            label='{} offset'.format(name), unit='V',
                            set_cmd=self._uhfqc_setter(awg, id, 'offset'),
                            get_cmd=self._uhfqc_getter(awg, id, 'offset'),
                            vals=vals.Numbers(-1.5, 1.5),
                            initial_value=0)
        self.add_parameter('{}_distortion'.format(name),
                            label='{} distortion mode'.format(name),
                            initial_value='off',
                            vals=vals.Enum('off', 'precalculate'),
                            parameter_class=ManualParameter)
        self.add_parameter('{}_distortion_dict'.format(name),
                            label='{} distortion dictionary'.format(name),
                            vals=vals.Dict(),
                            parameter_class=ManualParameter)
        self.add_parameter('{}_charge_buildup_compensation'.format(name),
                            parameter_class=ManualParameter,
                            vals=vals.Bool(), initial_value=False)
        self.add_parameter('{}_compensation_pulse_scale'.format(name),
                            parameter_class=ManualParameter,
                            vals=vals.Numbers(0., 1.), initial_value=0.5)
        self.add_parameter('{}_compensation_pulse_delay'.format(name), 
                           initial_value=0, unit='s',
                           parameter_class=ManualParameter)

    @staticmethod
    def _uhfqc_setter(obj, id, par):
        if par == 'offset':
            def s(val):
                obj.set('sigouts_{}_offset'.format(int(id[2])-1), val)
        elif par == 'amp':
            def s(val):
                obj.set('sigouts_{}_range'.format(int(id[2])-1), val)
        else:
            raise NotImplementedError('Unknown parameter {}'.format(par))
        return s

    def _uhfqc_getter(self, obj, id, par):
        if par == 'offset':
            def g():
                return obj.get('sigouts_{}_offset'.format(int(id[2])-1))
        elif par == 'amp':
            def g():
                if self._awgs_prequeried_state:
                    return obj.parameters['sigouts_{}_range' \
                        .format(int(id[2])-1)].get_latest()/2
                else:
                    return obj.get('sigouts_{}_range' \
                        .format(int(id[2])-1))/2
        else:
            raise NotImplementedError('Unknown parameter {}'.format(par))
        return g 

    def _program_awg(self, obj, awg_sequence, waveforms, repeat_pattern=None):
        if not isinstance(obj, UHFQCPulsar._supportedAWGtypes):
            return super()._program_awg(obj, awg_sequence, waveforms, repeat_pattern)

        if not self._zi_waves_cleared:
            _zi_clear_waves()
            self._zi_waves_cleared = True
        waves_to_upload = {h: waveforms[h]
                               for codewords in awg_sequence.values() 
                                   if codewords is not None
                               for cw, chids in codewords.items()
                                   if cw != 'metadata'
                               for h in chids.values()}
        self._zi_write_waves(waves_to_upload)

        defined_waves = set()
        wave_definitions = []
        playback_strings = []

        ch_has_waveforms = {'ch1': False, 'ch2': False}

        current_segment = 'no_segment'

        def play_element(element, playback_strings, wave_definitions):
            if awg_sequence[element] is None:
                current_segment = element
                playback_strings.append(f'// Segment {current_segment}')
                return playback_strings, wave_definitions
            playback_strings.append(f'// Element {element}')

            metadata = awg_sequence[element].pop('metadata', {})
            if list(awg_sequence[element].keys()) != ['no_codeword']:
                raise NotImplementedError('UHFQC sequencer does currently\
                                                       not support codewords!')
            chid_to_hash = awg_sequence[element]['no_codeword']

            wave = (chid_to_hash.get('ch1', None), None,
                    chid_to_hash.get('ch2', None), None)
            wave_definitions += self._zi_wave_definition(wave,
                                                         defined_waves)

            acq = metadata.get('acq', False)
            playback_strings += self._zi_playback_string('uhf', wave,
                                                         acq=acq)

            ch_has_waveforms['ch1'] |= wave[0] is not None
            ch_has_waveforms['ch2'] |= wave[2] is not None
            return playback_strings, wave_definitions

        if repeat_pattern is None:
            for element in awg_sequence:
                playback_strings, wave_definitions = play_element(element,
                                                                  playback_strings,
                                                                  wave_definitions)
        else:
            real_indicies = []
            for index, element in enumerate(awg_sequence):
                if awg_sequence[element] is not None:
                    real_indicies.append(index)
            el_total = len(real_indicies)

            def repeat_func(n, el_played, index, playback_strings, wave_definitions):
                if isinstance(n, tuple):
                    el_played_list = []
                    if n[0] > 1:
                        playback_strings.append('repeat ('+str(n[0])+') {')
                    for t in n[1:]:
                        el_cnt, playback_strings, wave_definitions = repeat_func(t,
                                                               el_played,
                                                               index + np.sum(
                                                                  el_played_list),
                                                               playback_strings,
                                                               wave_definitions)
                        el_played_list.append(el_cnt)
                    if n[0] > 1:
                        playback_strings.append('}')
                    return int(n[0] * np.sum(el_played_list)), playback_strings, wave_definitions
                else:
                    for k in range(n):
                        el_index = real_indicies[int(index)+k]
                        element = list(awg_sequence.keys())[el_index]
                        playback_strings, wave_definitions = play_element(element,
                                                                playback_strings,
                                                                wave_definitions)
                        el_played = el_played + 1
                    return el_played, playback_strings, wave_definitions



            el_played, playback_strings, wave_definitions = repeat_func(repeat_pattern, 0, 0,
                                                  playback_strings, wave_definitions)


            if int(el_played) != int(el_total):
                log.error(el_played, ' is not ', el_total)
                raise ValueError('Check number of sequences in repeat pattern')


        if not (ch_has_waveforms['ch1'] or ch_has_waveforms['ch2']):
            return
        self.awgs_with_waveforms(obj.name)
        
        awg_str = self._uhf_sequence_string_template.format(
            wave_definitions='\n'.join(wave_definitions),
            playback_string='\n  '.join(playback_strings),
        )

        obj.awg_string(awg_str, timeout=600)

    def _is_awg_running(self, obj):
        if not isinstance(obj, UHFQCPulsar._supportedAWGtypes):
            return super()._is_awg_running(obj)
        return obj.awgs_0_enable() != 0

    def _clock(self, obj, cid=None):
        if not isinstance(obj, UHFQCPulsar._supportedAWGtypes):
            return super()._clock(obj)
        return obj.clock_freq()

class HDAWG8Pulsar:
    """
    Defines the Zurich Instruments HDAWG8 specific functionality for the Pulsar
    class
    """
    _supportedAWGtypes = (ZI_HDAWG8, VirtualAWG8, )

    _hdawg_sequence_string_template = (
        "{wave_definitions}\n"
        "\n"
        "{codeword_table_defs}\n"
        "\n"
        "while (1) {{\n"
        "  {playback_string}\n"
        "}}\n"
    )

    def _create_awg_parameters(self, awg, channel_name_map):
        if not isinstance(awg, HDAWG8Pulsar._supportedAWGtypes):
            return super()._create_awg_parameters(awg, channel_name_map)
        
        name = awg.name

        self.add_parameter('{}_reuse_waveforms'.format(awg.name),
                           initial_value=True, vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_minimize_sequencer_memory'.format(awg.name),
                           initial_value=False, vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           docstring="Minimizes the sequencer "
                                     "memory by repeating specific sequence "
                                     "patterns (eg. readout) passed in "
                                     "'repeat dictionary'")
        self.add_parameter('{}_granularity'.format(awg.name),
                           get_cmd=lambda: 16)
        self.add_parameter('{}_element_start_granularity'.format(awg.name),
                           initial_value=8/(2.4e9),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_min_length'.format(awg.name),
                           initial_value=16 /(2.4e9),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_inter_element_deadtime'.format(awg.name),
                           # get_cmd=lambda: 80 / 2.4e9)
                           get_cmd=lambda: 8 / (2.4e9))
                           # get_cmd=lambda: 0 / 2.4e9)
        self.add_parameter('{}_precompile'.format(awg.name), 
                           initial_value=False, vals=vals.Bool(),
                           label='{} precompile segments'.format(awg.name),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_delay'.format(awg.name), 
                           initial_value=0, label='{} delay'.format(name), 
                           unit='s', parameter_class=ManualParameter,
                           docstring='Global delay applied to this '
                                     'channel. Positive values move pulses'
                                     ' on this channel forward in time')
        self.add_parameter('{}_trigger_channels'.format(awg.name), 
                           initial_value=None, 
                           label='{} trigger channel'.format(awg.name), 
                           parameter_class=ManualParameter)
        self.add_parameter('{}_active'.format(awg.name), initial_value=True,
                           label='{} active'.format(awg.name),
                           vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_compensation_pulse_min_length'.format(name), 
                           initial_value=0, unit='s',
                           parameter_class=ManualParameter)

        for ch_nr in range(8):
            id = 'ch{}'.format(ch_nr + 1)
            name = channel_name_map.get(id, awg.name + '_' + id)
            self._hdawg_create_analog_channel_parameters(id, name, awg)
            self.channels.add(name)
            id = 'ch{}m'.format(ch_nr + 1)
            name = channel_name_map.get(id, awg.name + '_' + id)
            self._hdawg_create_marker_channel_parameters(id, name, awg)
            self.channels.add(name)

    def _hdawg_create_analog_channel_parameters(self, id, name, awg):
        self.add_parameter('{}_id'.format(name), get_cmd=lambda _=id: _)
        self.add_parameter('{}_awg'.format(name), get_cmd=lambda _=awg.name: _)
        self.add_parameter('{}_type'.format(name), get_cmd=lambda: 'analog')
        self.add_parameter('{}_offset'.format(name),
                           label='{} offset'.format(name), unit='V',
                           set_cmd=self._hdawg_setter(awg, id, 'offset'),
                           get_cmd=self._hdawg_getter(awg, id, 'offset'),
                           vals=vals.Numbers())
        self.add_parameter('{}_amp'.format(name),
                            label='{} amplitude'.format(name), unit='V',
                            set_cmd=self._hdawg_setter(awg, id, 'amp'),
                            get_cmd=self._hdawg_getter(awg, id, 'amp'),
                            vals=vals.Numbers(0.01, 5.0))
        self.add_parameter('{}_distortion'.format(name),
                            label='{} distortion mode'.format(name),
                            initial_value='off',
                            vals=vals.Enum('off', 'precalculate'),
                            parameter_class=ManualParameter)
        self.add_parameter('{}_distortion_dict'.format(name),
                            label='{} distortion dictionary'.format(name),
                            vals=vals.Dict(),
                            parameter_class=ManualParameter)
        self.add_parameter('{}_charge_buildup_compensation'.format(name),
                            parameter_class=ManualParameter,
                            vals=vals.Bool(), initial_value=False)
        self.add_parameter('{}_compensation_pulse_scale'.format(name),
                            parameter_class=ManualParameter,
                            vals=vals.Numbers(0., 1.), initial_value=0.5)
        self.add_parameter('{}_compensation_pulse_delay'.format(name), 
                           initial_value=0, unit='s',
                           parameter_class=ManualParameter)
    
    def _hdawg_create_marker_channel_parameters(self, id, name, awg):
        self.add_parameter('{}_id'.format(name), get_cmd=lambda _=id: _)
        self.add_parameter('{}_awg'.format(name), get_cmd=lambda _=awg.name: _)
        self.add_parameter('{}_type'.format(name), get_cmd=lambda: 'marker')
        self.add_parameter('{}_offset'.format(name),
                           label='{} offset'.format(name), unit='V',
                           set_cmd=self._hdawg_setter(awg, id, 'offset'),
                           get_cmd=self._hdawg_getter(awg, id, 'offset'),
                           vals=vals.Numbers())
        self.add_parameter('{}_amp'.format(name),
                            label='{} amplitude'.format(name), unit='V',
                            set_cmd=self._hdawg_setter(awg, id, 'amp'),
                            get_cmd=self._hdawg_getter(awg, id, 'amp'),
                            vals=vals.Numbers(0.01, 5.0))
        
    @staticmethod
    def _hdawg_setter(obj, id, par):
        if par == 'offset':
            if id[-1] != 'm':
                def s(val):
                    obj.set('sigouts_{}_offset'.format(int(id[2])-1), val)
            else:
                s = None
        elif par == 'amp':
            if id[-1] != 'm':
                def s(val):
                    obj.set('sigouts_{}_range'.format(int(id[2])-1), 2*val)
            else:
                s = None
        else:
            raise NotImplementedError('Unknown parameter {}'.format(par))
        return s

    def _hdawg_getter(self, obj, id, par):
        if par == 'offset':
            if id[-1] != 'm':
                def g():
                    return obj.get('sigouts_{}_offset'.format(int(id[2])-1))
            else:
                return lambda: 0
        elif par == 'amp':
            if id[-1] != 'm':
                def g():
                    if self._awgs_prequeried_state:
                        return obj.parameters['sigouts_{}_range' \
                            .format(int(id[2])-1)].get_latest()/2
                    else:
                        return obj.get('sigouts_{}_range' \
                            .format(int(id[2])-1))/2
            else:
                return lambda: 1
        else:
            raise NotImplementedError('Unknown parameter {}'.format(par))
        return g 

    def _program_awg(self, obj, awg_sequence, waveforms, repeat_pattern=None):
        if not isinstance(obj, HDAWG8Pulsar._supportedAWGtypes):
            return super()._program_awg(obj, awg_sequence, waveforms, repeat_pattern)
        
        if not self._zi_waves_cleared:
            _zi_clear_waves()
            self._zi_waves_cleared = True
        waves_to_upload = {h: waveforms[h]
                               for codewords in awg_sequence.values() 
                                   if codewords is not None 
                               for cw, chids in codewords.items() 
                                   if cw != 'metadata'
                               for h in chids.values()}
        self._zi_write_waves(waves_to_upload)
        
        ch_has_waveforms = {'ch{}{}'.format(i + 1, m): False 
                                for i in range(8) for m in ['','m']}

        for awg_nr in self._hdawg_active_awgs(obj):
            defined_waves = set()
            codeword_table = {}
            wave_definitions = []
            codeword_table_defs = []
            playback_strings = []

            prev_dio_valid_polarity = obj.get(
                'awgs_{}_dio_valid_polarity'.format(awg_nr))
            
            added_cw = set()
            ch1id = 'ch{}'.format(awg_nr * 2 + 1)
            ch1mid = 'ch{}m'.format(awg_nr * 2 + 1)
            ch2id = 'ch{}'.format(awg_nr * 2 + 2)
            ch2mid = 'ch{}m'.format(awg_nr * 2 + 2)

            codeword_el = set()

            current_segment = 'no_segment'
            for element in awg_sequence:
                if awg_sequence[element] is None:
                    current_segment = element
                    playback_strings.append(f'// Segment {current_segment}')
                    continue
                playback_strings.append(f'// Element {element}')
                
                metadata = awg_sequence[element].pop('metadata', {})
                
                nr_cw = len(set(awg_sequence[element].keys()) - \
                            {'no_codeword'})

                if nr_cw == 1:
                    log.warning(
                        f'Only one codeword has been set for {element}')
                else:
                    for cw in awg_sequence[element]:
                        if cw == 'no_codeword':
                            if nr_cw != 0:
                                continue
                        chid_to_hash = awg_sequence[element][cw]
                        wave = tuple(chid_to_hash.get(ch, None)
                                    for ch in [ch1id, ch1mid, ch2id, ch2mid])
                        wave_definitions += self._zi_wave_definition(wave,
                                                                defined_waves)
                        
                        if nr_cw != 0:
                            w1, w2 = self._zi_waves_to_wavenames(wave)
                            if cw not in codeword_table:
                                codeword_table_defs += \
                                    self._zi_codeword_table_entry(cw, wave)
                                codeword_table[cw] = (w1, w2)
                            elif codeword_table[cw] != (w1, w2) \
                                    and self.reuse_waveforms():
                                log.warning('Same codeword used for different '
                                            'waveforms. Using first waveform. '
                                            f'Ignoring element {element}.')

                        ch_has_waveforms[ch1id] |= wave[0] is not None
                        ch_has_waveforms[ch1mid] |= wave[1] is not None
                        ch_has_waveforms[ch2id] |= wave[2] is not None
                        ch_has_waveforms[ch2mid] |= wave[3] is not None

                    playback_strings += self._zi_playback_string(
                        'hdawg', wave, codeword=(nr_cw != 0))
                
            if not any([ch_has_waveforms[ch] 
                    for ch in [ch1id, ch1mid, ch2id, ch2mid]]):
                continue
            
            awg_str = self._hdawg_sequence_string_template.format(
                wave_definitions='\n'.join(wave_definitions),
                codeword_table_defs='\n'.join(codeword_table_defs),
                playback_string='\n  '.join(playback_strings),
            )

            obj.configure_awg_from_string(awg_nr, awg_str, timeout=600)

            obj.set('awgs_{}_dio_valid_polarity'.format(awg_nr),
                    prev_dio_valid_polarity)

        for ch in range(8):
            obj.set('sigouts_{}_on'.format(ch), ch_has_waveforms[f'ch{ch+1}'])

        if any(ch_has_waveforms.values()):
            self.awgs_with_waveforms(obj.name)

    def _is_awg_running(self, obj):
        if not isinstance(obj, HDAWG8Pulsar._supportedAWGtypes):
            return super()._is_awg_running(obj)

        return all([obj.get('awgs_{}_enable'.format(awg_nr)) for awg_nr in
                    self._hdawg_active_awgs(obj)])

    def _clock(self, obj, cid):
        if not isinstance(obj, HDAWG8Pulsar._supportedAWGtypes):
            return super()._clock(obj, cid)
        return obj.clock_freq(0)

    def _hdawg_active_awgs(self, obj):
        return [0,1,2,3]

class AWG5014Pulsar:
    """
    Defines the Tektronix AWG5014 specific functionality for the Pulsar class
    """
    _supportedAWGtypes = (Tektronix_AWG5014, VirtualAWG5014, )

    def _create_awg_parameters(self, awg, channel_name_map):
        if not isinstance(awg, AWG5014Pulsar._supportedAWGtypes):
            return super()._create_awg_parameters(awg, channel_name_map)
        
        self.add_parameter('{}_reuse_waveforms'.format(awg.name),
                           initial_value=True, vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_minimize_sequencer_memory'.format(awg.name),
                           initial_value=False, vals=vals.Bool(),
                           parameter_class=ManualParameter,
                           docstring="Minimizes the sequencer "
                                     "memory by repeating specific sequence "
                                     "patterns (eg. readout) passed in "
                                     "'repeat dictionary'")
        self.add_parameter('{}_granularity'.format(awg.name),
                           get_cmd=lambda: 4)
        self.add_parameter('{}_element_start_granularity'.format(awg.name),
                           initial_value=4/(1.2e9),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_min_length'.format(awg.name),
                           get_cmd=lambda: 256/(1.2e9)) # Can not be triggered 
                                                        # faster than 210 ns.
        self.add_parameter('{}_inter_element_deadtime'.format(awg.name),
                           get_cmd=lambda: 0)
        self.add_parameter('{}_precompile'.format(awg.name), 
                           initial_value=False, 
                           label='{} precompile segments'.format(awg.name),
                           parameter_class=ManualParameter, vals=vals.Bool())
        self.add_parameter('{}_delay'.format(awg.name), initial_value=0,
                           label='{} delay'.format(awg.name), unit='s',
                           parameter_class=ManualParameter,
                           docstring="Global delay applied to this channel. "
                                     "Positive values move pulses on this "
                                     "channel forward in  time")
        self.add_parameter('{}_trigger_channels'.format(awg.name), 
                           initial_value=None,
                           label='{} trigger channels'.format(awg.name), 
                           parameter_class=ManualParameter)
        self.add_parameter('{}_active'.format(awg.name), initial_value=True,
                           label='{} active'.format(awg.name), 
                           vals=vals.Bool(),
                           parameter_class=ManualParameter)
        self.add_parameter('{}_compensation_pulse_min_length'.format(awg.name), 
                           initial_value=0, unit='s',
                           parameter_class=ManualParameter)

        for ch_nr in range(4):
            id = 'ch{}'.format(ch_nr + 1)
            name = channel_name_map.get(id, awg.name + '_' + id)
            self._awg5014_create_analog_channel_parameters(id, name, awg)
            self.channels.add(name)
            id = 'ch{}m1'.format(ch_nr + 1)
            name = channel_name_map.get(id, awg.name + '_' + id)
            self._awg5014_create_marker_channel_parameters(id, name, awg)
            self.channels.add(name)
            id = 'ch{}m2'.format(ch_nr + 1)
            name = channel_name_map.get(id, awg.name + '_' + id)
            self._awg5014_create_marker_channel_parameters(id, name, awg)
            self.channels.add(name)

    def _awg5014_create_analog_channel_parameters(self, id, name, awg):
        self.add_parameter('{}_id'.format(name), get_cmd=lambda _=id: _)
        self.add_parameter('{}_awg'.format(name), get_cmd=lambda _=awg.name: _)
        self.add_parameter('{}_type'.format(name), get_cmd=lambda: 'analog')
        self.add_parameter('{}_offset_mode'.format(name), 
                           parameter_class=ManualParameter, 
                           vals=vals.Enum('software', 'hardware'))
        offset_mode_func = self.parameters['{}_offset_mode'.format(name)]
        self.add_parameter('{}_offset'.format(name),
                           label='{} offset'.format(name), unit='V',
                           set_cmd=self._awg5014_setter(awg, id, 'offset', 
                                                        offset_mode_func),
                           get_cmd=self._awg5014_getter(awg, id, 'offset', 
                                                        offset_mode_func),
                           vals=vals.Numbers())
        self.add_parameter('{}_amp'.format(name),
                            label='{} amplitude'.format(name), unit='V',
                            set_cmd=self._awg5014_setter(awg, id, 'amp'),
                            get_cmd=self._awg5014_getter(awg, id, 'amp'),
                            vals=vals.Numbers(0.01, 2.25))
        self.add_parameter('{}_distortion'.format(name),
                            label='{} distortion mode'.format(name),
                            initial_value='off',
                            vals=vals.Enum('off', 'precalculate'),
                            parameter_class=ManualParameter)
        self.add_parameter('{}_distortion_dict'.format(name),
                            label='{} distortion dictionary'.format(name),
                            vals=vals.Dict(),
                            parameter_class=ManualParameter)
        self.add_parameter('{}_charge_buildup_compensation'.format(name),
                            parameter_class=ManualParameter,
                            vals=vals.Bool(), initial_value=False)
        self.add_parameter('{}_compensation_pulse_scale'.format(name),
                            parameter_class=ManualParameter,
                            vals=vals.Numbers(0., 1.), initial_value=0.5)
        self.add_parameter('{}_compensation_pulse_delay'.format(name), 
                           initial_value=0, unit='s',
                           parameter_class=ManualParameter)
    
    def _awg5014_create_marker_channel_parameters(self, id, name, awg):
        self.add_parameter('{}_id'.format(name), get_cmd=lambda _=id: _)
        self.add_parameter('{}_awg'.format(name), get_cmd=lambda _=awg.name: _)
        self.add_parameter('{}_type'.format(name), get_cmd=lambda: 'marker')
        self.add_parameter('{}_offset'.format(name),
                           label='{} offset'.format(name), unit='V',
                           set_cmd=self._awg5014_setter(awg, id, 'offset'),
                           get_cmd=self._awg5014_getter(awg, id, 'offset'),
                           vals=vals.Numbers(-2.7, 2.7))
        self.add_parameter('{}_amp'.format(name),
                            label='{} amplitude'.format(name), unit='V',
                            set_cmd=self._awg5014_setter(awg, id, 'amp'),
                            get_cmd=self._awg5014_getter(awg, id, 'amp'),
                            vals=vals.Numbers(-5.4, 5.4))

    @staticmethod
    def _awg5014_setter(obj, id, par, offset_mode_func=None):
        if id in ['ch1', 'ch2', 'ch3', 'ch4']:
            if par == 'offset':
                def s(val):
                    if offset_mode_func() == 'software':
                        obj.set('{}_offset'.format(id), val)
                    elif offset_mode_func() == 'hardware':
                        obj.set('{}_DC_out'.format(id), val)
                    else:
                        raise ValueError('Invalid offset mode for AWG5014: '
                                        '{}'.format(offset_mode_func()))
            elif par == 'amp':
                def s(val):
                    obj.set('{}_amp'.format(id), 2*val)
            else:
                raise NotImplementedError('Unknown parameter {}'.format(par))
        else:
            id_raw = id[:3] + '_' + id[3:]  # convert ch1m1 to ch1_m1
            if par == 'offset':
                def s(val):
                    h = obj.get('{}_high'.format(id_raw))
                    l = obj.get('{}_low'.format(id_raw))
                    obj.set('{}_high'.format(id_raw), val + h - l)
                    obj.set('{}_low'.format(id_raw), val)
            elif par == 'amp':
                def s(val):
                    l = obj.get('{}_low'.format(id_raw))
                    obj.set('{}_high'.format(id_raw), l + val)
            else:
                raise NotImplementedError('Unknown parameter {}'.format(par))
        return s

    def _awg5014_getter(self, obj, id, par, offset_mode_func=None):
        if id in ['ch1', 'ch2', 'ch3', 'ch4']:
            if par == 'offset':
                def g():
                    if offset_mode_func() == 'software':
                        return obj.get('{}_offset'.format(id))
                    elif offset_mode_func() == 'hardware':
                        return obj.get('{}_DC_out'.format(id))
                    else:
                        raise ValueError('Invalid offset mode for AWG5014: '
                                         '{}'.format(offset_mode_func()))
                                    
            elif par == 'amp':
                def g():
                    if self._awgs_prequeried_state:
                        return obj.parameters['{}_amp'.format(id)] \
                                   .get_latest()/2
                    else:
                        return obj.get('{}_amp'.format(id))/2
            else:
                raise NotImplementedError('Unknown parameter {}'.format(par))
        else:
            id_raw = id[:3] + '_' + id[3:]  # convert ch1m1 to ch1_m1
            if par == 'offset':
                def g():
                    return obj.get('{}_low'.format(id_raw))
            elif par == 'amp':
                def g():
                    if self._awgs_prequeried_state:
                        h = obj.get('{}_high'.format(id_raw))
                        l = obj.get('{}_low'.format(id_raw))
                    else:
                        h = obj.parameters['{}_high'.format(id_raw)]\
                            .get_latest()
                        l = obj.parameters['{}_low'.format(id_raw)]\
                            .get_latest()
                    return h - l
            else:
                raise NotImplementedError('Unknown parameter {}'.format(par))
        return g

    def _program_awg(self, obj, awg_sequence, waveforms, repeat_pattern=None):
        if not isinstance(obj, AWG5014Pulsar._supportedAWGtypes):
            return super()._program_awg(obj, awg_sequence, waveforms, repeat_pattern)

        pars = {
            'ch{}_m{}_low'.format(ch + 1, m + 1)
            for ch in range(4) for m in range(2)
        }
        pars |= {
            'ch{}_m{}_high'.format(ch + 1, m + 1)
            for ch in range(4) for m in range(2)
        }
        pars |= {
            'ch{}_offset'.format(ch + 1) for ch in range(4)
        }
        old_vals = {}
        for par in pars:
            old_vals[par] = obj.get(par)

        packed_waveforms = {}
        wfname_l = []

        grp_has_waveforms = {f'ch{i+1}': False for i in range(4)}

        for element in awg_sequence:
            if awg_sequence[element] is None:
                continue
            metadata = awg_sequence[element].pop('metadata', {})
            if list(awg_sequence[element].keys()) != ['no_codeword']:
                raise NotImplementedError('AWG5014 sequencer does '
                                          'not support codewords!')
            chid_to_hash = awg_sequence[element]['no_codeword']

            if not any(chid_to_hash):
                continue  # no waveforms
            
            maxlen = max([len(waveforms[h]) for h in chid_to_hash.values()])
            maxlen = max(maxlen, 256)

            wfname_l.append([])
            for grp in [f'ch{i + 1}' for i in range(4)]:
                wave = (chid_to_hash.get(grp, None),
                        chid_to_hash.get(grp + 'm1', None), 
                        chid_to_hash.get(grp + 'm2', None))
                grp_has_waveforms[grp] |= (wave != (None, None, None))
                wfname = self._hash_to_wavename((maxlen, wave))
                grp_wfs = [np.pad(waveforms.get(h, [0]), 
                                  (0, maxlen - len(waveforms.get(h, [0]))), 
                                  'constant', constant_values=0) for h in wave]
                packed_waveforms[wfname] = obj.pack_waveform(*grp_wfs)
                wfname_l[-1].append(wfname)
                if any([wf[0] != 0 for wf in grp_wfs]):
                    log.warning(f'Element {element} starts with non-zero ' 
                                f'entry on {obj.name}.')

        if not any(grp_has_waveforms.values()):
            for grp in ['ch1', 'ch2', 'ch3', 'ch4']:
                obj.set('{}_state'.format(grp), grp_has_waveforms[grp])
            return None

        self.awgs_with_waveforms(obj.name)

        nrep_l = [1] * len(wfname_l)
        goto_l = [0] * len(wfname_l)
        goto_l[-1] = 1
        wait_l = [1] * len(wfname_l)
        logic_jump_l = [0] * len(wfname_l)

        filename = 'pycqed_pulsar.awg'

        awg_file = obj.generate_awg_file(packed_waveforms, np.array(wfname_l).transpose().copy(),
                                         nrep_l, wait_l, goto_l, logic_jump_l,
                                         self._awg5014_chan_cfg(obj.name))
        obj.send_awg_file(filename, awg_file)
        obj.load_awg_file(filename)

        for par in pars:
            obj.set(par, old_vals[par])

        time.sleep(.1)
        # Waits for AWG to be ready
        obj.is_awg_ready()

        for grp in ['ch1', 'ch2', 'ch3', 'ch4']:
            obj.set('{}_state'.format(grp), 1*grp_has_waveforms[grp])

        hardware_offsets = 0
        for grp in ['ch1', 'ch2', 'ch3', 'ch4']:
            cname = self._id_channel(grp, obj.name)
            offset_mode = self.get('{}_offset_mode'.format(cname))
            if offset_mode == 'hardware':
                hardware_offsets = 1
            obj.DC_output(hardware_offsets)

        return awg_file

    def _is_awg_running(self, obj):
        if not isinstance(obj, AWG5014Pulsar._supportedAWGtypes):
            return super()._is_awg_running(obj)

        return obj.get_state() != 'Idle'

    def _clock(self, obj, cid=None):
        if not isinstance(obj, AWG5014Pulsar._supportedAWGtypes):
            return super()._clock(obj, cid)
        return obj.clock_freq()

    @staticmethod
    def _awg5014_group_ids(cid):
        """
        Returns all id-s corresponding to a single channel group.
        For example `Pulsar._awg5014_group_ids('ch2')` returns `['ch2',
        'ch2m1', 'ch2m2']`.

        Args:
            cid: An id of one of the AWG5014 channels.

        Returns: A list of id-s corresponding to the same group as `cid`.
        """
        return [cid[:3], cid[:3] + 'm1', cid[:3] + 'm2'] 

    def _awg5014_chan_cfg(self, awg):
        channel_cfg = {}
        for channel in self.channels:
            if self.get('{}_awg'.format(channel)) != awg:
                continue
            cid = self.get('{}_id'.format(channel))
            amp = self.get('{}_amp'.format(channel))
            off = self.get('{}_offset'.format(channel))
            if self.get('{}_type'.format(channel)) == 'analog':
                offset_mode = self.get('{}_offset_mode'.format(channel))
                channel_cfg['ANALOG_METHOD_' + cid[2]] = 1
                channel_cfg['ANALOG_AMPLITUDE_' + cid[2]] = amp * 2
                if offset_mode == 'software':
                    channel_cfg['ANALOG_OFFSET_' + cid[2]] = off
                    channel_cfg['DC_OUTPUT_LEVEL_' + cid[2]] = 0
                    channel_cfg['EXTERNAL_ADD_' + cid[2]] = 0
                else:
                    channel_cfg['ANALOG_OFFSET_' + cid[2]] = 0
                    channel_cfg['DC_OUTPUT_LEVEL_' + cid[2]] = off
                    channel_cfg['EXTERNAL_ADD_' + cid[2]] = 1
            else:
                channel_cfg['MARKER1_METHOD_' + cid[2]] = 2
                channel_cfg['MARKER2_METHOD_' + cid[2]] = 2
                channel_cfg['MARKER{}_LOW_{}'.format(cid[-1], cid[2])] = \
                    off
                channel_cfg['MARKER{}_HIGH_{}'.format(cid[-1], cid[2])] = \
                    off + amp
            channel_cfg['CHANNEL_STATE_' + cid[2]] = 0

        for channel in self.channels:
            if self.get('{}_awg'.format(channel)) != awg:
                continue
            if self.get('{}_active'.format(awg)):
                cid = self.get('{}_id'.format(channel))
                channel_cfg['CHANNEL_STATE_' + cid[2]] = 1
        return channel_cfg


class Pulsar(AWG5014Pulsar, HDAWG8Pulsar, UHFQCPulsar, Instrument):
    """
    A meta-instrument responsible for all communication with the AWGs.
    Contains information about all the available awg-channels in the setup.
    Starting, stopping and programming and changing the parameters of the AWGs
    should be done through Pulsar. Supports Tektronix AWG5014 and partially
    ZI UHFLI.

    Args:
        master_awg: Name of the AWG that triggers all the other AWG-s and
                    should be started last (after other AWG-s are already
                    waiting for a trigger.
    """
    def __init__(self, name='Pulsar', master_awg=None):
        super().__init__(name)

        self.add_parameter('master_awg', 
                           parameter_class=InstrumentRefParameter,
                           initial_value=master_awg)
        self.add_parameter('inter_element_spacing',
                           vals=vals.MultiType(vals.Numbers(0),
                                               vals.Enum('auto')),
                           set_cmd=self._set_inter_element_spacing,
                           get_cmd=self._get_inter_element_spacing)
        self.add_parameter('reuse_waveforms', initial_value=False,
                           parameter_class=ManualParameter, vals=vals.Bool())
                           
        self._inter_element_spacing = 'auto'
        self.channels = set() # channel names
        self.awgs = set() # AWG names
        self.last_sequence = None
        self.last_elements = None
        self._awgs_with_waveforms = set()

        self._awgs_prequeried_state = False

        self._zi_waves_cleared = False
        self._hash_to_wavename_table = {}

        Pulsar._instance = self

    @staticmethod
    def get_instance():
        return Pulsar._instance

    # channel handling
    def define_awg_channels(self, awg, channel_name_map=None):
        """
        The AWG object must be created before creating channels for that AWG

        Args:
            awg: AWG object to add to the pulsar.
            channel_name_map: A dictionary that maps channel ids to channel
                              names. (default {})
        """
        if channel_name_map is None:
            channel_name_map = {}

        for channel_name in channel_name_map.values():
            if channel_name in self.channels:
                raise KeyError("Channel named '{}' already defined".format(
                    channel_name))
        if awg.name in self.awgs:
            raise KeyError("AWG '{}' already added to pulsar".format(awg.name))

        fail = None
        super()._create_awg_parameters(awg, channel_name_map)
        # try:
        #     super()._create_awg_parameters(awg, channel_name_map)
        # except AttributeError as e:
        #     fail = e
        # if fail is not None:
        #     raise TypeError('Unsupported AWG instrument: {}. '
        #                     .format(awg.name) + str(fail))
        
        self.awgs.add(awg.name)

    def find_awg_channels(self, awg):
        channel_list = []
        for channel in self.channels:
            if self.get('{}_awg'.format(channel)) == awg:
                channel_list.append(channel)

        return channel_list

    def AWG_obj(self, **kw):
        """
        Return the AWG object corresponding to a channel or an AWG name.

        Args:
            awg: Name of the AWG Instrument.
            channel: Name of the channel

        Returns: An instance of Instrument class corresponding to the AWG
                 requested.
        """
        awg = kw.get('awg', None)
        chan = kw.get('channel', None)
        if awg is not None and chan is not None:
            raise ValueError('Both `awg` and `channel` arguments passed to '
                             'Pulsar.AWG_obj()')
        elif awg is None and chan is not None:
            name = self.get('{}_awg'.format(chan))
        elif awg is not None and chan is None:
            name = awg
        else:
            raise ValueError('Either `awg` or `channel` argument needs to be '
                             'passed to Pulsar.AWG_obj()')
        return Instrument.find_instrument(name)

    def clock(self, channel=None, awg=None):
        """
        Returns the clock rate of channel or AWG 'instrument_ref' 
        Args:
            isntrument_ref: name of the channel or AWG
        Returns: clock rate in samples per second
        """
        if channel is not None and awg is not None:
            raise ValueError('Both channel and awg arguments passed to '
                             'Pulsar.clock()')
        if channel is None and awg is None:
            raise ValueError('Neither channel nor awg arguments passed to '
                             'Pulsar.clock()')

        if channel is not None:
            awg = self.get('{}_awg'.format(channel))
     
        if self._awgs_prequeried_state:
            return self._clocks[awg]
        else:
            fail = None
            obj = self.AWG_obj(awg=awg)
            try:
                return super()._clock(obj)
            except AttributeError as e:
                fail = e
            if fail is not None:
                raise TypeError('Unsupported AWG instrument: {} of type {}. '
                                .format(obj.name, type(obj)) + str(fail))

    def active_awgs(self):
        """
        Returns:
            A set of the names of the active AWGs registered

            Inactive AWGs don't get started or stopped. Also the waveforms on
            inactive AWGs don't get updated.
        """
        return {awg for awg in self.awgs if self.get('{}_active'.format(awg))}

    def awgs_with_waveforms(self, awg=None):
        """
        Adds an awg to the set of AWGs with waveforms programmed, or returns 
        set of said AWGs.
        """
        if awg == None:
            return self._awgs_with_waveforms
        else:
            self._awgs_with_waveforms.add(awg)

    def start(self, exclude=None):
        """
        Start the active AWGs. If multiple AWGs are used in a setup where the
        slave AWGs are triggered by the master AWG, then the slave AWGs must be
        running and waiting for trigger when the master AWG is started to
        ensure synchronous playback.
        """
        if exclude is None:
            exclude = []

        # Start only the AWGs which have at least one channel programmed, i.e.
        # where at least one channel has state = 1. 
        awgs_with_waveforms = self.awgs_with_waveforms()
        used_awgs = set(self.active_awgs()) & awgs_with_waveforms
        
        for awg in used_awgs:
            self._stop_awg(awg)

        if self.master_awg() is None:
            for awg in used_awgs:
                if awg not in exclude:
                    self._start_awg(awg)
        else:
            if self.master_awg() not in exclude:
                self.master_awg.get_instr().stop()
            for awg in used_awgs:
                if awg != self.master_awg() and awg not in exclude:
                    self._start_awg(awg)
            tstart = time.time()
            for awg in used_awgs:
                if awg == self.master_awg() or awg in exclude:
                    continue
                good = False
                while not (good or time.time() > tstart + 10):
                    if self._is_awg_running(awg):
                        good = True
                    else:
                        time.sleep(0.1)
                if not good:
                    raise Exception('AWG {} did not start in 10s'
                                    .format(awg))
            if self.master_awg() not in exclude:
                self.master_awg.get_instr().start()

    def stop(self):
        """
        Stop all active AWGs.
        """

        awgs_with_waveforms = set(self.awgs_with_waveforms())
        used_awgs = set(self.active_awgs()) & awgs_with_waveforms

        for awg in used_awgs:
            self._stop_awg(awg)
    
    def program_awgs(self, sequence, awgs='all'):

        # Stores the last uploaded sequence for easy access and plotting
        self.last_sequence = sequence

        if awgs == 'all':
            awgs = self.active_awgs()

        # initializes the set of AWGs with waveforms
        self._awgs_with_waveforms -= awgs


        # prequery all AWG clock values and AWG amplitudes
        self.AWGs_prequeried(True)

        waveforms, awg_sequences = sequence.generate_waveforms_sequences()

        if self.get("{}_minimize_sequencer_memory"):
            channels_used = self._channels_in_awg_sequences(awg_sequences)
            repeat_dict = self._generate_awg_repeat_dict(sequence.repeat_patterns,
                                                         channels_used)
        else:
            repeat_dict = {}
        self._zi_waves_cleared = False
        self._hash_to_wavename_table = {}

        for awg in awgs:
            if awg in repeat_dict.keys():
                self._program_awg(self.AWG_obj(awg=awg),
                                  awg_sequences.get(awg, {}), waveforms,
                                  repeat_pattern=repeat_dict[awg])
            else:
                self._program_awg(self.AWG_obj(awg=awg),
                                  awg_sequences.get(awg, {}), waveforms)
        
        self.AWGs_prequeried(False)

    def _program_awg(self, obj, awg_sequence, waveforms, repeat_pattern=None):
        """
        Program the AWG with a sequence of segments.

        Args:
            obj: the instance of the AWG to program
            sequence: the `Sequence` object that determines the segment order,
                      repetition and trigger wait
            el_wfs: A dictionary from element name to a dictionary from channel
                    id to the waveform.
            loop: Boolean flag, whether the segments should be looped over.
                  Default is `True`.
        """
        # fail = None
        # try:
        #     super()._program_awg(obj, awg_sequence, waveforms)
        # except AttributeError as e:
        #     fail = e
        # if fail is not None:
        #     raise TypeError('Unsupported AWG instrument: {} of type {}. '
        #                     .format(obj.name, type(obj)) + str(fail))
        if repeat_pattern is not None:
            super()._program_awg(obj, awg_sequence, waveforms,
                                 repeat_pattern=repeat_pattern)
        else:
            super()._program_awg(obj, awg_sequence, waveforms)

    def _hash_to_wavename(self, h):
        alphabet = 'abcdefghijklmnopqrstuvwxyz'
        if h not in self._hash_to_wavename_table:
            hash_int = abs(hash(h))
            wname = ''.join(to_base(hash_int, len(alphabet), alphabet))[::-1]
            while wname in self._hash_to_wavename_table.values():
                hash_int += 1
                wname = ''.join(to_base(hash_int, len(alphabet), alphabet)) \
                    [::-1]
            self._hash_to_wavename_table[h] = wname
        return self._hash_to_wavename_table[h]

    def _zi_wave_definition(self, wave, defined_waves=None):
        if defined_waves is None:
            defined_waves = set()
        wave_definition = []
        w1, w2 = self._zi_waves_to_wavenames(wave)
        for analog, marker, wc in [(wave[0], wave[1], w1), 
                                   (wave[2], wave[3], w2)]:
            if analog is not None:
                wa = self._hash_to_wavename(analog)
                if wa not in defined_waves:
                    wave_definition.append(f'wave {wa} = "{wa}";')
                    defined_waves.add(wa)
            if marker is not None:        
                wm = self._hash_to_wavename(marker)
                if wm not in defined_waves:
                    wave_definition.append(f'wave {wm} = "{wm}";')
                    defined_waves.add(wm)
            if analog is not None and marker is not None:
                if wc not in defined_waves:
                    wave_definition.append(f'wave {wc} = {wa} + {wm};')
                    defined_waves.add(wc)
        return wave_definition

    def _zi_playback_string(self, device, wave, acq=False, codeword=False):
        playback_string = []
        w1, w2 = self._zi_waves_to_wavenames(wave)
        if not codeword:
            if w1 is None and w2 is not None:
                # This hack is needed due to a bug on the HDAWG. 
                # Remove this if case once the bug is fixed.
                if not acq:
                    playback_string.append(
                        f'prefetch(zeros(1) + marker(1, 0), {w2});')
            elif w1 is not None or w2 is not None:
                if not acq:
                    playback_string.append('prefetch({});'.format(', '.join(
                            [wn for wn in [w1, w2] if wn is not None])))
        playback_string.append(
            'waitDigTrigger(1{});'.format(', 1' if device == 'uhf' else ''))
        if codeword:
            playback_string.append('playWaveDIO();')
        else:
            if w1 is None and w2 is not None:
                # This hack is needed due to a bug on the HDAWG. 
                # Remove this if case once the bug is fixed.
                playback_string.append(
                    f'playWave(zeros(1) + marker(1, 0), {w2});')
            elif w1 is not None or w2 is not None:
                playback_string.append('playWave({});'.format(
                    _zi_wavename_pair_to_argument(w1, w2)))
        if acq:
            playback_string.append('setTrigger(RO_TRIG);')
            playback_string.append('setTrigger(WINT_EN);')
        return playback_string

    def _zi_codeword_table_entry(self, codeword, wave):
        w1, w2 = self._zi_waves_to_wavenames(wave)
        if w1 is None and w2 is not None:
            # This hack is needed due to a bug on the HDAWG. 
            # Remove this if case once the bug is fixed.
            return [f'setWaveDIO({codeword}, zeros(1) + marker(1, 0), {w2});']
        else:
            return ['setWaveDIO({}, {});'.format(codeword, 
                        _zi_wavename_pair_to_argument(w1, w2))]

    def _zi_waves_to_wavenames(self, wave):
        wavenames = []
        for analog, marker in [(wave[0], wave[1]), (wave[2], wave[3])]:
            if analog is None and marker is None:
                wavenames.append(None)
            elif analog is None and marker is not None:
                wavenames.append(self._hash_to_wavename(marker))
            elif analog is not None and marker is None:
                wavenames.append(self._hash_to_wavename(analog))
            else:
                wavenames.append(self._hash_to_wavename((analog, marker)))
        return wavenames

    def _zi_write_waves(self, waveforms):
        wave_dir = _zi_wave_dir()
        for h, wf in waveforms.items():
            filename = os.path.join(wave_dir, self._hash_to_wavename(h)+'.csv')
            fmt = '%.18e' if wf.dtype == np.float else '%d'
            np.savetxt(filename, wf, delimiter=",", fmt=fmt)

    def _start_awg(self, awg):
        obj = self.AWG_obj(awg=awg)
        obj.start()

    def _stop_awg(self, awg):
        obj = self.AWG_obj(awg=awg)
        obj.stop()

    def _is_awg_running(self, awg):
        fail = None
        obj = self.AWG_obj(awg=awg)
        try:
            return super()._is_awg_running(obj)
        except AttributeError as e:
            fail = e
        if fail is not None:
            raise TypeError('Unsupported AWG instrument: {} of type {}. '
                            .format(obj.name, type(obj)) + str(fail))

    def _set_inter_element_spacing(self, val):
        self._inter_element_spacing = val

    def _get_inter_element_spacing(self):
        if self._inter_element_spacing != 'auto':
            return self._inter_element_spacing
        else:
            max_spacing = 0
            for awg in self.awgs:
                max_spacing = max(max_spacing, self.get(
                    '{}_inter_element_deadtime'.format(awg)))
            return max_spacing

    def AWGs_prequeried(self, status=None):
        if status is None:
            return self._awgs_prequeried_state
        elif status:
            self._awgs_prequeried_state = False
            self._clocks = {}
            for awg in self.awgs:
                self._clocks[awg] = self.clock(awg=awg)
            for c in self.channels:
                # prequery also the output amplitude values
                self.get(c + '_amp')
            self._awgs_prequeried_state = True
        else:
            self._awgs_prequeried_state = False

    def _id_channel(self, cid, awg):
        """
        Returns the channel name corresponding to the channel with id `cid` on
        the AWG `awg`.

        Args:
            cid: An id of one of the channels.
            awg: The name of the AWG.

        Returns: The corresponding channel name. If the channel is not found,
                 returns `None`.
        """
        for cname in self.channels:
            if self.get('{}_awg'.format(cname)) == awg and \
               self.get('{}_id'.format(cname)) == cid:
                return cname
        return None

    @staticmethod
    def _channels_in_awg_sequences(awg_sequences):
        """
        identifies all channels used in the given awg keyed sequence
        :param awg_sequences (dict): awg sequences keyed by awg name, i.e. as
        returned by sequence.generate_sequence_waveforms()
        :return: dictionary keyed by awg of with all channel used during the sequence
        """
        channels_used = dict()
        for awg in awg_sequences:
            channels_used[awg] = []
            for segname in awg_sequences[awg]:
                if awg_sequences[awg][segname] is None:
                    pass
                else:
                    elements = awg_sequences[awg][segname]
                    for cw in elements:
                        if cw == "metadata":
                            pass
                        else:
                            [channels_used[awg].append(ch) for ch in elements[cw]
                             if ch not in channels_used[awg]]
        return channels_used

    def _generate_awg_repeat_dict(self, repeat_dict_per_ch, channels_used):
        """
        Translates a repeat dictionary keyed by channels to a repeat dictionary
        keyed by awg. Checks whether all channels in channels_used have an entry.
        :param repeat_dict_per_ch: keys: channels_id, values: repeat pattern
        :param channels_used (dict): list of channel used on each awg
        :return:
        """
        awg_ch_repeat_dict = dict()
        repeat_dict_per_awg = dict()
        for cname in repeat_dict_per_ch:
            awg = self.get(f"{cname}_awg")
            chid = self.get(f"{cname}_id")

            if not awg in awg_ch_repeat_dict.keys():
                awg_ch_repeat_dict[awg] = []
            awg_ch_repeat_dict[awg].append(chid)
            if repeat_dict_per_awg.get(awg, repeat_dict_per_ch[cname]) \
                    != repeat_dict_per_ch[cname]:
                raise NotImplementedError(f"Repeat pattern on {cname} is "
                f"different from at least one other channel on {awg}:"
                f"{repeat_dict_per_ch[cname]} vs {repeat_dict_per_awg[awg]}")
            repeat_dict_per_awg[awg] = repeat_dict_per_ch[cname]
            
        for awg_repeat, chs_repeat in awg_ch_repeat_dict.items():
            for ch in channels_used[awg_repeat]:
                assert ch in chs_repeat, f"Repeat pattern " \
                    f"provided for {awg_repeat} but no pattern was given on " \
                    f"{ch}. All used channels on the same awg must have a " \
                    f"repeat pattern."

        return repeat_dict_per_awg


def to_base(n, b, alphabet=None, prev=None):
    if prev is None: prev = []
    if n == 0: 
        if alphabet is None: return prev
        else: return [alphabet[i] for i in prev]
    return to_base(n//b, b, alphabet, prev+[n%b])

def _zi_wave_dir():
    if os.name == 'nt':
        dll = ctypes.windll.shell32
        buf = ctypes.create_unicode_buffer(ctypes.wintypes.MAX_PATH + 1)
        if dll.SHGetSpecialFolderPathW(None, buf, 0x0005, False):
            _basedir = buf.value
        else:
            log.warning('Could not extract my documents folder')
    else:
        _basedir = os.path.expanduser('~')
    return os.path.join(_basedir, 'Zurich Instruments', 'LabOne', 
        'WebServer', 'awg', 'waves')

def _zi_clear_waves():
    wave_dir = _zi_wave_dir()
    for f in os.listdir(wave_dir):
        if f.endswith(".csv"):
            os.remove(os.path.join(wave_dir, f))
        elif f.endswith('.cache'):
            shutil.rmtree(os.path.join(wave_dir, f))

def _zi_wavename_pair_to_argument(w1, w2):
    if w1 is not None and w2 is not None:
        return f'{w1}, {w2}'
    elif w1 is not None and w2 is None:
        return f'1, {w1}'
    elif w1 is None and w2 is not None:
        return f'2, {w2}'
    else:
        return ''