"""
File:       QuTech_AWG_Module.py
Author:     Wouter Vlothuizen, TNO/QuTech,
            edited by Adriaan Rol, Gerco Versloot
Purpose:    Instrument driver for Qutech QWG
Usage:
Notes:      It is possible to view the QWG log using ssh. To do this connect
            using ssh e.g., "ssh root@192.168.0.10"
            Logging can be enabled using "tailf /var/qwg.log"
Bugs:
"""

from .SCPI import SCPI
from qcodes.instrument.base import Instrument

import numpy as np
import struct
import json
import logging
from qcodes import validators as vals
import warnings
from qcodes.utils.helpers import full_class
from qcodes.instrument.parameter import ManualParameter
from typing import List, Sequence, Dict

from qcodes.instrument.parameter import Parameter
from qcodes.instrument.parameter import Command
import os


# Note: the HandshakeParameter is a temporary param that should be replaced
# once qcodes issue #236 is closed
class HandshakeParameter(Parameter):

    """
    If a string is specified as a set command it will append '*OPC?' and use
    instrument.ask instead of instrument.write
    """
    # pass

    def _set_set(self, set_cmd, set_parser):
        exec_str = self._instrument.ask if self._instrument else None
        if isinstance(set_cmd, str):
            set_cmd += '\n *OPC?'
        self._set = Command(arg_count=1, cmd=set_cmd, exec_str=exec_str,
                            input_parser=set_parser)

        self.has_set = set_cmd is not None


# These docstrings are both used in the QWG __init__ and for the parameters
_run_mode_doc = 'Run mode:\n' \
                '\t- NONE: No mode selected (default)\n' \
                '\t- CODeword: Codeword mode, will play wave based on codewords input' \
                'via IORearDIO or IORearMT board\n' \
                '\t- CONt: Continues mode, plays defined wave back to back\n' \
                '\t- SEQ: (Not implemented)'

_dio_mode_doc = 'Get or set the DIO input operation mode\n' \
                '\tOptions:\n' \
                '\t- MASTER: Use DIO codeword (lower 14 bits) input ' \
                'from its own IORearDIO board (Default)\n' \
                '\t\tEnables single-ended (SE) and differential (DIFF) inputs\n' \
                '\t- SLAVE: Use DIO codeword (upper 14 bits) input ' \
                'from the connected master IORearDIO board\n' \
                '\t\tDisables single-ended (SE) and differential (DIFF) inputs'

_codeword_protocol_doc = 'Configures the codeword input bits/channels per channel. These are predefined sets of ' \
                         'bit maps.\n \tOptions:\n' \
                         '\t- MICROWAVE: bit map preset for microwave (Default)\n' \
                         '\t- FLUX: bit map preset for flux\n' \
                         '\tNote: at the moment the presets are created for CCL use which only allows calibration of ' \
                         '8 bits, the QWG can support up to 14 bits of which 10 are selectable'


class QuTech_AWG_Module(SCPI):
    __doc__ = f"""
    Driver for a Qutech AWG Module (QWG) instrument. Will establish a connection to a module via ethernet.
    :param name: Name of the instrument
    :param address: Ethernet address of the device
    :param port: Device port
    :param reset: Set device to the default settings
    :param run_mode: {_run_mode_doc}
    :param dio_mode: {_dio_mode_doc}
    :param codeword_protocol: {_codeword_protocol_doc}
    :param kwargs: base class parameters (Instruments)
    """

    def __init__(self,
                 name: str,
                 address: str,
                 port: int = 5025,
                 reset: bool = False,
                 run_mode: str = None,
                 dio_mode: str = None,
                 codeword_protocol: str = None,
                 **kwargs):
        super().__init__(name, address, port, **kwargs)

        # AWG properties
        self.device_descriptor = type('', (), {})()
        self.device_descriptor.model = 'QWG'
        self.device_descriptor.numChannels = 4
        self.device_descriptor.numDacBits = 12
        self.device_descriptor.numMarkersPerChannel = 2
        self.device_descriptor.numMarkers = 8
        self.device_descriptor.numTriggers = 8

        self._nr_cw_bits_cmd = "SYSTem:CODEwords:BITs?"
        self.device_descriptor.numMaxCwBits = int(self.ask(self._nr_cw_bits_cmd))

        self._nr_cw_inp_cmd = "SYSTem:CODEwords:SELect?"
        self.device_descriptor.numSelectCwInputs = int(self.ask(self._nr_cw_inp_cmd))
        self.device_descriptor.numCodewords = pow(2, self.device_descriptor.numSelectCwInputs)

        # valid values
        self.device_descriptor.mvals_trigger_impedance = vals.Enum(50),
        self.device_descriptor.mvals_trigger_level = vals.Numbers(0, 5.0)

        # Codeword protocols: Pre-defined per channel bit maps
        cw_protocol_dio = {
            # FIXME: CCLight is limited to 8 cw bits output, QWG can have up to cw 14 bits input of which 10 are
            #  selectable
            'MICROWAVE': [[0, 1, 2, 3, 4, 5, 6, 7],  # Ch1
                          [0, 1, 2, 3, 4, 5, 6, 7],  # Ch2
                          [0, 1, 2, 3, 4, 5, 6, 7],  # Ch3
                          [0, 1, 2, 3, 4, 5, 6, 7]],  # Ch4

            'MICROWAVE_NO_VSM': [[0, 1, 2, 3, 4, 5, 6],  # Ch1
                                 [0, 1, 2, 3, 4, 5, 6],  # Ch2
                                 [7, 8, 9, 10, 11, 12, 13],  # Ch3
                                 [7, 8, 9, 10, 11, 12, 13]],  # Ch4

            'FLUX':      [[0, 1, 2],  # Ch1
                          [3, 4, 5],  # Ch2
                          [6, 7, 8],  # Ch3
                          [9, 10, 11]],  # Ch4  # See limitation/fixme; will use ch 3's bitmap
        }

        # Marker trigger protocol
        cw_protocol_mt = {
            # Name
            'MICROWAVE': [[0, 1, 2, 3, 4, 5, 6, 7],  # Ch1
                          [0, 1, 2, 3, 4, 5, 6, 7],  # Ch2
                          [0, 1, 2, 3, 4, 5, 6, 7],  # Ch3
                          [0, 1, 2, 3, 4, 5, 6, 7]],  # Ch4

            'FLUX':      [[0, 1, 2, 3, 4, 5, 6, 7],  # Ch1
                          [0, 1, 2, 3, 4, 5, 6, 7],  # Ch2
                          [0, 1, 2, 3, 4, 5, 6, 7],  # Ch3
                          [0, 1, 2, 3, 4, 5, 6, 7]],  # Ch4
        }

        if self.device_descriptor.numMaxCwBits <= 7:
            self.codeword_protocols = cw_protocol_mt
        else:
            self.codeword_protocols = cw_protocol_dio

        # FIXME: Remove when QCodes PR #1653 is merged, see PycQED_py3 issue #566
        self._params_exclude_snapshot = []

        self._params_to_skip_update = []
        self.add_parameters()
        self.connect_message()

        if reset:
            self.reset()

        if run_mode:
            self.run_mode(run_mode)

        if dio_mode:
            self.dio_mode(dio_mode)

        if codeword_protocol:
            self.codeword_protocol(codeword_protocol)

    def add_parameters(self):
        #######################################################################
        # QWG specific
        #######################################################################

        # Channel pair parameters
        for i in range(self.device_descriptor.numChannels//2):
            ch_pair = i*2+1
            sfreq_cmd = f'qutech:output{ch_pair}:frequency'
            sph_cmd = f'qutech:output{ch_pair}:phase'
            # NB: sideband frequency has a resolution of ~0.23 Hz:
            self.add_parameter(f'ch_pair{ch_pair}_sideband_frequency',
                               parameter_class=HandshakeParameter,
                               unit='Hz',
                               label=('Sideband frequency channel ' +
                                      'pair {} (Hz)'.format(i)),
                               get_cmd=sfreq_cmd + '?',
                               set_cmd=sfreq_cmd + ' {}',
                               vals=vals.Numbers(-300e6, 300e6),
                               get_parser=float,
                               docstring='Set the frequency of the sideband modulator\n'
                                         'Resolution: ~0.23 Hz\n'
                                         'Effective immediately when send')
            self.add_parameter(f'ch_pair{ch_pair}_sideband_phase',
                               parameter_class=HandshakeParameter,
                               unit='deg',
                               label=('Sideband phase channel' +
                                      ' pair {} (deg)'.format(i)),
                               get_cmd=sph_cmd + '?',
                               set_cmd=sph_cmd + ' {}',
                               vals=vals.Numbers(-180, 360),
                               get_parser=float,
                               docstring='Sideband phase differance between channels\n'
                                         'Effective immediately when send')

            self.add_parameter(f'ch_pair{ch_pair}_transform_matrix',
                               parameter_class=HandshakeParameter,
                               unit='%',
                               label=('Transformation matrix channel' +
                                      'pair {}'.format(i)),
                               get_cmd=self._gen_ch_get_func(
                                    self._getMatrix, ch_pair),
                               set_cmd=self._gen_ch_set_func(
                                    self._setMatrix, ch_pair),
                               # NB range is not a hardware limit
                               vals=vals.Arrays(-2, 2, shape=(2, 2)),
                               docstring='Q & I transformation per channel pair.\n'
                                         'Used for mixer correction\n'
                                         'Effective immediately when send')

        # Triggers parameter
        for trigger in range(1, self.device_descriptor.numTriggers+1):
            triglev_cmd = f'qutech:trigger{trigger}:level'
            triglev_name = f'tr{trigger}_trigger_level'
            # individual trigger level per trigger input:
            self.add_parameter(triglev_name,
                               unit='V',
                               label=f'Trigger level channel {trigger} (V)',
                               get_cmd=triglev_cmd + '?',
                               set_cmd=triglev_cmd + ' {}',
                               vals=self.device_descriptor.mvals_trigger_level,
                               get_parser=float,
                               snapshot_exclude=True)

            # FIXME: Remove when QCodes PR #1653 is merged, see PycQED_py3 issue #566
            self._params_exclude_snapshot.append(triglev_name)

        self.add_parameter('run_mode',
                           get_cmd='AWGC:RMO?',
                           set_cmd='AWGC:RMO ' + '{}',
                           vals=vals.Enum('NONE', 'CONt', 'SEQ', 'CODeword'),
                           docstring=_run_mode_doc + '\n Effective after start command')
        # NB: setting mode "CON" (valid SCPI abbreviation) reads back as "CONt"

        self.add_parameter('dio_mode',
                           unit='',
                           label='DIO input operation mode',
                           get_cmd='DIO:MODE?',
                           set_cmd='DIO:MODE ' + '{}',
                           vals=vals.Enum('MASTER', 'SLAVE'),
                           val_mapping={'MASTER': 'MASter', 'SLAVE': 'SLAve'},
                           docstring=_dio_mode_doc + '\nEffective immediately when send')

        self.add_parameter('dio_is_calibrated',
                           unit='',
                           label='DIO calibration status',
                           get_cmd='DIO:CALibrate?',
                           val_mapping={True: '1', False: '0'},
                           docstring='Get DIO calibration status\n'
                                     'Result:\n'
                                     '\tTrue: DIO is calibrated\n'
                                     '\tFalse: DIO is not calibrated'
                           )

        self.add_parameter('dio_active_index',
                           unit='',
                           label='DIO calibration index',
                           get_cmd='DIO:INDexes:ACTive?',
                           set_cmd='DIO:INDexes:ACTive {}',
                           get_parser=np.uint32,
                           vals=vals.Ints(0, 20),
                           docstring='Get and set DIO calibration index\n'
                                     'See dio_calibrate() parameter\n'
                                     'Effective immediately when send'
                           )

        self.add_parameter('dio_suitable_indexes',
                           unit='',
                           label='DIO suitable indexes',
                           get_cmd='DIO:INDexes?',
                           get_parser=self._int_to_array,
                           docstring='Get DIO all suitable indexes\n'
                                     '\t- The array is ordered by most preferable index first\n'
                           )

        self.add_parameter('dio_calibrated_inputs',
                           unit='',
                           label='DIO calibrated inputs',
                           get_cmd='DIO:INPutscalibrated?',
                           get_parser=int,
                           docstring='Get all DIO inputs which are calibrated\n'
                           )

        self.add_parameter('dio_lvds',
                           unit='bool',
                           label='LVDS DIO connection detected',
                           get_cmd='DIO:LVDS?',
                           val_mapping={True: '1', False: '0'},
                           docstring='Get the DIO LVDS connection status.\n'
                                     'Result:\n'
                                     '\tTrue: Cable detected\n'
                                     '\tFalse: No cable detected'
                           )

        self.add_parameter('dio_interboard',
                           unit='bool',
                           label='DIO interboard detected',
                           get_cmd='DIO:IB?',
                           val_mapping={True: '1', False: '0'},
                           docstring='Get the DIO interboard status.\n'
                                     'Result:\n'
                                     '\tTrue:  To master interboard connection detected\n'
                                     '\tFalse: No interboard connection detected'
                           )

        # Channel parameters #
        for ch in range(1, self.device_descriptor.numChannels+1):
            amp_cmd = f'SOUR{ch}:VOLT:LEV:IMM:AMPL'
            offset_cmd = f'SOUR{ch}:VOLT:LEV:IMM:OFFS'
            state_cmd = f'OUTPUT{ch}:STATE'
            waveform_cmd = f'SOUR{ch}:WAV'
            output_voltage_cmd = f'QUTEch:OUTPut{ch}:Voltage'
            dac_temperature_cmd = f'STATus:DAC{ch}:TEMperature'
            gain_adjust_cmd = f'DAC{ch}:GAIn:DRIFt:ADJust'
            dac_digital_value_cmd = f'DAC{ch}:DIGitalvalue'
            # Set channel first to ensure sensible sorting of pars
            # Compatibility: 5014, QWG
            self.add_parameter(f'ch{ch}_state',
                               label=f'Status channel {ch}',
                               get_cmd=state_cmd + '?',
                               set_cmd=state_cmd + ' {}',
                               val_mapping={True: '1', False: '0'},
                               vals=vals.Bool(),
                               docstring='Enables or disables the output of channels\n'
                                         'Default: Disabled\n'
                                         'Effective immediately when send')

            self.add_parameter(
                f'ch{ch}_amp',
                parameter_class=HandshakeParameter,
                label=f'Channel {ch} Amplitude ',
                unit='Vpp',
                docstring=f'Amplitude channel {ch} (Vpp into 50 Ohm) \n'
                          'Effective immediately when send',
                get_cmd=amp_cmd + '?',
                set_cmd=amp_cmd + ' {:.6f}',
                vals=vals.Numbers(-1.6, 1.6),
                get_parser=float)

            self.add_parameter(f'ch{ch}_offset',
                               # parameter_class=HandshakeParameter,
                               label=f'Offset channel {ch}',
                               unit='V',
                               docstring=f'Offset channel {ch}\n'
                               'Effective immediately when send',
                               get_cmd=offset_cmd + '?',
                               set_cmd=offset_cmd + ' {:.3f}',
                               vals=vals.Numbers(-.25, .25),
                               get_parser=float)

            self.add_parameter(f'ch{ch}_default_waveform',
                               get_cmd=waveform_cmd+'?',
                               set_cmd=waveform_cmd+' "{}"',
                               vals=vals.Strings())

            self.add_parameter(f'status_dac{ch}_temperature',
                               unit='C',
                               label=f'DAC {ch} temperature',
                               get_cmd=dac_temperature_cmd + '?',
                               get_parser=float,
                               docstring='Reads the temperature of a DAC.\n'
                                         'Temperature measurement interval is 10 seconds\n'
                                         'Return:\n     float with temperature in Celsius')

            self.add_parameter(f'output{ch}_voltage',
                               unit='V',
                               label=f'Channel {ch} voltage output',
                               get_cmd=output_voltage_cmd + '?',
                               get_parser=float,
                               docstring='Reads the output voltage of a channel.\n'
                                         'Notes:\n    Measurement interval is 10 seconds.\n'
                                         '    The output voltage will only be read if the channel is disabled:\n'
                                         '    E.g.: qwg.chX_state(False)\n'
                                         '    If the channel is enabled it will return an low value: >0.1\n'
                                         'Return:\n   float in voltage')

            self.add_parameter(f'dac{ch}_gain_drift_adjust',
                               unit='',
                               label=f'DAC {ch}, gain drift adjust',
                               get_cmd=gain_adjust_cmd + '?',
                               set_cmd=gain_adjust_cmd + ' {}',
                               vals=vals.Ints(0, 4095),
                               get_parser=int,
                               docstring='Gain drift adjust setting of the DAC of a channel.\n'
                                         'Used for calibration of the DAC. Do not use to set the gain of a channel!\n'
                                         'Notes:\n  The gain setting is from 0 to 4095 \n'
                                         '    Where 0 is 0 V and 4095 is 3.3V \n'
                                         'Get Return:\n   Setting of the gain in interger (0 - 4095)\n'
                                         'Set parameter:\n   Integer: Gain of the DAC in , min: 0, max: 4095')

            self.add_parameter(f'_dac{ch}_digital_value',
                               unit='',
                               label=f'DAC {ch}, set digital value',
                               set_cmd=dac_digital_value_cmd + ' {}',
                               vals=vals.Ints(0, 4095),
                               docstring='FOR DEVELOPMENT ONLY: Set a digital value directly into the DAC\n'
                                         'Used for testing the DACs.\n'
                                         'Notes:\n\tThis command will also set the '
                                         '\tinternal correction matrix (Phase and amplitude) of the channel pair '
                                         'to [0,0,0,0], '
                                         'disabling any influence from the wave memory.'
                                         'This will also stop the wave the other channel of the pair!\n\n'
                                         'Set parameter:\n\tInteger: Value to write to the DAC, min: 0, max: 4095\n'
                                         '\tWhere 0 is minimal DAC scale and 4095 is maximal DAC scale \n')

            self.add_parameter(f'ch{ch}_bit_map',
                               unit='',
                               label=f'Channel {ch}, set bit map for this channel',
                               get_cmd=f"DAC{ch}:BITmap?",
                               set_cmd=self._gen_ch_set_func(
                                   self._set_bit_map, ch),
                               get_parser=self._int_to_array,
                               docstring='Codeword bit map for a channel, 14 bits available of which 10 are '
                                         'selectable \n'
                                         'The codeword bit map specifies which bits of the codeword (coming from a '
                                         'central controller) are used for the codeword of a channel. This allows to '
                                         'split up the codeword into sections for each channel\n'
                                         'Effective immediately when send')

            # Trigger parameters
            self.add_parameter(f'ch{ch}_triggers_logic_input',
                               label='Read triggers input value',
                               get_cmd=f'QUTEch:TRIGgers{ch}:LOGIcinput?',
                               get_parser=np.uint32,  # Did not convert to readable
                                                      # string because a uint32 is more
                                                      # useful when other logic is needed
                               docstring='Reads the current input values on the all the trigger '
                                         'inputs for a channel, after the bitSelect.\nReturn:'
                                         '\n\tuint32 where rigger 1 (T1) '
                                         'is on the Least significant bit (LSB), T2 on the second  '
                                         'bit after LSB, etc.\n\n For example, if only T3 is '
                                         'connected to a high signal, the return value is: '
                                         '4 (0b0000100)\n\n Note: To convert the return value '
                                         'to a readable '
                                         'binary output use: `print(\"{0:#010b}\".format(qwg.'
                                         'triggers_logic_input()))`')

        # Single parameters
        self.add_parameter('status_frontIO_temperature',
                           unit='C',
                           label='FrontIO temperature',
                           get_cmd='STATus:FrontIO:TEMperature?',
                           get_parser=float,
                           docstring='Reads the temperature of the frontIO.\n'
                                     'Temperature measurement interval is 10 seconds\n'
                                     'Return:\n     float with temperature in Celsius')

        self.add_parameter('status_fpga_temperature',
                           unit='C',
                           label='FPGA temperature',
                           get_cmd='STATus:FPGA:TEMperature?',
                           get_parser=int,
                           docstring='Reads the temperature of the FPGA.\n'
                                     'Temperature measurement interval is 10 seconds\n'
                                     'Return:\n     float with temperature in Celsius')

        # Parameter for codeword per channel
        for cw in range(self.device_descriptor.numCodewords):
            for j in range(self.device_descriptor.numChannels):
                ch = j+1
                # Codeword 0 corresponds to bitcode 0
                cw_cmd = 'sequence:element{:d}:waveform{:d}'.format(cw, ch)
                cw_param = f'codeword_{cw}_ch{ch}_waveform'
                self.add_parameter(cw_param,
                                   get_cmd=cw_cmd+'?',
                                   set_cmd=cw_cmd+' "{:s}"',
                                   vals=vals.Strings(),
                                   snapshot_exclude=True)
                # FIXME: Remove when QCodes PR #1653 is merged, see PycQED_py3 issue #566
                self._params_exclude_snapshot.append(cw_param)

        # Waveform parameters
        self.add_parameter('WlistSize',
                           label='Waveform list size',
                           unit='#',
                           get_cmd='wlist:size?',
                           get_parser=int,
                           snapshot_exclude=True)
        # TODO: Remove when QCodes PR #1653 is merged, see PycQED_py3 issue #566
        self._params_exclude_snapshot.append('WlistSize')

        self.add_parameter('Wlist',
                           label='Waveform list',
                           get_cmd=self._getWlist,
                           snapshot_exclude=True)
        # TODO: Remove when QCodes PR #1653 is merged, see PycQED_py3 issue #566
        self._params_exclude_snapshot.append('Wlist')

        self.add_parameter('get_system_status',
                           unit='JSON',
                           label="System status",
                           get_cmd='SYSTem:STAtus?',
                           vals=vals.Strings(),
                           get_parser=self.JSON_parser,
                           docstring='Reads the current system status. E.q. channel '
                                     'status: on or off, overflow, underdrive.\n'
                                     'Return:\n     JSON object with system status')

        self.add_parameter('get_max_codeword_bits',
                           unit='',
                           label='Max codeword bits',
                           get_cmd=self._nr_cw_bits_cmd,
                           vals=vals.Strings(),
                           get_parser=int,
                           docstring='Reads the maximal number of codeword bits for all channels')

        self.add_parameter('codeword_protocol',
                           unit='',
                           label='Codeword protocol',
                           get_cmd=self._getCodewordProtocol,
                           set_cmd=self._setCodewordProtocol,
                           vals=vals.Enum('MICROWAVE', 'FLUX', 'MICROWAVE_NO_VSM'),
                           docstring=_codeword_protocol_doc + '\nEffective immediately when send')

        self._add_codeword_parameters()

        self.add_function('deleteWaveformAll',
                          call_cmd='wlist:waveform:delete all')

        self.add_function('syncSidebandGenerators',
                          call_cmd='QUTEch:OUTPut:SYNCsideband',
                          docstring='Synchronize both sideband frequency '
                                    'generators, i.e. restart them with their defined phases.\n'
                                    'Effective immediately when send')

    def stop(self):
        """
        Shutsdown output on channels. When stopped will check for errors or overflow
        """
        self.write('awgcontrol:stop:immediate')

        self.getErrors()

    def _add_codeword_parameters(self):
        docst = 'Specifies a waveform for a specific codeword. \n' \
                'The channel number corresponds' \
                ' to the channel as indicated on the device (1 is lowest).'
        for j in range(self.device_descriptor.numChannels):
            for cw in range(self.device_descriptor.numCodewords):
                ch = j+1

                parname = 'wave_ch{}_cw{:03}'.format(ch, cw)
                self.add_parameter(
                    parname,
                    label='Waveform channel {} codeword {:03}'.format(ch, cw),
                    vals=vals.Arrays(min_value=-1, max_value=1),
                    set_cmd=self._gen_ch_cw_set_func(
                        self._set_cw_waveform, ch, cw),
                    get_cmd=self._gen_ch_cw_get_func(
                        self._get_cw_waveform, ch, cw),
                    snapshot_exclude=True,
                    docstring=docst)
                # FIXME: Remove when QCodes PR #1653 is merged, see PycQED_py3 issue #566
                self._params_exclude_snapshot.append(parname)

    def _set_cw_waveform(self, ch: int, cw: int, waveform):
        wf_name = 'wave_ch{}_cw{:03}'.format(ch, cw)
        cw_cmd = 'sequence:element{:d}:waveform{:d}'.format(cw, ch)
        self.createWaveformReal(wf_name, waveform)
        self.write(cw_cmd + ' "{:s}"'.format(wf_name))

    def _get_cw_waveform(self, ch: int, cw: int):
        wf_name = 'wave_ch{}_cw{:03}'.format(ch, cw)
        return self.getWaveformDataFloat(wf_name)

    def start(self):
        """
        Activates output on channels with the current settings. When started this function will check for
        possible warnings
        """
        run_mode = self.run_mode()
        if run_mode == 'NONE':
            raise RuntimeError('No run mode is specified')
        self.write('awgcontrol:run:immediate')

        self.getErrors()

        status = self.get_system_status()
        warn_msg = self.detect_underdrive(status)

        if(len(warn_msg) > 0):
            warnings.warn(', '.join(warn_msg))

    def _setMatrix(self, chPair, mat):
        """
        Args:
            chPair(int): ckannel pair for operation, 1 or 3

            matrix(np.matrix): 2x2 matrix for mixer calibration
        """
        # function used internally for the parameters because of formatting
        self.write('qutech:output{:d}:matrix {:f},{:f},{:f},{:f}'.format(
                   chPair, mat[0, 0], mat[1, 0], mat[0, 1], mat[1, 1]))

    def _getMatrix(self, chPair):
        # function used internally for the parameters because of formatting
        mstring = self.ask(f'qutech:output{chPair}:matrix?')
        M = np.zeros(4)
        for i, x in enumerate(mstring.split(',')):
            M[i] = x
        M = M.reshape(2, 2, order='F')
        return(M)

    def _setCodewordProtocol(self, protocol_name):
        """
        Args:
            protocol_name(string): Name of the predefined protocol
        """
        # function used internally for the parameters because of formatting
        protocol = self.codeword_protocols.get(protocol_name)
        if protocol is None:
            allowed_protocols = ", ".join(f'{protocol_name}' for protocols_name in self.codeword_protocols)
            raise ValueError(f"Invalid protocol: actual: {protocol_name}, expected: {allowed_protocols}")

        for ch, bitMap in enumerate(protocol):
            self.set(f"ch{ch+1}_bit_map", bitMap)

    def _getCodewordProtocol(self):
        channels_bit_maps = []
        result = "Custom"  # Default, if no protocol matches
        for ch in range(1, self.device_descriptor.numChannels + 1):
            channels_bit_maps.append(list(map(int, self.get(f"ch{ch}_bit_map"))))

        for prtc_name, prtc_bit_map in self.codeword_protocols.items():
            if channels_bit_maps == prtc_bit_map:
                result = prtc_name
                break

        return result

    def detect_underdrive(self, status):
        """
        Will raise an warning if on a channel underflow is detected
        """
        msg = []
        for channel in status["channels"]:
            if(channel["on"] == True) and (channel["underdrive"] == True):
                msg.append(f"Possible wave underdrive detected on channel: {channel['id']}")
        return msg

    def getErrors(self):
        """
        The SCPI protocol by default does not return errors. Therefore the user needs
        to ask for errors. This function retrieves all errors and will raise them.
        """
        errNr = self.getSystemErrorCount()

        if errNr > 0:
            errMgs = []
            for i in range(errNr):
                errMgs.append(self.getError())
            raise RuntimeError(', '.join(errMgs))

    def JSON_parser(self, msg):
        """
        Converts the result of a SCPI message to a JSON.

        msg: SCPI message where the body is a JSON
        return: JSON object with the data of the SCPI message
        """
        result = str(msg)[1:-1]
        # SCPI/visa adds additional quotes
        result = result.replace('\"\"', '\"')
        return json.loads(result)

    @staticmethod
    def _int_to_array(msg):
        """
        Convert a scpi array of ints into a python int array
        :param msg: scpi result
        :return: array of ints
        """
        if msg == "\"\"":
            return []
        return msg.split(',')

    def _set_bit_map(self, ch: int, bit_map: List[int]):
        """
        Helper function to set a bitMap
        :param ch:  int, channel of the bitmap
        :param bit_map:  array of ints, element determines the codeword input
        :return: none
        """
        if len(bit_map) > self.device_descriptor.numSelectCwInputs:
            raise ValueError(f'Cannot set bit map; Number of codeword bits inputs are too high; '
                             f'max: {self.device_descriptor.numSelectCwInputs}, actual: {len(bit_map)}')
        invalid_inputs = list(x for x in bit_map if x > (
            self.device_descriptor.numMaxCwBits - 1))
        if invalid_inputs:
            err_msg = ', '.join(f"input {cw_bit_input} at index {bit_map.index(cw_bit_input) + 1}"
                                for index, cw_bit_input in enumerate(invalid_inputs))
            raise ValueError(f'Cannot set bit map; invalid codeword bit input(s); '
                             f'max: {self.device_descriptor.numMaxCwBits - 1}, actual: {err_msg}')

        array_raw = ''
        if bit_map:
            array_raw = ',' + ','.join(str(x) for x in bit_map)
        self.write(f"DAC{ch}:BITmap {len(bit_map)}{array_raw}")

    def dio_calibrate(self, target_index: int = ''):
        """
        Calibrate the DIO input signals.\n

        Will analyze the input signals for each DIO
        inputs (used to transfer codeword bits), secondly,
        the most preferable index (active index) is set.\n\n'

        Each signal is sampled and divided into sections.
        These sections are analyzed to find a stable
        stable signal. These stable sections
        are addressed by there index.\n\n

        After calibration the suitable indexes list (see dio_suitable_indexes()) contains all indexes which are stable.

        Parameters:
        :param target_index: unsigned int, optional: When provided the calibration will select an active index based
        on the target index. Used to determine the new index before or after the edge. This parameter is commonly used
        to calibrate a DIO slave where the target index is the active index after calibration of the DIO master

        Note 1: Expects a DIO calibration signal on the inputs:\n
        \tAn all codewords bits high followed by an all codeword
        bits low in a continues repetition. This results in a
        square wave of 25 MHz on the DIO inputs of the
        DIO connection. Individual DIO inputs where no
        signal is detected will not be calibrated (See
         dio_calibrated_inputs())\n\n

        Note 2: The QWG will continuously validate if
        the active index is still stable.\n\n

        Note 3: If no suitable indexes are found
        is empty and an error is pushed onto the error stack\n
        """
        self.write(f'DIO:CALibrate {target_index}')

    def dio_calibration_rapport(self, extended: bool=False) -> str:
        """
        Return a string containing the latest DIO calibration rapport (successful and failed calibrations). Includes:
        selected index, dio mode, valid indexes, calibrated DIO bits and the DIO bitDiff table.
        :param extended: Adds more information about DIO: interboard and LVDS
        :return: String of DIO calibration rapport
        """
        info = f'- Calibrated:          {self.dio_is_calibrated()}\n' \
               f'- Mode:                {self.dio_mode()}\n' \
               f'- Selected index:      {self.dio_active_index()}\n' \
               f'- Suitable indexes:    {self.dio_suitable_indexes()}\n' \
               f'- Calibrated DIO bits: {bin(self.dio_calibrated_inputs())}\n' \
               f'- DIO bit diff table:\n{self._dio_bit_diff_table()}'

        if extended:
            info += f'- LVDS detected:       {self.dio_lvds()}\n' \
                    f'- Interboard detected: {self.dio_interboard()}'

        return info

    ##########################################################################
    # AWG5014 functions: SEQUENCE
    ##########################################################################

    def setSeqLength(self, length):
        """
        Args:
            length (int): 0..max. Allocates new, or trims existing sequence
        """
        self.write('sequence:length %d' % length)

    def setSeqElemLoopInfiniteOn(self, element):
        """
        Args:
            element(int): 1..length
        """
        self.write('sequence:element%d:loop:infinite on' % element)

    ##########################################################################
    # AWG5014 functions: WLIST (Waveform list)
    ##########################################################################
    # def getWlistSize(self):
    #     return self.ask_int('wlist:size?')

    def _getWlistName(self, idx):
        """
        Args:
            idx(int): 0..size-1
        """
        return self.ask('wlist:name? %d' % idx)

    def _getWlist(self):
        """
        NB: takes a few seconds on 5014: our fault or Tek's?
        """
        size = self.WlistSize()
        wlist = []                                  # empty list
        for k in range(size):                       # build list of names
            wlist.append(self._getWlistName(k+1))
        return wlist

    def deleteWaveform(self, name):
        """
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            'test'
        """
        self.write('wlist:waveform:delete "%s"' % name)

    def getWaveformType(self, name):
        """
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            '*Sine100'

        Returns:
            'INT' or 'REAL'
        """
        return self.ask('wlist:waveform:type? "%s"' % name)

    def getWaveformLength(self, name):
        """
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            '*Sine100'
        """
        return self.ask_int('wlist:waveform:length? "%s"' % name)

    def newWaveformReal(self, name, len):
        """
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            '*Sine100'

        NB: seems to do nothing (on Tek5014) if waveform already exists
        """
        self.write('wlist:waveform:new "%s",%d,real' % (name, len))

    def getWaveformDataFloat(self, name):
        """
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            '*Sine100'

        Returns:
            waveform  (np.array of float): waveform data

        Compatibility: QWG
        """
        self.write('wlist:waveform:data? "%s"' % name)
        binBlock = self.binBlockRead()
        # extract waveform
        if 1:   # high performance
            waveform = np.frombuffer(binBlock, dtype=np.float32)
        else:   # more generic
            waveformLen = int(len(binBlock)/4)   # 4 bytes per record
            waveform = np.array(range(waveformLen), dtype=float)
            for k in range(waveformLen):
                val = struct.unpack_from('<f', binBlock, k*4)
                waveform[k] = val[0]
        return waveform

    def sendWaveformDataReal(self, name, waveform):
        """
        send waveform and markers directly to AWG memory, i.e. not to a file
        on the AWG disk.
        NB: uses real data normalized to the range from -1 to 1 (independent
        of number of DAC bits of AWG)

        Args:
            name (string): waveform name excluding double quotes, e.g. 'test'.
            Must already exits in AWG

            waveform (np.array of float)): vector defining the waveform,
            normalized between -1.0 and 1.0

        Compatibility:  QWG

        Based on:
            Tektronix_AWG5014.py::send_waveform, which sends data to an AWG
            _file_, not a memory waveform
            'awg_transferRealDataWithMarkers', Author = Stefano Poletto,
            Compatibility = Tektronix AWG5014, AWG7102
        """

        # generate the binblock
        if 1:   # high performance
            arr = np.asarray(waveform, dtype=np.float32)
            binBlock = arr.tobytes()
        else:   # more generic
            binBlock = b''
            for i in range(len(waveform)):
                binBlock = binBlock + struct.pack('<f', waveform[i])

        # write binblock
        hdr = f'wlist:waveform:data "{name}",'
        self.binBlockWrite(binBlock, hdr)

    def createWaveformReal(self, name, waveform):
        """
        Convenience function to create a waveform in the AWG and then send
        data to it

        Args:
            name(string): name of waveform for internal use by the AWG

            waveform (float[numpoints]): vector defining the waveform,
            normalized between -1.0 and 1.0


        Compatibility:  QWG
        """
        wv_val = vals.Arrays(min_value=-1, max_value=1)
        wv_val.validate(waveform)

        maxWaveLen = 2**17-4  # FIXME: this is the hardware max

        waveLen = len(waveform)
        if waveLen > maxWaveLen:
            raise ValueError(f'Waveform length ({waveLen}) must be < {maxWaveLen}')

        self.newWaveformReal(name, waveLen)
        self.sendWaveformDataReal(name, waveform)

    def _dio_bit_diff_table(self):
        """
        FOR DEVELOPMENT ONLY: Get the bit diff table of the last calibration
        :return: String of the bitDiff table
        """
        return self.ask("DIO:BDT").replace("\"", '').replace(",", "\n")

    def _dio_calibrate_param(self, meas_time: float, nr_itr: int, target_index: int = ""):
        """
        FOR DEVELOPMENT ONLY: Calibrate the DIO input signals with extra arguments.\n
        Parameters:
        \t meas_time: Measurement time between indexes in seconds, resolution of 1e-6 s
        \tNote that when select a measurement time longer than 25e-2 S the scpi connection
        will timeout, but the calibration is than still running. The timeout will happen on the
        first `get` parameter after this call\n
        \tnr_itr: Number of DIO signal data (bitDiffs) gathering iterations\n
        \ttarget_index: DIO index which determines on which side of the edge to select the active index from\n
        Calibration duration = meas_time * nr_itr * 20 * 1.1 (10% to compensate for log printing time)\n
        """
        if meas_time < 1e-6:
            raise ValueError(f"Cannot calibration inputs: meas time is too low; min 1e-6, actual: {meas_time}")

        if nr_itr < 1:
            raise ValueError(f"Cannot calibration inputs: nr_itr needs to be positive; actual: {nr_itr}")

        if target_index is not "":
            target_index = f",{target_index}"

        self.write(f'DIO:CALibrate:PARam {meas_time},{nr_itr}{target_index}')

    def _dio_set_signals(self, signals: List):
        """
        FOR DEVELOPMENT ONLY: Set DIO simulation signals. Only works if kernel module and application are build with the
        'OPT_DBG_SIM_DIO' define enabled
        :param signals: list of unsigned int, the signal on a input (note: not the bitDiffs, the system will calculate
        the bitDiffs). Need to contain 16 elements. Example: signal[0]=0xFFF00 where LSB is the oldest in time
        :return: None
        """

        if not len(signals) == 16:
            raise ValueError(f"Invalid number of DIO signals; expected 16, actual: {len(signals)}")
        self.write("DIO:DBG:SIG {}".format(','.join(map(str, signals))))

    def snapshot_base(self, update=False,
                      params_to_skip_update: Sequence[str] = None,
                      params_to_exclude: Sequence[str] = None) -> Dict:
        """
        State of the instrument as a JSON-compatible dict.

        Args:
            update: If True, update the state by querying the
                instrument. If False, just use the latest values in memory.
            params_to_skip_update: List of parameter names that will be skipped
                in update even if update is True. This is useful if you have
                parameters that are slow to update but can be updated in a
                different way (as in the qdac)
            params_to_exclude: List of parameter names that will be excluded from the snapshot

        Returns:
            dict: base snapshot
        """

        if params_to_skip_update is None:
            params_to_skip_update = self._params_to_skip_update

        # FIXME: Enable when QCodes PR #1653 is merged, see PycQED_py3 issue #566
        # snap = super().snapshot_base(update=update,
        #                              params_to_skip_update=params_to_skip_update)
        # return snap

        # FIXME: Workaround, remove when QCodes PR #1653 is merged, see PycQED_py3 issue #566
        if params_to_exclude is None:
            params_to_exclude = self._params_exclude_snapshot
        #
        snap = {
            "functions": {name: func.snapshot(update=update)
                          for name, func in self.functions.items()},
            "submodules": {name: subm.snapshot(update=update)
                           for name, subm in self.submodules.items()},
            "__class__": full_class(self)
        }

        snap['parameters'] = {}
        for name, param in self.parameters.items():
            if params_to_exclude and name in params_to_exclude:
                continue
            if params_to_skip_update and name in params_to_skip_update:
                update_par = False
            else:
                update_par = update

            try:
                snap['parameters'][name] = param.snapshot(update=update_par)
            except:
                # really log this twice. Once verbose for the UI and once
                # at lower level with more info for file based loggers
                logging.info(
                    "Snapshot: Could not update parameter: {}".format(name))
                self.log.info(f"Details for Snapshot:",
                              exc_info=True)
                snap['parameters'][name] = param.snapshot(update=False)

        for attr in set(self._meta_attrs):
            if hasattr(self, attr):
                snap[attr] = getattr(self, attr)
        snap['port'] = self._port
        snap['confirmation'] = self._confirmation
        snap['address'] = self._address
        snap['terminator'] = self._terminator
        snap['timeout'] = self._timeout
        snap['persistent'] = self._persistent
        return snap
        # FIXME: End remove

    ##########################################################################
    # Generic (i.e. at least AWG520 and AWG5014) Tektronix AWG functions
    ##########################################################################

    # Tek_AWG functions: menu Setup|Waveform/Sequence
    def loadWaveformOrSequence(self, awgFileName):
        """
        awgFileName:        name referring to AWG file system
        """
        self.write('source:def:user "%s"' % awgFileName)
        # NB: we only  support default Mass Storage Unit Specifier "Main",
        # which is the internal harddisk

    # Used for setting the channel pairs
    def _gen_ch_set_func(self, fun, ch):
        def set_func(val):
            return fun(ch, val)
        return set_func

    def _gen_ch_get_func(self, fun, ch):
        def get_func():
            return fun(ch)
        return get_func

    def _gen_ch_cw_set_func(self, fun, ch, cw):
        def set_func(val):
            return fun(ch, cw, val)
        return set_func

    def _gen_ch_cw_get_func(self, fun, ch, cw):
        def get_func():
            return fun(ch, cw)
        return get_func


class QWGMultiDevices:
    """
    QWG helper class to execute parameters/functions on multiple devices. E.g.: DIO calibration
    Usually all methods are static
    """
    from pycqed.instrument_drivers.physical_instruments import QuTech_QCC

    @staticmethod
    def dio_calibration(cc: QuTech_QCC, qwgs: List[QuTech_AWG_Module],
            verbose: bool = False):
        """
        Calibrate multiple QWG using a CCLight
        First QWG will be used als base DIO calibration for all other QWGs. First QWG in the list needs to be a DIO
        master.
        On failure of calibration an exception is raised.
        Will stop all QWGs before calibration

        Note: Will use the QWG_DIO_Calibration.qisa, cs.txt and qisa_opcodes.qmap
        files to assemble a  calibration program for the CCLight. These files
        should be located in the _QWG subfolder in the path of this file.
        :param ccl: CCLight device, connection has to be active
        :param qwgs: List of QWG which will be calibrated, all QWGs are expected to have an active connection
        :param verbose: Print the DIO calibration rapport of all QWGs
        :return: None
        """
        # The CCL will start sending codewords to calibrate. To make sure the QWGs will not play waves a stop is send
        for qwg in qwgs:
            qwg.stop()

        if not cc:
            raise ValueError("Cannot calibrate QWGs; No CC provided")

        if cc.ask("QUTech:RUN?") == '1':
            cc.stop()

        _qwg_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '_QWG'))


        qisa_qwg_dio_calibrate = os.path.join(_qwg_path,
            'QCC_DIO_Calibration.qisa')

        cs_qwg_dio_calibrate = os.path.join(_qwg_path, 'qcc_cs.txt')

        qisa_opcode_qwg_dio_calibrate = os.path.join(_qwg_path,
            'qcc_qisa_opcodes.qmap')

        old_cs = cc.control_store()
        old_qisa_opcode = cc.qisa_opcode()

        cc.control_store(cs_qwg_dio_calibrate)
        cc.qisa_opcode(qisa_opcode_qwg_dio_calibrate)

        cc.eqasm_program(qisa_qwg_dio_calibrate)
        cc.start()
        cc.getOperationComplete()

        if not qwgs:
            raise ValueError("Can not calibrate QWGs; No QWGs provided")

        def try_errors(qwg):
            try:
                qwg.getErrors()
            except Exception as e:
                raise type(e)(f'{qwg.name}: {e}')

        main_qwg = qwgs[0]
        if main_qwg.dio_mode() is not 'MASTER':
            raise ValueError(f"First QWG ({main_qwg.name}) is not a DIO MASTER, therefor it is not save the use it "
                             f"as base QWG for calibration of multiple QWGs.")
        main_qwg.dio_calibrate()
        try_errors(main_qwg)
        active_index = main_qwg.dio_active_index()

        for qwg in qwgs[1:]:
            qwg.dio_calibrate(active_index)
            try_errors(qwg)
        if verbose:
            for qwg in qwgs:
                print(f'QWG ({qwg.name}) calibration rapport\n{qwg.dio_calibration_rapport()}\n')
        cc.stop()

        #Set the control store
        cc.control_store(old_cs)
        cc.qisa_opcode(old_qisa_opcode)



class Mock_QWG(QuTech_AWG_Module):
    """
    Mock QWG instrument designed to mock QWG interface for testing purposes.
    """

    def __init__(self, name, **kwargs):
        Instrument.__init__(self, name=name,  **kwargs)

        # AWG properties
        self.device_descriptor = type('', (), {})()
        self.device_descriptor.model = 'QWG'
        self.device_descriptor.numChannels = 4
        self.device_descriptor.numDacBits = 12
        self.device_descriptor.numMarkersPerChannel = 2
        self.device_descriptor.numMarkers = 8
        self.device_descriptor.numTriggers = 8

        self._nr_cw_bits_cmd = "SYSTem:CODEwords:BITs?"
        self.device_descriptor.numMaxCwBits = 32  # Some random mock val

        self.device_descriptor.numSelectCwInputs = 10  # mock val based on DIO
        self.device_descriptor.numCodewords = pow(2, 5)  # Some random mock val

        # valid values
        self.device_descriptor.mvals_trigger_impedance = vals.Enum(50),
        self.device_descriptor.mvals_trigger_level = vals.Numbers(0, 5.0)

        cw_protocol_mt = {
            # Name          Ch1,    Ch2,    Ch3,    Ch4
            'FLUX':         [0x5F,  0x5F,   0x5F,   0x5F],
            'MICROWAVE':    [0x5F,  0x5F,   0x5F,   0x5F]
        }

        cw_protocol_dio = {
            # Name          Ch1,   Ch2,  Ch3,  Ch4
            'FLUX':         [0x07, 0x38, 0x1C0, 0xE00],
            'MICROWAVE':    [0x3FF, 0x3FF, 0x3FF, 0x3FF]
        }

        if self.device_descriptor.numMaxCwBits <= 7:
            self.codeword_protocols = cw_protocol_mt
        else:
            self.codeword_protocols = cw_protocol_dio

        # FIXME: Remove when QCodes PR #1653 is merged, see PycQED_py3 issue #566
        self._params_exclude_snapshot = []

        self._params_to_skip_update = []
        self.add_parameters()
        # self.connect_message()

    def add_parameter(self, name: str,
                      parameter_class: type=Parameter, **kwargs) -> None:

        kwargs.pop('get_cmd', None)
        kwargs.pop('set_cmd', None)
        return super().add_parameter(name=name,
                                     parameter_class=ManualParameter, **kwargs)

    def stop(self):
        pass

    def start(self):
        pass
