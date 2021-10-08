'''
File:       QuTech_AWG_Module.py
Author:     Wouter Vlothuizen, TNO/QuTech,
            edited by Adriaan Rol, Gerco Versloot
Purpose:    Instrument driver for Qutech QWG
Usage:
Notes:      It is possible to view the QWG log using ssh. To do this connect
            using ssh e.g., "ssh root@192.168.0.10"
            Logging can be enabled using "tail -f /tmpLog/qwg.log"
Bugs:
'''

from .SCPI import SCPI

import numpy as np
import struct
import json
from qcodes import validators as vals
import warnings
from typing import List


from qcodes.instrument.parameter import Parameter
from qcodes.instrument.parameter import Command


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


class QuTech_AWG_Module(SCPI):

    def __init__(self, name, address, port, **kwargs):
        super().__init__(name, address, port, **kwargs)

        # AWG properties
        self.device_descriptor = type('', (), {})()
        self.device_descriptor.model = 'QWG'
        self.device_descriptor.numChannels = 4
        self.device_descriptor.numDacBits = 12
        self.device_descriptor.numMarkersPerChannel = 2
        self.device_descriptor.numMarkers = 8
        self.device_descriptor.numTriggers = 14

        self._nr_cw_bits_cmd = "SYSTem:CODEwords:BITs?"
        self.device_descriptor.numMaxCwBits = int(self.ask(self._nr_cw_bits_cmd))

        self._nr_cw_inp_cmd = "SYSTem:CODEwords:SELect?"
        self.device_descriptor.numSelectCwInputs = int(self.ask(self._nr_cw_inp_cmd))
        self.device_descriptor.numCodewords = pow(2, self.device_descriptor.numSelectCwInputs)

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


        # FIXME: not in [V]

        self.add_parameters()
        self.connect_message()

    def add_parameters(self):
        #######################################################################
        # QWG specific
        #######################################################################

        # Channel pair parameters
        for i in range(self.device_descriptor.numChannels//2):
            ch_pair = i*2+1
            sfreq_cmd = 'qutech:output{}:frequency'.format(ch_pair)
            sph_cmd = 'qutech:output{}:phase'.format(ch_pair)
            # NB: sideband frequency has a resolution of ~0.23 Hz:
            self.add_parameter('ch_pair{}_sideband_frequency'.format(ch_pair),
                               parameter_class=HandshakeParameter,
                               unit='Hz',
                               label=('Sideband frequency channel ' +
                                      'pair {} (Hz)'.format(i)),
                               get_cmd=sfreq_cmd + '?',
                               set_cmd=sfreq_cmd + ' {}',
                               vals=vals.Numbers(-300e6, 300e6),
                               get_parser=float)
            self.add_parameter('ch_pair{}_sideband_phase'.format(ch_pair),
                               parameter_class=HandshakeParameter,
                               unit='deg',
                               label=('Sideband phase channel' +
                                      ' pair {} (deg)'.format(i)),
                               get_cmd=sph_cmd + '?',
                               set_cmd=sph_cmd + ' {}',
                               vals=vals.Numbers(-180, 360),
                               get_parser=float)

            self.add_parameter('ch_pair{}_transform_matrix'.format(ch_pair),
                               parameter_class=HandshakeParameter,
                               label=('Transformation matrix channel' +
                                      'pair {}'.format(i)),
                               get_cmd=self._gen_ch_get_func(
                self._getMatrix, ch_pair),
                set_cmd=self._gen_ch_set_func(
                self._setMatrix, ch_pair),
                # NB range is not a hardware limit
                vals=vals.Arrays(-2, 2, shape=(2, 2)))

        # Triggers parameter
        for i in range(1, self.device_descriptor.numTriggers+1):
            triglev_cmd = 'qutech:trigger{}:level'.format(i)
            # individual trigger level per trigger input:
            self.add_parameter('tr{}_trigger_level'.format(i),
                               unit='V',
                               label='Trigger level channel {} (V)'.format(i),
                               get_cmd=triglev_cmd + '?',
                               set_cmd=triglev_cmd + ' {}',
                               vals=self.device_descriptor.mvals_trigger_level,
                               get_parser=float)

        self.add_parameter('run_mode',
                           get_cmd='AWGC:RMO?',
                           set_cmd='AWGC:RMO ' + '{}',
                           vals=vals.Enum('NONE', 'CONt', 'SEQ', 'CODeword'))
        # NB: setting mode "CON" (valid SCPI abbreviation) reads back as "CONt"

        self.add_parameter('dio_mode',
                           unit='',
                           label='DIO input operation mode',
                           get_cmd='DIO:MODE?',
                           set_cmd='DIO:MODE ' + '{}',
                           vals=vals.Enum('MASTER', 'SLAVE'),
                           val_mapping={'MASTER': 'MASter', 'SLAVE': 'SLAve'},
                           docstring='Get or set the DIO input operation mode\n' \
                               'Paramaters:\n' \
                               '\tMASTER: Use DIO codeword (lower 14 bits) input '\
                               'from its own IORearDIO board' \
                                   '\t\tEnables single-ended and differential inputs\n' \
                               '\tSLAVE; Use DIO codeword (upper 14 bits) input '\
                               'from the connected master IORearDIO board\n'
                                   '\t\tDisables SE and DIFF inputs\n' )

        self.add_parameter('dio_is_calibrated',
                           unit='',
                           label='DIO calibration status',
                           get_cmd='DIO:CALibrate?',
                           val_mapping={True: '1', False: '0'},
                           docstring="""Get DIO calibration status\n 
                                        Result:\n
                                       \tTrue: DIO is calibrated\n
                                       \tFalse: DIO is not calibrated"""
                           )

        self.add_parameter('dio_active_index',
                           unit='',
                           label='DIO calibration index',
                           get_cmd='DIO:INDexes:ACTive?',
                           set_cmd='DIO:INDexes:ACTive {}',
                           get_parser=np.uint32,
                           vals=vals.Ints(0, 20),
                           docstring='Get and set DIO calibration index\n' \
                               'See dio_calibrate() paramater\n'
                           )

        self.add_parameter('dio_suitable_indexes',
                           unit='',
                           label='DIO suitable indexes',
                           get_cmd='DIO:INDexes?',
                           get_parser=self._int_to_array,
                           docstring='Get DIO all suitable indexes\n' \
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
            amp_cmd = 'SOUR{}:VOLT:LEV:IMM:AMPL'.format(ch)
            offset_cmd = 'SOUR{}:VOLT:LEV:IMM:OFFS'.format(ch)
            state_cmd = 'OUTPUT{}:STATE'.format(ch)
            waveform_cmd = 'SOUR{}:WAV'.format(ch)
            output_voltage_cmd = 'QUTEch:OUTPut{}:Voltage'.format(ch)
            dac_temperature_cmd = 'STATus:DAC{}:TEMperature'.format(ch)
            gain_adjust_cmd = 'DAC{}:GAIn:DRIFt:ADJust'.format(ch)
            dac_digital_value_cmd = 'DAC{}:DIGitalvalue'.format(ch)
            # Set channel first to ensure sensible sorting of pars
            # Compatibility: 5014, QWG
            self.add_parameter('ch{}_state'.format(ch),
                               label='Status channel {}'.format(ch),
                               get_cmd=state_cmd + '?',
                               set_cmd=state_cmd + ' {}',
                               val_mapping={True: '1', False: '0'},
                               vals=vals.Bool())

            self.add_parameter(
                'ch{}_amp'.format(ch),
                parameter_class=HandshakeParameter,
                label='Channel {} Amplitude '.format(ch),
                unit='Vpp',
                docstring='Amplitude channel {} (Vpp into 50 Ohm)'.format(ch),
                get_cmd=amp_cmd + '?',
                set_cmd=amp_cmd + ' {:.6f}',
                vals=vals.Numbers(-1.6, 1.6),
                get_parser=float)

            self.add_parameter('ch{}_offset'.format(ch),
                               # parameter_class=HandshakeParameter,
                               label='Offset channel {}'.format(ch),
                               unit='V',
                               get_cmd=offset_cmd + '?',
                               set_cmd=offset_cmd + ' {:.3f}',
                               vals=vals.Numbers(-.25, .25),
                               get_parser=float)

            self.add_parameter('ch{}_default_waveform'.format(ch),
                               get_cmd=waveform_cmd+'?',
                               set_cmd=waveform_cmd+' "{}"',
                               vals=vals.Strings())

            self.add_parameter('status_dac{}_temperature'.format(ch),
                               unit='C',
                               label=('DAC {} temperature'.format(ch)),
                               get_cmd=dac_temperature_cmd + '?',
                               get_parser=float,
                               docstring='Reads the temperature of a DAC.\n' \
                                 +'Temperature measurement interval is 10 seconds\n' \
                                 +'Return:\n     float with temperature in Celsius')

            self.add_parameter('output{}_voltage'.format(ch),
                               unit='V',
                               label=('Channel {} voltage output').format(ch),
                               get_cmd=output_voltage_cmd + '?',
                               get_parser=float,
                               docstring='Reads the output voltage of a channel.\n' \
                                 +'Notes:\n    Measurement interval is 10 seconds.\n' \
                                 +'    The output voltage will only be read if the channel is disabled:\n' \
                                 +'    E.g.: qwg.chX_state(False)\n' \
                                 +'    If the channel is enabled it will return an low value: >0.1\n' \
                                 +'Return:\n   float in voltage')

            self.add_parameter('dac{}_gain_drift_adjust'.format(ch),
                               unit='',
                               label=('DAC {}, gain drift adjust').format(ch),
                               get_cmd=gain_adjust_cmd + '?',
                               set_cmd=gain_adjust_cmd + ' {}',
                               vals=vals.Ints(0, 4095),
                               get_parser=int,
                               docstring='Gain drift adjust setting of the DAC of a channel.\n' \
                                 +'Used for calibration of the DAC. Do not use to set the gain of a channel!\n' \
                                 +'Notes:\n  The gain setting is from 0 to 4095 \n' \
                                 +'    Where 0 is 0 V and 4095 is 3.3V \n' \
                                 +'Get Return:\n   Setting of the gain in interger (0 - 4095)\n'\
                                 +'Set parameter:\n   Integer: Gain of the DAC in , min: 0, max: 4095')

            self.add_parameter('_dac{}_digital_value'.format(ch),
                               unit='',
                               label=('DAC {}, set digital value').format(ch),
                               set_cmd=dac_digital_value_cmd + ' {}',
                               vals=vals.Ints(0, 4095),
                               docstring='FOR DEVELOPMENT ONLY: Set a digital value directly into the DAC\n' \
                                 +'Used for testing the DACs.\n' \
                                 +'Notes:\n\tThis command will also set the ' \
                                 +'\tinternal correction matrix (Phase and amplitude) of the channel pair to [0,0,0,0], ' \
                                 +'disabling any influence from the wave memory.' \
                                 +'This will also stop the wave the other channel of the pair!\n\n' \
                                 +'Set parameter:\n\tInteger: Value to write to the DAC, min: 0, max: 4095\n' \
                                 +'\tWhere 0 is minimal DAC scale and 4095 is maximal DAC scale \n')

            self.add_parameter('ch{}_bit_select'.format(ch),
                               unit='',
                               label=('Channel {}, set bit selection for this channel').format(ch),
                               get_cmd=self._gen_ch_get_func(
                               self._get_bit_select, ch),
                               set_cmd=self._gen_ch_set_func(
                               self._set_bit_select, ch),
                               get_parser=np.uint32,
                               docstring='Codeword bit select for a channel\n' \
                                 +'Set: \n' \
                                 +'\tParamater: Integer, the bit select\n' \
                                 +'\nWhen a bit is enabled (1) in the bitSelect, this bit is used as part of the codeword for that channel. ' \
                                 +' If a bit is disabled (0), it will be ignored.\n' \
                                 +'This can be used to control individual channels with a their own codeword.\n' \
                                 +'Note that codeword 1 will start on the first enabled bit. Bit endianness: LSB, lowest bit right \n' \
                                 +'\nExamples:\n' \
                                 +'\tCh1: 0b000011(0x03); Only the first and second bit will be used as codeword for channel 1.\n'\
                                 +'\tCh2: 0b001100(0x0C); Only the third and forth bit will be used as codeword for channel 2.\n'\
                                 +'\tCh3: 0b110000(0x30); Only the fifth and sixth bit will be used as codeword for channel 3.\n'\
                                 +'\tCh4: 0b110000(0x30); Only the fifth and sixth bit will be used as codeword for channel 4.\n'\
                                 +'The bit select of different channels are only allowed to overlap each other if their least significant bit is the same.\n' \
                                 +'So a bitSelect of ch1: 0b011, and ch2: 0b010 is not allowed. This will be checked on `start()`. Errors are reported by `getError()` or `getErrors()`.' \
                                 +'\n\n Get:\n' \
                                 +'\tResult:  Integer that represent the bit select of the channel\n')

            self.add_parameter('ch{}_bit_map'.format(ch),
                               unit='',
                               label='Channel {}, set bit map for this channel'.format(ch),
                               get_cmd=f"DAC{ch}:BITmap?",
                               set_cmd=self._gen_ch_set_func(
                                   self._set_bit_map, ch),
                               get_parser=self._int_to_array,
                               docstring='Codeword bit map for a channel\n')

            # Trigger parameters
            doc_trgs_log_inp = 'Reads the current input values on the all the trigger ' \
                        +'inputs for a channel, after the bitSelect.\nReturn:\n    uint32 where trigger 1 (T1) ' \
                        +'is on the Least significant bit (LSB), T2 on the second  ' \
                        +'bit after LSB, etc.\n\n For example, if only T3 is ' \
                        +'connected to a high signal, the return value is: ' \
                        +'4 (0b0000100)\n\n Note: To convert the return value ' \
                        +'to a readable ' \
                        +'binary output use: `print(\"{0:#010b}\".format(qwg.' \
                        +'triggers_logic_input()))`'
            self.add_parameter('ch{}_triggers_logic_input'.format(ch),
                               label='Read triggers input value',
                               get_cmd='QUTEch:TRIGgers{}:LOGIcinput?'.format(ch),
                               get_parser=np.uint32, # Did not convert to readable
                                                     # string because a uint32 is more
                                                     # usefull when other logic is needed
                               docstring=doc_trgs_log_inp)

        # Single parameters
        self.add_parameter('status_frontIO_temperature',
                           unit='C',
                           label='FrontIO temperature',
                           get_cmd='STATus:FrontIO:TEMperature?',
                           get_parser=float,
                           docstring='Reads the temperature of the frontIO.\n' \
                             +'Temperature measurement interval is 10 seconds\n' \
                             +'Return:\n     float with temperature in Celsius')

        self.add_parameter('status_fpga_temperature',
                           unit='C',
                           label=('FPGA temperature'),
                           get_cmd='STATus:FPGA:TEMperature?',
                           get_parser=int,
                           docstring='Reads the temperature of the FPGA.\n' \
                             +'Temperature measurement interval is 10 seconds\n' \
                             +'Return:\n     float with temperature in Celsius')

        # Paramater for codeword per channel
        for cw in range(self.device_descriptor.numCodewords):
            for j in range(self.device_descriptor.numChannels):
                ch = j+1
                # Codeword 0 corresponds to bitcode 0
                cw_cmd = 'sequence:element{:d}:waveform{:d}'.format(cw, ch)
                self.add_parameter('codeword_{}_ch{}_waveform'.format(cw, ch),
                                   get_cmd=cw_cmd+'?',
                                   set_cmd=cw_cmd+' "{:s}"',
                                   vals=vals.Strings())
        # Waveform parameters
        self.add_parameter('WlistSize',
                           label='Waveform list size',
                           unit='#',
                           get_cmd='wlist:size?',
                           get_parser=int)
        self.add_parameter('Wlist',
                           label='Waveform list',
                           get_cmd=self._getWlist)

        self.add_parameter('get_system_status',
                           unit='JSON',
                           label="System status",
                           get_cmd='SYSTem:STAtus?',
                           vals=vals.Strings(),
                           get_parser=self.JSON_parser,
                           docstring='Reads the current system status. E.q. channel ' \
                             +'status: on or off, overflow, underdrive.\n' \
                             +'Return:\n     JSON object with system status')

        self.add_parameter('get_max_codeword_bits',
                           unit='',
                           label=('Max codeword bits'),
                           get_cmd=self._nr_cw_bits_cmd,
                           vals=vals.Strings(),
                           get_parser=int,
                           docstring='Reads the maximal number of codeword bits for all channels')

        self.add_parameter('codeword_protocol',
                           unit='',
                           label='Codeword protocol',
                           get_cmd=self._getCodewordProtocol,
                           set_cmd=self._setCodewordProtocol,
                           vals=vals.Enum('MICROWAVE', 'FLUX'),
                           docstring='Reads the current system status. E.q. channel ' \
                             +'status: on or off, overflow, underdrive.\n' \
                             +'Return:\n     JSON object with system status')

        self._add_codeword_parameters()

        self.add_function('deleteWaveformAll',
                          call_cmd='wlist:waveform:delete all')

        doc_sSG = "Synchronize both sideband frequency" \
            + " generators, i.e. restart them with their defined phases."
        self.add_function('syncSidebandGenerators',
                          call_cmd='QUTEch:OUTPut:SYNCsideband',
                          docstring=doc_sSG)


    def stop(self):
        '''
        Shutsdown output on channels. When stoped will check for errors or overflow
        '''
        self.write('awgcontrol:stop:immediate')

        self.getErrors()

    def _add_codeword_parameters(self):
        self._params_to_skip_update = []
        docst = ('Specifies a waveform for a specific codeword. ' +
                 'The channel number corresponds' +
                 ' to the channel as indicated on the device (1 is lowest).')
        for j in range(self.device_descriptor.numChannels):
            for cw in range(self.device_descriptor.numCodewords):
                ch = j+1

                parname = 'wave_ch{}_cw{:03}'.format(ch, cw)
                self.add_parameter(
                    parname,
                    label='Waveform channel {} codeword {:03}'.format(
                        ch, cw),
                    vals=vals.Arrays(min_value=-1, max_value=1),
                    set_cmd=self._gen_ch_cw_set_func(
                        self._set_cw_waveform, ch, cw),
                    get_cmd=self._gen_ch_cw_get_func(
                        self._get_cw_waveform, ch, cw),
                    docstring=docst)
                self._params_to_skip_update.append(parname)

    def _set_cw_waveform(self, ch: int, cw: int, waveform):
        wf_name = 'wave_ch{}_cw{:03}'.format(ch, cw)
        cw_cmd = 'sequence:element{:d}:waveform{:d}'.format(cw, ch)
        self.createWaveformReal(wf_name, waveform)
        self.write(cw_cmd + ' "{:s}"'.format(wf_name))

    def _get_cw_waveform(self, ch: int, cw: int):
        wf_name = 'wave_ch{}_cw{:03}'.format(ch, cw)
        return self.getWaveformDataFloat(wf_name)

    def start(self):
        '''
        Activates output on channels with the current settings. When started this function will check for possible warnings
        '''
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
        '''
        Args:
            chPair(int): ckannel pair for operation, 1 or 3

            matrix(np.matrix): 2x2 matrix for mixer calibration
        '''
        # function used internally for the parameters because of formatting
        self.write('qutech:output{:d}:matrix {:f},{:f},{:f},{:f}'.format(
                   chPair, mat[0, 0], mat[1, 0], mat[0, 1], mat[1, 1]))

    def _getMatrix(self, chPair):
        # function used internally for the parameters because of formatting
        mstring = self.ask('qutech:output{}:matrix?'.format(chPair))
        M = np.zeros(4)
        for i, x in enumerate(mstring.split(',')):
            M[i] = x
        M = M.reshape(2, 2, order='F')
        return(M)

    def _setCodewordProtocol(self, protocol_name):
        '''
        Args:
            protocol_name(string): Name of the predefined protocol
        '''
        # function used internally for the parameters because of formatting
        protocol = self.codeword_protocols.get(protocol_name)
        if protocol is None:
            allowed_protocols = ", ".join("{}".format(protocols_name) for protocols_name in self.codeword_protocols)
            raise ValueError(f"Invalid protocol: actual: {protocol_name}, expected: {allowed_protocols}")

        for ch, bitSelect in enumerate(protocol):
            self.set("ch{}_bit_select".format(ch+1), bitSelect)

    def _getCodewordProtocol(self):
        channels_bit_sels = []
        result = "Custom"  # Default, if no protocol matches
        for ch in range(1, self.device_descriptor.numChannels + 1):
            channels_bit_sels.append(self.get("ch{}_bit_select".format(ch)))

        for prtc_name, prtc_bitSels in self.codeword_protocols.items():
            if channels_bit_sels == prtc_bitSels:
                result = prtc_name
                break

        return result

    def detect_underdrive(self, status):
        '''
        Will raise an warning if on a channel underflow is detected
        '''
        msg = []
        for channel in status["channels"]:
            if((channel["on"] == True) and (channel["underdrive"] == True)):
                msg.append("Possible wave underdrive detected on channel: {}".format(channel["id"]))
        return msg

    def getErrors(self):
        '''
        The SCPI protocol by default does not return errors. Therefore the user needs
        to ask for errors. This function retrieves all errors and will raise them.
        '''
        errNr = self.getSystemErrorCount()

        if errNr > 0:
            errMgs = []
            for i in range(errNr):
                errMgs.append(self.getError())
            raise RuntimeError(', '.join(errMgs))

    def JSON_parser(self, msg):
        '''
        Converts the result of a SCPI message to a JSON.

        msg: SCPI message where the body is a JSON
        return: JSON object with the data of the SCPI message
        '''
        result = str(msg)[1:-1]
        result = result.replace('\"\"', '\"') # SCPI/visa adds additional quotes
        return json.loads(result)

    @staticmethod
    def _int_to_array(msg):
        """
        Converst a scpi array of ints into a python int array
        :param msg: scpi result
        :return: array of ints
        """
        if msg == "\"\"":
            return []
        return msg.split(',')

    def _set_bit_select(self, ch: int, selection: int):
        bit_map = []
        if selection.bit_length() > self.device_descriptor.numMaxCwBits:
            raise ValueError(f'Cannot set bit select; highest set bit is to high; '
                             f'max: {self.device_descriptor.numMaxCwBits}, actual: {selection.bit_length()}')

        for cw_bit_input in range(0, selection.bit_length()):
            if (1 << cw_bit_input) & selection:
                bit_map.append(cw_bit_input)

        self._set_bit_map(ch, bit_map)

    def _get_bit_select(self, ch: type = int):
        result = self.ask(f"DAC{ch}:BITmap?")
        return result.split(",")

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
        invalid_inputs = list(x for x in bit_map if x > (self.device_descriptor.numMaxCwBits - 1))
        if invalid_inputs:
            err_msg = ', '.join("input {} at index {}".format(cw_bit_input, bit_map.index(cw_bit_input) + 1)
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

    ##########################################################################
    # AWG5014 functions: SEQUENCE
    ##########################################################################

    def setSeqLength(self, length):
        '''
        Args:
            length (int): 0..max. Allocates new, or trims existing sequence
        '''
        self.write('sequence:length %d' % length)

    def setSeqElemLoopInfiniteOn(self, element):
        '''
        Args:
            element(int): 1..length
        '''
        self.write('sequence:element%d:loop:infinite on' % element)

    ##########################################################################
    # AWG5014 functions: WLIST (Waveform list)
    ##########################################################################
    # def getWlistSize(self):
    #     return self.ask_int('wlist:size?')

    def _getWlistName(self, idx):
        '''
        Args:
            idx(int): 0..size-1
        '''
        return self.ask('wlist:name? %d' % idx)

    def _getWlist(self):
        '''
        NB: takes a few seconds on 5014: our fault or Tek's?
        '''
        size = self.WlistSize()
        wlist = []                                  # empty list
        for k in range(size):                       # build list of names
            wlist.append(self._getWlistName(k+1))
        return wlist

    def deleteWaveform(self, name):
        '''
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            'test'
        '''
        self.write('wlist:waveform:delete "%s"' % name)

    def getWaveformType(self, name):
        '''
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            '*Sine100'

        Returns:
            'INT' or 'REAL'
        '''
        return self.ask('wlist:waveform:type? "%s"' % name)

    def getWaveformLength(self, name):
        '''
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            '*Sine100'
        '''
        return self.ask_int('wlist:waveform:length? "%s"' % name)

    def newWaveformReal(self, name, len):
        '''
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            '*Sine100'

        NB: seems to do nothing (on Tek5014) if waveform already exists
        '''
        self.write('wlist:waveform:new "%s",%d,real' % (name, len))

    def getWaveformDataFloat(self, name):
        '''
        Args:
            name (string):  waveform name excluding double quotes, e.g.
            '*Sine100'

        Returns:
            waveform  (np.array of float): waveform data

        Compatibility: QWG
        '''
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
        hdr = 'wlist:waveform:data "{}",'.format(name)
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
            raise ValueError('Waveform length ({}) must be < {}'.format(
                             waveLen, maxWaveLen))

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

    ##########################################################################
    # Generic (i.e. at least AWG520 and AWG5014) Tektronix AWG functions
    ##########################################################################

    # Tek_AWG functions: menu Setup|Waveform/Sequence
    def loadWaveformOrSequence(self, awgFileName):
        ''' awgFileName:        name referring to AWG file system
        '''
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
