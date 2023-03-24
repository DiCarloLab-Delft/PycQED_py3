



"""
    File:       QWG.py
    Author:     Wouter Vlothuizen, TNO/QuTech,
                edited by Adriaan Rol, Gerco Versloot
    Purpose:    QCoDeS instrument driver for Qutech QWG
    Usage:
    Notes:      Must use QWGCore.py to write SCPI syntax to QWG
                This file replaces QuTech_AWG_Module.py
                It is possible to view the QWG log using ssh. To do this:
                - connect using ssh e.g., "ssh root@192.168.0.10"
                - view log using "tail -f /var/log/qwg.log"
    Bugs:       - requires QWG software version > 1.5.0, which isn't officially released yet
"""

import numpy as np
import logging
from typing import List, Tuple

from .QWGCore import QWGCore
import pycqed.instrument_drivers.library.DIO as DIO
from pycqed.instrument_drivers.library.Transport import Transport

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
try:
    from qcodes.instrument.parameter import Command
except ImportError:
    from qcodes.parameters.command import Command
from qcodes import validators as vals

log = logging.getLogger(__name__)

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
_run_mode_doc = '''
Run mode:\n
\t- NONE: No mode selected (default)\n
\t- CODeword: Codeword mode, will play wave based on codewords input via IORearDIO or IORearMT board\n
\t- CONt: Continuous mode, plays defined wave back to back\n
\t- SEQ: (Not implemented)'''

_dio_mode_doc = '''
Get or set the DIO input operation mode\n
\tOptions:\n
\t- MASTER: Use DIO codeword (lower 14 bits) input from its own IORearDIO board (Default)\n
\t\tEnables single-ended (SE) and differential (DIFF) inputs\n
\t- SLAVE: Use DIO codeword (upper 14 bits) input from the connected master IORearDIO board\n
\t\tDisables single-ended (SE) and differential (DIFF) inputs'''

# FIXME: modes outdated:
_codeword_protocol_doc = '''
Configures the codeword input bits/channels per channel. These are predefined sets of bit maps.\n 
\tOptions:\n
\t- MICROWAVE: bit map preset for microwave (Default)\n
\t- FLUX: bit map preset for flux\n
\tNote: at the moment the presets are created for CCL use which only allows calibration of
8 bits, the QWG can support up to 14 bits of which 10 are selectable'''


##########################################################################
# class
##########################################################################

class QWG(QWGCore, Instrument):
    def __init__(self,
                 name: str,
                 transport: Transport
                 ) -> None:
        super().__init__(name, transport)  # calls CCCore
        Instrument.__init__(self, name)  # calls Instrument

        # validator values
        self._dev_desc.mvals_trigger_impedance = vals.Enum(50),
        self._dev_desc.mvals_trigger_level = vals.Numbers(0, 5.0)

        self._add_parameters()
#        self.connect_message()

    ##########################################################################
    # QCoDeS parameter definitions: AWG related
    ##########################################################################

    def _add_awg_parameters(self):
        # Channel pair parameters
        for i in range(self._dev_desc.numChannels//2):
            ch_pair = i*2+1
            self.add_parameter(
                f'ch_pair{ch_pair}_sideband_frequency',
                parameter_class=HandshakeParameter,
                unit='Hz',
                label=('Sideband frequency channel pair {} (Hz)'.format(i)),
                get_cmd=_gen_get_func_1par(self.get_sideband_frequency, ch_pair),
                set_cmd=_gen_set_func_1par(self.set_sideband_frequency, ch_pair),
                vals=vals.Numbers(-300e6, 300e6),
                get_parser=float,
                docstring='Frequency of the sideband modulator\n'
                          'Resolution: ~0.23 Hz')

            self.add_parameter(
                f'ch_pair{ch_pair}_sideband_phase',
                parameter_class=HandshakeParameter,
                unit='deg',
                label=('Sideband phase channel pair {} (deg)'.format(i)),
                get_cmd=_gen_get_func_1par(self.get_sideband_phase, ch_pair),
                set_cmd=_gen_set_func_1par(self.set_sideband_phase, ch_pair),
                vals=vals.Numbers(-180, 360),
                get_parser=float,
                docstring='Sideband phase difference between channels')

            self.add_parameter(
                f'ch_pair{ch_pair}_transform_matrix',
                parameter_class=HandshakeParameter,
                unit='%',
                label=('Transformation matrix channel pair {}'.format(i)),
                get_cmd=_gen_get_func_1par(self._get_matrix, ch_pair),
                set_cmd=_gen_set_func_1par(self._set_matrix, ch_pair),
                vals=vals.Arrays(-2, 2, shape=(2, 2)),                  # NB range is not a hardware limit
                docstring='Transformation matrix for mixer correction per channel pair')

        # Channel parameters
        for ch in range(1, self._dev_desc.numChannels+1):
            self.add_parameter(
                f'ch{ch}_state',
                label=f'Output state channel {ch}',
                get_cmd=_gen_get_func_1par(self.get_output_state, ch),
                set_cmd=_gen_set_func_1par(self.set_output_state, ch),
                val_mapping={True: '1', False: '0'},
                vals=vals.Bool(),
                docstring='Enables or disables the output of channels\n'
                          'Default: Disabled')

            self.add_parameter(
                f'ch{ch}_amp',
                parameter_class=HandshakeParameter,
                label=f'Channel {ch} Amplitude ',
                unit='Vpp',
                get_cmd=_gen_get_func_1par(self.get_amplitude, ch),
                set_cmd=_gen_set_func_1par(self.set_amplitude, ch),
                vals=vals.Numbers(-1.6, 1.6),
                get_parser=float,
                docstring=f'Amplitude channel {ch} (Vpp into 50 Ohm)')

            self.add_parameter(
                f'ch{ch}_offset',
                # parameter_class=HandshakeParameter, FIXME: was commented out
                label=f'Offset channel {ch}',
                unit='V',
                get_cmd=_gen_get_func_1par(self.get_offset, ch),
                set_cmd=_gen_set_func_1par(self.set_offset, ch),
                vals=vals.Numbers(-.25, .25),
                get_parser=float,
                docstring=f'Offset channel {ch}')

            self.add_parameter(
                f'ch{ch}_default_waveform',
                get_cmd=_gen_get_func_1par(self.get_waveform, ch),
                set_cmd=_gen_set_func_1par(self.set_waveform, ch),
                # FIXME: docstring
                vals=vals.Strings())

            # end for(ch...

        # FIXME: enable if/when support for marker/trigger board is added again
        # Triggers parameter
        # for trigger in range(1, self._dev_desc.numTriggers+1):
        #     triglev_cmd = f'qutech:trigger{trigger}:level'
        #     # individual trigger level per trigger input:
        #     self.add_parameter(
        #         f'tr{trigger}_trigger_level',
        #         unit='V',
        #         label=f'Trigger level channel {trigger} (V)',
        #         # FIXME
        #         get_cmd=triglev_cmd + '?',
        #         set_cmd=triglev_cmd + ' {}',
        #         vals=self._dev_desc.mvals_trigger_level,
        #         get_parser=float,
        #         snapshot_exclude=True)
        #     # FIXME: docstring

        # Single parameters
        self.add_parameter(
            'run_mode',
            get_cmd=self.get_run_mode,
            set_cmd=self.set_run_mode,
            vals=vals.Enum('NONE', 'CONt', 'SEQ', 'CODeword'),
            docstring=_run_mode_doc + '\n Effective after start command')
        # NB: setting mode "CON" (valid SCPI abbreviation) reads back as "CONt"

        # FIXME: apparently not used, in favour of 'wave_ch{}_cw{:03}'
        # # Parameter for codeword per channel
        # for cw in range(self._dev_desc.numCodewords):  # FIXME: this may give 1024 parameters per channel
        #     for j in range(self._dev_desc.numChannels):
        #         ch = j+1
        #         # Codeword 0 corresponds to bitcode 0
        #         cw_cmd = 'sequence:element{:d}:waveform{:d}'.format(cw, ch)
        #         cw_param = f'codeword_{cw}_ch{ch}_waveform'
        #         self.add_parameter(
        #             cw_param,
        #             get_cmd=cw_cmd+'?',
        #             set_cmd=cw_cmd+' "{:s}"',
        #             vals=vals.Strings(),
        #             snapshot_exclude=True)

        # FIXME: replace by future SCPI status
        # self.add_parameter(
        #     'get_system_status',
        #     unit='JSON',
        #     label="System status",
        #     get_cmd='SYSTem:STAtus?',
        #     vals=vals.Strings(),
        #     get_parser=self._JSON_parser,
        #     docstring='Reads the current system status. E.q. channel '
        #               'status: on or off, overflow, underdrive.\n'
        #               'Return:\n     JSON object with system status')

    def _add_parameters(self):
        self._add_awg_parameters()
        self._add_codeword_parameters()
        self._add_dio_parameters()  # FIXME: conditional on QWG SW version?

    ##########################################################################
    # QCoDeS parameter definitions: codewords
    ##########################################################################

    def _add_codeword_parameters(self, add_extra: bool = True):
        self.add_parameter(
            'cfg_codeword_protocol',    # NB: compatible with ZI drivers
            unit='',
            label='Codeword protocol',
            get_cmd=self._get_codeword_protocol,
            set_cmd=self._set_codeword_protocol,
            #vals=vals.Enum('MICROWAVE', 'FLUX', 'MICROWAVE_NO_VSM'),
            docstring=_codeword_protocol_doc + '\nEffective immediately when sent')

        docst = 'Specifies a waveform for a specific codeword. \n' \
                'The channel number corresponds' \
                ' to the channel as indicated on the device (counting from 1).'
        for j in range(self._dev_desc.numChannels):
            for cw in range(self._dev_desc.numCodewords):
                ch = j + 1

                parname = 'wave_ch{}_cw{:03}'.format(ch, cw)
                self.add_parameter(
                    parname,
                    label='Waveform channel {} codeword {:03}'.format(ch, cw),
                    vals=vals.Arrays(min_value=-1, max_value=1),
                    get_cmd=_gen_get_func_2par(self._get_cw_waveform, ch, cw),
                    set_cmd=_gen_set_func_2par(self._set_cw_waveform, ch, cw),
#                    snapshot_exclude=True,
                    docstring=docst)

    ##########################################################################
    # QCoDeS parameter definitions: DIO
    ##########################################################################

    def _add_dio_parameters(self):
        self.add_parameter(
            'dio_mode',
            unit='',
            label='DIO input operation mode',
            get_cmd=self.get_dio_mode,
            set_cmd=self.set_dio_mode,
            vals=vals.Enum('MASTER', 'SLAVE'),
            val_mapping={'MASTER': 'MASter', 'SLAVE': 'SLAve'},
            docstring=_dio_mode_doc + '\nEffective immediately when sent')  # FIXME: no way, not a HandshakeParameter

        # FIXME: handle through SCPI status (once implemented on QWG)
        self.add_parameter(
            'dio_is_calibrated',
            unit='',
            label='DIO calibration status',
            get_cmd='DIO:CALibrate?',
            val_mapping={True: '1', False: '0'},
            docstring='Get DIO calibration status\n'
                      'Result:\n'
                      '\tTrue: DIO is calibrated\n'
                      '\tFalse: DIO is not calibrated'
        )

        self.add_parameter(
            'dio_active_index',
            unit='',
            label='DIO calibration index',
            get_cmd=self.get_dio_active_index,
            set_cmd=self.set_dio_active_index,
            get_parser=np.uint32,
            vals=vals.Ints(0, 20),
            docstring='Get and set DIO calibration index\n'
                      'See dio_calibrate() parameter\n'
                      'Effective immediately when sent'  # FIXME: no way, not a HandshakeParameter
        )

    ##########################################################################
    # QCoDeS parameter helpers
    ##########################################################################

    def _set_cw_waveform(self, ch: int, cw: int, waveform):
        wf_name = 'wave_ch{}_cw{:03}'.format(ch, cw)
        cw_cmd = 'sequence:element{:d}:waveform{:d}'.format(cw, ch)
        self.create_waveform_real(wf_name, waveform)
        self._transport.write(cw_cmd + ' "{:s}"'.format(wf_name))   # FIXME: move to QWGCore

    def _get_cw_waveform(self, ch: int, cw: int):
        wf_name = 'wave_ch{}_cw{:03}'.format(ch, cw)
        return self.get_waveform_data_float(wf_name)

    def _set_matrix(self, chPair, mat):
        """
        Args:
            chPair(int): channel pair for operation, 1 or 3

            matrix(np.matrix): 2x2 matrix for mixer calibration
        """
        # function used internally for the parameters because of formatting
        self._transport.write('qutech:output{:d}:matrix {:f},{:f},{:f},{:f}'.format(  # FIXME: move to QWGCore
            chPair, mat[0, 0], mat[1, 0], mat[0, 1], mat[1, 1]))

    def _get_matrix(self, chPair):
        # function used internally for the parameters because of formatting
        mstring = self._ask(f'qutech:output{chPair}:matrix?')
        M = np.zeros(4)
        for i, x in enumerate(mstring.split(',')):
            M[i] = x
        M = M.reshape(2, 2, order='F')
        return (M)

    def _set_codeword_protocol(self, protocol_name):
        """
        Args:
            protocol_name(string): Name of the predefined protocol
        """
        # function used internally for the parameters because of formatting
        protocol = self._codeword_protocol.get(protocol_name)
        if protocol is None:
            allowed_protocols = ", ".join(f'{protocol_name}' for protocols_name in self._codeword_protocol)
            raise ValueError(f"Invalid protocol: actual: {protocol_name}, expected: {allowed_protocols}")

        for ch, bit_map in enumerate(protocol):
            self._set_bit_map(ch, bit_map)

    def _get_codeword_protocol(self):
        channels_bit_maps = []
        result = "Custom"  # Default, if no protocol matches
        for ch in range(1, self._dev_desc.numChannels + 1):
            channels_bit_maps.append(list(map(int, self.get(f"ch{ch}_bit_map"))))   # FIXME: ch{}bitmap was removed

        for prtc_name, prtc_bit_map in self._codeword_protocol.items():
            if channels_bit_maps == prtc_bit_map:
                result = prtc_name
                break

        return result

    def _set_bit_map(self, ch: int, bit_map: List[int]):
        """
        Helper function to set a bitMap
        :param ch:  int, channel of the bitmap
        :param bit_map:  array of ints, element determines the codeword input
        :return: none
        """
        # FIXME: leave checking to QWG
        if len(bit_map) > self._dev_desc.numSelectCwInputs:
            raise ValueError(f'Cannot set bit map; Number of codeword bits inputs are too high; '
                             f'max: {self._dev_desc.numSelectCwInputs}, actual: {len(bit_map)}')
        invalid_inputs = list(x for x in bit_map if x > (
            self._dev_desc.numMaxCwBits - 1))
        if invalid_inputs:
            err_msg = ', '.join(f"input {cw_bit_input} at index {bit_map.index(cw_bit_input) + 1}"
                                for index, cw_bit_input in enumerate(invalid_inputs))
            raise ValueError(f'Cannot set bit map; invalid codeword bit input(s); '
                             f'max: {self._dev_desc.numMaxCwBits - 1}, actual: {err_msg}')

        array_raw = ''
        if bit_map:
            array_raw = ',' + ','.join(str(x) for x in bit_map)
        self._transport.write(f"DAC{ch+1}:BITmap {len(bit_map)}{array_raw}")  # FIXME: move to QWGCore

    # def _JSON_parser(self, msg):
    #     """
    #     Converts the result of a SCPI message to a JSON.
    #
    #     msg: SCPI message where the body is a JSON
    #     return: JSON object with the data of the SCPI message
    #     """
    #     result = str(msg)[1:-1]
    #     # SCPI/visa adds additional quotes
    #     result = result.replace('\"\"', '\"')
    #     return json.loads(result)


##########################################################################
# helpers
##########################################################################


# helpers for Instrument::add_parameter.set_cmd
def _gen_set_func_1par(fun, par1):
    def set_func(val):
        return fun(par1, val)

    return set_func


def _gen_set_func_2par(fun, par1, par2):
    def set_func(val):
        return fun(par1, par2, val)

    return set_func


# helpers for Instrument::add_parameter.get_cmd
def _gen_get_func_1par(fun, par1):
    def get_func():
        return fun(par1)

    return get_func


def _gen_get_func_2par(fun, par1, par2):
    def get_func():
        return fun(par1, par2)

    return get_func

##########################################################################
# Multi device timing calibration
##########################################################################

class QWGMultiDevices(DIO.CalInterface):
    """
    QWG helper class to execute parameters/functions on multiple devices
    """
    def __init__(self, qwgs: List[QWG]) -> None:
        self.qwgs = qwgs

    @staticmethod
    def dio_calibration(cc, qwgs: List[QWG], verbose: bool = False):
        raise DeprecationWarning("calibrate_CC_dio_protocol is deprecated, use instrument_drivers.library.DIO.calibrate")

    ##########################################################################
    # overrides for CalInterface interface
    ##########################################################################

    def calibrate_dio_protocol(self, dio_mask: int, expected_sequence: List, port: int=0):
        """
        Calibrate multiple QWG using a CCLight, QCC or other CC-like device.
        First QWG will be used as base DIO calibration for all other QWGs. First QWG in the list needs to be a DIO
        master.
        On failure of calibration an exception is raised.
        Will stop all QWGs before calibration
        """

        if not self.qwgs:
            raise ValueError("Can not calibrate QWGs; No QWGs provided")

        # Stop the QWGs to make sure they don't play the codewords used to calibrate DIO (FIXME: move to DIO.py)
        for qwg in self.qwgs:
            qwg.stop()

        main_qwg = self.qwgs[0]
        if main_qwg.dio_mode() != 'MASTER':
            raise ValueError(f"First QWG ({main_qwg.name}) is not a DIO MASTER, therefore it is not possible the use it "
                             f"as base QWG for calibration of multiple QWGs.")
        main_qwg.dio_calibrate()
        main_qwg.check_errors()
        active_index = main_qwg.dio_active_index()

        for qwg in self.qwgs[1:]:
            qwg.dio_calibrate(active_index)
            qwg.check_errors()

        for qwg in self.qwgs:
            print(f'QWG ({qwg.name}) calibration report\n{qwg.dio_calibration_report()}\n')

    def output_dio_calibration_data(self, dio_mode: str, port: int=0) -> Tuple[int, List]:
        raise RuntimeError("QWGMultiDevices cannot output calibration data (because QWG cannot)")

