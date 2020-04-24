# FIXME: do not use yet, work in progress

"""
    File:       QWG.py
    Author:     Wouter Vlothuizen, TNO/QuTech,
                edited by Adriaan Rol, Gerco Versloot
    Purpose:    QCoDeS instrument driver for Qutech QWG
    Usage:
    Notes:      This file replaces QuTech_AWG_Module.py
                It is possible to view the QWG log using ssh. To do this:
                - connect using ssh e.g., "ssh root@192.168.0.10"
                - view log using "tail -f /var/log/qwg.log"
    Bugs:       - requires QWG software version > 1.5.0, which isn't officially released yet
    Todo:       - cleanup after https://github.com/QCoDeS/Qcodes/pull/1653  NB: was merged 20190807
                - cleanup after https://github.com/QCoDeS/Qcodes/issues/236  NB: looks stale
"""

import os
import numpy as np
import logging
#import json
from typing import List, Sequence, Dict

from .QWGCore import QWGCore
from pycqed.instrument_drivers.library.Transport import Transport

from qcodes.instrument.base import Instrument
from qcodes.instrument.parameter import Parameter
from qcodes.instrument.parameter import Command
from qcodes import validators as vals
from qcodes.utils.helpers import full_class

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
        super().__init__(name, transport) # calls CCCore
        Instrument.__init__(self, name) # calls Instrument

        # validator values
        self._dev_desc.mvals_trigger_impedance = vals.Enum(50),
        self._dev_desc.mvals_trigger_level = vals.Numbers(0, 5.0)

        # FIXME: Remove when QCodes PR #1653 is merged, see PycQED_py3 issue #566
        self._params_exclude_snapshot = []
        self._params_to_skip_update = []
        self._add_parameters()
#        self.connect_message()

    ##########################################################################
    # QCoDeS parameter definitions: AWG related
    ##########################################################################

    def _add_awg_parameters(self):
        # Channel pair parameters
        for i in range(self._dev_desc.numChannels//2):
            ch_pair = i*2+1
            sfreq_cmd = f'qutech:output{ch_pair}:frequency'
            sph_cmd = f'qutech:output{ch_pair}:phase'
            self.add_parameter(
                f'ch_pair{ch_pair}_sideband_frequency',
                parameter_class=HandshakeParameter,
                unit='Hz',
                label=('Sideband frequency channel pair {} (Hz)'.format(i)),
                get_cmd=sfreq_cmd + '?',
                set_cmd=sfreq_cmd + ' {}',
                vals=vals.Numbers(-300e6, 300e6),
                get_parser=float,
                docstring='Set the frequency of the sideband modulator\n'
                          'Resolution: ~0.23 Hz\n'
                          'Effective immediately when sent')

            self.add_parameter(
                f'ch_pair{ch_pair}_sideband_phase',
                parameter_class=HandshakeParameter,
                unit='deg',
                label=('Sideband phase channel pair {} (deg)'.format(i)),
                get_cmd=sph_cmd + '?',
                set_cmd=sph_cmd + ' {}',
                vals=vals.Numbers(-180, 360),
                get_parser=float,
                docstring='Sideband phase differance between channels\n'
                          'Effective immediately when sent')

            self.add_parameter(
                f'ch_pair{ch_pair}_transform_matrix',
                parameter_class=HandshakeParameter,
                unit='%',
                label=('Transformation matrix channel pair {}'.format(i)),
                get_cmd=_gen_get_func_1par(self._get_matrix, ch_pair),
                set_cmd=_gen_set_func_1par(self._set_matrix, ch_pair),
                # NB range is not a hardware limit
                vals=vals.Arrays(-2, 2, shape=(2, 2)),
                docstring='transformation matrix per channel pair.\n'
                          'Used for mixer correction\n'
                          'Effective immediately when sent')

        # Channel parameters
        for ch in range(1, self._dev_desc.numChannels+1):
            amp_cmd = f'SOUR{ch}:VOLT:LEV:IMM:AMPL'
            offset_cmd = f'SOUR{ch}:VOLT:LEV:IMM:OFFS'
            state_cmd = f'OUTPUT{ch}:STATE'
            waveform_cmd = f'SOUR{ch}:WAV'

            # Compatibility: 5014, QWG
            self.add_parameter(
                f'ch{ch}_state',
                label=f'Status channel {ch}',
                get_cmd=state_cmd + '?',
                set_cmd=state_cmd + ' {}',
                val_mapping={True: '1', False: '0'},
                vals=vals.Bool(),
                docstring='Enables or disables the output of channels\n'
                          'Default: Disabled\n'
                          'Effective immediately when sent') # FIXME: no way, not a HandshakeParameter

            self.add_parameter(
                f'ch{ch}_amp',
                parameter_class=HandshakeParameter,
                label=f'Channel {ch} Amplitude ',
                unit='Vpp',
                get_cmd=amp_cmd + '?',
                set_cmd=amp_cmd + ' {:.6f}',
                vals=vals.Numbers(-1.6, 1.6),
                get_parser=float,
                docstring=f'Amplitude channel {ch} (Vpp into 50 Ohm) \n'
                          'Effective immediately when sent')

            self.add_parameter(
                f'ch{ch}_offset',
                # parameter_class=HandshakeParameter, FIXME: was commented out
                label=f'Offset channel {ch}',
                unit='V',
                get_cmd=offset_cmd + '?',
                set_cmd=offset_cmd + ' {:.3f}',
                vals=vals.Numbers(-.25, .25),
                get_parser=float,
                docstring = f'Offset channel {ch}\n'
                            'Effective immediately when sent')  # FIXME: only if HandshakeParameter

            self.add_parameter(
                f'ch{ch}_default_waveform',
                get_cmd=waveform_cmd+'?',
                set_cmd=waveform_cmd+' "{}"',
                vals=vals.Strings())
                # FIXME: docstring

            # end for(ch...

        # Triggers parameter
        for trigger in range(1, self._dev_desc.numTriggers+1):
            triglev_cmd = f'qutech:trigger{trigger}:level'
            triglev_name = f'tr{trigger}_trigger_level'
            # individual trigger level per trigger input:
            self.add_parameter(
                triglev_name,
                unit='V',
                label=f'Trigger level channel {trigger} (V)',
                get_cmd=triglev_cmd + '?',
                set_cmd=triglev_cmd + ' {}',
                vals=self._dev_desc.mvals_trigger_level,
                get_parser=float)
#                snapshot_exclude=True)
                # FIXME: docstring

            # FIXME: Remove when QCodes PR #1653 is merged, see PycQED_py3 issue #566
            self._params_exclude_snapshot.append(triglev_name)

        # Single parameters
        self.add_parameter(
            'run_mode',
            get_cmd='AWGC:RMO?',
            set_cmd='AWGC:RMO ' + '{}',
            vals=vals.Enum('NONE', 'CONt', 'SEQ', 'CODeword'),
            docstring=_run_mode_doc + '\n Effective after start command')
        # NB: setting mode "CON" (valid SCPI abbreviation) reads back as "CONt"

        # Parameter for codeword per channel
        for cw in range(self._dev_desc.numCodewords):  # FIXME: this may give 1024 parameters per channel
            for j in range(self._dev_desc.numChannels):
                ch = j+1
                # Codeword 0 corresponds to bitcode 0
                cw_cmd = 'sequence:element{:d}:waveform{:d}'.format(cw, ch)
                cw_param = f'codeword_{cw}_ch{ch}_waveform'
                self.add_parameter(
                    cw_param,
                    get_cmd=cw_cmd+'?',
                    set_cmd=cw_cmd+' "{:s}"',
                    vals=vals.Strings())
#                    snapshot_exclude=True)
                # FIXME: Remove when QCodes PR #1653 is merged, see PycQED_py3 issue #566
                self._params_exclude_snapshot.append(cw_param)

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
            'codeword_protocol',
            unit='',
            label='Codeword protocol',
            get_cmd=self._get_codeword_protocol,
            set_cmd=self._set_codeword_protocol,
            vals=vals.Enum('MICROWAVE', 'FLUX', 'MICROWAVE_NO_VSM'),
            docstring=_codeword_protocol_doc + '\nEffective immediately when sent')
        # FIXME: HDAWG uses cfg_codeword_protocol, with different options

        docst = 'Specifies a waveform for a specific codeword. \n' \
                'The channel number corresponds' \
                ' to the channel as indicated on the device (1 is lowest).'
        for j in range(self._dev_desc.numChannels):
            for cw in range(self._dev_desc.numCodewords):
                ch = j + 1

                parname = 'wave_ch{}_cw{:03}'.format(ch, cw)
                self.add_parameter(
                    parname,
                    label='Waveform channel {} codeword {:03}'.format(ch, cw),
                    vals=vals.Arrays(min_value=-1, max_value=1),
                    set_cmd=_gen_set_func_2par(
                        self._set_cw_waveform, ch, cw),
                    get_cmd=_gen_get_func_2par(
                        self._get_cw_waveform, ch, cw),
                    #                    snapshot_exclude=True,
                    docstring=docst)
                # FIXME: Remove when QCodes PR #1653 is merged, see PycQED_py3 issue #566
                self._params_exclude_snapshot.append(parname)


    ##########################################################################
    # QCoDeS parameter definitions: DIO
    ##########################################################################

    def _add_dio_parameters(self):
        self.add_parameter(
            'dio_mode',
            unit='',
            label='DIO input operation mode',
            get_cmd='DIO:MODE?',
            set_cmd='DIO:MODE ' + '{}',
            vals=vals.Enum('MASTER', 'SLAVE'),
            val_mapping={'MASTER': 'MASter', 'SLAVE': 'SLAve'},
            docstring=_dio_mode_doc + '\nEffective immediately when sent') # FIXME: no way, not a HandshakeParameter

        # FIXME: handle through SCPI status
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
            get_cmd='DIO:INDexes:ACTive?',
            set_cmd='DIO:INDexes:ACTive {}',
            get_parser=np.uint32,
            vals=vals.Ints(0, 20),
            docstring='Get and set DIO calibration index\n'
                      'See dio_calibrate() parameter\n'
                      'Effective immediately when sent' # FIXME: no way, not a HandshakeParameter
            )


    ##########################################################################
    # QCoDeS override for InstrumentBase
    ##########################################################################

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
                log.info("Snapshot: Could not update parameter: {}".format(name))
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
    # QCoDeS parameter helpers
    ##########################################################################

    def _set_cw_waveform(self, ch: int, cw: int, waveform):
        wf_name = 'wave_ch{}_cw{:03}'.format(ch, cw)
        cw_cmd = 'sequence:element{:d}:waveform{:d}'.format(cw, ch)
        self.createWaveformReal(wf_name, waveform)
        self._transport.write(cw_cmd + ' "{:s}"'.format(wf_name))

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
        self._transport.write('qutech:output{:d}:matrix {:f},{:f},{:f},{:f}'.format(
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
        protocol = self.codeword_protocols.get(protocol_name)
        if protocol is None:
            allowed_protocols = ", ".join(f'{protocol_name}' for protocols_name in self.codeword_protocols)
            raise ValueError(f"Invalid protocol: actual: {protocol_name}, expected: {allowed_protocols}")

        for ch, bitMap in enumerate(protocol):
            self.set(f"ch{ch + 1}_bit_map", bitMap)

    def _get_codeword_protocol(self):
        channels_bit_maps = []
        result = "Custom"  # Default, if no protocol matches
        for ch in range(1, self._dev_desc.numChannels + 1):
            channels_bit_maps.append(list(map(int, self.get(f"ch{ch}_bit_map"))))

        for prtc_name, prtc_bit_map in self.codeword_protocols.items():
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
        self._transport.write(f"DAC{ch}:BITmap {len(bit_map)}{array_raw}")

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
# Calibration with CC. FIXME: move out of driver
##########################################################################

class QWGMultiDevices:
    """
    QWG helper class to execute parameters/functions on multiple devices. E.g.: DIO calibration
    Usually all methods are static
    """

    @staticmethod
    def dio_calibration(cc, qwgs: List[QWG], verbose: bool = False):
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

        _qwg_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), '_QWG'))

        CC_model = cc.IDN()['model']
        if 'QCC' in CC_model:
            qisa_qwg_dio_calibrate = os.path.join(_qwg_path,
                'QCC_DIO_Calibration.qisa')
            cs_qwg_dio_calibrate = os.path.join(_qwg_path, 'qcc_cs.txt')
            qisa_opcode_qwg_dio_calibrate = os.path.join(_qwg_path,
                'qcc_qisa_opcodes.qmap')
        elif 'CCL' in CC_model:
            qisa_qwg_dio_calibrate = os.path.join(_qwg_path,
                'QWG_DIO_Calibration.qisa')
            cs_qwg_dio_calibrate = os.path.join(_qwg_path, 'cs.txt')
            qisa_opcode_qwg_dio_calibrate = os.path.join(_qwg_path,
                'qisa_opcodes.qmap')
        else:
            raise ValueError('CC model ({}) not recognized.'.format(CC_model))

        if cc._ask("QUTech:RUN?") == '1':
            cc.stop()

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
