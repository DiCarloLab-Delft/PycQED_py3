# Note to Xiang, remove the imports that are not used :)
import time
import numpy as np
import sys
# import visa
import unittest
import logging

qcpath = 'D:\GitHubRepos\Qcodes'
if qcpath not in sys.path:
    sys.path.append(qcpath)

from qcodes.instrument.visa import VisaInstrument
from qcodes.utils import validators as vals

# cython drivers for encoding and decoding
import pyximport
pyximport.install(setup_args={"script_args": ["--compiler=msvc"],
                              "include_dirs": np.get_include()},
                  reload_support=True)

from ._controlbox import codec as c
from ._controlbox import Assembler
from . import QuTech_ControlBoxdriver as qcb
from ._controlbox import defHeaders_CBox_v3 as defHeaders

'''
@author: Xiang Fu
The driver for ControlBox version 3. This is inherited from the driver of
ControlBox version 2.
'''


class QuTech_ControlBox_v3(qcb.QuTech_ControlBox):
    def __init__(self, *args, **kwargs):
        super(QuTech_ControlBox_v3, self).__init__(*args, **kwargs)

        # self.add_parameter('acquisition_mode',
        #                    set_cmd=self._do_set_acquisition_mode,
        #                    get_cmd=self._do_get_acquisition_mode,
        #                    vals=vals.Anything())
        if not('core_state' in self.parameters):
            self.add_parameter('core_state',
                               set_cmd=self._do_set_core_state,
                               get_cmd=self._do_get_core_state,
                               vals=vals.Anything())

        if not('trigger_source' in self.parameters):
            self.add_parameter('trigger_source',
                               set_cmd=self._do_set_trigger_source,
                               get_cmd=self._do_get_trigger_source,
                               vals=vals.Anything())

        if not('demodulation_mode' in self.parameters):
            self.add_parameter('demodulation_mode',
                   set_cmd=self._do_set_demodulation_mode,
                   get_cmd=self._do_get_demodulation_mode,
                   vals=vals.Anything())

    def run_test_suite(self):
            from importlib import reload  # Useful for testing
            from ._controlbox import test_suite_v3 as test_suite
            reload(test_suite)
            # pass the CBox to the module so it can be used in the tests
            self.c = c  # make the codec callable from the testsuite
            test_suite.CBox = self
            suite = unittest.TestLoader().loadTestsFromTestCase(
                test_suite.CBox_tests_v3)
            unittest.TextTestRunner(verbosity=2).run(suite)

    def _do_get_firmware_version(self):
        # raise NotImplementedError("This function is obselete, please try\
        # function get_master_controller_params.")
        v = self.get_master_controller_params()
        # print(v)
        return v

    def get_master_controller_params(self):
        message = self.create_message(defHeaders.ReadVersion)
        (stat, mesg) = self.serial_write(message)
        if stat:
            param_msg = self.serial_read()

        # Decoding the message
        # version number, first 3 bytes
        v_list = []
        for byte in bytes(param_msg):
            v_list.append(int(byte) - 128)  # Remove the MSb 1 for data_bytes
            # print(byte, ": ", "{0:{fill}8b}".format(byte, fill='0'))

        version = 'v' + str(v_list[0])+'.'+str(v_list[1]) + \
            '.'+str(v_list[2])

        self._acquisition_mode = v_list[3]
        self._core_state = v_list[4]
        self._trigger_source = v_list[5]
        self._adc_offset = (v_list[6] << 4) + v_list[7]
        self._signal_delay = (v_list[8] << 4) + v_list[9]
        self._integration_length = (v_list[10] << 7) + v_list[11]
        a11 = (v_list[12] << 7) + v_list[13]
        a12 = (v_list[14] << 7) + v_list[15]
        a21 = (v_list[16] << 7) + v_list[17]
        a22 = (v_list[18] << 7) + v_list[19]
        self._lin_trans_coeffs = np.array([a11, a12, a21, a22])
        self._sig_thres[0] = (v_list[20] << 21) + (v_list[21] << 14) + \
                          (v_list[22] << 7) + v_list[23]
        self._sig_thres[1] = (v_list[24] << 21) + (v_list[25] << 14) + \
                          (v_list[26] << 7) + v_list[27]
        self._log_length = (v_list[28] << 7) + v_list[29]
        self._nr_samples = (v_list[30] << 7) + v_list[31]
        self._avg_size = v_list[32]
        self._nr_averages = 2**self._avg_size

        # self.snapshot()
        # print("version: ", version)
        return version

    def _do_set_log_length(self, length):
        '''
        set the number of measurements of the log in test mode.

        @param length : the number of measurements range (1, 8000)
        @return stat : 0 if the upload succeeded and 1 if the upload failed.
        '''
        v = self.get_master_controller_params()
        if (int(v[1]) == 2) and (int(int(v[3:5])) >= 15):
            n_bytes = 3
        elif (int(v[1]) == 3) and (int(int(v[3])) > 1):
            n_bytes = 3
        else:
            n_bytes = 2

        cmd = defHeaders.UpdateLoggerMaxCounterHeader
        data_bytes = c.encode_byte(length-1, 7,
                                   expected_number_of_bytes=n_bytes)
        # Changed from version 2.15 onwards
        message = c.create_message(cmd, data_bytes)
        (stat, mesg) = self.serial_write(message)
        if stat:
            self._log_length = length
        else:
            raise Exception('Failed to set log_length')
        return (stat, message)

    def _do_set_core_state(self, core_state):
        if self.get('acquisition_mode') is not None:
            tmp_acquisition_mode = self.get('acquisition_mode')
        else:
            tmp_acquisition_mode = 0   # idle state

        if self.get('trigger_source') is not None:
            tmp_trigger_source = self.get('trigger_source')
        else:
            tmp_trigger_source = 0      # internal trigger

        if self.get('demodulation_mode') is not None:
            tmp_demodulation_mode = self.get('demodulation_mode')
        else:
            tmp_demodulation_mode = 0   # double side band demodulation
        self.set_master_controller_working_state(core_state,
                                                 tmp_acquisition_mode,
                                                 tmp_trigger_source,
                                                 tmp_demodulation_mode)

    def _do_set_acquisition_mode(self, acquisition_mode):
        # this function can be invoked by CBox_v2 driver, so the parameter
        # needs to be added.
        if not('core_state' in self.parameters):
            self.add_parameter('core_state',
                               set_cmd=self._do_set_core_state,
                               get_cmd=self._do_get_core_state,
                               vals=vals.Anything())
            tmp_core_state = 0
        elif self.get('core_state') is not None:
            tmp_core_state = self.get('core_state')
        else:
            tmp_core_state = 0   # idle state

        # this function can be invoked by CBox_v2 driver, so the parameter
        # needs to be added.
        if not('trigger_source' in self.parameters):
            self.add_parameter('trigger_source',
                               set_cmd=self._do_set_trigger_source,
                               get_cmd=self._do_get_trigger_source,
                               vals=vals.Anything())
            tmp_trigger_source = 0
        elif self.get('trigger_source') is not None:
            tmp_trigger_source = self.get('trigger_source')
        else:
            tmp_trigger_source = 0      # internal trigger

        if not('demodulation_mode' in self.parameters):
            self.add_parameter('demodulation_mode',
                               set_cmd=self._do_set_demodulation_mode,
                               get_cmd=self._do_get_demodulation_mode,
                               vals=vals.Anything())
            tmp_demodulation_mode = 0
        elif self.get('demodulation_mode') is not None:
            tmp_demodulation_mode = self.get('demodulation_mode')
        else:
            tmp_demodulation_mode = 0      # internal trigger

        self.set_master_controller_working_state(tmp_core_state,
                                                 acquisition_mode,
                                                 tmp_trigger_source,
                                                 tmp_demodulation_mode)

    def _do_set_trigger_source(self, trigger_source):
        if self.get('core_state') is not None:
            tmp_core_state = self.get('core_state')
            # print('_do_set_trigger_source\got_core_state: ', tmp_core_state)
        else:
            tmp_core_state = 0   # idle state

        if self.get('acquisition_mode') is not None:
            tmp_acquisition_mode = self.get('acquisition_mode')
        else:
            tmp_acquisition_mode = 0   # idle state

        if self.get('demodulation_mode') is not None:
            tmp_demodulation_mode = self.get('demodulation_mode')
        else:
            tmp_demodulation_mode = 0   # double side band demodulation

        self.set_master_controller_working_state(tmp_core_state,
                                                 tmp_acquisition_mode,
                                                 trigger_source,
                                                 tmp_demodulation_mode)

    def _do_set_demodulation_mode(self, demodulation_mode):
        if self.get('core_state') is not None:
            tmp_core_state = self.get('core_state')
            # print('_do_set_trigger_source\got_core_state: ', tmp_core_state)
        else:
            tmp_core_state = 0   # idle state

        if self.get('acquisition_mode') is not None:
            tmp_acquisition_mode = self.get('acquisition_mode')
        else:
            tmp_acquisition_mode = 0   # idle state

        if self.get('trigger_source') is not None:
            tmp_trigger_source = self.get('trigger_source')
        else:
            tmp_trigger_source = 0      # internal trigger

        self.set_master_controller_working_state(tmp_core_state,
                                                 tmp_acquisition_mode,
                                                 tmp_trigger_source,
                                                 demodulation_mode)
                                                 
    def _do_get_core_state(self):
        return self._core_state

    def _do_get_trigger_source(self):
        return self._trigger_source

    def _do_get_acquisition_mode(self):
        return self._acquisition_mode

    def _do_get_demodulation_mode(self):
        return self._demodulation_mode
    def _set_touch_n_go_parameters(self):

        '''
        Touch 'n Go is only valid for ControlBox version 2, so this function is
        obselete.
        '''
        raise NotImplementedError(
            "This is CBox_v3 driver." +
            "Touch 'n Go is only valid for ControlBox version 2.")

    def set_master_controller_working_state(self, 
                                            core_state='idle', 
                                            acquisition_mode='idle',
                                            trigger_source='internal', 
                                            demodulation_mode=
                                            'double side band demodulation'):
        '''
        @param core_states: activate the core or disable it,
                        0 = idle,
                        1 = active.
        @param acquisition_modes: the data collection mode in MasterController,
                        0 = idle (default),
                        1 = integration logging mode,
                        2 = integration averaging mode,
                        3 = input averaging mode,
                        4 = integration streaming mode.
        @param trigger_sources: trigger source of the MasterController,
                        0 = internal trigger (default),
                        1 = external trigger,
                        2 = mixed trigger.
        @param demodulation_mode: the method to demodulate the signal,
                        0 = double side band demodulation,
                        1 = single side band demodulation.

        @return stat : True if the upload succeeded and False if the upload
                       failed
        '''
        core_state = str(core_state)
        acquisition_mode = str(acquisition_mode)
        trigger_source = str(trigger_source)
        demodulation_mode = str(demodulation_mode)

        acquisition_mode_int = None
        for i in range(len(defHeaders.acquisition_modes)):
            if acquisition_mode.upper() in \
               defHeaders.acquisition_modes[i].upper():
                acquisition_mode_int = i
                break
        if acquisition_mode_int is None:
            raise KeyError('acquisition_mode %s not recognized')
        if acquisition_mode_int == 1 and self._log_length > 8000:
            logging.warning('Log length can be max 8000 in int. log. mode')

        core_state_int = None
        for i in range(len(defHeaders.core_states)):
            if core_state.upper() in defHeaders.core_states[i].upper():
                core_state_int = i
                break
        if core_state_int is None:
            raise KeyError('core_state %s not recognized')

        trigger_source_int = None
        for i in range(len(defHeaders.trigger_sources)):
            if trigger_source.upper() in defHeaders.trigger_sources[i].upper():
                trigger_source_int = i
                break
        if trigger_source_int is None:
            raise KeyError('trigger_state %s not recognized')

        demodulation_mode_int = None
        for i in range(len(defHeaders.demodulation_modes)):
            if demodulation_mode.upper() in defHeaders.demodulation_modes[i].upper():
                demodulation_mode_int = i
                break
        if demodulation_mode_int is None:
            raise KeyError('demodulation_mode %s not recognized')

        # Here the actual acquisition_mode is set
        cmd = defHeaders.UpdateMCWorkingState
        data_bytes = bytes()
        data_bytes += c.encode_byte(acquisition_mode_int, 7,
                                    expected_number_of_bytes=1)
        data_bytes += c.encode_byte(demodulation_mode_int, 7,

                                    expected_number_of_bytes=1)
        data_bytes += c.encode_byte(trigger_source_int, 7,
                                    expected_number_of_bytes=1)
        data_bytes += c.encode_byte(core_state_int, 7,
                                   expected_number_of_bytes=1)
        message = c.create_message(cmd, data_bytes)
        print("set master controller command: ",  message)

        (stat, mesg) = self.serial_write(message)
        if stat:
            self._acquisition_mode = defHeaders.acquisition_modes[
                                    acquisition_mode_int]
            self._core_state = defHeaders.core_states[core_state_int]
            self._trigger_source = defHeaders.trigger_sources[
                                    trigger_source_int]
            self._demodulation_mode = defHeaders.demodulation_modes[
                                    demodulation_mode_int]
        else:
            raise Exception('Failed to set acquisition_mode')

        return (stat, message)

    def load_instructions(self, asm_file, PrintHex=False):
        '''
        set the weights of the integregration

        @param instructions : the instructions, an array of 32-bit instructions
        @return stat : 0 if the upload succeeded and 1 if the upload failed.
        '''

        asm = Assembler.Assembler(asm_file)

        instructions = asm.convert_to_instructions()

        if instructions is False:
            print("Error: the assembly file is of errors.")
            return False

        if PrintHex:
            i = 0
            for instruction in instructions:
                print(i, ": ",  format(instruction, 'x').zfill(8))
                i = i + 4

        # Check the instruction list length
        if len(instructions) == 0:
            raise ValueError("The instruction list is empty.")

        cmd = defHeaders.LoadInstructionsHeader
        data_bytes = bytearray()

        instr_length = 32
        data_bytes.extend(c.encode_array(
                        self.convert_arrary_to_signed(instructions,
                                                      instr_length),
                        data_bits_per_byte=4,
                        bytes_per_value=8))

        message = c.create_message(cmd, bytes(data_bytes))
        (stat, mesg) = self.serial_write(message)

        if not stat:
            raise Exception('Failed to load instructions')

        return (stat, mesg)
