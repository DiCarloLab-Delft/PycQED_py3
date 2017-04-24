import numpy as np
import unittest
import logging

from qcodes.utils import validators as vals
from ._controlbox import Assembler
from . import QuTech_ControlBoxdriver as qcb
from ._controlbox import defHeaders_CBox_v3 as defHeaders

from qcodes.instrument.parameter import ManualParameter
# cython drivers for encoding and decoding
import pyximport
pyximport.install(setup_args={"script_args": ["--compiler=msvc"],
                              "include_dirs": np.get_include()},
                  reload_support=True)
# important codec is after pyximport.install
from ._controlbox import codec as c

'''
@author: Xiang Fu
The driver for ControlBox version 3. This is inherited from the driver of
ControlBox version 2.
'''


class QuTech_ControlBox_v3(qcb.QuTech_ControlBox):

    def __init__(self, *args, **kwargs):
        super(QuTech_ControlBox_v3, self).__init__(*args, **kwargs)

    def add_params(self):
        super(QuTech_ControlBox_v3, self).add_params()
        self.add_parameter('core_state',
                           set_cmd=self._do_set_core_state,
                           get_cmd=self._do_get_core_state,
                           vals=vals.Enum('idle', 'active'))
        self.add_parameter('trigger_source',
                           set_cmd=self._do_set_trigger_source,
                           get_cmd=self._do_get_trigger_source,
                           vals=vals.Enum('internal', 'external',
                                          'mixed'))
        self._pulse_queue_state = defHeaders.pulse_queue_states[0]
        self.add_parameter(
            'pulse_queue_state',
            docstring=('returns 0 if OK. returns 1 if instructions are not'
                       ' executed fast enough.'),
            get_cmd=self._do_get_pulse_queue_state)
        self._max_instruction_address = 0
        self.add_parameter(
            'max_instruction_address',
            docstring=('returns the highest address (index) of instructions '
                       'that was exectued by the program, should be equal '
                       'to uploaded_program_length.'),
            get_cmd=self._do_get_max_instruction_address)
        self._uploaded_program_length = 0
        self.add_parameter('uploaded_program_length',
                           get_cmd=self._do_get_uploaded_program_length)
        self.add_parameter('instr_mem_size',
                           unit='#',
                           label='instruction memory size',
                           get_cmd=self._get_instr_mem_size)
        self.add_parameter('last_loaded_instructions',
                           vals=vals.Strings(),
                           parameter_class=ManualParameter)
        # hardcoded memory limit, depends on firmware of the CBox
        self._instr_mem_size = 2**15

    def init_params(self):
        self.add_params()
        self.get_master_controller_params()
        self.set('acquisition_mode', 'idle')
        self.set('demodulation_mode', 'double')
        self.set('trigger_source', 'internal')
        self.set('core_state', 'idle')
        self.set('run_mode', 'idle')
        self.set('signal_delay', 0)
        self.set('integration_length', 100)
        self.set('adc_offset', 0)
        self.set('log_length', 100)
        self.set('nr_samples', 100)
        self.set('nr_averages', 512)
        self.set('lin_trans_coeffs', [1, 0, 0, 1])

    def run_test_suite(self):
        from importlib import reload  # Useful for testing
        from ._controlbox import test_suite_v3 as test_suite
        reload(test_suite)
        # pass the CBox to the module so it can be used in the tests
        self.c = c  # make the codec callable from the testsuite
        test_suite.CBox = self
        suite = unittest.TestLoader().loadTestsFromTestCase(
            test_suite.CBox_tests_v3)
        print('test suite got:',
              unittest.TestLoader().getTestCaseNames(test_suite.CBox_tests_v3))
        unittest.TextTestRunner(verbosity=2).run(suite)

    def start(self):
        self.core_state('active')
        self.run_mode('run')

    def stop(self):
        self.run_mode('idle')
        self.core_state('idle')

    def _do_get_firmware_version(self):
        v = self.get_master_controller_params()
        return v

    def _do_get_pulse_queue_state(self):
        self.get_master_controller_params()
        return self._pulse_queue_state

    def _do_get_max_instruction_address(self):
        self.get_master_controller_params()
        return self._max_instruction_address

    def _do_get_uploaded_program_length(self):
        return self._uploaded_program_length

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
        version = 'v' + str(v_list[0])+'.'+str(v_list[1]) + \
            '.'+str(v_list[2])

        acquisition_mode_int = v_list[3]
        core_state_int = v_list[4]
        trigger_source_int = v_list[5]
        self._acquisition_mode = defHeaders.acquisition_modes[
            acquisition_mode_int]
        self._core_state = defHeaders.core_states[core_state_int]
        self._trigger_source = defHeaders.trigger_sources[trigger_source_int]

        self._adc_offset = (v_list[6] << 4) + v_list[7]
        self._signal_delay = (v_list[8] << 4) + v_list[9]
        self._integration_length = (v_list[10] << 7) + v_list[11]
        a11 = (v_list[12] << 7) + v_list[13]
        a12 = (v_list[14] << 7) + v_list[15]
        a21 = (v_list[16] << 7) + v_list[17]
        a22 = (v_list[18] << 7) + v_list[19]
        self._lin_trans_coeffs = np.array([a11, a12, a21, a22])/2**12
        self._sig_thres[0] = (v_list[20] << 21) + (v_list[21] << 14) + \
            (v_list[22] << 7) + v_list[23]
        self._sig_thres[1] = (v_list[24] << 21) + (v_list[25] << 14) + \
            (v_list[26] << 7) + v_list[27]
        self._log_length = (v_list[28] << 7) + v_list[29] + 1
        self._nr_samples = (v_list[30] << 7) + v_list[31]
        self._avg_size = v_list[32]
        self._nr_averages = 2**self._avg_size
        # self._offset_calc_delay = (v_list[33] << 4) + v_list[34]
        queue_state = v_list[35]
        self._pulse_queue_state = defHeaders.pulse_queue_states[
            queue_state]
        self._max_instruction_address = int(
            ((v_list[36] << 14) + (v_list[37] << 7) + v_list[38])/4)

        return version

    def _do_set_core_state(self, core_state):
        if not isinstance(core_state, str):
            raise KeyError('core_state %s not recognized.'
                           'It should be a string: \'idle\' or \'active.\'')

        if self.get('acquisition_mode') is not None:
            tmp_acquisition_mode = self.get('acquisition_mode')
        else:
            tmp_acquisition_mode = 'idle'

        if self.get('trigger_source') is not None:
            tmp_trigger_source = self.get('trigger_source')
        else:
            tmp_trigger_source = 'internal'  # internal MC trigger

        if self.get('demodulation_mode') is not None:
            tmp_demodulation_mode = self.get('demodulation_mode')
        else:
            tmp_demodulation_mode = 'double'  # double side band demodulation
        self._set_master_controller_working_state(core_state,
                                                  tmp_acquisition_mode,
                                                  tmp_trigger_source,
                                                  tmp_demodulation_mode)

    def _do_set_acquisition_mode(self, acquisition_mode):
        # this function can be invoked by CBox_v2 driver, so the parameter
        # needs to be added.
        if not isinstance(acquisition_mode, str):
            raise KeyError('acquisition_mode %s not recognized.'
                           'It should be one of the following mode names: '
                           '\'idle\', '
                           '\'integration logging\', '
                           '\'integration averaging\', '
                           '\'input averaging\', or '
                           '\'integration streaming\'.')

        if self.get('core_state') is not None:
            tmp_core_state = self.get('core_state')
        else:
            tmp_core_state = 'idle'   # idle state

        if self.get('trigger_source') is not None:
            tmp_trigger_source = self.get('trigger_source')
        else:
            tmp_trigger_source = 'internal'  # internal trigger

        if self.get('demodulation_mode') is not None:
            tmp_demodulation_mode = self.get('demodulation_mode')
        else:
            tmp_demodulation_mode = 'double'

        self._set_master_controller_working_state(tmp_core_state,
                                                  acquisition_mode,
                                                  tmp_trigger_source,
                                                  tmp_demodulation_mode)

    def _do_set_trigger_source(self, trigger_source):
        if not isinstance(trigger_source, str):
            raise KeyError('trigger_source %s not recognized.'
                           'It should be a string: \'internal\','
                           ' \'external\' or \'mixed\'.')

        if self.get('core_state') is not None:
            tmp_core_state = self.get('core_state')
        else:
            tmp_core_state = 'idle'

        if self.get('acquisition_mode') is not None:
            tmp_acquisition_mode = self.get('acquisition_mode')
        else:
            tmp_acquisition_mode = 'idle'

        if self.get('demodulation_mode') is not None:
            tmp_demodulation_mode = self.get('demodulation_mode')
        else:
            tmp_demodulation_mode = 'double'   # double side band demodulation

        self._set_master_controller_working_state(tmp_core_state,
                                                  tmp_acquisition_mode,
                                                  trigger_source,
                                                  tmp_demodulation_mode)

    def _do_set_demodulation_mode(self, demodulation_mode):
        if not isinstance(demodulation_mode, str):
            raise KeyError('demodulation_mode %s not recognized.'
                           'It should be a string:\'double\', or \'single\'')

        if self.get('core_state') is not None:
            tmp_core_state = self.get('core_state')
        else:
            tmp_core_state = 'idle'

        if self.get('acquisition_mode') is not None:
            tmp_acquisition_mode = self.get('acquisition_mode')
        else:
            tmp_acquisition_mode = 'idle'

        if self.get('trigger_source') is not None:
            tmp_trigger_source = self.get('trigger_source')
        else:
            tmp_trigger_source = 'internal'  # internal MC trigger

        self._set_master_controller_working_state(tmp_core_state,
                                                  tmp_acquisition_mode,
                                                  tmp_trigger_source,
                                                  demodulation_mode)

    def _get_instr_mem_size(self):
        return self._instr_mem_size

    def _do_get_core_state(self):
        return self._core_state[3:]

    def _do_get_trigger_source(self):
        return self._trigger_source[3:]

    def _do_get_acquisition_mode(self):
        return self._acquisition_mode[3:]

    def _do_get_demodulation_mode(self):
        return self._demodulation_mode[3:]

    def _set_touch_n_go_parameters(self):
        '''
        Touch 'n Go is only valid for ControlBox version 2, so this function is
        obselete.
        '''
        raise NotImplementedError(
            "This is CBox_v3 driver." +
            "Touch 'n Go is only valid for ControlBox version 2.")

    def _set_master_controller_working_state(
            self, core_state='idle', acquisition_mode='idle',
            trigger_source='internal',
            demodulation_mode='double side band demodulation'):
        '''
        @param core_states: activate the core or disable it:
                        > idle,
                        > active.
        @param acquisition_modes: the data collection mode in MasterController:
                        > idle (default),
                        > integration logging mode,
                        > integration averaging mode,
                        > input averaging mode,
                        > integration streaming mode.
        @param trigger_sources: trigger source of the MasterController:
                        > internal trigger (default),
                        > external trigger,
                        > mixed trigger.
        @param demodulation_mode: the method to demodulate the signal:
                        > double side band demodulation (default),
                        > single side band demodulation.

        @return stat : True if the upload succeeded and False if the upload
                       failed
        '''
        if not isinstance(core_state, str):
            raise KeyError('core_state %s not recognized.'
                           'It should be a string: \'idle\' or \'active.\'')

        if not isinstance(acquisition_mode, str):
            raise KeyError('acquisition_mode %s not recognized.'
                           'It should be one of the following mode names: '
                           '\'idle\', '
                           '\'integration logging\', '
                           '\'integration averaging\', '
                           '\'input averaging\', or '
                           '\'integration streaming\'.')

        if not isinstance(trigger_source, str):
            raise KeyError('trigger_source %s not recognized.'
                           'It should be a string: \'internal\','
                           ' \'external\' or \'mixed\'.')

        if not isinstance(demodulation_mode, str):
            raise KeyError('demodulation_mode %s not recognized.'
                           'It should be a string:\'double\', or \'single\'')

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

    def load_instructions(self, asm_filename, PrintHex=False):
        '''
        @param instructions : the instructions, an array of 32-bit instructions
        @return stat : 0 if the upload succeeded and 1 if the upload failed.

        Additionally it starts by setting the core state to idle before
        uploading instructions and ends by setting the core state to active.
        '''

        asm = Assembler.Assembler(asm_filename)

        instructions = asm.convert_to_instructions()
        if len(instructions) > self.instr_mem_size():
            raise MemoryError(
                'asm file contains too many "{}" instructions,'.format(
                    len(instructions)) + ' max number of instructions ' +
                'is {}'.format(self.instr_mem_size()))

        if PrintHex:
            i = 0
            for instruction in instructions:
                print(i, ": ",  format(instruction, 'x').zfill(8))
                i = i + 4

        # Check the instruction list length
        if len(instructions) == 0:
            raise ValueError("The instruction list is empty.")

        self.set('core_state', 'idle')

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

        appending_instr_num = 3
        self._uploaded_program_length = len(instructions) - appending_instr_num
        self.set('core_state', 'active')
        self.last_loaded_instructions(asm_filename)
        return (stat, mesg)
