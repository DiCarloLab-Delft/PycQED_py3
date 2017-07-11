
import sys
import numpy as np
from io import StringIO
from unittest import TestCase
from os.path import join, dirname
from copy import deepcopy

# from qcodes import Instrument

from pycqed.utilities.general import mopen
from pycqed.measurement.waveform_control_CC import \
    single_qubit_qasm_seqs as sq_qasm
from pycqed.measurement.waveform_control_CC import \
    multi_qubit_qasm_seqs as mq_qasm

# from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object \
#     import Transmon
from pycqed.instrument_drivers.physical_instruments._controlbox \
    import Assembler

from pycqed.measurement.waveform_control_CC import qasm_to_asm as qta


class Test_single_qubit_seqs(TestCase):

    @classmethod
    def setUpClass(self):
        self.qubit_name = 'q0'  # self.qubit.name
        self.times = np.linspace(20e-9, 50e-6, 61)
        # a sample operation dictionary for testing
        self.operation_dict = {
            'init_all': {'instruction': 'WaitReg r0 \n'},
            'X180 {}'.format(self.qubit_name): {
                'duration': 2, 'instruction': 'Trigger 1000000, 2 \n'},
            'X90 {}'.format(self.qubit_name): {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'},
            'Y180 {}'.format(self.qubit_name): {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'},
            'Y90 {}'.format(self.qubit_name): {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'},

            'mX180 {}'.format(self.qubit_name): {
                'duration': 2, 'instruction': 'Trigger 1000000, 2 \n'},
            'mX90 {}'.format(self.qubit_name): {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'},
            'mY180 {}'.format(self.qubit_name): {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'},
            'mY90 {}'.format(self.qubit_name): {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'},

            'I {}'.format(self.qubit_name): {
                'duration': None, 'instruction': 'wait {} \n'},
            'I': {
                'duration': None, 'instruction': 'wait {} \n'},
            'RO {}'.format(self.qubit_name): {
                'duration': 8, 'instruction': 'Trigger 0010000, 2 \n'}}

        def Rx_codeword(amp, min_amp=-.5, max_amp=0.5):
            """
            maps an amp to a codeword, defining a function like this
            is NOT the responsibility of the qasm compiler
            """
            amp = float(amp)
            codeword = int((amp-min_amp)/(max_amp - min_amp) * 127)
            return 'Trigger {:07b}, 2 \n'.format(codeword)
        self.operation_dict['Rx {}'.format(self.qubit_name)] = {
            'instruction': Rx_codeword, 'duration': 2}

        for op in ['X180', 'X90', 'Y180', 'Y90']:
            # This is not the way to do this in  a real sequence but enough
            # to test if the test motzoi sequence compiles
            self.operation_dict[op+'_Motz {}'.format(self.qubit_name)] = {
                'instruction': Rx_codeword, 'duration': 2}

        def Rphi_codeword(phase):
            """
            maps an phase to a codeword, defining a function like this
            is NOT the responsibility of the qasm compiler
            """
            phase = float(phase)  # input is in degrees
            codeword = int(phase/10+10)  # resolution will be up to 10 deg
            return 'Trigger {:07b}, 2 \n'.format(codeword)

        self.operation_dict['R90_phi {}'.format(self.qubit_name)] = {
            'instruction': Rphi_codeword, 'duration': 2}

    def test_qasm_seq_T1(self):

        qasm_file = sq_qasm.T1(self.qubit_name, self.times)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_single_elt_on(self):
        qasm_file = sq_qasm.single_elt_on(self.qubit_name)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_two_elt_MotzoiXY(self):
        qasm_file = sq_qasm.two_elt_MotzoiXY(self.qubit_name)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_OffOn(self):
        qasm_file = sq_qasm.off_on(self.qubit_name)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_AllXY(self):
        qasm_file = sq_qasm.AllXY(self.qubit_name)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_Rabi(self):
        amps = np.linspace(-.4, .5, 51)  # in V
        qasm_file = sq_qasm.Rabi(self.qubit_name, amps)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_Ramsey(self):
        qasm_file = sq_qasm.Ramsey(self.qubit_name, self.times)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.Ramsey(self.qubit_name, self.times,
                                   artificial_detuning=4/self.times[-4])
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.Ramsey(self.qubit_name, self.times,
                                   artificial_detuning=None)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.Ramsey(self.qubit_name, self.times,
                                   cal_points=False)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_echo(self):
        qasm_file = sq_qasm.echo(self.qubit_name, self.times)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.echo(self.qubit_name, self.times)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.echo(self.qubit_name, self.times,
                                 artificial_detuning=4/self.times[-4])
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.echo(self.qubit_name, self.times,
                                 artificial_detuning=None)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.echo(self.qubit_name, self.times,
                                 cal_points=False)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_butterfly(self):
        qasm_file = sq_qasm.butterfly(self.qubit_name, initialize=False)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()
        qasm_file = sq_qasm.butterfly(self.qubit_name, initialize=True)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_randomized_benchmarking(self):
        nr_cliffords = (2**(np.arange(10)+1))
        nr_seeds = 10
        qasm_file = sq_qasm.randomized_benchmarking(self.qubit_name,
                                                    nr_cliffords, nr_seeds,
                                                    double_curves=True)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.randomized_benchmarking(self.qubit_name,
                                                    nr_cliffords, nr_seeds,
                                                    double_curves=False)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_MotzoiXY(self):
        qasm_file = sq_qasm.MotzoiXY(
            self.qubit_name, motzois=np.linspace(-.5, .5, 21),
            cal_points=False)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.MotzoiXY(
            self.qubit_name, motzois=np.linspace(-.5, .5, 21),
            cal_points=True)
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()


class Test_multi_qubit_seqs(TestCase):

    @classmethod
    def setUpClass(self):
        self.qubits = ['q0', 'q1']
        self.times = np.linspace(20e-9, 50e-6, 61)
        # a sample operation dictionary for testing
        self.operation_dict = {}
        for q in self.qubits:
            self.operation_dict['init_all'] = {'instruction': 'WaitReg r0 \n'}
            self.operation_dict['X180 {}'.format(q)] = {
                'duration': 2, 'instruction': 'Trigger 1000000, 2 \n'}
            self.operation_dict['X90 {}'.format(q)] = {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'}
            self.operation_dict['Y180 {}'.format(q)] = {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'}
            self.operation_dict['Y90 {}'.format(q)] = {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'}

            self.operation_dict['mX180 {}'.format(q)] = {
                'duration': 2, 'instruction': 'Trigger 1000000, 2 \n'}
            self.operation_dict['mX90 {}'.format(q)] = {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'}
            self.operation_dict['mY180 {}'.format(q)] = {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'}
            self.operation_dict['mY90 {}'.format(q)] = {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'}

            # FIXME: It should be possible to translate 'I' to 'wait for some
            # time', which it is not when it is translated to an empty string
            # like this.
            self.operation_dict['I {}'.format(q)] = {
                'duration': None, 'instruction': ''}
            self.operation_dict['RO {}'.format(q)] = {
                'duration': 8, 'instruction': 'Trigger 0010000, 2 \n'}
        self.operation_dict['RO all'] = {
            'duration': 8, 'instruction': 'Trigger 0010000, 2 \n'}

    def test_two_qubit_off_on(self):
        qasm_file = mq_qasm.two_qubit_off_on(self.qubits[0], self.qubits[1])
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        asm.convert_to_instructions()

    def test_two_qubit_tomo_cardinal(self):
        qasm_file = mq_qasm.two_qubit_tomo_cardinal(1, self.qubits[0],
                                                    self.qubits[1])
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        asm.convert_to_instructions()

    def test_two_qubit_AllXY(self):
        qasm_file = mq_qasm.two_qubit_AllXY(self.qubits[0], self.qubits[1])
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        asm.convert_to_instructions()


class Test_qasm_to_asm(TestCase):

    @classmethod
    def setUpClass(self):
        self.base_qasm_path = join(dirname(__file__), 'qasm_files')
        self.qubit_name = 'q0'  # self.qubit.name

        # a sample operation dictionary for testing
        self.operation_dict = {
            'init_all': {'instruction': 'WaitReg r0 \n'},
            'X180 {}'.format(self.qubit_name): {
                'duration': 2, 'instruction': 'Trigger 1000000, 2 \n'},
            'Y180 {}'.format(self.qubit_name): {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'},
            'I {}'.format(self.qubit_name): {
                'duration': None, 'instruction': 'wait {} \n'},
            'RO {}'.format(self.qubit_name): {
                'duration': 8, 'instruction': 'Trigger 0010000, 2 \n'}}

    def test_empty_qasm_file(self):
        filename = join(self.base_qasm_path, 'empty.qasm')
        qasm_file = mopen(filename, mode='w')
        qasm_file.writelines('qubit {} \n'.format(self.qubit_name))
        qasm_file.close()
        asm_file = qta.qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_invalid_command(self):
        filename = join(self.base_qasm_path, 'invalid_command.qasm')
        qasm_file = mopen(filename, mode='w')
        qasm_file.writelines('qubit {} \n'.format(self.qubit_name))
        qasm_file.writelines('Xbla {}     # invalid cmd\n'.format(
                             self.qubit_name))
        qasm_file.close()
        with self.assertRaises(ValueError):
            qta.qasm_to_asm(qasm_file.name, self.operation_dict)

    def test_qasm_function_with_string_format_arg(self):
        ext_op_dict = deepcopy(self.operation_dict)

        filename = join(self.base_qasm_path, 'argument.qasm')
        qasm_file = mopen(filename, mode='w')
        qasm_file.writelines('qubit {} \n'.format(self.qubit_name))
        # test wait argument
        qasm_file.writelines('I {} 12\n'.format(
                             self.qubit_name))
        qasm_file.writelines('I {} 4\n'.format(
                             self.qubit_name))
        qasm_file.close()
        qta.qasm_to_asm(qasm_file.name, ext_op_dict)

    def test_qasm_function_with_function_arg(self):
        ext_op_dict = deepcopy(self.operation_dict)

        def Rx_codeword(amp, min_amp=-.5, max_amp=0.5):
            """
            maps an amp to a codeword, defining a function like this
            is NOT the responsibility of the qasm compiler
            """
            amp = float(amp)
            codeword = int((amp-min_amp)/(max_amp - min_amp) * 127)
            return 'Trigger {:07b}, 2 \n'.format(codeword)

        ext_op_dict['Rx'] = {
            self.qubit_name: {'instruction': Rx_codeword,
                              'duration': 2}}
        filename = join(self.base_qasm_path, 'argument.qasm')
        qasm_file = mopen(filename, mode='w')
        qasm_file.writelines('qubit {} \n'.format(self.qubit_name))
        # test wait argument
        qasm_file.writelines('I {} 12\n'.format(
                             self.qubit_name))
        qasm_file.writelines('I {} 4\n'.format(
                             self.qubit_name))
        qasm_file.close()
        qta.qasm_to_asm(qasm_file.name, ext_op_dict)

        # I need to test here if the file written contains the right commands

    def test_too_many_args_command(self):
        filename = join(self.base_qasm_path, 'too_many_args.qasm')
        qasm_file = mopen(filename, mode='w')
        qasm_file.writelines('qubit {} \n'.format(self.qubit_name))
        # leaving out the \n prevents the line from breaking
        qasm_file.writelines('X180 {}'.format(
                             self.qubit_name))
        qasm_file.writelines('X180 {}'.format(
                             self.qubit_name))
        qasm_file.writelines('Y180 {}'.format(
                             self.qubit_name))

        qasm_file.close()
        with self.assertRaises(ValueError):
            qta.qasm_to_asm(qasm_file.name, self.operation_dict)


class Test_qasm_waveform_management(TestCase):

    """
    Waveform management,
        uploading the required waveforms and keeping track of the waveforms
        is tested here for the example of a Rabi.
    """

    @classmethod
    def setUpClass(self):
        self.base_qasm_path = join(dirname(__file__), 'qasm_files')
        self.qubit_name = 'q0'  # self.qubit.name

        self.pulse_pars = {
            'control_pulse q0': {
                'duration': 2,
                'I_channel': 'ch1',
                'Q_channel': 'ch2',
                'amplitude': 0.5,
                'amp90_scale': 0.48,
                'sigma': 5e-9,
                'nr_sigma': 4,
                'motzoi': -0.2,
                'mod_frequency': -100e6,
                'phi_skew': 3.2,
                'alpha': 1.05,
                'qubit': 'q0',
                'prepare_function': 'mock_control_pulse_prepare'},
            'RO q0': {
                'duration': 60,
                'I_channel': 'ch3',
                'Q_channel': 'ch4',
                'amplitude': 0.4,
                'length': 200e-9,
                'pulse_delay': 15e-9,
                'mod_frequency': 400e6,
                'acq_marker_delay': -20e-9,
                'acq_marker_channel': 'ch',
                'prepare_function': 'mock_control_pulse_prepare'}
        }

        # a sample operation dictionary for testing
        self.operation_dict = {
            'init_all': {'instruction': 'WaitReg r0 \n',
                         'duration': None,
                         'prepare_function': None,
                         'prepare_function_kwargs': None},
            'X180 {}'.format(self.qubit_name): {
                'duration': 2, 'instruction': 'Trigger 1000000, 2 \n',
                'prepare_function': 'mock_control_pulse_prepare',
                'prepare_function_kwargs': {'test': 0}},
            'Y180 {}'.format(self.qubit_name): {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n',
                'prepare_function': 'mock_control_pulse_prepare',
                'prepare_function_kwargs': {'test': 0}},
            'I {}'.format(self.qubit_name): {
                'duration': None, 'instruction': 'wait {} \n',
                'prepare_function': 'mock_control_pulse_prepare',
                'prepare_function_kwargs': None},
            'RO {}'.format(self.qubit_name): {
                'duration': 8, 'instruction': 'Trigger 0010000, 2 \n',
                'prepare_function': 'mock_control_pulse_prepare',
                'prepare_function_kwargs': {'test': 0}}}

        self.basic_ops = ['init_all', 'RO q0', 'X180 q0', 'Y180 q0',
                          'X90 q0', 'Y90 q0']

        self.amps = np.linspace(-.5, .5, 11)  # in V
        self.Rabi_qasm_file = sq_qasm.Rabi(self.qubit_name, self.amps)
        self.AllXY_qasm_file = sq_qasm.AllXY(self.qubit_name)

    def test_qasm_extract_required_ops(self):
        operations = qta.extract_required_operations(
            self.AllXY_qasm_file.name)
        self.assertCountEqual(operations, self.basic_ops)

        operations = qta.extract_required_operations(
            self.Rabi_qasm_file.name)
        rabi_ops = ['init_all', 'RO q0']
        for amp in self.amps:
            rabi_ops.append('Rx q0 {}'.format(amp))
        self.assertCountEqual(operations, rabi_ops)

    def test_create_qasm_operation_dict(self):
        ops = self.basic_ops + ['mY180 q0']

        operation_dict = qta.create_operation_dict(ops, self.pulse_pars)
        self.assertTrue(valid_operation_dictionary(operation_dict))
        self.assertCountEqual(operation_dict.keys(), ['init_all',
                                                      'RO q0', 'X180 q0',
                                                      'Y180 q0', 'X90 q0',
                                                      'Y90 q0', 'mY180 q0'])
        self.assertEqual(
            operation_dict['X180 q0']['prepare_function_kwargs']['amplitude'],
            0.5)
        self.assertEqual(
            operation_dict['X90 q0']['prepare_function_kwargs']['amplitude'],
            0.5*0.48)
        self.assertEqual(
            operation_dict['Y180 q0']['prepare_function_kwargs']['amplitude'],
            0.5)
        self.assertEqual(
            operation_dict['Y90 q0']['prepare_function_kwargs']['amplitude'],
            0.5*0.48)

        self.assertEqual(
            operation_dict['X180 q0']['prepare_function_kwargs']['phase'], 0)
        self.assertEqual(
            operation_dict['X90 q0']['prepare_function_kwargs']['phase'], 0)
        self.assertEqual(
            operation_dict['mY180 q0']['prepare_function_kwargs']['phase'],
            270)
        self.assertEqual(
            operation_dict['Y90 q0']['prepare_function_kwargs']['phase'], 90)

    def test_prepare_operations(self):
        assert(valid_operation_dictionary(self.operation_dict))
        with Capturing() as output:
            qta.prepare_operations(self.operation_dict)
        self.assertTrue("mock called with {'test': 0}" in output)

    def test_complete_sequence_loading_simple(self):
        '''
        Tests all the loading with an AllXY sequence that
        contains all the basic steps but does not require mapping
        the allowed operations.
        '''
        qasm_file = self.AllXY_qasm_file
        ops = qta.extract_required_operations(qasm_file.name)

        # # config needs to contain enough info to generate mapping

        operation_dict = qta.create_operation_dict(ops, self.pulse_pars)
        # operation_mapping = qta.create_operation_mapping(required_ops)

        # # uploads all operations in op dict
        with Capturing() as output:
            qta.prepare_operations(operation_dict)

        qta.qasm_to_asm(qasm_file.name, operation_dict)

    def test_complete_sequence_loading_dynamic(self):
        '''
        Tests all the loading with a Rabi sequence that requires
        dynamic defining of the configuration file
        '''
        # Still mostly psuedo code
        pass
        # self.qasm_file

        # required_ops = extract_required_operations(self.qasm_file.name)

        # # config needs to contain enough info to generate mapping
        # operation_mapping = get_operation_mapping(required_ops, config_a)

        # for operation in operation_mapping:
        #     waveform = generate_waveform(operation, args)
        #     upload_waveform(waveform, location)


def valid_operation_dictionary(operation_dict):
    '''
    checks if the operation dictionary is valid
    '''
    if not isinstance(operation_dict, dict):
        return False
    for key, item in operation_dict.items():
        if not isinstance(key, str):
            return False
        if sorted(item.keys()) != ['duration', 'instruction',
                                   'prepare_function',
                                   'prepare_function_kwargs']:
            print(sorted([item.keys()]), '!=',
                  "['duration', 'instruction', 'prepare_function', 'prepare_function_kwargs']")
            return False
    return True


class Capturing(list):

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout
