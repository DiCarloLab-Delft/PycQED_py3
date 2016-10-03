import numpy as np
from unittest import TestCase
from os.path import join, dirname
from copy import deepcopy

# from qcodes import Instrument

from pycqed.utilities.general import mopen
from pycqed.measurement.waveform_control_CC import \
    single_qubit_qasm_seqs as sq_qasm

# from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object \
#     import Transmon
from pycqed.instrument_drivers.physical_instruments._controlbox \
    import Assembler

from pycqed.measurement.waveform_control_CC.qasm_to_asm_converter import \
    qasm_to_asm


class Test_single_qubit_seqs(TestCase):

    @classmethod
    def setUpClass(self):
        # try:
        #     self.qubit = Transmon('q0_test', server_name=None)
        # except:
        #     self.qubit = Instrument.find_instrument('q0_test').close()
        #     self.qubit = Transmon('q0_test', server_name=None)
        self.qubit_name = 'q0'  # self.qubit.name
        self.times = np.linspace(20e-9, 50e-6, 61)
        # a sample operation dictionary for testing
        self.operation_dict = {
            'init_all': {'instruction': 'WaitReg r0 \n'},
            'X180': {self.qubit_name: {
                'duration': 2, 'instruction': 'Trigger 1000000, 2 \n'}},
            'X90': {self.qubit_name: {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'}},
            'Y180': {self.qubit_name: {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'}},
            'Y90': {self.qubit_name: {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'}},

            'mX180': {self.qubit_name: {
                'duration': 2, 'instruction': 'Trigger 1000000, 2 \n'}},
            'mX90': {self.qubit_name: {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'}},
            'mY180': {self.qubit_name: {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'}},
            'mY90': {self.qubit_name: {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'}},

            'I': {self.qubit_name: {
                'duration': None, 'instruction': 'wait {} \n'}},
            'RO': {self.qubit_name: {
                'duration': 8, 'instruction': 'Trigger 0010000, 2 \n'}}
        }

        def Rx_codeword(amp, min_amp=-.5, max_amp=0.5):
            """
            maps an amp to a codeword, defining a function like this
            is NOT the responsibility of the qasm compiler
            """
            amp = float(amp)
            codeword = int((amp-min_amp)/(max_amp - min_amp) * 127)
            return 'Trigger {:07b}, 2 , \n'.format(codeword)
        self.operation_dict['Rx'] = {
            self.qubit_name: {'instruction': Rx_codeword,
                              'duration': 2}}

        for op in ['X180', 'X90', 'Y180', 'Y90']:
            # This is not the way to do this in  a real sequence but enough
            # to test if the test motzoi sequence compiles
            self.operation_dict[op+'_M'] = {
                self.qubit_name: {'instruction': Rx_codeword,
                                  'duration': 2}}

        def Rphi_codeword(phase):
            """
            maps an phase to a codeword, defining a function like this
            is NOT the responsibility of the qasm compiler
            """
            phase = float(phase)  # input is in degrees
            codeword = int(phase/10+10)  # resolution will be up to 10 deg
            return 'Trigger {:07b}, 2 , \n'.format(codeword)

        self.operation_dict['R90_phi'] = {
            self.qubit_name: {'instruction': Rphi_codeword,
                              'duration': 2}}

    def test_qasm_seq_T1(self):

        qasm_file = sq_qasm.T1(self.qubit_name, self.times)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_OffOn(self):
        qasm_file = sq_qasm.off_on(self.qubit_name)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_AllXY(self):
        qasm_file = sq_qasm.AllXY(self.qubit_name)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_Rabi(self):
        amps = np.linspace(-.4, .5, 51)  # in V
        qasm_file = sq_qasm.Rabi(self.qubit_name, amps)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_Ramsey(self):
        qasm_file = sq_qasm.Ramsey(self.qubit_name, self.times)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.Ramsey(self.qubit_name, self.times)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.Ramsey(self.qubit_name, self.times,
                                   artificial_detuning=4/self.times[-4])
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.Ramsey(self.qubit_name, self.times,
                                   artificial_detuning=None)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.Ramsey(self.qubit_name, self.times,
                                   cal_points=False)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_echo(self):
        qasm_file = sq_qasm.echo(self.qubit_name, self.times)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.echo(self.qubit_name, self.times)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.echo(self.qubit_name, self.times,
                                 artificial_detuning=4/self.times[-4])
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.echo(self.qubit_name, self.times,
                                 artificial_detuning=None)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.echo(self.qubit_name, self.times,
                                 cal_points=False)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_butterfly(self):
        qasm_file = sq_qasm.butterfly(self.qubit_name, initialize=False)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()
        qasm_file = sq_qasm.butterfly(self.qubit_name, initialize=True)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_randomized_benchmarking(self):
        nr_cliffords = (2**(np.arange(10)+1))
        nr_seeds = 10
        qasm_file = sq_qasm.randomized_benchmarking(self.qubit_name,
                                                    nr_cliffords, nr_seeds,
                                                    double_curves=True)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.randomized_benchmarking(self.qubit_name,
                                                    nr_cliffords, nr_seeds,
                                                    double_curves=False)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_MotzoiXY(self):
        qasm_file = sq_qasm.MotzoiXY(
            self.qubit_name, motzois=np.linspace(-.5, .5, 21),
            cal_points=False)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

        qasm_file = sq_qasm.MotzoiXY(
            self.qubit_name, motzois=np.linspace(-.5, .5, 21),
            cal_points=True)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()


class Test_qasm_to_asm(TestCase):

    @classmethod
    def setUpClass(self):
        self.base_qasm_path = join(dirname(__file__), 'qasm_files')
        self.qubit_name = 'q0'  # self.qubit.name

        # a sample operation dictionary for testing
        self.operation_dict = {
            'init_all': {'instruction': 'WaitReg r0 \n'},
            'X180': {self.qubit_name: {
                     'duration': 2, 'instruction': 'Trigger 1000000, 2 \n'}},
            'Y180': {self.qubit_name: {
                'duration': 2, 'instruction': 'Trigger 0100000, 2 \n'}},
            'I': {self.qubit_name: {
                'duration': None, 'instruction': 'wait {} \n'}},
            'RO': {self.qubit_name: {
                'duration': 8, 'instruction': 'Trigger 0010000, 2 \n'}}
        }

    def test_empty_qasm_file(self):
        filename = join(self.base_qasm_path, 'empty.qasm')
        qasm_file = mopen(filename, mode='w')
        qasm_file.writelines('qubit {} \n'.format(self.qubit_name))
        qasm_file.close()
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
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
            qasm_to_asm(qasm_file.name, self.operation_dict)

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
        qasm_to_asm(qasm_file.name, ext_op_dict)

    def test_qasm_function_with_function_arg(self):
        ext_op_dict = deepcopy(self.operation_dict)

        def Rx_codeword(amp, min_amp=-.5, max_amp=0.5):
            """
            maps an amp to a codeword, defining a function like this
            is NOT the responsibility of the qasm compiler
            """
            amp = float(amp)
            codeword = int((amp-min_amp)/(max_amp - min_amp) * 127)
            return 'Trigger {:07b}, 2 , \n'.format(codeword)

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
        qasm_to_asm(qasm_file.name, ext_op_dict)

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
            qasm_to_asm(qasm_file.name, self.operation_dict)
