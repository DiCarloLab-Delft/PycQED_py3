import numpy as np
from unittest import TestCase
from os.path import join, dirname

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

        # a sample operation dictionary for testing
        self.operation_dict = {
            'init_all': {'instruction': 'WaitReg r0 \n'},
            'X': {self.qubit_name: {'duration': 2,
                                    'instruction': 'Trigger 1000000, 2 \n'}},
            'Y': {self.qubit_name: {'duration': 2,
                                    'instruction': 'Trigger 0100000, 2 \n'}},
            'I': {self.qubit_name: {'duration': None,
                                    'instruction': 'wait {} \n'}},
            'RO': {self.qubit_name: {'duration': 8,
                                     'instruction': 'Trigger 0010000, 2 \n'}}
                                }

    def test_qasm_seq_T1(self):
        times = np.linspace(20e-9, 50e-6, 61)
        qasm_file = sq_qasm.T1(self.qubit_name, times)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_OffOn(self):
        qasm_file = sq_qasm.off_on(self.qubit_name)
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    # def test_qasm_seq_AllXY(self):
    #     qasm_file = sq_qasm.AllXY(self.qubit_name, times)
    #     asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
    #     asm = Assembler.Assembler(asm_file.name)
    #     instructions = asm.convert_to_instructions()

    # def test_qasm_seq_Rabi(self):
    #     qasm_file = sq_qasm.Rabi(self.qubit_name, amps)
    #     asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
    #     asm = Assembler.Assembler(asm_file.name)
    #     instructions = asm.convert_to_instructions()

    # def test_qasm_seq_Ramsey(self):
    #     qasm_file = sq_qasm.Ramsey(self.qubit_name, times)
    #     asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
    #     asm = Assembler.Assembler(asm_file.name)
    #     instructions = asm.convert_to_instructions()

    # def test_qasm_seq_echo(self):
    #     qasm_file = sq_qasm.echo(self.qubit_name, times)
    #     asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
    #     asm = Assembler.Assembler(asm_file.name)
    #     instructions = asm.convert_to_instructions()

    # def test_qasm_seq_off_on(self):
    #     qasm_file = sq_qasm.off_on(self.qubit_name)
    #     asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
    #     asm = Assembler.Assembler(asm_file.name)
    #     instructions = asm.convert_to_instructions()

    # def test_qasm_seq_butterfly(self):
    #     qasm_file = sq_qasm.butterfly(self.qubit_name)
    #     asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
    #     asm = Assembler.Assembler(asm_file.name)
    #     instructions = asm.convert_to_instructions()

    # def test_qasm_seq_randomized_benchmarking(self):
    #     qasm_file = sq_qasm.randomized_benchmarking(self.qubit_name)
    #     asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
    #     asm = Assembler.Assembler(asm_file.name)
    #     instructions = asm.convert_to_instructions()

    # def test_qasm_seq_MotzoiXY(self):
    #     qasm_file = sq_qasm.MotzoiXY(self.qubit_name)
    #     asm_file=qasm_to_asm(qasm_file.name, self.operation_dict)
    #     asm = Assembler.Assembler(asm_file.name)
    #     instructions = asm.convert_to_instructions()



    @classmethod
    def tearDownClass(self):
        pass
        # self.qubit.close()




class Test_qasm_to_asm(TestCase):
    @classmethod
    def setUpClass(self):
        # try:
        #     self.qubit = Transmon('q0_test', server_name=None)
        # except:
        #     self.qubit = Instrument.find_instrument('q0_test').close()
        #     self.qubit = Transmon('q0_test', server_name=None)
        self.base_qasm_path = join(dirname(__file__), 'qasm_files')
        self.qubit_name = 'q0'  # self.qubit.name

        # a sample operation dictionary for testing
        self.operation_dict = {
            'init_all': {'instruction': 'WaitReg r0 \n'},
            'X': {self.qubit_name: {'duration': 2,
                                    'instruction': 'Trigger 1000000, 2 \n'}},
            'Y': {self.qubit_name: {'duration': 2,
                                    'instruction': 'Trigger 0100000, 2 \n'}},
            'I': {self.qubit_name: {'duration': None,
                                    'instruction': 'wait {} \n'}},
            'RO': {self.qubit_name: {'duration': 8,
                                     'instruction': 'Trigger 0010000, 2 \n'}}
                                }

    def test_empty_qasm_file(self):
        filename = join(self.base_qasm_path, 'T1.qasm')
        qasm_file = mopen(filename, mode='w')
        qasm_file.writelines('qubit {} \n'.format(self.qubit_name))
        qasm_file.close()
        asm_file = qasm_to_asm(qasm_file.name, self.operation_dict)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_invalid_command(self):
        filename = join(self.base_qasm_path, 'T1.qasm')
        qasm_file = mopen(filename, mode='w')
        qasm_file.writelines('qubit {} \n'.format(self.qubit_name))
        qasm_file.writelines('Xbla {}     # invalid cmd\n'.format(
                             self.qubit_name))
        qasm_file.close()
        with self.assertRaises(ValueError):
            qasm_to_asm(qasm_file.name, self.operation_dict)







