import numpy as np
from unittest import TestCase
# from qcodes import Instrument

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

    def test_T1_sequence(self):
        times = np.linspace(20e-9, 50e-6, 61)
        qasm_file = sq_qasm.T1(self.qubit_name, times)
        asm_file = qasm_to_asm(qasm_file.name)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()

    def test_OffOn_sequence(self):
        qasm_file = sq_qasm.off_on(self.qubit_name)
        asm_file = qasm_to_asm(qasm_file.name)
        asm = Assembler.Assembler(asm_file.name)
        instructions = asm.convert_to_instructions()


    @classmethod
    def tearDownClass(self):
        pass
        # self.qubit.close()
