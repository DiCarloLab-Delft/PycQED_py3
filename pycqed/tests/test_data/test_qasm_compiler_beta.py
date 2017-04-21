
import sys
import numpy as np
from io import StringIO
from unittest import TestCase
from os.path import join, dirname
from copy import deepcopy

# from qcodes import Instrument

from pycqed.instrument_drivers.physical_instruments._controlbox \
    import qasm_compiler as qasm_compiler

from pycqed.utilities.general import mopen
from pycqed.measurement.waveform_control_CC import \
    single_qubit_qasm_seqs as sq_qasm

# from pycqed.instrument_drivers.meta_instrument.qubit_objects.qubit_object \
#     import Transmon
from pycqed.instrument_drivers.physical_instruments._controlbox \
    import Assembler

from pycqed.measurement.waveform_control_CC import qasm_to_asm as qta


class Test_Parallel_QASM_Compiler(TestCase):

    @classmethod
    def setUpClass(self):
        print("hello world.")

    def setCompiler(self, qasm_file_name=None):
        test_file_dir = os.path.join(
            pq.__path__[0], 'tests', 'test_data', "qasm_test")
        if qasm_file_name != None:
            self.qasm_file_name = os.path.join(test_file_dir,
                                                qasm_file_name)
        else:
            self.qasm_file_name = os.path.join(test_file_dir,
                                                "test_qasm.qasm")

        self.compiler = qasm_compiler.QASM_QuMIS_Compiler(self.qasm_file_name)

    def test_remove_comment(self):
        self.assertTrue(True)