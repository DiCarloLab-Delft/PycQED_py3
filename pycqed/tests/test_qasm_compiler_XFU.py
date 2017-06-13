"""
This module contains tests for the QASM compiler by Xiang Fu
"""

import sys
import numpy as np
import pycqed as pq
from io import StringIO
from unittest import TestCase
from pycqed.instrument_drivers.physical_instruments._controlbox import qasm_compiler as qc
from os.path import join, dirname
from copy import deepcopy


class Test_single_qubit_seqs(TestCase):

    @classmethod
    def setUpClass(self):
        self.test_file_dir = join(
            pq.__path__[0], 'tests', 'qasm_files')
        self.config_fn = join(self.test_file_dir, 'config.json')

    def test_compiler_example(self):
            qasm_fn = join(self.test_file_dir, 'dev_test.qasm')
            qumis_fn = join(self.test_file_dir, "output.qumis")
            compiler = qc.QASM_QuMIS_Compiler(self.config_fn,
                                              verbosity_level=6)
            compiler.compile(qasm_fn, qumis_fn)

    def test_methods_of_compiler(self):
        compiler = qc.QASM_QuMIS_Compiler()

        c_methods = set(dir(compiler))
        printing_methods = {'print_event_list',
                            'print_hw_timing_grid',
                            'print_lines',
                            'print_op_dict',
                            'print_qumis',
                            'print_raw_events',
                            'print_timing_events',
                            'print_timing_grid'}
        self.assertTrue(printing_methods.issubset(c_methods))


    # def test_set_config(self):
        # config_fn = r"D:\GitHub\PycQED_py3\pycqed\instrument_drivers\physical_instruments\_controlbox\config.json"
        # compiler = qc.QASM_QuMIS_Compiler(config_fn, 6)

    # def test_example_compilation(self):
    #     config_fn = r"D:\GitHub\PycQED_py3\pycqed\instrument_drivers\physical_instruments\_controlbox\config.json"
    #     qasm_fn = r"D:/Github/PycQED_py3/pycqed/tests/test_data/qasm_test/dev_test.qasm"
    #     qumis_fn = "output.qumis"
    #     compiler = qc.QASM_QuMIS_Compiler(config_fn, 6)
    #     compiler.compile(qasm_fn, qumis_fn)


def valid_operation_dictionary(operation_dict):
    '''
    checks if the operation dictionary is valid
    '''

    return True


class Capturing(list):

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout
