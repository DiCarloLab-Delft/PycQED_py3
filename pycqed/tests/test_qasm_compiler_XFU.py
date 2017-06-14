"""
This module contains tests for the QASM compiler by Xiang Fu
"""
import unittest
import sys
import numpy as np
import pycqed as pq
from io import StringIO
from unittest import TestCase
from pycqed.instrument_drivers.physical_instruments._controlbox import qasm_compiler as qc
from pycqed.instrument_drivers.physical_instruments._controlbox.Assembler \
    import Assembler
from os.path import join, dirname
from copy import deepcopy
from pycqed.measurement.waveform_control_CC import \
    single_qubit_qasm_seqs as sq_qasm


class Test_compiler(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.test_file_dir = join(
            pq.__path__[0], 'tests', 'qasm_files')
        self.config_fn = join(self.test_file_dir, 'config.json')

        self.jump_to_start = "beq r14, r14, Exp_Start \t# Infinite loop"

    def test_compiler_example(self):
        qasm_fn = join(self.test_file_dir, 'dev_test.qasm')
        qumis_fn = join(self.test_file_dir, "output.qumis")
        compiler = qc.QASM_QuMIS_Compiler(self.config_fn,
                                          verbosity_level=6)
        compiler.compile(qasm_fn, qumis_fn)
        qumis = compiler.qumis_instructions
        m = open(compiler.qumis_fn).read()
        qumis_from_file = m.splitlines()
        self.assertEqual(qumis, qumis_from_file)
        # finally test that it can be converted into valid instructions
        asm = Assembler(qumis_fn)
        instructions = asm.convert_to_instructions()
        self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
        self.assertEqual(compiler.qumis_instructions[-1], self.jump_to_start)


    def test_methods_of_compiler(self):
        compiler = qc.QASM_QuMIS_Compiler()

        c_methods = set(dir(compiler))
        printing_methods = {'print_hw_timing_grid',
                            'print_program_lines',
                            'print_raw_lines',
                            'print_op_dict',
                            'print_qumis',
                            'print_raw_events',
                            'print_timing_events',
                            'print_timing_grid'}
        self.assertTrue(printing_methods.issubset(c_methods))

    def test_loading_config(self):
        with Capturing() as output:
            compiler = qc.QASM_QuMIS_Compiler()

        self.assertIn(
            'Configuration not specified. Default configuration file instrument_drivers\physical_instruments\_controlbox\config.json used.', output)
        self.assertNotEqual(compiler.config_filename, self.config_fn)
        compiler.load_config(self.config_fn)
        self.assertEqual(compiler.config_filename, self.config_fn)

        hardware_spec_keys = {'qubit list', 'init time',
                              'cycle time', 'channels'}
        self.assertEqual(set(compiler.hardware_spec.keys()),
                         hardware_spec_keys)

        self.assertEqual(compiler.qubit_map, {'q0': 0, 'q1': 1})

        self.assertEqual(len(compiler.luts), 2)  # MW and Flux
        allowed_single_q_ops = {'x180',
                                'x90',
                                'y180',
                                'y90',
                                'mx180',
                                'mx90',
                                'my180',
                                'my90'}
        self.assertEqual(
            set(compiler.luts[0].keys()), allowed_single_q_ops)  # MW and Flux


class Test_single_qubit_seqs(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.test_file_dir = join(
            pq.__path__[0], 'tests', 'qasm_files')
        self.config_fn = join(self.test_file_dir, 'config.json')
        self.qubit_name = 'q0'
        self.jump_to_start = "beq r14, r14, Exp_Start \t# Infinite loop"


    @unittest.skip('no identity')
    def test_qasm_seq_T1(self):
        times = np.linspace(20e-9, 50e-6, 61)
        qasm_file = sq_qasm.T1(self.qubit_name, times)
        qasm_fn = qasm_file.name
        qumis_fn = join(self.test_file_dir, "T1_xf.qumis")
        compiler = qc.QASM_QuMIS_Compiler(self.config_fn,
                                          verbosity_level=0)
        compiler.compile(qasm_fn, qumis_fn)
        asm = Assembler(qumis_fn)
        instructions = asm.convert_to_instructions()

    def test_qasm_seq_allxy(self):
        for q_name in ['q0', 'q1']:
            qasm_file = sq_qasm.AllXY(q_name)
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir,
                            "allxy_{}.qumis".format(q_name))
            compiler = qc.QASM_QuMIS_Compiler(self.config_fn,
                                              verbosity_level=0)
            compiler.compile(qasm_fn, qumis_fn)
            asm = Assembler(qumis_fn)
            instructions = asm.convert_to_instructions()

            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(compiler.qumis_instructions[-1], self.jump_to_start)



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
