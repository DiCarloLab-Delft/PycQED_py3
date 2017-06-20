"""
This module contains tests for the QASM compiler by Xiang Fu
"""
import unittest
import sys
import numpy as np
import pycqed as pq
from io import StringIO
from pycqed.utilities import general as gen
from pycqed.instrument_drivers.physical_instruments._controlbox import qasm_compiler as qc
from pycqed.instrument_drivers.physical_instruments._controlbox.Assembler \
    import Assembler
from os.path import join
from pycqed.measurement.waveform_control_CC import \
    single_qubit_qasm_seqs as sq_qasm


class Test_compiler(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.test_file_dir = join(
            pq.__path__[0], 'tests', 'qasm_files')
        self.config_fn = join(self.test_file_dir, 'config.json')

        self.jump_to_start = ("beq r14, r14, Exp_Start " +
                              "\t# Jump to start ad nauseam")

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
        self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
        self.assertEqual(compiler.qumis_instructions[-1], self.jump_to_start)

        # finally test that it can be converted into valid instructions
        asm = Assembler(qumis_fn)
        asm.convert_to_instructions()

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
                                'mx90',
                                'my90'}
        self.assertEqual(
            set(compiler.luts[0].keys()), allowed_single_q_ops)  # MW and Flux

    def test_converting_CBox_pulses_to_qumis(self):

        qasm_fn = join(self.test_file_dir, 'single_op.qasm')
        qumis_fn = join(self.test_file_dir, "output.qumis")
        compiler = qc.QASM_QuMIS_Compiler(self.config_fn,
                                          verbosity_level=6)
        compiler.compile(qasm_fn, qumis_fn)
        qumis = compiler.qumis_instructions
        x180_q0 = qumis[3]
        y180_q0 = qumis[5]
        x90_q0 = qumis[7]
        y90_q0 = qumis[9]
        mx90_q0 = qumis[11]
        my90_q0 = qumis[13]

        self.assertEqual(x180_q0, 'pulse 0000, 0000, 1001')
        self.assertEqual(y180_q0, 'pulse 0000, 0000, 1010')
        self.assertEqual(x90_q0, 'pulse 0000, 0000, 1011')
        self.assertEqual(y90_q0, 'pulse 0000, 0000, 1100')
        self.assertEqual(mx90_q0, 'pulse 0000, 0000, 1101')
        self.assertEqual(my90_q0, 'pulse 0000, 0000, 1110')

    def test_converting_triggers_to_qumis(self):
        qasm_fn = join(self.test_file_dir, 'single_op.qasm')
        qumis_fn = join(self.test_file_dir, "output.qumis")
        compiler = qc.QASM_QuMIS_Compiler(self.config_fn,
                                          verbosity_level=0)
        compiler.compile(qasm_fn, qumis_fn)
        qumis = compiler.qumis_instructions

        x180_q1 = [qumis[15], qumis[17]]
        y180_q1 = [qumis[19], qumis[21]]
        x90_q1 = [qumis[23], qumis[25]]
        y90_q1 = [qumis[27], qumis[29]]
        mx90_q1 = [qumis[31], qumis[33]]
        my90_q1 = [qumis[35], qumis[37]]

        self.assertEqual(x180_q1[0], 'trigger 0100000, 1')
        self.assertEqual(x180_q1[1], 'trigger 1100000, 2')

        self.assertEqual(y180_q1[0], 'trigger 0010000, 1')
        self.assertEqual(y180_q1[1], 'trigger 1010000, 2')

        self.assertEqual(x90_q1[0], 'trigger 0110000, 1')
        self.assertEqual(x90_q1[1], 'trigger 1110000, 2')

        self.assertEqual(y90_q1[0], 'trigger 0001000, 1')
        self.assertEqual(y90_q1[1], 'trigger 1001000, 2')

        self.assertEqual(mx90_q1[0], 'trigger 0101000, 1')
        self.assertEqual(mx90_q1[1], 'trigger 1101000, 2')

        self.assertEqual(my90_q1[0], 'trigger 0011000, 1')
        self.assertEqual(my90_q1[1], 'trigger 1011000, 2')


class Test_single_qubit_seqs(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.test_file_dir = join(
            pq.__path__[0], 'tests', 'qasm_files')
        self.config_fn = join(self.test_file_dir, 'config.json')
        self.qubit_name = 'q0'
        self.jump_to_start = ("beq r14, r14, Exp_Start " +
                              "\t# Jump to start ad nauseam")

        self.times = gen.gen_sweep_pts(start=100e-9, stop=5e-6, step=200e-9)

    def test_qasm_seq_T1(self):
        qasm_file = sq_qasm.T1(self.qubit_name, self.times)
        qasm_fn = qasm_file.name
        qumis_fn = join(self.test_file_dir, "T1_xf.qumis")
        compiler = qc.QASM_QuMIS_Compiler(self.config_fn,
                                          verbosity_level=6)
        compiler.compile(qasm_fn, qumis_fn)


        asm = Assembler(qumis_fn)
        asm.convert_to_instructions()

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
            asm.convert_to_instructions()

            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(
                compiler.qumis_instructions[-1], self.jump_to_start)

    def test_qasm_seq_MotzoiXY(self):
        for q_name in ['q0', 'q1']:
            qasm_file = sq_qasm.two_elt_MotzoiXY(q_name)
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir,
                            "motzoi_{}.qumis".format(q_name))
            compiler = qc.QASM_QuMIS_Compiler(self.config_fn,
                                              verbosity_level=0)
            compiler.compile(qasm_fn, qumis_fn)
            asm = Assembler(qumis_fn)
            asm.convert_to_instructions()

            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(
                compiler.qumis_instructions[-1], self.jump_to_start)

    def test_qasm_seq_OffOn(self):
        for q_name in ['q0', 'q1']:
            qasm_file = sq_qasm.off_on(q_name)
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir,
                            "off_on_{}.qumis".format(q_name))
            compiler = qc.QASM_QuMIS_Compiler(self.config_fn,
                                              verbosity_level=0)
            compiler.compile(qasm_fn, qumis_fn)
            asm = Assembler(qumis_fn)
            asm.convert_to_instructions()

            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(
                compiler.qumis_instructions[-1], self.jump_to_start)

    @unittest.expectedFailure
    def test_qasm_seq_ramsey(self):
        for q_name in ['q0', 'q1']:
            qasm_file = sq_qasm.Ramsey(q_name, times=self.times)
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir,
                            "Ramsey_{}.qumis".format(q_name))
            compiler = qc.QASM_QuMIS_Compiler(self.config_fn,
                                              verbosity_level=0)
            compiler.compile(qasm_fn, qumis_fn)
            asm = Assembler(qumis_fn)
            asm.convert_to_instructions()

            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(
                compiler.qumis_instructions[-1], self.jump_to_start)

    def test_qasm_seq_echo(self):
        for q_name in ['q0', 'q1']:
            qasm_file = sq_qasm.echo(q_name, times=self.times)
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir,
                            "echo_{}.qumis".format(q_name))
            compiler = qc.QASM_QuMIS_Compiler(self.config_fn,
                                              verbosity_level=0)
            compiler.compile(qasm_fn, qumis_fn)
            asm = Assembler(qumis_fn)
            asm.convert_to_instructions()

            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(
                compiler.qumis_instructions[-1], self.jump_to_start)

    def test_qasm_seq_butterfly(self):
        for q_name in ['q0', 'q1']:
            qasm_file = sq_qasm.butterfly(q_name)
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir,
                            "butterfly_{}.qumis".format(q_name))
            compiler = qc.QASM_QuMIS_Compiler(self.config_fn,
                                              verbosity_level=0)
            compiler.compile(qasm_fn, qumis_fn)
            asm = Assembler(qumis_fn)
            asm.convert_to_instructions()

            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(
                compiler.qumis_instructions[-1], self.jump_to_start)

    def test_qasm_seq_randomized_benchmarking(self):
        ncl = [2, 4, 6, 20]
        nr_seeds = 10
        for q_name in ['q0', 'q1']:
            qasm_file = sq_qasm.randomized_benchmarking(q_name, ncl, nr_seeds)
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir,
                            "randomized_benchmarking_{}.qumis".format(q_name))
            compiler = qc.QASM_QuMIS_Compiler(self.config_fn,
                                              verbosity_level=0)
            compiler.compile(qasm_fn, qumis_fn)
            asm = Assembler(qumis_fn)
            asm.convert_to_instructions()

            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(
                compiler.qumis_instructions[-1], self.jump_to_start)


class Capturing(list):

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout
