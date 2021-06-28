"""
This module contains tests for the QASM compiler by Xiang Fu
"""
import unittest
import sys
import numpy as np
import pycqed as pq
from io import StringIO
from pycqed.utilities import general as gen
from pycqed.measurement.waveform_control_CC import qasm_compiler as qcx
from pycqed.instrument_drivers.physical_instruments._controlbox.Assembler \
    import Assembler
from os.path import join
from pycqed.measurement.waveform_control_CC import \
    single_qubit_qasm_seqs as sqqs

from pycqed.measurement.waveform_control_CC import \
    multi_qubit_qasm_seqs as mqqs
from pycqed.measurement.waveform_control_CC import \
    QWG_fluxing_seqs as qwfs

from pycqed.measurement.waveform_control_CC.qasm_compiler_helpers import \
    get_timepoints_from_label


@unittest.skip(reason="As decided in #635 this tests are considered non-important")
class Test_compiler(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.test_file_dir = join(
            pq.__path__[0], 'tests', 'qasm_files')
        self.config_fn = join(self.test_file_dir, 'config.json')

        self.times = gen.gen_sweep_pts(start=100e-9, stop=5e-6, step=200e-9)
        self.clocks = np.round(self.times/5e-9).astype(int)
        self.simple_config_fn = join(self.test_file_dir, 'config_simple.json')
        self.jump_to_start = ("beq r14, r14, Exp_Start " +
                              "\t# Jump to start ad nauseam")

    def test_compiler_example(self):
        qasm_fn = join(self.test_file_dir, 'dev_test.qasm')
        qumis_fn = join(self.test_file_dir, "output.qumis")
        compiler = qcx.QASM_QuMIS_Compiler(self.config_fn,
                                           verbosity_level=6)
        compiler.compile(qasm_fn, qumis_fn)
        qumis = compiler.qumis_instructions
        m = open(compiler.qumis_fn).read()
        qumis_from_file = m.splitlines()
        self.assertEqual(qumis, qumis_from_file)
        self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')

        # Test that the final wait after the last instruction is not trimmed
        self.assertEqual(compiler.qumis_instructions[-2], 'wait 60')
        self.assertEqual(compiler.qumis_instructions[-1], self.jump_to_start)

        # finally test that it can be converted into valid instructions
        asm = Assembler(qumis_fn)
        asm.convert_to_instructions()

    def test_methods_of_compiler(self):
        compiler = qcx.QASM_QuMIS_Compiler()

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
        compiler = qcx.QASM_QuMIS_Compiler()
        self.assertEqual(compiler.config_filename, '')
        self.assertNotEqual(compiler.config_filename, self.config_fn)
        compiler.load_config(self.config_fn)
        self.assertEqual(compiler.config_filename, self.config_fn)

        hardware_spec_keys = {'qubit list', 'init time',
                              'cycle time', 'qubit_cfgs'}
        self.assertEqual(set(compiler.hardware_spec.keys()),
                         hardware_spec_keys)

        self.assertEqual(compiler.qubit_map, {'q0': 0, 'q1': 1,
                                              'ql': 0, 'qr': 1})

        self.assertEqual(len(compiler.luts), 2)  # MW and Flux
        allowed_single_q_ops = {'i',
                                'x180',
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
        compiler = qcx.QASM_QuMIS_Compiler(self.config_fn,
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
        compiler = qcx.QASM_QuMIS_Compiler(self.config_fn,
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

    def test_qasm_wait_timing_pulse_T1(self):
        # Tests the timing of the qasm sequences using a T1 sequence
        qasm_file = sqqs.T1('q0', self.times)
        qasm_fn = qasm_file.name
        qumis_fn = join(self.test_file_dir, "T1_xf.qumis")
        compiler = qcx.QASM_QuMIS_Compiler(self.simple_config_fn,
                                           verbosity_level=2)
        compiler.compile(qasm_fn, qumis_fn)
        asm = Assembler(qumis_fn)
        asm.convert_to_instructions()
        instrs = compiler.qumis_instructions
        self.assertEqual(instrs[2], 'Exp_Start: ')
        wait_instrs = instrs[5:-4:4]
        # -4 in slicing exists to take out the calibration points
        for clock, wait_instr in zip(self.clocks[:-4], wait_instrs[:-4]):
            exp_wait = clock + 4  # +4 includes the length of the pulse
            instr_wait = int(wait_instr.split()[1])
            self.assertEqual(instr_wait, exp_wait)

        init_instrs = instrs[3:-4:4]
        init_time = 200e-6/5e-9
        RO_time = 300e-9/5e-9

        self.assertEqual(init_instrs[0],
                         'wait {:d}'.format(int(init_time)))
        for init_instr in init_instrs[1:]:
            self.assertEqual(init_instr,
                             'wait {:d}'.format(int(init_time+RO_time)))

        self.assertEqual(
            instrs[-1], self.jump_to_start)

    def test_get_timepoints(self):
        qasm_file = qwfs.chevron_block_seq('q0', 'q1', RO_target='q0',
                                           no_of_points=5)
        qasm_fn = qasm_file.name
        qumis_fn = join(self.test_file_dir, "output.qumis")
        compiler = qcx.QASM_QuMIS_Compiler(self.simple_config_fn,
                                           verbosity_level=2)
        compiler.compile(qasm_fn, qumis_fn)
        time_pts = get_timepoints_from_label(
            compiler.timing_grid, 'cz',
            start_label='qwg_trigger_1',
            end_label='ro')

        time_pts_ns = get_timepoints_from_label(
            compiler.timing_grid, 'cz',
            start_label='qwg_trigger_1',
            end_label='ro', convert_clk_to_ns=True)

        self.assertEqual(time_pts['start_tp'].absolute_time*5,
                         time_pts_ns['start_tp'].absolute_time)
        for tp, tp_ns in zip(time_pts['target_tps'],
                             time_pts_ns['target_tps']):
            self.assertEqual(tp.absolute_time*5, tp_ns.absolute_time)
        self.assertEqual(time_pts['end_tp'].absolute_time*5,
                         time_pts_ns['end_tp'].absolute_time)

        cz_pts = time_pts['target_tps']
        self.assertEqual(len(cz_pts), 1)
        t_cz = cz_pts[0].absolute_time
        # Time of time points is in clocks
        # 500ns specified
        self.assertEqual(cz_pts[0].following_waiting_time, 100)
        t_ro = time_pts['end_tp'].absolute_time
        self.assertEqual(t_ro-t_cz, 100)

        t_trigg = time_pts['start_tp'].absolute_time
        self.assertEqual(t_cz-t_trigg, 7)

        time_pts = get_timepoints_from_label(
            target_label='cz', timing_grid=compiler.timing_grid,
            start_label=None,
            end_label=None)
        self.assertEqual(len(time_pts['target_tps']), 5)

    def test_qasm_wait_timing_trigger_T1(self):
        # Tests the timing of the qasm sequences using a T1 sequence
        # 'q1' contains "trigger" instructions
        qasm_file = sqqs.T1('q1', self.times)
        qasm_fn = qasm_file.name
        qumis_fn = join(self.test_file_dir, "T1_xf.qumis")
        compiler = qcx.QASM_QuMIS_Compiler(self.simple_config_fn,
                                           verbosity_level=2)
        compiler.compile(qasm_fn, qumis_fn)
        asm = Assembler(qumis_fn)
        asm.convert_to_instructions()
        instrs = compiler.qumis_instructions
        self.assertEqual(instrs[2], 'Exp_Start: ')
        wait_instrs = instrs[7:-4:6]
        # -4 in slicing exists to take out the calibration points
        for clock, wait_instr in zip(self.clocks[:-4], wait_instrs[:-4]):
            exp_wait = clock + 4  # +4 includes the length of the pulse
            instr_wait = int(wait_instr.split()[1])
            self.assertEqual(instr_wait, exp_wait)

        init_instrs = instrs[3:-4:6]
        init_time = 200e-6/5e-9 - 1  # -1 is for the prepare codeword clock
        RO_time = 300e-9/5e-9

        self.assertEqual(init_instrs[0],
                         'wait {:d}'.format(int(init_time)))
        for init_instr in init_instrs[1:-4]:
            self.assertEqual(init_instr,
                             'wait {:d}'.format(int(init_time+RO_time)))

        self.assertEqual(
            instrs[-1], self.jump_to_start)

    def test_equivalent_maps_custom_qubit_name(self):
        # generate the same qasm and compile using two qubit names that
        # refer to the same qubit_cfg in the hardware spec
        qumis_instrs = [[], []]
        for i, q_name in enumerate(['q0', 'ql']):
            qasm_file = sqqs.AllXY(q_name)
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir,
                            "allxy_{}.qumis".format(q_name))
            compiler = qcx.QASM_QuMIS_Compiler(self.config_fn,
                                               verbosity_level=0)
            compiler.compile(qasm_fn, qumis_fn)
            qumis_instrs[i] = compiler.qumis_instructions
        self.assertEqual(qumis_instrs[0], qumis_instrs[1])


@unittest.skip(reason="As decided in #635 this tests are considered non-important")
class Test_single_qubit_seqs(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.test_file_dir = join(
            pq.__path__[0], 'tests', 'qasm_files')
        self.config_fn = join(self.test_file_dir, 'config.json')
        self.simple_config_fn = join(self.test_file_dir, 'config_simple.json')
        self.qubit_name = 'q0'
        self.jump_to_start = ("beq r14, r14, Exp_Start " +
                              "\t# Jump to start ad nauseam")
        self.times = gen.gen_sweep_pts(start=100e-9, stop=5e-6, step=200e-9)
        self.clocks = np.round(self.times/5e-9).astype(int)

    def test_qasm_seq_allxy(self):
        for q_name in ['q0', 'q1']:
            qasm_file = sqqs.AllXY(q_name)
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir,
                            "allxy_{}.qumis".format(q_name))
            compiler = qcx.QASM_QuMIS_Compiler(self.config_fn,
                                               verbosity_level=0)
            compiler.compile(qasm_fn, qumis_fn)
            asm = Assembler(qumis_fn)
            asm.convert_to_instructions()

            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(
                compiler.qumis_instructions[-1], self.jump_to_start)

    def test_qasm_seq_MotzoiXY(self):
        for q_name in ['q0', 'q1']:
            qasm_file = sqqs.two_elt_MotzoiXY(q_name)
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir,
                            "motzoi_{}.qumis".format(q_name))
            compiler = qcx.QASM_QuMIS_Compiler(self.config_fn,
                                               verbosity_level=0)
            compiler.compile(qasm_fn, qumis_fn)
            asm = Assembler(qumis_fn)
            asm.convert_to_instructions()

            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(
                compiler.qumis_instructions[-1], self.jump_to_start)

    def test_qasm_seq_OffOn(self):
        for q_name in ['q0', 'q1']:
            qasm_file = sqqs.off_on(q_name)
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir,
                            "off_on_{}.qumis".format(q_name))
            compiler = qcx.QASM_QuMIS_Compiler(self.config_fn,
                                               verbosity_level=0)
            compiler.compile(qasm_fn, qumis_fn)
            asm = Assembler(qumis_fn)
            asm.convert_to_instructions()

            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(
                compiler.qumis_instructions[-1], self.jump_to_start)

    def test_qasm_seq_ramsey(self):
        for q_name in ['q0', 'q1']:
            qasm_file = sqqs.Ramsey(q_name, times=self.times)
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir,
                            "Ramsey_{}.qumis".format(q_name))
            compiler = qcx.QASM_QuMIS_Compiler(self.simple_config_fn,
                                               verbosity_level=0)
            compiler.compile(qasm_fn, qumis_fn)
            asm = Assembler(qumis_fn)
            asm.convert_to_instructions()

            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(
                compiler.qumis_instructions[-1], self.jump_to_start)

    def test_qasm_seq_echo(self):
        for q_name in ['q0', 'q1']:
            qasm_file = sqqs.echo(q_name, times=self.times)
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir,
                            "echo_{}.qumis".format(q_name))
            compiler = qcx.QASM_QuMIS_Compiler(self.config_fn,
                                               verbosity_level=0)
            compiler.compile(qasm_fn, qumis_fn)
            asm = Assembler(qumis_fn)
            asm.convert_to_instructions()

            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(
                compiler.qumis_instructions[-1], self.jump_to_start)

    def test_qasm_seq_butterfly(self):
        for q_name in ['q1', 'q1']:
            qasm_file = sqqs.butterfly(q_name, initialize=True)
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir,
                            "butterfly_{}.qumis".format(q_name))
            compiler = qcx.QASM_QuMIS_Compiler(self.simple_config_fn,
                                               verbosity_level=2)
            compiler.compile(qasm_fn, qumis_fn)
            asm = Assembler(qumis_fn)
            asm.convert_to_instructions()
            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(
                compiler.qumis_instructions[-1], self.jump_to_start)
            # The sequence should contain 6 RO instructions
            self.assertEqual(
                compiler.qumis_instructions.count('trigger 0000001, 3'), 6)

    def test_qasm_seq_randomized_benchmarking(self):
        ncl = [2, 4, 6, 20]
        nr_seeds = 10
        for q_name in ['q0', 'q1']:
            qasm_file = sqqs.randomized_benchmarking(q_name, ncl, nr_seeds)
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir,
                            "randomized_benchmarking_{}.qumis".format(q_name))
            compiler = qcx.QASM_QuMIS_Compiler(self.config_fn,
                                               verbosity_level=0)
            compiler.compile(qasm_fn, qumis_fn)
            asm = Assembler(qumis_fn)
            asm.convert_to_instructions()

            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(
                compiler.qumis_instructions[-1], self.jump_to_start)

    def test_restless_RB_seq(self):
        ncl = [10]
        nr_seeds = 15
        for q_name in ['q0', 'q1']:
            qasm_file = sqqs.randomized_benchmarking(
                q_name, ncl, nr_seeds, restless=True, cal_points=False)
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir,
                            "randomized_benchmarking_{}.qumis".format(q_name))
            compiler = qcx.QASM_QuMIS_Compiler(self.simple_config_fn,
                                               verbosity_level=0)
            compiler.compile(qasm_fn, qumis_fn)
            asm = Assembler(qumis_fn)
            asm.convert_to_instructions()

            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(
                compiler.qumis_instructions[-1], self.jump_to_start)

            self.assertEqual(
                compiler.qumis_instructions.count('trigger 0000001, 3'), 15)


@unittest.skip(reason="As decided in #635 this tests are considered non-important")
class Test_multi_qubit_seqs(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.test_file_dir = join(
            pq.__path__[0], 'tests', 'qasm_files')
        self.config_fn = join(self.test_file_dir, 'config.json')
        self.simple_config_fn = join(self.test_file_dir, 'config_simple.json')
        self.qubit_name = 'q0'
        self.jump_to_start = ("beq r14, r14, Exp_Start " +
                              "\t# Jump to start ad nauseam")
        self.times = gen.gen_sweep_pts(start=100e-9, stop=5e-6, step=200e-9)
        self.clocks = np.round(self.times/5e-9).astype(int)

    def test_two_qubit_off_on(self):
        for RO_target in ['q0', 'q1', 'all']:
            qasm_file = mqqs.two_qubit_off_on('q0', 'q1', RO_target='all')
            qasm_fn = qasm_file.name
            qumis_fn = join(self.test_file_dir, "TwoQ_off_on.qumis")
            compiler = qcx.QASM_QuMIS_Compiler(self.simple_config_fn,
                                               verbosity_level=2)
            compiler.compile(qasm_fn, qumis_fn)
            asm = Assembler(qumis_fn)
            asm.convert_to_instructions()

            self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
            self.assertEqual(
                compiler.qumis_instructions[-1], self.jump_to_start)

            self.assertEqual(
                compiler.qumis_instructions.count('trigger 0000001, 3'), 4)

    def test_chevron_block_seq(self):
        qasm_file = qwfs.chevron_block_seq('q0', 'q1', RO_target='q0',
                                           no_of_points=5)
        qasm_fn = qasm_file.name
        qumis_fn = join(self.test_file_dir, "chevron_block.qumis")
        compiler = qcx.QASM_QuMIS_Compiler(self.simple_config_fn,
                                           verbosity_level=6)
        compiler.compile(qasm_fn, qumis_fn)
        asm = Assembler(qumis_fn)
        asm.convert_to_instructions()

        self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
        self.assertEqual(
            compiler.qumis_instructions[-1], self.jump_to_start)

        self.assertEqual(
            compiler.qumis_instructions.count('trigger 0000001, 3'), 5)

        compiler.timing_event_list


@unittest.skip(reason="As decided in #635 this tests are considered non-important")
class Capturing(list):

    def __enter__(self):
        self._stdout = sys.stdout
        sys.stdout = self._stringio = StringIO()
        return self

    def __exit__(self, *args):
        self.extend(self._stringio.getvalue().splitlines())
        sys.stdout = self._stdout
