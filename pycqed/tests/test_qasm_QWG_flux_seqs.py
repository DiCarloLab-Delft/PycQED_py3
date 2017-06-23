"""
This module contains tests for the QASM compiler by Xiang Fu
"""
import unittest
import sys
import numpy as np
import pycqed as pq
from io import StringIO
from pycqed.utilities import general as gen
from qcodes.instrument.base import Instrument
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
    get_timepoints_from_label, get_timetuples_since_event


class Test_QWG_flux_seqs(unittest.TestCase):

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

    def test_qwg_chevron(self):
        qasm_file = qwfs.chevron_block_seq('q0', 'q1', RO_target='q0',
                                           no_of_points=5)
        qasm_fn = qasm_file.name
        qumis_fn = join(self.test_file_dir, "output.qumis")
        compiler = qcx.QASM_QuMIS_Compiler(self.simple_config_fn,
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

    def test_qwg_swapN(self):
        nr_pulses = [1, 3, 5, 11]
        qasm_file = qwfs.SWAPN('q0', 'q1', RO_target='q0',
                               nr_pulses=nr_pulses)
        qasm_fn = qasm_file.name
        qumis_fn = join(self.test_file_dir, "output.qumis")
        compiler = qcx.QASM_QuMIS_Compiler(self.simple_config_fn,
                                           verbosity_level=2)
        compiler.compile(qasm_fn, qumis_fn)
        qumis = compiler.qumis_instructions
        m = open(compiler.qumis_fn).read()
        qumis_from_file = m.splitlines()
        self.assertEqual(qumis, qumis_from_file)
        self.assertEqual(compiler.qumis_instructions[2], 'Exp_Start: ')
        self.assertEqual(compiler.qumis_instructions[-1], self.jump_to_start)

        for i in range(3):
            time_pts = get_timepoints_from_label(
                target_label='square', timing_grid=compiler.timing_grid,
                start_label='qwg_trigger_{}'.format(i),
                end_label='ro')
            print(len(time_pts['target_tps']))
            self.assertEqual(len(time_pts['target_tps']), nr_pulses[i])

        time_pts = get_timepoints_from_label(
            target_label='ro', timing_grid=compiler.timing_grid)
        self.assertEqual(len(time_pts['target_tps']), len(nr_pulses)+4)
        # finally test that it can be converted into valid instructions
        asm = Assembler(qumis_fn)
        asm.convert_to_instructions()

    def test_time_tuples_since_event(self):
        nr_pulses = [1, 3, 5, 11]
        qasm_file = qwfs.SWAPN('q0', 'q1', RO_target='q0',
                               nr_pulses=nr_pulses)
        qasm_fn = qasm_file.name
        qumis_fn = join(self.test_file_dir, "output.qumis")
        compiler = qcx.QASM_QuMIS_Compiler(self.simple_config_fn,
                                           verbosity_level=2)
        compiler.compile(qasm_fn, qumis_fn)

        for i in range(3):
            time_tuples = get_timetuples_since_event(
                start_label='qwg_trigger_{}'.format(i),
                target_labels=['square', 'cz'],
                timing_grid=compiler.timing_grid, end_label='ro')

            self.assertEqual(len(time_tuples), nr_pulses[i])
            for time_tuple in time_tuples:
                self.assertEqual(time_tuple[1], 'square')
                self.assertGreater(time_tuple[0], 0)

    @unittest.expectedFailure
    def test_QWG_flux_QASM_sweep(self):
        qasm_file = qwfs.chevron_block_seq('QL', 'QR', no_of_points=10)
        qasm_fn = qasm_file.name

        s = swf.QWG_flux_QASM_Sweep(qasm_fn, config_fn,
                                    CBox=None, QWG_flux_lutman=None)
