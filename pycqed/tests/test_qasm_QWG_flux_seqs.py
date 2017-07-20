"""
This module contains tests for the QASM compiler by Xiang Fu
"""
import json
import unittest
import sys
import numpy as np
from pycqed.measurement.waveform_control_CC import waveform as wf
import pycqed as pq
from io import StringIO
from pycqed.utilities import general as gen
from qcodes.instrument.base import Instrument
from pycqed.measurement import sweep_functions as swf
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


from pycqed.instrument_drivers.meta_instrument import QWG_LookuptableManager \
    as qlm


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
        self.QWG_flux_lutman = qlm.QWG_FluxLookuptableManager(
            'QWG_flux_lutman')

        with open(self.simple_config_fn) as data_file:
            self.config_simple = json.load(data_file)

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
            time_tuples, end_time = get_timetuples_since_event(
                start_label='qwg_trigger_{}'.format(i),
                target_labels=['square', 'cz'],
                timing_grid=compiler.timing_grid, end_label='ro')

            self.assertEqual(len(time_tuples), nr_pulses[i])
            for time_tuple in time_tuples:
                self.assertEqual(time_tuple[1], 'square')
                self.assertGreater(time_tuple[0], 0)

    def test_QWG_flux_QASM_sweep(self):
        qasm_file = qwfs.chevron_block_seq('Q0', 'Q1', no_of_points=6)
        qasm_fn = qasm_file.name

        s = swf.QWG_flux_QASM_Sweep(
            qasm_fn=qasm_fn, config=self.config_simple,
            CBox=None, QWG_flux_lutmans=[self.QWG_flux_lutman[0]],
            verbosity_level=1, upload=False)
        s.sweep_points = np.arange(6)
        s.prepare()

    def test_QWG_flux_waveform_time_tuples(self):
        self.QWG_flux_lutman.F_amp(0.5)
        self.QWG_flux_lutman.F_length(20e-9)
        self.QWG_flux_lutman.Z_amp(0.05)  # amp of Z_compensation

        end_time = 300
        time_tuples = []
        waveform = self.QWG_flux_lutman.generate_composite_flux_pulse(
            time_tuples, end_time)
        np.testing.assert_array_equal(np.zeros(300), waveform)

        time_tuples = [(5, 'cz'), (50, 'cz'), (120, 'square')]
        waveform = self.QWG_flux_lutman.generate_composite_flux_pulse(
            time_tuples, end_time)
        # test_QWG_flux_lutman_basic_waveforms below tests that the
        # CZ_expected is what we want
        std_wfs = self.QWG_flux_lutman.standard_waveforms()
        CZ_exp = std_wfs['adiabatic_Z']
        sq_exp = std_wfs['square']

        expected_waveform = np.zeros(300)
        expected_waveform[5:5+len(CZ_exp)] = CZ_exp
        expected_waveform[50:50+len(CZ_exp)] = CZ_exp
        expected_waveform[120:120+len(sq_exp)] = sq_exp
        np.testing.assert_array_equal(waveform, expected_waveform)

    def test_QWG_flux_lutman_basic_wfs_square(self):
        """
        Test the correct generation of the basic waveforms.
        It tests the generation of the block pulse and the adiabatic pulse
        based on the parameters specified in the QWG_flux_lutman
        """
        wf_dict = self.QWG_flux_lutman.generate_standard_pulses()
        test_block = np.concatenate([0.5*np.ones(1000), np.zeros(500),
                                     -0.5*np.ones(1000)])
        np.testing.assert_array_equal(wf_dict['square'], test_block)

        simple_block = self.QWG_flux_lutman.standard_waveforms()['square']
        np.testing.assert_array_equal(simple_block, 0.5*np.ones(1000))

    def test_QWG_flux_lutman_basic_wfs_CZ(self):
        """
        Test the basic waveform CZ and CZ with phase correction
        """
        self.QWG_flux_lutman.sampling_rate(1e9)
        self.QWG_flux_lutman.F_amp(0.5)
        self.QWG_flux_lutman.F_length(1e-6)
        self.QWG_flux_lutman.F_compensation_delay(500e-9)
        self.QWG_flux_lutman.Z_amp(0.05)
        corr_len = 10
        self.QWG_flux_lutman.Z_length(corr_len*1e-9)

        CZ = self.QWG_flux_lutman.standard_waveforms()['adiabatic']
        CZ_expected = wf.martinis_flux_pulse_v2(
            length=self.QWG_flux_lutman.F_length(),
            lambda_2=self.QWG_flux_lutman.F_lambda_2(),
            lambda_3=self.QWG_flux_lutman.F_lambda_3(),
            theta_f=self.QWG_flux_lutman.F_theta_f(),
            f_01_max=self.QWG_flux_lutman.F_f_01_max(),
            J2=self.QWG_flux_lutman.F_J2(),
            E_c=self.QWG_flux_lutman.F_E_c(),
            V_per_phi0=self.QWG_flux_lutman.V_per_phi0(),
            f_interaction=self.QWG_flux_lutman.F_f_interaction(),
            f_bus=None,
            asymmetry=self.QWG_flux_lutman.F_asymmetry(),
            sampling_rate=self.QWG_flux_lutman.sampling_rate(),
            return_unit='V')
        np.testing.assert_array_equal(CZ, CZ_expected)
        CZ_phase_corr = \
            self.QWG_flux_lutman.standard_waveforms()['adiabatic_Z']

        # Test CZ_with_phase_correction
        Z_corr = np.ones(corr_len) * 0.05
        CZ_phase_corr_expected = np.concatenate([CZ, Z_corr])
        np.testing.assert_array_equal(CZ_phase_corr, CZ_phase_corr_expected)
