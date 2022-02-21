import io
import unittest
import contextlib
import numpy as np

import pycqed.instrument_drivers.physical_instruments.ZurichInstruments.UHFQA_core as UHF

class Test_UHFQA_core(unittest.TestCase):
    @classmethod
    def setup_class(cls):
        cls.uhf = UHF.UHFQA_core(name='MOCK_UHF', server='emulator',
                            device='dev2109', interface='1GbE')

        cls.uhf.reset_waveforms_zeros()

    @classmethod
    def teardown_class(cls):
        cls.uhf.close()

    def test_instantiation(self):
        self.assertEqual(Test_UHFQA_core.uhf.devname, 'dev2109')

    def test_assure_ext_clock(self):
        self.uhf.assure_ext_clock()
        self.assertEqual(self.uhf.system_extclk(), 1)

    def test_clock_freq(self):
        self.assertEqual(self.uhf.clock_freq(), 1.8e9)

    def test_load_default_settings(self):
        self.uhf.load_default_settings()
        self.assertEqual(self.uhf.download_crosstalk_matrix().tolist(), np.eye(10).tolist())

    def test_print_overview(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.uhf.print_overview()
        f.seek(0)
        self.assertIn('Crosstalk overview', f.read())

    def test_print_correlation_overview(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.uhf.print_correlation_overview()
        f.seek(0)
        self.assertIn('Correlations overview', f.read())

    def test_print_deskew_overview(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.uhf.print_deskew_overview()
        f.seek(0)
        self.assertIn('Deskew overview', f.read())

    def test_print_crosstalk_overview(self):
      f = io.StringIO()
      with contextlib.redirect_stdout(f):
          self.uhf.print_crosstalk_overview()
      f.seek(0)
      self.assertIn('Crosstalk overview', f.read())

    def test_print_integration_overview(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.uhf.print_integration_overview()
        f.seek(0)
        self.assertIn('Integration overview', f.read())

    def test_print_rotations_overview(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.uhf.print_rotations_overview()
        f.seek(0)
        self.assertIn('Rotations overview', f.read())

    def test_print_thresholds_overview(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.uhf.print_thresholds_overview()
        f.seek(0)
        self.assertIn('Thresholds overview', f.read())

    def test_print_user_regs_overview(self):
        f = io.StringIO()
        with contextlib.redirect_stdout(f):
            self.uhf.print_user_regs_overview()
        f.seek(0)
        self.assertIn('User registers overview', f.read())

    def test_minimum_holdoff(self):
        # Test without averaging
        self.uhf.qas_0_integration_length(128)
        self.uhf.qas_0_result_averages(1)
        self.uhf.qas_0_delay(0)
        assert self.uhf.minimum_holdoff() == 800/1.8e9
        self.uhf.qas_0_delay(896)
        assert self.uhf.minimum_holdoff() == (896+16)/1.8e9
        self.uhf.qas_0_integration_length(2048)
        assert self.uhf.minimum_holdoff() == (2048)/1.8e9

        # Test with averaging
        self.uhf.qas_0_result_averages(16)
        self.uhf.qas_0_delay(0)
        self.uhf.qas_0_integration_length(128)
        assert self.uhf.minimum_holdoff() == 2560/1.8e9
        self.uhf.qas_0_delay(896)
        assert self.uhf.minimum_holdoff() == 2560/1.8e9
        self.uhf.qas_0_integration_length(4096)
        assert self.uhf.minimum_holdoff() == 4096/1.8e9

    def test_crosstalk_matrix(self):
        mat = np.random.random((10, 10))
        self.uhf.upload_crosstalk_matrix(mat)
        new_mat = self.uhf.download_crosstalk_matrix()
        assert np.allclose(mat, new_mat)

    def test_reset_crosstalk_matrix(self):
        mat = np.random.random((10, 10))
        self.uhf.upload_crosstalk_matrix(mat)
        self.uhf.reset_crosstalk_matrix()
        reset_mat = self.uhf.download_crosstalk_matrix()
        assert np.allclose(np.eye(10), reset_mat)

    def test_reset_acquisition_params(self):
        for i in range(16):
            self.uhf.set(f'awgs_0_userregs_{i}', i)

        self.uhf.reset_acquisition_params()
        values = [self.uhf.get(f'awgs_0_userregs_{i}') for i in range(16)]
        assert values == [0]*16

    def test_correlation_settings(self):
        self.uhf.qas_0_correlations_5_enable(1)
        self.uhf.qas_0_correlations_5_source(3)

        assert self.uhf.qas_0_correlations_5_enable() == 1
        assert self.uhf.qas_0_correlations_5_source() == 3

    def test_thresholds_correlation_settings(self):
        self.uhf.qas_0_thresholds_5_correlation_enable(1)
        self.uhf.qas_0_thresholds_5_correlation_source(3)

        assert self.uhf.qas_0_thresholds_5_correlation_enable() == 1
        assert self.uhf.qas_0_thresholds_5_correlation_source() == 3

    def test_reset_correlation_settings(self):
        self.uhf.qas_0_correlations_5_enable(1)
        self.uhf.qas_0_correlations_5_source(3)
        self.uhf.qas_0_thresholds_5_correlation_enable(1)
        self.uhf.qas_0_thresholds_5_correlation_source(3)

        self.uhf.reset_correlation_params()

        assert self.uhf.qas_0_correlations_5_enable() == 0
        assert self.uhf.qas_0_correlations_5_source() == 0
        assert self.uhf.qas_0_thresholds_5_correlation_enable() == 0
        assert self.uhf.qas_0_thresholds_5_correlation_source() == 0

    def test_reset_rotation_params(self):
        self.uhf.qas_0_rotations_3(1-1j)
        assert self.uhf.qas_0_rotations_3() == (1-1j)
        self.uhf.reset_rotation_params()
        assert self.uhf.qas_0_rotations_3() == (1+1j)

    def test_start(self):
        self.uhf.start()