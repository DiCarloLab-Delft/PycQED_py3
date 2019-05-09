import unittest
import numpy as np
import pycqed.instrument_drivers.meta_instrument.lfilt_kernel_object as lko
import pycqed.instrument_drivers.virtual_instruments.virtual_AWG8 as v8

class Test_LinDistortionKernelObject(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.k0 = lko.LinDistortionKernel('k0')
        self.AWG = v8.VirtualAWG8('DummyAWG8')
        self.k0.instr_AWG(self.AWG.name)
        self.k0.cfg_awg_channel(1)

    def setUp(self):
        self.k0.cfg_sampling_rate(2.4e9)
        self.k0.filter_model_00(
            {'model': 'high-pass', 'params': {'tau': 4.071755778296734e-05}})
        self.k0.filter_model_01({'model': 'exponential',
                                 'params': {'amp': 4.2035373039155806, 'tau': 5.9134605614601521e-06}})

    def test_print_overview(self):
        # only test that it doesn't raise errors
        self.k0.print_overview()

    def test_distort_waveform(self):
        my_square = np.ones(20)
        self.k0.distort_waveform(my_square)

        self.k0.distort_waveform(my_square, length_samples=1000)

    def test_get_first_empty_kernel(self):
        first_empty = self.k0.get_first_empty_filter()
        self.assertEqual(first_empty, 2)

    def test_reset_kernels(self):

        mod00 = {'model': 'high-pass', 'params':
                 {'tau': 4.071755778296734e-05}}
        self.k0.filter_model_00(mod00)

        read_back = self.k0.filter_model_00()
        self.assertEqual(mod00, read_back)

        self.k0.reset_kernels()
        read_back = self.k0.filter_model_00()
        self.assertEqual({}, read_back)

    def test_reset_kernel_resets_hardware(self):

        self.k0.filter_model_00(
            {'model': 'exponential',
             'real-time': True,
             'params': {'tau': 1e-6, 'amp': 0.1}})
        self.k0.filter_model_01(
            {'model': 'FIR',
             'real-time': True,
             'params': {'weights': np.linspace(.2, 0, 40)}})

        my_square = np.ones(20)
        distorted_square = self.k0.distort_waveform(my_square)

        exp_amp = self.AWG.sigouts_0_precompensation_exponentials_0_amplitude()
        fir_coeffs = self.AWG.sigouts_0_precompensation_fir_coefficients()
        assert exp_amp != 0
        assert (fir_coeffs == np.linspace(.2, 0, 40)).all()
        self.k0.set_realtime_distortions_zero()

        exp_amp = self.AWG.sigouts_0_precompensation_exponentials_0_amplitude()
        fir_coeffs = self.AWG.sigouts_0_precompensation_fir_coefficients()
        assert exp_amp == 0
        imp_resp = np.zeros(40)
        imp_resp[0] = 1
        assert (fir_coeffs == imp_resp).all()


    def test_setting_realtime_filter(self):
        self.k0.reset_kernels()
        self.k0.cfg_gain_correction(1)

        tau = 4.071755778296734e-05
        self.k0.filter_model_00(
            {'model': 'exponential',
             'real-time': True,
             'params': {'tau': tau, 'amp': 0.1}})
        my_square = np.ones(20)
        distorted_square = self.k0.distort_waveform(my_square)
        assert (my_square == distorted_square[:20]).all()

        self.k0.cfg_awg_channel(1)
        amp = self.AWG.sigouts_0_precompensation_exponentials_0_amplitude()
        assert amp == 0.1
        set_tau = self.AWG.sigouts_0_precompensation_exponentials_0_timeconstant()
        assert set_tau == tau

        # Add tests for enabling realtime fiters






    @classmethod
    def tearDownClass(self):
        self.k0.close()
        self.AWG.close()


