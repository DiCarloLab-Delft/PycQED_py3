import unittest
import numpy as np
import pycqed.instrument_drivers.meta_instrument.lfilt_kernel_object as lko


class Test_LinDistortionKernelObject(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.k0 = lko.LinDistortionKernel('k0')

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
        my_sqaure = np.ones(20)
        self.k0.distort_waveform(my_sqaure)

        self.k0.distort_waveform(my_sqaure, length_samples=1000)

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

    @classmethod
    def tearDownClass(self):
        self.k0.close()
