import unittest
import numpy as np
from scipy import signal

import pycqed.measurement.kernel_functions_ZI as ZI_kf

class Test_Kernel_functions_ZI(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.time_start = -100e-9
        self.time_end = 100e-9
        self.sampling_rate = 2.4e9
        self.time = np.arange(self.time_start, self.time_end, 1/self.sampling_rate)
        self.bounce_delay = 10e-9
        self.bounce_amp = 0.1
        self.sawtooth_period = 50e-9

        self.generate_test_waveform(self)
        self.compute_distorted_waveform(self)

    def generate_test_waveform(self):
        # Sawtooth test waveform
        self.ideal_waveform = np.remainder(2*self.time/self.sawtooth_period, 1)

    def compute_distorted_waveform(self):
        a = ZI_kf.first_order_bounce_kern(self.bounce_delay, self.bounce_amp, self.sampling_rate)
        self.distorted_waveform = signal.lfilter([1.0], a, self.ideal_waveform)

    def test_first_order_bounce_correction(self):
        hw_corr = ZI_kf.first_order_bounce_corr(self.distorted_waveform, self.bounce_delay, self.bounce_amp, self.sampling_rate)
        b = ZI_kf.first_order_bounce_kern(self.bounce_delay, ZI_kf.coef_round(self.bounce_amp, force_bshift=0), self.sampling_rate)
        first_order_corr = signal.lfilter(b, 1.0, self.distorted_waveform)
        np.testing.assert_almost_equal(hw_corr, first_order_corr, 6)

    def test_ideal_bounce_correction(self):
        # Construct impulse response
        impulse = np.zeros(len(self.time))
        zero_ind = np.argmin(np.abs(self.time))
        impulse[zero_ind] = 1.0
        a = ZI_kf.first_order_bounce_kern(self.bounce_delay, self.bounce_amp, self.sampling_rate)
        impulse_response = signal.lfilter([1.0], a, impulse)
        b_inv = ZI_kf.ideal_inverted_fir_kernel(impulse_response, zero_ind)
        ideal_corr = signal.lfilter(b_inv, 1.0, self.distorted_waveform)
        np.testing.assert_almost_equal(ideal_corr, self.ideal_waveform, 6)

