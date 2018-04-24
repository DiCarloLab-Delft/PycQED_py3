import unittest
import numpy as np
import pycqed as pq
import os
from pycqed.analysis_v2 import measurement_analysis as ma


class Test_Cryoscope_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    # def test_Cryoscope_Analysis(self):
    #     a = ma.Cryoscope_Analysis(
    #         t_start='20180423_114715',
    #         polycoeffs_freq_conv='Snapshot/instruments/FL_LutMan_QR/parameters/polycoeffs_freq_conv/value',
    #         derivative_window_length=2e-9)

    #     expected_figs = ['raw_data', 'demod_data', 'norm_data_circ',
    #                      'demod_phase', 'frequency_detuning',
    #                      'cryoscope_amplitude', 'short_time_fft']
        # self.assertIn(expected_figs, list(a.figs.keys()))
        # self.assertIn(expected_figs, list(a.axs.keys()))
        # Does not actually check for the content

    def test_RamZFluxArc(self):
        a = ma.RamZFluxArc(t_start='20180205_105633', t_stop='20180205_120210',
                           ch_idx_cos=2, ch_idx_sin=3)

        # test dac arc conversion
        # For this to work all other parts have to work
        amps = a.freq_to_amp([.5e9, .6e9, .8e9])
        exp_amps = np.array([0.67,  0.73,  0.83])
        np.testing.assert_array_almost_equal(amps, exp_amps, decimal=2)

        freqs = a.amp_to_freq([.3, .4, .5])
        exp_freqs = np.array(
            [9.42122560e+07,   1.67210367e+08,  2.67461142e+08])
        np.testing.assert_array_almost_equal(freqs, exp_freqs, decimal=-7)

        poly_coeffs = a.proc_data_dict['poly_coeffs']
        exp_poly_coeffs = np.array(
            [1.36263320e+09, -2.23862128e+08, 3.87339064e+07])
        print(poly_coeffs)
        np.testing.assert_array_almost_equal(poly_coeffs, exp_poly_coeffs,
                                             decimal=-7)

    # def test_sliding_pulses_analysis(self):

    #     a = ma.SlidingPulses_Analysis(t_start='20180221_195729')

    #     exp_phase = np.array(
    #         [119.29377743,  287.53327004,  295.54002558,  299.90242333,
    #          302.68848956,  307.55146448,  312.51101486,  316.98690607,
    #          317.64290854,  321.73396087,  324.67631183,  328.34611014,
    #          331.73351818,  332.78362899,  331.43704918,  332.3188191,
    #          333.25853922,  336.82341857,  336.50881579,  333.68404155,
    #          330.77646598,  327.25968537,  329.38652223,  329.42998052,
    #          329.45415504])
    #     np.testing.assert_array_almost_equal(
    #         a.proc_data_dict['phase'], exp_phase)

    #     # a reference curve is needed to convert to amps
    #     da = ma.RamZFluxArc(t_start='20180205_105633',
    #                         t_stop='20180205_120210',
    #                         ch_idx_cos=2, ch_idx_sin=3)
    #     a = ma.SlidingPulses_Analysis(
    #         t_start='20180221_195729',
    #         freq_to_amp=da.freq_to_amp, amp_to_freq=da.amp_to_freq)

    #     exp_amps = np.array(
    #         [0.69900528,  0.69979251,  0.69982995,  0.69985035,  0.69986337,
    #          0.69988611,  0.6999093,  0.69993023,  0.69993329,  0.69995242,
    #          0.69996618,  0.69998333,  0.69999917,  0.70000408,  0.69999778,
    #          0.7000019,  0.7000063,  0.70002296,  0.70002149,  0.70000829,
    #          0.69999469,  0.69997825,  0.6999882,  0.6999884,  0.69998851])
    #     np.testing.assert_array_almost_equal(a.proc_data_dict['amp'], exp_amps,
    #                                          decimal=2)
