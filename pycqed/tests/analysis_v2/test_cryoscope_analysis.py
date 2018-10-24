import unittest
import numpy as np
import pycqed as pq
import os
import matplotlib.pyplot as plt
from pycqed.analysis_v2 import measurement_analysis as ma


class Test_Cryoscope_analysis(unittest.TestCase):

    @classmethod
    def tearDownClass(self):
        plt.close('all')

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_Cryoscope_Analysis(self):
        a = ma.Cryoscope_Analysis(
            t_start='20180423_114715',
            polycoeffs_freq_conv='Snapshot/instruments/FL_LutMan_QR/parameters/polycoeffs_freq_conv/value',
            derivative_window_length=2e-9)

        expected_figs = {'raw_data', 'demod_data', 'norm_data_circ',
                         'demod_phase', 'frequency_detuning',
                         'cryoscope_amplitude', 'short_time_fft'}
        self.assertTrue(expected_figs.issubset(set(a.figs.keys())))
        self.assertTrue(expected_figs.issubset(set(a.axs.keys())))
        # Does not actually check for the content

    def test_RamZFluxArc(self):
        a = ma.RamZFluxArc(t_start='20180205_105633', t_stop='20180205_120210',
                           ch_idx_cos=2, ch_idx_sin=3)
        poly_coeffs = a.proc_data_dict['poly_coeffs']

        # test dac arc conversion
        # For this to work all other parts have to work
        amps = np.linspace(.1, 1, 21)
        freqs = a.amp_to_freq(amps)

        rec_amps = a.freq_to_amp(freqs, kind='interpolate')
        np.testing.assert_array_almost_equal(amps, rec_amps, decimal=2)
        rec_amps = a.freq_to_amp(freqs, kind='root')
        np.testing.assert_array_almost_equal(amps, rec_amps, decimal=2)
        rec_amps = a.freq_to_amp(freqs, kind='root_parabola')
        np.testing.assert_array_almost_equal(amps, rec_amps, decimal=2)

        np.testing.assert_array_almost_equal(amps, rec_amps, decimal=2)

        poly_coeffs = a.proc_data_dict['poly_coeffs']
        exp_poly_coeffs = np.array(
            [1.36263320e+09, -2.23862128e+08, 3.87339064e+07])
        print(poly_coeffs)
        np.testing.assert_array_almost_equal(poly_coeffs, exp_poly_coeffs,
                                             decimal=-7)

    def test_sliding_pulses_analysis(self):

        a = ma.SlidingPulses_Analysis(t_start='20180221_195729')

        exp_phase = np.array(
            [132.48846657, 288.37102808, 298.68161824, 307.02336668,
             303.94512662, 306.71370643, 305.59951102, 303.79221692,
             311.98804177, 318.1734892, 331.79725518, 322.90068287,
             341.15829614, 328.38539928, 337.929674, 335.46041175,
             310.84851162, 333.47238641, 343.83919864, 339.75778735,
             350.46377994, 311.97060112, 327.71100615, 343.67186721,
             326.73144141])
        np.testing.assert_array_almost_equal(
            a.proc_data_dict['phase'], exp_phase)

        # a reference curve is needed to convert to amps
        da = ma.RamZFluxArc(t_start='20180205_105633',
                            t_stop='20180205_120210',
                            ch_idx_cos=2, ch_idx_sin=3)
        a = ma.SlidingPulses_Analysis(
            t_start='20180221_195729',
            freq_to_amp=da.freq_to_amp, amp_to_freq=da.amp_to_freq)

        exp_amps = np.array(
            [0.87382635, 0.87473807, 0.87479834, 0.8748471, 0.87482911,
             0.87484529, 0.87483878, 0.87482821, 0.87487611, 0.87491226,
             0.87499188, 0.87493989, 0.87504658, 0.87497194, 0.87502771,
             0.87501328, 0.87486945, 0.87500167, 0.87506224, 0.87503839,
             0.87510095, 0.87487601, 0.874968,   0.87506126, 0.87496228])

        np.testing.assert_array_almost_equal(a.proc_data_dict['amp'], exp_amps,
                                             decimal=2)
