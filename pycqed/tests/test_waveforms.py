import numpy as np
import unittest
from pycqed.measurement.waveform_control_CC import waveform as wf

# These are test waveforms
g_env = np.array([0., 0.00582638, 0.0120768, 0.01876077, 0.0258853,
                  0.03345463, 0.04146991, 0.04992889, 0.05882567, 0.06815045,
                  0.07788926, 0.08802383, 0.0985314, 0.10938465, 0.12055162,
                  0.13199574, 0.14367589, 0.15554655, 0.16755795, 0.17965637,
                  0.19178445, 0.20388161, 0.21588445, 0.22772732, 0.23934285,
                  0.25066257, 0.26161757, 0.27213918, 0.28215966, 0.29161297,
                  0.30043541, 0.3085664, 0.31594915, 0.32253129, 0.32826554,
                  0.33311022, 0.33702981, 0.33999533, 0.34198473, 0.34298317,
                  0.34298317, 0.34198473, 0.33999533, 0.33702981, 0.33311022,
                  0.32826554, 0.32253129, 0.31594915, 0.3085664, 0.30043541,
                  0.29161297, 0.28215966, 0.27213918, 0.26161757, 0.25066257,
                  0.23934285, 0.22772732, 0.21588445, 0.20388161, 0.19178445,
                  0.17965637, 0.16755795, 0.15554655, 0.14367589, 0.13199574,
                  0.12055162, 0.10938465, 0.0985314, 0.08802383, 0.07788926,
                  0.06815045, 0.05882567, 0.04992889, 0.04146991, 0.03345463,
                  0.0258853, 0.01876077, 0.0120768, 0.00582638, 0.])

d_env = np.array(
    [0.08202382,  0.08813478,  0.09440084,  0.1007882,  0.10725849,
     0.1137688,  0.12027183,  0.1267161,  0.13304622,  0.13920334,
     0.14512556,  0.15074853,  0.15600609,  0.16083094,  0.1651555,
     0.16891272,  0.17203698,  0.17446503,  0.17613698,  0.1769972,
     0.17699536,  0.17608728,  0.17423586,  0.1714119,  0.16759478,
     0.16277318,  0.15694552,  0.15012041,  0.14231687,  0.13356447,
     0.12390323,  0.11338342,  0.10206522,  0.09001814,  0.07732035,
     0.06405784,  0.05032349,  0.03621595,  0.02183849,  0.00729772,
     -0.00729772, -0.02183849, -0.03621595, -0.05032349, -0.06405784,
     -0.07732035, -0.09001814, -0.10206522, -0.11338342, -0.12390323,
     -0.13356447, -0.14231687, -0.15012041, -0.15694552, -0.16277318,
     -0.16759478, -0.1714119, -0.17423586, -0.17608728, -0.17699536,
     -0.1769972, -0.17613698, -0.17446503, -0.17203698, -0.16891272,
     -0.1651555, -0.16083094, -0.15600609, -0.15074853, -0.14512556,
     -0.13920334, -0.13304622, -0.1267161, -0.12027183, -0.1137688,
     -0.10725849, -0.1007882, -0.09440084, -0.08813478, -0.08202382])


class Test_Waveforms(unittest.TestCase):

    def test_gauss_pulse(self):
        amplitude = .4  # something not equal to one to prevent some bugs
        motzoi = .73
        sigma = 20e-9

        I, Q = wf.gauss_pulse(amplitude, sigma, axis='x', nr_sigma=4,
                              sampling_rate=1e9,
                              motzoi=motzoi, delay=0)
        self.assertEqual(np.shape(I), np.shape(g_env))

        self.assertEqual(np.shape(Q), np.shape(d_env))

        np.testing.assert_almost_equal(I, g_env)
        np.testing.assert_almost_equal(Q, d_env)

        I, Q = wf.gauss_pulse(amplitude, sigma, axis='y', nr_sigma=4,
                              sampling_rate=1e9,
                              motzoi=motzoi, delay=0)
        np.testing.assert_almost_equal(I, -d_env)
        np.testing.assert_almost_equal(Q, g_env)

        I, Q = wf.gauss_pulse(amplitude, sigma, axis='x', phase=90,
                              nr_sigma=4,
                              sampling_rate=1e9,
                              motzoi=motzoi, delay=0)
        np.testing.assert_almost_equal(I, -d_env)
        np.testing.assert_almost_equal(Q, g_env)

    def test_mod_gauss(self):
        amplitude = .4  # something not equal to one to prevent some bugs
        motzoi = .73
        sigma = 20e-9
        I, Q = wf.mod_gauss(amplitude, sigma, axis='x', nr_sigma=4,
                            sampling_rate=1e9, f_modulation=0,
                            motzoi=motzoi, delay=0)

        np.testing.assert_almost_equal(I, g_env)
        np.testing.assert_almost_equal(Q, d_env)

    def test_mod_gauss_VSM(self):
        amplitude = .4  # something not equal to one to prevent some bugs
        motzoi = .73
        sigma = 20e-9
        G_I, G_Q, D_I, D_Q = wf.mod_gauss_VSM(
            amplitude, sigma, axis='x', nr_sigma=4,
            sampling_rate=1e9, f_modulation=0,
            motzoi=motzoi, delay=0)

        np.testing.assert_almost_equal(G_I, g_env)
        np.testing.assert_almost_equal(G_Q, np.zeros(len(g_env)))
        np.testing.assert_almost_equal(D_I, np.zeros(len(g_env)))
        np.testing.assert_almost_equal(D_Q, d_env)

    def test_mod_square_VSM(self):
        waveform = wf.mod_square_VSM(1, 0,
                                     20e-9, f_modulation=0, sampling_rate=1e9)

        np.testing.assert_almost_equal(waveform[0], np.ones(20))
        np.testing.assert_almost_equal(waveform[1], np.zeros(20))
        np.testing.assert_almost_equal(waveform[2], np.zeros(20))
        np.testing.assert_almost_equal(waveform[3], np.zeros(20))

        waveform = wf.mod_square_VSM(1, 1,
                                     20e-9, f_modulation=0, sampling_rate=1e9)

        np.testing.assert_almost_equal(waveform[0], np.ones(20))
        np.testing.assert_almost_equal(waveform[1], np.zeros(20))
        np.testing.assert_almost_equal(waveform[2], np.ones(20))
        np.testing.assert_almost_equal(waveform[3], np.zeros(20))

        waveform = wf.mod_square_VSM(0, 1,
                                     20e-9, f_modulation=0, sampling_rate=1e9)
        np.testing.assert_almost_equal(waveform[0], np.zeros(20))
        np.testing.assert_almost_equal(waveform[1], np.zeros(20))
        np.testing.assert_almost_equal(waveform[2], np.ones(20))
        np.testing.assert_almost_equal(waveform[3], np.zeros(20))
