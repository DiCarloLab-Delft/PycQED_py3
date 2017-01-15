import numpy as np
import unittest

from pycqed.measurement.waveform_control_CC import waveform as wf


class Test_Waveforms(unittest.TestCase):

    def test_gauss_pulse(self):

        g_env = np.array([0.,  0.0056186,  0.01165567,
                          0.01812183,  0.02502537,
                          0.03237195,  0.04016432,  0.04840195,
                          0.05708081,  0.06619307,
                          0.07572687,  0.08566613,  0.09599033,
                          0.10667444,  0.11768883,
                          0.12899923,  0.14056679,  0.15234816,
                          0.16429566,  0.17635752,
                          0.18847815,  0.20059853,  0.21265661,
                          0.2245878,  0.2363255,
                          0.24780173,  0.2589477,  0.26969455,
                          0.27997397,  0.28971899,
                          0.29886465,  0.30734872,  0.31511243,
                          0.32210111,  0.32826488,
                          0.33355918,  0.33794536,  0.3413911,
                          0.34387088,  0.3453662,
                          0.34586589,  0.3453662,  0.34387088,
                          0.3413911,  0.33794536,
                          0.33355918,  0.32826488,  0.32210111,
                          0.31511243,  0.30734872,
                          0.29886465,  0.28971899,  0.27997397,
                          0.26969455,  0.2589477,
                          0.24780173,  0.2363255,  0.2245878,
                          0.21265661,  0.20059853,
                          0.18847815,  0.17635752,  0.16429566,
                          0.15234816,  0.14056679,
                          0.12899923,  0.11768883,  0.10667444,
                          0.09599033,  0.08566613,
                          0.07572687,  0.06619307,  0.05708081,
                          0.04840195,  0.04016432,
                          0.03237195,  0.02502537,  0.01812183,
                          0.01165567,  0.0056186,  0.])

        d_env = np.array([0.07903581,  0.08505798,
                          0.09125043,  0.09758165,  0.10401556,
                          0.1105115,  0.11702435,  0.12350468,
                          0.12989903,  0.13615021,
                          0.14219778,  0.14797855,  0.15342718,
                          0.15847683,  0.16305997,
                          0.16710918,  0.17055799,  0.17334187,
                          0.17539911,  0.17667183,
                          0.17710695,  0.17665709,  0.17528151,
                          0.17294695,  0.16962842,
                          0.16530987,  0.15998481,  0.1536567,
                          0.14633934,  0.13805702,
                          0.12884455,  0.11874711,  0.10781999,
                          0.0961281,  0.08374538,
                          0.07075403,  0.0572436,  0.04331001,
                          0.02905436,  0.01458176,
                          -0., -0.01458176, -0.02905436, -
                          0.04331001, -0.0572436,
                          -0.07075403, -0.08374538, -
                          0.0961281, -0.10781999, -0.11874711,
                          -0.12884455, -0.13805702, -
                          0.14633934, -0.1536567, -0.15998481,
                          -0.16530987, -0.16962842, -
                          0.17294695, -0.17528151, -0.17665709,
                          -0.17710695, -0.17667183, -
                          0.17539911, -0.17334187, -0.17055799,
                          -0.16710918, -0.16305997, -
                          0.15847683, -0.15342718, -0.14797855,
                          -0.14219778, -0.13615021, -
                          0.12989903, -0.12350468, -0.11702435,
                          -0.1105115, -0.10401556, -
                          0.09758165, -0.09125043, -0.08505798,
                          -0.07903581])

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

    def test_martinis_flux_pulse(self):
        g2 = 1/(120e-9/(14.5/2))
        f_bus = 4.8e9
        f_01_max = 5.94e9
        dac_flux_coefficient = 0.679
        E_c = 369.2e6
        theta_f = .4
        length=40e-9
        lambda_coeffs_list = [[.1, 0], [.4, .2, .1, .01, .2]]
        for lambda_coeffs in lambda_coeffs_list:

            th_pulse = wf.martinis_flux_pulse(
                length=length, theta_f=theta_f, lambda_coeffs=lambda_coeffs,
                g2=g2, E_c=E_c, f_01_max=f_01_max, f_bus=f_bus,
                dac_flux_coefficient=dac_flux_coefficient,
                return_unit='theta')
            V_pulse = wf.martinis_flux_pulse(
                length=length, theta_f=theta_f, lambda_coeffs=lambda_coeffs,
                g2=g2, E_c=E_c, f_01_max=f_01_max, f_bus=f_bus,
                dac_flux_coefficient=dac_flux_coefficient,
                return_unit='V')

            theta_0 = np.arctan(2*g2/(f_01_max-E_c-f_bus))
            np.testing.assert_almost_equal(theta_0, th_pulse[0])
            np.testing.assert_almost_equal(0, V_pulse[0])

            self.assertEqual(len(th_pulse), 40)
            np.testing.assert_almost_equal(np.max(th_pulse), theta_f)

            self.assertEqual(np.argmax(th_pulse), 20)
            self.assertEqual(np.argmax(V_pulse), 20)
