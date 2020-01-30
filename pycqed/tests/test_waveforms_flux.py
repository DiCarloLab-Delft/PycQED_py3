import numpy as np
import unittest
from pycqed.measurement.waveform_control_CC import waveforms_flux as wfl


class Test_waveforms_flux(unittest.TestCase):

    def test_eps_theta_conversion(self):

        eps = 800e6
        J2 = 40e6
        theta = wfl.eps_to_theta(eps=eps, g=J2)
        theta_exp = 5.71059
        self.assertAlmostEqual(np.rad2deg(theta), theta_exp, places=4)

        eps_inv = wfl.theta_to_eps(theta=theta, g=J2)
        self.assertAlmostEqual(eps, eps_inv, places=4)

    def test_eps_zero_conversion(self):

        eps = 0.0
        J2 = 40e6
        theta = wfl.eps_to_theta(eps=eps, g=J2)
        self.assertAlmostEqual(theta, np.pi/2)

        eps = np.zeros(5)
        J2 = 40e6
        thetas = wfl.eps_to_theta(eps=eps, g=J2)
        np.testing.assert_array_almost_equal(thetas, 0.5*np.pi*np.ones(5))

    def test_eps_theta_conversion_arrays(self):
        eps = np.linspace(800e6, 0, 5)
        J2 = 40e6
        thetas = wfl.eps_to_theta(eps=eps, g=J2)
        thetas_exp = np.array(
            [0.09966865, 0.13255153, 0.19739556, 0.38050638, 1.57079633])
        np.testing.assert_array_almost_equal(thetas, thetas_exp)

        eps_inv = wfl.theta_to_eps(thetas, g=J2)
        np.testing.assert_array_almost_equal(eps, eps_inv)

    def test_martinis_flux_pulse_theta_bounds(self):
        """
        Tests that the constraint setting theta_f and theta_i is working
        correctly
        """

        for theta_f in [40, 60, 80, 90]:
            theta_f = np.deg2rad(theta_f)
            for lambda_2 in np.linspace(-.2, .2, 5):
                for lambda_3 in np.linspace(-.2, .2, 5):
                    theta_i = wfl.eps_to_theta(800e6, 25e6)
                    thetas = wfl.martinis_flux_pulse(
                        35e-9, theta_i=theta_i, theta_f=theta_f,
                        lambda_2=lambda_2, lambda_3=lambda_3, sampling_rate=1e9)
                    np.testing.assert_almost_equal(
                        thetas[0], theta_i, decimal=3)
                    np.testing.assert_almost_equal(
                        thetas[-1], theta_i, decimal=2)
                    np.testing.assert_almost_equal(
                        thetas[len(thetas)//2], theta_f, decimal=3)

        # The martinis_flux_pulse was change to always clip values
        # It breaks sometimes running optmizations if the optmizer tries
        # certains values that are not allowed. It is a well know "issue"
        # It is ok to go almost silent
        with self.assertLogs("", level='DEBUG') as cm:
            # with self.assertRaises(ValueError):
            theta_i = np.deg2rad(40)
            theta_f = np.deg2rad(30)
            thetas = wfl.martinis_flux_pulse(
                35e-9, theta_i=theta_i, theta_f=theta_f,
                lambda_2=lambda_2, lambda_3=lambda_3, sampling_rate=1e9)
            msg0 = "final coupling weaker than initial coupling"
            msg1 = "Martinis flux wave form has been clipped to"
            self.assertIn(msg0, cm.output[0])
            self.assertIn(msg1, cm.output[1])
