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

