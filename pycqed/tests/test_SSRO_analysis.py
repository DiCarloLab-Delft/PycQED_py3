import unittest
import numpy as np
import pycqed as pq
import os
# # hack for badly installed matplotlib on maserati pc
# import matplotlib
# matplotlib.use('QT4Agg')
from pycqed.analysis import measurement_analysis as ma

from pycqed.analysis.tools.data_manipulation import rotate_complex

class Test_SSRO_discrimination_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0],'tests','test_data')
        ma.a_tools.datadir = self.datadir
        self.a_discr = ma.SSRO_discrimination_analysis(label='dummy_Butterfly')
        self.a_discr_rot = ma.SSRO_discrimination_analysis(
            label='dummy_Butterfly', theta_in=-self.a_discr.theta,
            plot_2D_histograms=False)

    def test_discrimination_fidelity(self):
        # Test the correct file is loaded
        self.assertEqual(
            self.a_discr.folder,
            os.path.join(self.datadir,'20161214','120000_dummy_Butterfly'))
        mu_a = self.a_discr.mu_a
        mu_b = self.a_discr.mu_b

        # Test if the fit gives the expected means
        self.assertAlmostEqual(mu_a.real, -6719.6, places=1)
        self.assertAlmostEqual(mu_a.imag, 20024.2, places=1)
        self.assertAlmostEqual(mu_b.real, 1949.4, places=1)
        self.assertAlmostEqual(mu_b.imag, 37633.0, places=1)

        # Test identifying the rotation vector
        self.assertAlmostEqual(self.a_discr.theta%180, 63.8, places=1)
        self.assertAlmostEqual(self.a_discr.theta%180,
                               np.angle(self.a_discr.mu_b-self.a_discr.mu_a, deg=True), places=1)
        diff_v_r = rotate_complex((mu_b-mu_a), -self.a_discr.theta)
        self.assertAlmostEqual(diff_v_r.imag, 0)


        self.assertAlmostEqual(self.a_discr.opt_I_threshold,
                               np.mean([mu_a.real, mu_b.real]), places=1)
        self.assertAlmostEqual(self.a_discr.F_discr, 0.908, places=3)
        self.assertAlmostEqual(self.a_discr.F_discr_I, 0.5427, places=3)

    def test_rotated_discrimination_fidelity(self):
        self.assertEqual(
            self.a_discr_rot.folder,
            os.path.join(self.datadir,'20161214','120000_dummy_Butterfly'))

        # self.assertAlmostEqual(self.a_discr_rot.theta, 0)
        mu_a = self.a_discr_rot.mu_a
        mu_b = self.a_discr_rot.mu_b

        self.assertAlmostEqual((mu_b-mu_a).imag/10, 0, places=0)

        self.assertAlmostEqual(self.a_discr_rot.F_discr, self.a_discr_rot.F_discr_I,
                               places=3)
        self.assertAlmostEqual(self.a_discr_rot.F_discr, 0.908, places=3)

