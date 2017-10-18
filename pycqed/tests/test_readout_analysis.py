import unittest
import numpy as np
import pycqed as pq
import os
# # hack for badly installed matplotlib on maserati pc
# import matplotlib
# matplotlib.use('QT4Agg')
from pycqed.analysis_v2 import measurement_analysis as ma


class Test_SSRO_discrimination_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_SSRO_analysis_basic_1D(self):
        t_start = '20171016_135112'
        t_stop = t_start
        a = ma.Singleshot_Readout_Analysis(t_start=t_start, t_stop=t_stop,
                                           options_dict={'plot_init': True})

        np.testing.assert_almost_equal(a.proc_data_dict['threshold_raw'],
                                       -3.66, decimal=2)
        np.testing.assert_almost_equal(a.proc_data_dict['F_assignment_raw'],
                                       0.922, decimal=3)
        np.testing.assert_almost_equal(a.proc_data_dict['threshold_fit'],
                                       -3.65, decimal=2)
        np.testing.assert_almost_equal(a.proc_data_dict['F_assignment_fit'],
                                       0.920, decimal=3)
        np.testing.assert_almost_equal(a.proc_data_dict['threshold_discr'],
                                       -3.64, decimal=2)
        np.testing.assert_almost_equal(a.proc_data_dict['F_discr'],
                                       0.996, decimal=3)

    @unittest.skip('NotImplemented')
    def test_discrimination_fidelity(self):
        # Test the correct file is loaded
        pass
        # a = ma.SSRO_discrimination_analysis(label='dummy_Butterfly',
        #                                     plot_2D_histograms=False)

        # self.assertEqual(
        #     a.folder,
        #     os.path.join(self.datadir, '20161214', '120000_dummy_Butterfly'))
        # mu_a = a.mu_a
        # mu_b = a.mu_b

        # # Test if the fit gives the expected means
        # self.assertAlmostEqual(mu_a.real, -6719.6, places=1)
        # self.assertAlmostEqual(mu_a.imag, 20024.2, places=1)
        # self.assertAlmostEqual(mu_b.real, 1949.4, places=1)
        # self.assertAlmostEqual(mu_b.imag, 37633.0, places=1)

        # # Test identifying the rotation vector
        # self.assertAlmostEqual(a.theta % 180, 63.8, places=1)
        # self.assertAlmostEqual(a.theta % 180,
        #                        np.angle(a.mu_b-a.mu_a,
        #                                 deg=True), places=1)
        # diff_v_r = rotate_complex((mu_b-mu_a), -a.theta)
        # self.assertAlmostEqual(diff_v_r.imag, 0)

        # self.assertAlmostEqual(a.opt_I_threshold,
        #                        np.mean([mu_a.real, mu_b.real]), places=1)
        # self.assertAlmostEqual(a.F_discr, 0.954, places=3)
        # self.assertAlmostEqual(a.F_discr_I, 0.5427, places=3)

    @unittest.skip("NotImplemented")
    def test_rotated_discrimination_fidelity(self):
        # First is run to determine the theta to rotate with
        a = ma.SSRO_discrimination_analysis(
            label='dummy_Butterfly',
            plot_2D_histograms=False)

        a = ma.SSRO_discrimination_analysis(
            label='dummy_Butterfly', theta_in=-a.theta,
            plot_2D_histograms=True)
        self.assertEqual(
            a.folder,
            os.path.join(self.datadir, '20161214', '120000_dummy_Butterfly'))

        mu_a = a.mu_a
        mu_b = a.mu_b
        self.assertAlmostEqual((mu_b-mu_a).imag/10, 0, places=0)

        self.assertAlmostEqual(a.F_discr, a.F_discr,
                               places=3)

    @unittest.skip("NotImplemented")
    def test_discrimination_fidelity_small_vals(self):
        pass
        # a = ma.SSRO_discrimination_analysis(timestamp='20170716_144742')
        # self.assertAlmostEqual(a.F_discr, 0.934047, places=3)
        # self.assertAlmostEqual(a.F_discr_I, 0.8052, places=3)

    @unittest.skip("NotImplemented")
    def test_single_quadrature_discr_fid(self):
        a = ma.SSRO_single_quadrature_discriminiation_analysis(
            timestamp='20170716_134634')
        self.assertAlmostEqual(a.F_discr, 0.79633097)


class Test_multiplexed_SSRO_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    @unittest.skip('Not Implemented')
    def test_two_qubit_ssro(self):
        pass
        # res_dict = mra.two_qubit_ssro_fidelity(label='SSRO_QL_QR')

        # self.assertAlmostEqual(res_dict['Fa_q0'], 0.6169, places=2)
        # self.assertAlmostEqual(res_dict['Fa_q1'], 0.8504, places=2)

        # self.assertAlmostEqual(res_dict['Fd_q0'], 0.6559, places=2)
        # self.assertAlmostEqual(res_dict['Fd_q1'], 0.8728, places=2)

        # mu_mat_exp = np.array([[1.04126946, -0.00517882],
        #                        [-0.03172471,  1.00574731]])
        # np.testing.assert_almost_equal(res_dict['mu_matrix'], mu_mat_exp)
