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

    def assertBetween(self, value, min, max):
        """Fail if value is not between min and max (inclusive)."""
        self.assertGreaterEqual(value, min)
        self.assertLessEqual(value, max)

    def test_SSRO_analysis_basic_1D(self):
        t_start = '20171016_135112'
        t_stop = t_start
        a = ma.Singleshot_Readout_Analysis(t_start=t_start, t_stop=t_stop,
                                           options_dict={'plot_init': True})
        np.testing.assert_almost_equal(a.proc_data_dict['threshold_raw'],
                                       -3.66, decimal=2)
        np.testing.assert_almost_equal(a.proc_data_dict['F_assignment_raw'],
                                       0.922, decimal=3)
        self.assertBetween(a.proc_data_dict['threshold_fit'], -3.69, -3.62)
        np.testing.assert_almost_equal(a.proc_data_dict['F_assignment_fit'],
                                       0.920, decimal=2)
        np.testing.assert_almost_equal(a.proc_data_dict['threshold_discr'],
                                       -3.64, decimal=1)
        np.testing.assert_almost_equal(a.proc_data_dict['F_discr'],
                                       0.996, decimal=2)

    def test_SSRO_analysis_basic_1D_wrong_peak_selected(self):
        # This fit failed when I made a typo in the peak selection part
        t_start = '20171016_171715'
        t_stop = t_start
        a = ma.Singleshot_Readout_Analysis(t_start=t_start, t_stop=t_stop,
                                           extract_only=True)
        np.testing.assert_almost_equal(a.proc_data_dict['threshold_raw'],
                                       -3.30, decimal=2)
        np.testing.assert_almost_equal(a.proc_data_dict['F_assignment_raw'],
                                       0.944, decimal=3)
        np.testing.assert_almost_equal(a.proc_data_dict['threshold_fit'],
                                       -3.25, decimal=2)
        np.testing.assert_almost_equal(a.proc_data_dict['F_assignment_fit'],
                                       0.944, decimal=2)
        np.testing.assert_almost_equal(a.proc_data_dict['threshold_discr'],
                                       -3.2, decimal=1)
        np.testing.assert_almost_equal(a.proc_data_dict['F_discr'],
                                       0.99, decimal=2)

    def test_SSRO_analysis_basic_1D_misfit(self):
        # This dataset failed before I added additional constraints to the
        # guess
        t_start = '20171016_181021'
        t_stop = t_start
        a = ma.Singleshot_Readout_Analysis(t_start=t_start, t_stop=t_stop,
                                           extract_only=True)
        np.testing.assert_almost_equal(a.proc_data_dict['threshold_raw'],
                                       -.95, decimal=2)
        np.testing.assert_almost_equal(a.proc_data_dict['F_assignment_raw'],
                                       0.949, decimal=3)
        self.assertBetween(a.proc_data_dict['threshold_fit'], -1, -.9)
        np.testing.assert_almost_equal(a.proc_data_dict['F_assignment_fit'],
                                       0.945, decimal=2)
        self.assertBetween(a.proc_data_dict['threshold_discr'], -1, -.7)
        np.testing.assert_almost_equal(a.proc_data_dict['F_discr'],
                                       1.000, decimal=2)
        self.assertLess(a.proc_data_dict['residual_excitation'], 0.02)
        np.testing.assert_almost_equal(
            a.proc_data_dict['measurement_induced_relaxation'], 0.1,
            decimal=1)
