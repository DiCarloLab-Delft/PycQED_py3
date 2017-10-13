import numpy as np
import unittest
import pycqed as pq
import os
from pycqed.analysis_v2 import measurement_analysis as ma


class Test_RTE_Analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir
        self.start_timestamps = ['20171012_213543',
                                 '20171012_214325',
                                 '20171012_214657',
                                 '20171012_215028']

        self.stop_timestamps = [s[:-6] + str(int(s[-6:])-1)
                                for s in self.start_timestamps[1:]]

    def test_SQ_RTE_pi_no_FB(self):
        a = ma.Single_Qubit_RoundsToEvent_Analysis(
            t_start=self.start_timestamps[0],
            t_stop=self.stop_timestamps[0],
            options_dict={'typ_data_idx': 2})
        self.assertEqual(a.raw_data_dict['net_gate'][0], 'pi')
        self.assertEqual(a.raw_data_dict['sequence_type'][0], 'echo')
        self.assertEqual(a.raw_data_dict['feedback'][0], False)
        self.assertEqual(a.raw_data_dict['depletion_time'],
                         [1500, 1600, 1700, 1800])

        # Testing the data processing
        single_err_frac = np.array([0.55387247,  0.34473491,  0.30222331,
                                    0.26948449])
        np.testing.assert_array_almost_equal(
            single_err_frac, a.proc_data_dict['frac_single'])

        double_err_frac = np.array([0.36339198,  0.18963832,  0.18035191,
                                    0.15786901])
        np.testing.assert_array_almost_equal(
            double_err_frac, a.proc_data_dict['frac_double'])
        zero_err_frac = np.array([0.26868588,  0.41866146,  0.40937958,
                                  0.42476795])
        np.testing.assert_array_almost_equal(
            zero_err_frac, a.proc_data_dict['frac_zero'])

    def test_SQ_RTE_const_no_FB(self):
        a = ma.Single_Qubit_RoundsToEvent_Analysis(
            t_start=self.start_timestamps[2],
            t_stop=self.stop_timestamps[2],
            options_dict={'typ_data_idx': 2})
        self.assertEqual(a.raw_data_dict['net_gate'][0], 'i')
        self.assertEqual(a.raw_data_dict['feedback'][0], False)
        self.assertEqual(a.raw_data_dict['sequence_type'][0], 'echo')
        self.assertEqual(a.raw_data_dict['depletion_time'],
                         [1500, 1600, 1700, 1800])
        # Testing the data processing
        single_err_frac = np.array([0.26459809,  0.24383093,  0.23039335,
                                    0.18983631])
        np.testing.assert_array_almost_equal(
            single_err_frac, a.proc_data_dict['frac_single'])
        double_err_frac = np.array([0.11852395,  0.10215054,  0.10166178,
                                    0.07917889])
        np.testing.assert_array_almost_equal(
            double_err_frac, a.proc_data_dict['frac_double'])
        zero_err_frac = np.array([0.6802638,  0.60820713,  0.5456766,
                                  0.52149487])
        np.testing.assert_array_almost_equal(
            zero_err_frac, a.proc_data_dict['frac_zero'])

    def test_SQ_RTE_const_FB(self):
        a = ma.Single_Qubit_RoundsToEvent_Analysis(
            t_start=self.start_timestamps[1],
            t_stop=self.stop_timestamps[1],
            options_dict={'typ_data_idx': 2})
        self.assertEqual(a.raw_data_dict['net_gate'][0], 'i')
        self.assertEqual(a.raw_data_dict['feedback'][0], True)
        self.assertEqual(a.raw_data_dict['sequence_type'][0], 'echo')
        depletion_times = list(np.arange(1500, 3001, 100))
        self.assertEqual(a.raw_data_dict['depletion_time'],
                         depletion_times)
