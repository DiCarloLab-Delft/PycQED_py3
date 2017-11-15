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

    def test_SQ_RTE_flipping(self):
        a = ma.Single_Qubit_RoundsToEvent_Analysis(
            t_start=self.start_timestamps[0],
            t_stop=self.stop_timestamps[0],
            options_dict={'typ_data_idx': 2})
        self.assertEqual(a.raw_data_dict['net_gate'][0], 'pi')
        self.assertEqual(a.raw_data_dict['sequence_type'][0], 'echo')
        self.assertEqual(a.raw_data_dict['feedback'][0], False)
        self.assertEqual(a.raw_data_dict['depletion_time'],
                         [1500, 1600, 1700, 1800])

        self.assertEqual(a.proc_data_dict['exp_pattern'], 'flipping')
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

    def test_SQ_RTE_constant(self):
        a = ma.Single_Qubit_RoundsToEvent_Analysis(
            t_start=self.start_timestamps[2],
            t_stop=self.stop_timestamps[2],
            options_dict={'typ_data_idx': 2})
        self.assertEqual(a.raw_data_dict['net_gate'][0], 'i')
        self.assertEqual(a.raw_data_dict['feedback'][0], False)
        self.assertEqual(a.raw_data_dict['sequence_type'][0], 'echo')
        self.assertEqual(a.raw_data_dict['depletion_time'],
                         [1500, 1600, 1700, 1800])

        self.assertEqual(a.proc_data_dict['exp_pattern'], 'constant')
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

    def test_SQ_RTE_FB_to_ground(self):
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
        self.assertEqual(a.proc_data_dict['exp_pattern'], 'FB_to_ground')

        # Testing the data processing
        single_err_frac = np.array([0.55167359,  0.53970193,  0.57195211,
                                   0.56193501,  0.59271928,
                                    0.59736135,  0.60664549,  0.32494503,
                                    0.28634254,  0.34058148,
                                    0.30539946,  0.27974591,  0.24480821,
                                    0.27803567,  0.25849011,
                                    0.26679697, ])
        np.testing.assert_array_almost_equal(
            single_err_frac, a.proc_data_dict['frac_single'])

        double_err_frac = np.array([0.30351906,  0.3172043,  0.3184262,
                                   0.30913978,  0.33822092,
                                    0.33944282,  0.34335288,  0.2157869,
                                    0.18792766,  0.2370479,
                                    0.20039101,  0.16959922,  0.15102639,
                                    0.1769306,  0.13880743,
                                    0.15053763, ])
        np.testing.assert_array_almost_equal(
            double_err_frac, a.proc_data_dict['frac_double'])
        zero_err_frac = np.array([0.4482169,  0.46018564,  0.42794333,
                                 0.43795799,  0.40718124,
                                  0.4025403,  0.39325843,  0.67489008,
                                  0.71348315,  0.65925745,
                                  0.69443087,  0.72007816,  0.75500733,
                                  0.72178798,  0.74132877,
                                  0.73302394])
        np.testing.assert_array_almost_equal(
            zero_err_frac, a.proc_data_dict['frac_zero'])
