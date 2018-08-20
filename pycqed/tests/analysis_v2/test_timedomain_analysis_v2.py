import unittest
import pycqed as pq
import numpy as np
import os
from pycqed.analysis_v2 import measurement_analysis as ma


class Test_flipping_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_flipping_analysis(self):

        # this test is based on an experiment with a known
        # added detuning in the amplitude. The test tests that the analysis
        # works for a range of known scale factors.

        # 20% detuning only works for coarse
        self._check_scaling('20170726_164507', 0.8, 1)

        self._check_scaling('20170726_164536', 0.9, 1)
        self._check_scaling('20170726_164550', 0.9, 1)
        self._check_scaling('20170726_164605', 0.95, 2)
        self._check_scaling('20170726_164619', 0.95, 2)
        self._check_scaling('20170726_164635', 0.99, 2)
        self._check_scaling('20170726_164649', 0.99, 2)
        self._check_scaling('20170726_164704', 1, 2)
        self._check_scaling('20170726_164718', 1, 2)
        self._check_scaling('20170726_164733', 1.01, 2)
        self._check_scaling('20170726_164747', 1.01, 2)
        self._check_scaling('20170726_164802', 1.05, 1)
        self._check_scaling('20170726_164816', 1.05, 1)
        self._check_scaling('20170726_164831', 1.1, 1)
        self._check_scaling('20170726_164845', 1.1, 1)

        # 20% detuning only works for coarse
        self._check_scaling('20170726_164901', 1.2, 1)

        # Test running it once with showing the initial fit
        ma.FlippingAnalysis(t_start='20170726_164901',
                            options_dict={'plot_init': True})

    def _check_scaling(self, timestamp, known_detuning, places):
        a = ma.FlippingAnalysis(t_start=timestamp)
        s = a.get_scale_factor()
        self.assertAlmostEqual(s*known_detuning, 1, places=places)
        print('Scale factor {:.4f} known detuning {:.4f}'.format(
            s, known_detuning))


class Test_CZ_1QPhaseCal_Analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_zero_phase_diff_intersect(self):
        a = ma.CZ_1QPhaseCal_Analysis(t_start='20171126_180251',
                                      options_dict={'ch_idx':1})
        self.assertAlmostEqual(a.get_zero_phase_diff_intersect(),
                               .058, places=3)


        a = ma.CZ_1QPhaseCal_Analysis(t_start='20171126_181327',
                                      options_dict={'ch_idx':0})
        self.assertAlmostEqual(a.get_zero_phase_diff_intersect(),
                               .1218, places=3)

class Test_Idling_Error_Rate_Analyisis(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    @unittest.skip("TODO: fix this test")
    def test_error_rates_vary_N2(self):
        a=ma.Idling_Error_Rate_Analyisis(
            t_start ='20180210_181633',
            options_dict={'close_figs':True, 'vary_N2': True})


        expected_dict = {'A': 0.41685563870942149,
             'N1': 1064.7100611208791,
             'N2': 3644.550952436859,
             'offset': 0.52121402524448934}
        for key, value in expected_dict.items():
            np.testing.assert_almost_equal(
                a.fit_res['fit +'].best_values[key], value, decimal=2)

        expected_dict = {'A': -0.13013585779457398,
             'N1': 1138.3895116903586,
             'N2': 601415.64642756886,
             'offset': 0.14572799876310505}
        for key, value in expected_dict.items():
            np.testing.assert_almost_equal(
                a.fit_res['fit 0'].best_values[key], value, decimal=2)

        expected_dict = {'A': 0.74324542246644376,
             'N1': 939.61974247762646,
             'N2': 3566698.2870284803,
             'offset': 0.18301612896797623}
        for key, value in expected_dict.items():
            np.testing.assert_almost_equal(
                a.fit_res['fit 1'].best_values[key], value, decimal=2)



    def test_error_rates_fixed_N2(self):
        a=ma.Idling_Error_Rate_Analyisis(
            t_start ='20180210_181633',
            options_dict={'close_figs':True, 'vary_N2': False})


        expected_dict = {'A': 0.43481425072120633,
            'N1': 1034.9644095297574,
            'N2': 1e+21,
            'offset': 0.50671519356947314}
        for key, value in expected_dict.items():
            np.testing.assert_almost_equal(
                a.fit_res['fit +'].best_values[key], value, decimal=2)

        expected_dict = {'A': -0.13013614484482647,
            'N1': 1138.3896694924019,
            'N2': 1e+21,
            'offset': 0.1457282565842071}
        for key, value in expected_dict.items():
            np.testing.assert_almost_equal(
                a.fit_res['fit 0'].best_values[key], value, decimal=2)

        expected_dict = {'A': 0.7432454022744126,
            'N1': 939.61870748568992,
            'N2': 1e+21,
            'offset': 0.18301632862249007}
        for key, value in expected_dict.items():
            np.testing.assert_almost_equal(
                a.fit_res['fit 1'].best_values[key], value, decimal=2)


class Test_Conditional_Oscillation_Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_condition_oscillation_extracted_pars(self):

        a=ma.Conditional_Oscillation_Analysis(t_start='20180130_191229')


        extracted = np.array([a.proc_data_dict['phi_cond'][0],
            a.proc_data_dict['phi_cond'][1],
            a.proc_data_dict['phi_0'][0],
            a.proc_data_dict['phi_0'][1],
            a.proc_data_dict['phi_1'][0],
            a.proc_data_dict['phi_1'][1],
            a.proc_data_dict['osc_amp_0'][0],
            a.proc_data_dict['osc_amp_0'][1],
            a.proc_data_dict['osc_amp_1'][0],
            a.proc_data_dict['osc_amp_1'][1],
            a.proc_data_dict['offs_diff'][0],
            a.proc_data_dict['offs_diff'][1],
            a.proc_data_dict['osc_offs_0'][0],
            a.proc_data_dict['osc_offs_0'][1],
            a.proc_data_dict['osc_offs_1'][0],
            a.proc_data_dict['osc_offs_1'][1]])
        expected = np.array([178.48651885251698,
             2.9898741913646272,
             3.3113606223925696,
             1.691099641377918,
             181.79787947490954,
             2.4656702300023334,
             0.3236975712917689,
             0.009554015995104392,
             0.27746120839251875,
             0.011940273429767528,
             0.00150761346297007,
             0.010813168390439071,
             0.4868083822944365,
             0.0067557093031373455,
             0.4883159957574066,
             0.0084430446197739737])

        np.testing.assert_almost_equal(extracted, expected, decimal=3)
