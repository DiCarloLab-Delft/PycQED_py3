import unittest
import pycqed as pq
import numpy as np
import matplotlib.pyplot as plt
import os
from pycqed.analysis_v2 import measurement_analysis as ma


class Test_flipping_analysis(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        plt.close("all")

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], "tests", "test_data")
        ma.a_tools.datadir = self.datadir

    def test_flipping_analysis(self):

        # this test is based on an experiment with a known
        # added detuning in the amplitude. The test tests that the analysis
        # works for a range of known scale factors.

        # 20% detuning only works for coarse
        self._check_scaling("20170726_164507", 0.8, 1)

        self._check_scaling("20170726_164536", 0.9, 1)
        self._check_scaling("20170726_164550", 0.9, 1)
        self._check_scaling("20170726_164605", 0.95, 2)
        self._check_scaling("20170726_164619", 0.95, 2)
        self._check_scaling("20170726_164635", 0.99, 2)
        self._check_scaling("20170726_164649", 0.99, 2)
        self._check_scaling("20170726_164704", 1, 2)
        self._check_scaling("20170726_164718", 1, 2)
        self._check_scaling("20170726_164733", 1.01, 2)
        self._check_scaling("20170726_164747", 1.01, 2)
        self._check_scaling("20170726_164802", 1.05, 1)
        self._check_scaling("20170726_164816", 1.05, 1)
        self._check_scaling("20170726_164831", 1.1, 1)
        self._check_scaling("20170726_164845", 1.1, 1)

        # 20% detuning only works for coarse
        self._check_scaling("20170726_164901", 1.2, 1)

        # Test running it once with showing the initial fit
        ma.FlippingAnalysis(t_start="20170726_164901", options_dict={"plot_init": True})

    def _check_scaling(self, timestamp, known_detuning, places):
        a = ma.FlippingAnalysis(t_start=timestamp)
        s = a.get_scale_factor()
        self.assertAlmostEqual(s * known_detuning, 1, places=places)
        print("Scale factor {:.4f} known detuning {:.4f}".format(s, known_detuning))


class Test_Idling_Error_Rate_Analyisis(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], "tests", "test_data")
        ma.a_tools.datadir = self.datadir

    @unittest.skip("TODO: fix this test")
    def test_error_rates_vary_N2(self):
        a = ma.Idling_Error_Rate_Analyisis(
            t_start="20180210_181633",
            options_dict={"close_figs": True, "vary_N2": True},
        )

        expected_dict = {
            "A": 0.41685563870942149,
            "N1": 1064.7100611208791,
            "N2": 3644.550952436859,
            "offset": 0.52121402524448934,
        }
        for key, value in expected_dict.items():
            np.testing.assert_almost_equal(
                a.fit_res["fit +"].best_values[key], value, decimal=2
            )

        expected_dict = {
            "A": -0.13013585779457398,
            "N1": 1138.3895116903586,
            "N2": 601415.64642756886,
            "offset": 0.14572799876310505,
        }
        for key, value in expected_dict.items():
            np.testing.assert_almost_equal(
                a.fit_res["fit 0"].best_values[key], value, decimal=2
            )

        expected_dict = {
            "A": 0.74324542246644376,
            "N1": 939.61974247762646,
            "N2": 3566698.2870284803,
            "offset": 0.18301612896797623,
        }
        for key, value in expected_dict.items():
            np.testing.assert_almost_equal(
                a.fit_res["fit 1"].best_values[key], value, decimal=2
            )

    def test_error_rates_fixed_N2(self):
        a = ma.Idling_Error_Rate_Analyisis(
            t_start="20180210_181633",
            options_dict={"close_figs": True, "vary_N2": False},
        )

        expected_dict = {
            "A": 0.43481425072120633,
            "N1": 1034.9644095297574,
            "N2": 1e21,
            "offset": 0.50671519356947314,
        }
        for key, value in expected_dict.items():
            np.testing.assert_almost_equal(
                a.fit_res["fit +"].best_values[key], value, decimal=2
            )

        expected_dict = {
            "A": -0.13013614484482647,
            "N1": 1138.3896694924019,
            "N2": 1e21,
            "offset": 0.1457282565842071,
        }
        for key, value in expected_dict.items():
            np.testing.assert_almost_equal(
                a.fit_res["fit 0"].best_values[key], value, decimal=2
            )

        expected_dict = {
            "A": 0.7432454022744126,
            "N1": 939.61870748568992,
            "N2": 1e21,
            "offset": 0.18301632862249007,
        }
        for key, value in expected_dict.items():
            np.testing.assert_almost_equal(
                a.fit_res["fit 1"].best_values[key], value, decimal=2
            )


class Test_Conditional_Oscillation_Analysis(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], "tests", "test_data")
        ma.a_tools.datadir = self.datadir

    # [2020-08-05 Victor] Experiment code and analysis was upgraded
    # new tests are needed, including the case of measuring phase on the
    # parked qubit
    @unittest.skip("FIXME: test dataset has wrong channel convention")
    def test_condition_oscillation_extracted_pars(self):

        a = ma.Conditional_Oscillation_Analysis(
            t_start="20181126_131143", cal_points="gef"
        )
        qoi = a.proc_data_dict["quantities_of_interest"]
        print(qoi)
        extracted = np.array(
            [
                qoi["phi_cond"].nominal_value,
                qoi["phi_cond"].std_dev,
                qoi["phi_0"].nominal_value,
                qoi["phi_0"].std_dev,
                qoi["phi_1"].nominal_value,
                qoi["phi_1"].std_dev,
                qoi["osc_amp_0"].nominal_value,
                qoi["osc_amp_0"].std_dev,
                qoi["osc_amp_1"].nominal_value,
                qoi["osc_amp_1"].std_dev,
                qoi["offs_diff"].nominal_value,
                qoi["offs_diff"].std_dev,
                qoi["osc_offs_0"].nominal_value,
                qoi["osc_offs_0"].std_dev,
                qoi["osc_offs_1"].nominal_value,
                qoi["osc_offs_1"].std_dev,
            ]
        )
        expected = np.array(
            [
                7.139e01,
                1.077e00,
                8.753e01,
                5.926e-01,
                1.614e01,
                8.990e-01,
                4.859e-01,
                5.026e-03,
                4.792e-01,
                7.518e-03,
                1.225e-02,
                6.395e-03,
                4.869e-01,
                3.554e-03,
                4.992e-01,
                5.316e-03,
            ]
        )

        np.testing.assert_almost_equal(extracted, expected, decimal=2)
