import unittest
from matplotlib import rcParams
import pycqed as pq
import os
from pycqed.analysis_v2 import measurement_analysis as ma


class Test_RBAnalysis(unittest.TestCase):
    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], "tests", "test_data")
        ma.a_tools.datadir = self.datadir
        # to have fast tests
        rcParams["figure.dpi"] = 80

    def test_single_qubit_RB_analysis(self):
        a = ma.RandomizedBenchmarking_SingleQubit_Analysis(
            t_start="20180601_135117", classification_method="rates"
        )
        self.a = a

        leak_pars = a.fit_res["leakage_decay_raw w0"].params
        L1 = leak_pars["L1"].value
        L2 = leak_pars["L2"].value
        self.assertAlmostEqual(L1 * 100, 0.010309, places=2)
        self.assertAlmostEqual(L2 * 100, 0.392206, places=2)

        rb_pars = a.fit_res["rb_decay_raw w0"].params
        F = rb_pars["F"].value
        self.assertAlmostEqual(F, 0.997895, places=4)

    def test_int_cz_idle_two_qubit_RB_analysis(self):
        # Run full analysis to produce the plots
        # Commented out cause it takes a lot of time to generate plots
        # Here for debugging reference
        # ma.RandomizedBenchmarking_TwoQubit_Analysis(
        #     t_start="20200720_215813",
        #     rates_I_quad_ch_idxs=[0, 2],
        # )
        # ma.RandomizedBenchmarking_TwoQubit_Analysis(
        #     t_start="20200720_223359",
        #     rates_I_quad_ch_idxs=[0, 2],
        # )
        # ma.RandomizedBenchmarking_TwoQubit_Analysis(
        #     t_start="20200720_230928",
        #     rates_I_quad_ch_idxs=[0, 2],
        # )

        a = ma.InterleavedRandomizedBenchmarkingAnalysis(
            ts_base="20200720_215813",
            ts_int="20200720_223359",
            ts_int_idle="20200720_230928",
            rates_I_quad_ch_idxs=[0, 2],
        )
        qois = a.proc_data_dict["quantities_of_interest"]

        qois_values = {
            'eps_simple_2Q_ref': 0.04607,
            'eps_X1_2Q_ref': 0.05743,
            'L1_2Q_ref': 0.008230,
            'L2_2Q_ref': 0.0041501,
            'eps_simple_2Q_int': 0.058965,
            'eps_X1_2Q_int': 0.0784,
            'L1_2Q_int': 0.015203,
            'L2_2Q_int': 0.005343,
            'eps_CZ_X1': 0.02223,
            'eps_CZ_simple': 0.013515,
            'L1_CZ': 0.007031,
            'eps_simple_2Q_int_idle': 0.07064,
            'eps_X1_2Q_int_idle': 0.08811,
            'L1_2Q_int_idle': 0.006986,
            'L2_2Q_int_idle': 0.005116,
            'eps_idle_X1': 0.03254,
            'eps_idle_simple': 0.0257,
            'L1_idle': -0.0012543,
            'L1_CZ_naive': 0.005494,
            'eps_CZ_simple_naive': 0.030955,
            'eps_CZ_X1_naive': 0.03866,
        }
        for val_name, val in qois_values.items():
            self.assertAlmostEqual(qois[val_name].n, val, places=3)

    def test_int_cz_only_two_qubit_RB_analysis(self):
        a = ma.InterleavedRandomizedBenchmarkingAnalysis(
            ts_base="20200720_215813",
            ts_int="20200720_223359",
            rates_I_quad_ch_idxs=[0, 2],
        )
        qois = a.proc_data_dict["quantities_of_interest"]

        qois_values = {
            'eps_simple_2Q_ref': 0.04607,
            'eps_X1_2Q_ref': 0.05743,
            'L1_2Q_ref': 0.008230,
            'L2_2Q_ref': 0.0041501,
            'eps_simple_2Q_int': 0.058965,
            'eps_X1_2Q_int': 0.0784,
            'L1_2Q_int': 0.015203,
            'L2_2Q_int': 0.005343,
            'eps_CZ_X1': 0.02223,
            'eps_CZ_simple': 0.013515,
            'L1_CZ': 0.007031,
            'L1_CZ_naive': 0.005494,
            'eps_CZ_simple_naive': 0.030955,
            'eps_CZ_X1_naive': 0.03866,
        }
        for val_name, val in qois_values.items():
            self.assertAlmostEqual(qois[val_name].n, val, places=3)

    @unittest.skip(
        "[2020-07-12 Victor] This analysis requires to be "
        "upgraded to the new version of the 1Q-RB analysis."
    )
    def test_UnitarityBenchmarking_TwoQubit_Analysis(self):
        a = ma.UnitarityBenchmarking_TwoQubit_Analysis(
            t_start="20180926_110112",
            classification_method="rates",
            rates_ch_idxs=[0, 3],
            nseeds=200,
        )
        u_dec = a.fit_res["unitarity_decay"].params
        self.assertAlmostEqual(u_dec["u"].value, 0.7354, places=3)
        self.assertAlmostEqual(u_dec["eps"].value, 0.1068, places=3)


class Test_CharRBAnalysis:
    def test_char_rb_extract_data(self):

        ts = "20181129_170623"
        a = ma.CharacterBenchmarking_TwoQubit_Analysis(t_start=ts)
        df = a.raw_data_dict["df"]
        assert df.shape == (135, 12)
        assert {"pauli", "interleaving_cl", "ncl"} <= set(df.keys())

        char_df = a.proc_data_dict["char_df"]
        assert {
            "P00",
            "P01",
            "P10",
            "P11",
            "P00_CZ",
            "P01_CZ",
            "P10_CZ",
            "P11_CZ",
            "C1",
            "C2",
            "C12",
            "C1_CZ",
            "C2_CZ",
            "C12_CZ",
        } <= set(char_df.keys())
