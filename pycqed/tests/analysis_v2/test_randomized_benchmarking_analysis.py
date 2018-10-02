import unittest
import pycqed as pq
import os
from pycqed.analysis_v2 import measurement_analysis as ma


class Test_RBAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_single_qubit_RB_analysis(self):
        a = ma.RandomizedBenchmarking_SingleQubit_Analysis(
            t_start='20180601_135117',
            classification_method='rates', rates_ch_idx=1)

        leak_pars = a.fit_res['leakage_decay'].params
        L1 = leak_pars['L1'].value
        L2 = leak_pars['L2'].value
        self.assertAlmostEqual(L1*100, 0.010309, places=2)
        self.assertAlmostEqual(L2*100, 0.37824, places=2)

        rb_pars = a.fit_res['rb_decay'].params
        F = rb_pars['F'].value
        self.assertAlmostEqual(F, 0.997895, places=4)

    def test_single_qubit_RB_analysis_missing_f_cal(self):
        a = ma.RandomizedBenchmarking_SingleQubit_Analysis(
            t_start='20180815_150417',
            classification_method='rates', rates_ch_idx=0,
            ignore_f_cal_pts=True)

        rb_pars = a.fit_res['rb_decay'].params
        eps = rb_pars['eps'].value
        self.assertAlmostEqual(eps, 0.00236731, places=4)

    def test_two_qubit_RB_analysis_missing_f_cal(self):
        a = ma.RandomizedBenchmarking_TwoQubit_Analysis(
            t_start='20180727_182529',
            classification_method='rates', rates_ch_idxs=[1, 3])

        leak_pars = a.fit_res['leakage_decay'].params
        L1 = leak_pars['L1'].value
        L2 = leak_pars['L2'].value
        self.assertAlmostEqual(L1, 0.029, places=2)
        self.assertAlmostEqual(L2, 0.040, places=2)

        rb_pars = a.fit_res['rb_decay'].params
        eps = rb_pars['eps'].value
        self.assertAlmostEqual(eps, 0.205, places=3)

        rb_pars = a.fit_res['rb_decay_simple'].params
        eps = rb_pars['eps'].value
        self.assertAlmostEqual(eps, 0.157, places=3)

    def test_UnitarityBenchmarking_TwoQubit_Analysis(self):
        a = ma.PurityBenchmarking_TwoQubit_Analysis(
            t_start='20180926_110112',
            classification_method='rates', rates_ch_idxs=[0, 3],
            nseeds=200)
