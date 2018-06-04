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


