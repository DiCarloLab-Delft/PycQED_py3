import unittest
import pycqed as pq
import os
import matplotlib.pyplot as plt
from pycqed.analysis_v2 import measurement_analysis as ma
from pycqed.analysis_v2 import Two_state_T1_analysis as Ta


class Test_efT1_analysis(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        plt.close('all')

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_efT1_analysis(self):
        b = Ta.efT1_analysis(
            t_start='20180606_144110', auto=True, close_figs=False)
        t1_ef = b.fit_res['fit_res_P2'].params['tau'].value
        t1_eg = b.fit_res['fit_res_P1'].params['tau1'].value
        self.assertAlmostEqual(t1_ef*1e6, 25.698, places=1)
        self.assertAlmostEqual(t1_eg*1e6, 33.393, places=1)
