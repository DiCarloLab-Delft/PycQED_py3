import unittest
import pycqed as pq
import os
from pycqed.analysis_v2 import measurement_analysis as ma
from pycqed.analysis_v2 import Two_state_T1_analysis as Ta


class Test_efT1_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_efT1_analysis(self):
        Ta.efT1_analysis(
            t_start='20180606_144110', auto=True, close_figs=False)
        self.fit_res['fit_res_P0'].params['tau1'].value
