import unittest
import pycqed as pq
import os
from pycqed.analysis_v2 import measurement_analysis as ma
from pycqed.analysis_v2 import enh_T1_analysis as ea

class Test_ehn_T1_analysis(unittest.TestCase):

    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_enh_T1_analysis(self):
        ea.efT1_analysis(
            t_start='20180606_144110', auto=True, close_figs=False)
