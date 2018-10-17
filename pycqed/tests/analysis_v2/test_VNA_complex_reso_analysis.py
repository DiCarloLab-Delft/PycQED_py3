import unittest
import pycqed as pq
import numpy as np
import os
from pycqed.analysis_v2 import measurement_analysis as ma
import matplotlib.pyplot as plt



class Test_VNA_complex_reso_Analysis(unittest.TestCase):
    @classmethod
    def tearDownClass(self):
        plt.close('all')

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_VNA_complex_reso_analysis(self):
        opt_dict = {'fit_options':{'model':'complex'}}
        a = ma.VNA_analysis('20180807_120036', options_dict=opt_dict)
        expected_dict = {
            'f0':5529726415.878748,
            'Q': 9093.722828722679,
            'Qe': 8587.822332782061,
            'A': 1.2276125654908019,
            'theta': 0.43890226314761344,
            'slope':12.88398129325207,
            'phi_v':-4.483641207597703e-07,
            'phi_0':6.28318526488766}
        for key, val in expected_dict.items():
            np.testing.assert_almost_equal(
                a.fit_dicts['reso_fit']['fit_res'].params[key].value, val, decimal=2)

