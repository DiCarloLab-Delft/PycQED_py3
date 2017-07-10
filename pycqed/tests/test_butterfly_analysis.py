import unittest
import pycqed as pq
import os
from pycqed.analysis import measurement_analysis as ma


class Test_SSRO_discrimination_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_butterfly_postselected(self):
        # Test the correct file is loaded
        a = ma.butterfly_analysis(timestamp='20170710_180002',
                                  close_main_fig=False, initialize=True,
                                  threshold=0.5,
                                  digitize=False, case=True)

        self.assertAlmostEqual(a.butterfly_coeffs['F_a_butterfly'],
                               0.7998, places=3)

    def test_butterfly_simple(self):
        # Test the correct file is loaded
        a = ma.butterfly_analysis(timestamp='20170710_182949',
                                  close_main_fig=False, initialize=False,
                                  threshold=0.5,
                                  digitize=False, case=True)

        self.assertAlmostEqual(a.butterfly_coeffs['F_a_butterfly'],
                               0.819, places=3)
