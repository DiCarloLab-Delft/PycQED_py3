import unittest
import pycqed as pq
import os
import numpy as np
from pycqed.analysis import measurement_analysis as ma


class Test_TwoDAnalysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_incomplete_twoD(self):
        a = ma.TwoD_Analysis(timestamp='20180222_135055')

        extracted_vals = a.measured_values[0][4]

        # critical part here is that nan's get appended
        expected_vals = np.array(
            [148.17970988,  140.84035328,  135.46773387,  140.18987364, np.nan])

        np.testing.assert_array_almost_equal(extracted_vals, expected_vals)

        exp_sweep_points = np.array(
            [2.40000000e-07,   2.45000000e-07,   2.50000000e-07,
             2.55000000e-07,   2.60000000e-07,   2.65000000e-07,
             2.70000000e-07,   2.75000000e-07,   2.80000000e-07,
             2.85000000e-07,   2.90000000e-07,   2.95000000e-07,
             3.00000000e-07,   3.05000000e-07,   3.10000000e-07,
             3.15000000e-07,   3.20000000e-07,   3.25000000e-07,
             3.30000000e-07,   3.35000000e-07,   3.40000000e-07,
             3.45000000e-07,   3.50000000e-07,   3.55000000e-07,
             3.60000000e-07,   3.65000000e-07,   3.70000000e-07,
             3.75000000e-07,   3.80000000e-07,   3.85000000e-07,
             3.90000000e-07,   3.95000000e-07,   4.00000000e-07,
             4.01000000e-07,   4.51000000e-07,   5.01000000e-07,
             5.51000000e-07,   6.01000000e-07,   6.51000000e-07,
             7.01000000e-07,   7.51000000e-07,   8.01000000e-07,
             8.51000000e-07,   9.01000000e-07,   9.51000000e-07])
        np.testing.assert_array_almost_equal(a.sweep_points, exp_sweep_points)

        np.testing.assert_array_almost_equal(a.sweep_points_2D, np.arange(5.))
