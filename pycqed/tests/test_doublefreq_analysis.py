import unittest
import pycqed as pq
import os
from pycqed.analysis import measurement_analysis as ma


class Test_DoubleFreq_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir
        self.ma_obj = ma.DoubleFrequency(label='Ramsey')

    def test_doublefreq_fit(self):
        # Test the correct file is loaded
        self.assertEqual(
            self.ma_obj.folder,
            os.path.join(self.datadir, '20170201', '122018_Ramsey_AncT'))
        fit_res = self.ma_obj.fit_res.best_values

        # Test if the fit gives the expected means
        self.assertAlmostEqual(fit_res['osc_offset'], 0.507, places=3)
        self.assertAlmostEqual(fit_res['phase_1'], -0.006, places=3)
        self.assertAlmostEqual(fit_res['phase_2'], 6.293, places=3)
        self.assertAlmostEqual(fit_res['freq_1']*1e-6, 0.286, places=3)
        self.assertAlmostEqual(fit_res['freq_2']*1e-6, 0.235, places=3)
        self.assertAlmostEqual(fit_res['tau_1']*1e6, 23.8, places=1)
        self.assertAlmostEqual(fit_res['tau_2']*1e6, 15.1, places=1)
        self.assertAlmostEqual(fit_res['amp_1'], 0.25, places=2)
        self.assertAlmostEqual(fit_res['amp_2'], 0.25, places=2)
