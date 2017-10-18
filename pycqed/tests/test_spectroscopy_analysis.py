import unittest
import pycqed as pq
import os
from pycqed.analysis import measurement_analysis as ma


class Test_Qubit_spectroscopy_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_Qubit_spectroscopy_analysis_one_peak(self):

        a = ma.Qubit_Spectroscopy_Analysis(timestamp='20170929_175516')
        self.assertAlmostEqual(a.fit_res.values['f0']/1e9, 6.11181, places=2)
        self.assertAlmostEqual(a.fit_res.values['kappa']/1e6, 0.332, places=2)

    def test_Qubit_spectroscopy_analysis_two_peaks(self):

        a = ma.Qubit_Spectroscopy_Analysis(timestamp='20170929_144754', analyze_ef=True)
        self.assertAlmostEqual(a.fit_res.values['f0']/1e9, 6.11089, places=2)
        self.assertAlmostEqual(a.fit_res.values['kappa']/1e6, 16.647, places=2)
        self.assertAlmostEqual(a.fit_res.values['f0_gf_over_2']/1e9, 5.998, places=2)
        self.assertAlmostEqual(a.fit_res.values['kappa_gf_over_2']/1e6, 1.9275, places=2)



