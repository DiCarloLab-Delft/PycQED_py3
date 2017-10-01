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
        self.assertAlmostEqual(a.fit_res.values['f0'], 6.111810985426058E9, places=2)
        self.assertAlmostEqual(a.fit_res.values['kappa'], 332986.1894, places=2)

    def test_Qubit_spectroscopy_analysis_two_peaks(self):

        a = ma.Qubit_Spectroscopy_Analysis(timestamp='20170929_144754', analyze_ef=True)
        self.assertAlmostEqual(a.fit_res.values['f0'], 6.110897089343999E9, places=2)
        self.assertAlmostEqual(a.fit_res.values['kappa'], 1.664737312747661E7, places=2)
        self.assertAlmostEqual(a.fit_res.values['f0_gf_over_2'], 5.998399149274546E9, places=2)
        self.assertAlmostEqual(a.fit_res.values['kappa_gf_over_2'], 1927578.1062894785, places=2)

class Test_Homodyne_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir

    def test_Homodyne_analysis_hanger(self):
        #lorentzian model tested in test_heterodyne_analysis.py

        a = ma.Homodyne_Analysis(timestamp='20170929_174145', label='resonator_spec')
        try:
            self.assertAlmostEqual(a.fit_res.values['f0'], 7.1875, places=2)
            self.assertAlmostEqual(a.fit_res.values['Q'], 1523.119, places=2)
        except AttributeError:
            self.assertAlmostEqual(a.fit_results.values['f0'], 7.1875, places=2)
            self.assertAlmostEqual(a.fit_results.values['Q'], 1523.119, places=2)

        a = ma.Homodyne_Analysis(timestamp='20170929_120456', label='resonator_spec')
        try:
            self.assertAlmostEqual(a.fit_res.values['f0'], 7.4942, places=2)
            self.assertAlmostEqual(a.fit_res.values['Q'], 7430.27187, places=2)
        except AttributeError:
            self.assertAlmostEqual(a.fit_results.values['f0'], 7.4942, places=2)
            self.assertAlmostEqual(a.fit_results.values['Q'], 7430.27187, places=2)

