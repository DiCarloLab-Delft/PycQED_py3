import unittest
import pycqed as pq
import os
from pycqed.analysis import measurement_analysis as ma


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
            self.assertAlmostEqual(a.fit_res.values['Q'], 1523.119, places=1)
        except AttributeError:
            self.assertAlmostEqual(a.fit_results.values['f0'], 7.1875, places=2)
            self.assertAlmostEqual(a.fit_results.values['Q'], 1523.119, places=1)

        a = ma.Homodyne_Analysis(timestamp='20170929_120456', label='resonator_spec')
        try:
            self.assertAlmostEqual(a.fit_res.values['f0'], 7.4942, places=2)
            self.assertAlmostEqual(a.fit_res.values['Q'], 7430.27187, places=1)
        except AttributeError:
            self.assertAlmostEqual(a.fit_results.values['f0'], 7.4942, places=2)
            self.assertAlmostEqual(a.fit_results.values['Q'], 7430.27187, places=1)

