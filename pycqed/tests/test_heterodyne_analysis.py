import unittest
import numpy as np
import pycqed as pq
import os
from pycqed.analysis import measurement_analysis as ma
from pycqed.analysis import analysis_toolbox as a_tools

class Test_Heterodyne_analysis(unittest.TestCase):

    @classmethod
    def setUpClass(self):
        self.datadir = os.path.join(pq.__path__[0], 'tests', 'test_data')
        ma.a_tools.datadir = self.datadir
        self.a_spectroscopy = ma.Homodyne_Analysis(label='resonator_scan',
                                                   fitting_model='lorentzian')
        self.a_acq_delay = ma.Acquisition_Delay_Analysis(
            label='acquisition_delay_scan')

    def test_spectroscopy_analysis_lorentzian_model(self):
        # Test the correct file is loaded
        self.assertEqual(self.a_spectroscopy.folder,
                         os.path.join(self.datadir, '20170227',
                                      '115026_resonator_scan_qubit'))

        f0 = self.a_spectroscopy.fit_results.params['f0'].value
        Q = self.a_spectroscopy.fit_results.params['Q'].value

        print("f0 = {}, Q = {}".format(f0, Q))

        # Test if the fit gives the expected means
        self.assertAlmostEqual(f0/1e9, 8.110836918005137, places=3)
        self.assertAlmostEqual(Q/1000, 2.506147979874048, places=1)

    def test_acquisition_delay_analysis(self):
        # Test the correct file is loaded
        self.assertEqual(self.a_acq_delay.folder,
                         os.path.join(self.datadir, '20170227',
                                      '115118_acquisition_delay_scan_qubit'))

        id = a_tools.nearest_idx(self.a_acq_delay.sweep_points,
                                 self.a_acq_delay.max_delay)

        self.assertGreaterEqual(self.a_acq_delay.measured_values[0][id],
                                0.95*max(self.a_acq_delay.measured_values[0]))